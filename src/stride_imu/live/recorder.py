"""Host-side recording of a live stream, in APDM's own .h5 layout.

The file mirrors APDM file format v5 closely enough that the existing loaders
(``load_imu_recording``, ``list_sensors``, ``brock_functions.load_available_feet``
and the Annotations reader) open it unchanged:

    /Sensors/XI-<id>/{Accelerometer, Gyroscope, Magnetometer, Time}
    /Sensors/XI-<id>/Configuration  attrs: 'Label 0', 'Sample Rate'
    /Annotations  (Time u8, Sensor ID u4, Annotation S2048)

Samples arrive per device in blocks; each device's datasets grow
independently (sample alignment across devices is by Time, as in APDM files
produced from streaming). Data is flushed every ``flush_s`` seconds so a crash
loses at most that much.
"""
from __future__ import annotations

import time
from typing import Dict

import h5py
import numpy as np

from .sources import Event, SampleBlock

ANNOTATION_DTYPE = np.dtype([('Time', '<u8'), ('Sensor ID', '<u4'), ('Annotation', 'S2048')])


class HdfRecorder:
    def __init__(self, path: str, devices: Dict[int, str], period: float,
                 flush_s: float = 2.0):
        self.path = path
        self.f = h5py.File(path, 'w')
        self.f.attrs['FileFormatVersion'] = 5
        self.f.attrs['RecordedBy'] = 'stride_imu.live.HdfRecorder'
        self.f.create_group('Processed')
        self.ds = {}
        for dev, label in devices.items():
            g = self.f.create_group(f'Sensors/XI-{dev:06d}')
            cfg = g.create_group('Configuration')
            cfg.attrs['Label 0'] = label.encode()
            cfg.attrs['Sample Rate'] = int(round(1.0 / period))
            self.ds[dev] = {
                'Time': g.create_dataset('Time', (0,), '<u8', maxshape=(None,), chunks=(8000,)),
                'Accelerometer': g.create_dataset('Accelerometer', (0, 3), '<f8', maxshape=(None, 3), chunks=(8000, 3)),
                'Gyroscope': g.create_dataset('Gyroscope', (0, 3), '<f8', maxshape=(None, 3), chunks=(8000, 3)),
                'Magnetometer': g.create_dataset('Magnetometer', (0, 3), '<f8', maxshape=(None, 3), chunks=(8000, 3)),
            }
        self.ann = self.f.create_dataset('Annotations', (0,), ANNOTATION_DTYPE,
                                         maxshape=(None,), chunks=(64,))
        self.flush_s = flush_s
        self._last_flush = time.monotonic()

    @staticmethod
    def _append(ds, x):
        n = ds.shape[0]
        ds.resize(n + len(x), axis=0)
        ds[n:] = x

    def write(self, block: SampleBlock):
        d = self.ds.get(block.device_id)
        if d is None or len(block.t_us) == 0:
            return
        self._append(d['Time'], block.t_us)
        self._append(d['Accelerometer'], block.accel)
        self._append(d['Gyroscope'], block.gyro)
        self._append(d['Magnetometer'], block.mag)
        self._maybe_flush()

    def annotate(self, ev: Event):
        # Opal buttons and schedule cues keep their bare text ('Start'/'Stop'),
        # which is what the Brock annotation readers look for
        text = ev.text if ev.kind in ('button', 'cue') else f'{ev.kind}:{ev.text}'
        row = np.array([(ev.t_us, ev.device_id, text.encode()[:2048])],
                       dtype=ANNOTATION_DTYPE)
        self._append(self.ann, row)
        self.f.flush()                    # events are rare and precious

    def _maybe_flush(self):
        if time.monotonic() - self._last_flush > self.flush_s:
            self.f.flush()
            self._last_flush = time.monotonic()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


class EventLog:
    """Sidecar CSV of every event with its full context (one row per event,
    flushed immediately): host time, Opal-clock time, experiment time,
    trial/rep/bout/walker/attempt for cues, free text for notes."""

    COLUMNS = ['host_iso', 'host_s', 'sensor_us', 'exp_t', 'kind', 'text', 'event',
               'trial', 'rep', 'bout', 'walker', 'attempt', 'device_id']

    def __init__(self, path: str):
        import csv
        self.path = path
        self._f = open(path, 'a', newline='')
        self._w = csv.DictWriter(self._f, fieldnames=self.COLUMNS)
        if self._f.tell() == 0:
            self._w.writeheader()

    def write(self, **row):
        import datetime
        host = row.get('host_s', time.time())
        row.setdefault('host_s', host)
        row.setdefault('host_iso', datetime.datetime.fromtimestamp(host).isoformat(timespec='milliseconds'))
        self._w.writerow({k: row.get(k, '') for k in self.COLUMNS})
        self._f.flush()

    def close(self):
        self._f.close()
