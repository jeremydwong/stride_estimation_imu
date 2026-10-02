"""Where live samples come from.

Every source speaks the same small interface so the viewer / recorder never
know whether they are looking at hardware or a replay:

    src.start()
    src.devices    -> {device_id: label}        (after start)
    src.period     -> sampling period [s]
    src.poll()     -> (list[SampleBlock], list[Event])   non-blocking
    src.stop()

Samples are in the SENSOR frame and SI units, exactly as APDM stores them in
the .h5 (gyro rad/s, accel m/s², mag µT) with epoch-microsecond timestamps;
``to_body`` applies the same orientation mapping ``load_imu_recording`` uses.

* ``ReplaySource``  – plays a recorded session .h5 back at wall-clock speed
  (x ``speed``), including its Annotations (the 'event' Opal's Start/Stop
  button presses) as button events. Lets everything be developed and tested
  without the access point.
* ``BridgeSource``  – the real hardware: runs the Java bridge
  (bridge/ApdmBridge.java, APDM's own apdm.jar + libapdm.dylib) as a child
  process and parses its line protocol. See README.md in this folder.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..apdm import _apply_orientation


@dataclass
class SampleBlock:
    device_id: int
    t_us: np.ndarray          # (n,) uint64 epoch microseconds
    gyro: np.ndarray          # (n, 3) rad/s, sensor frame
    accel: np.ndarray         # (n, 3) m/s^2, sensor frame
    mag: np.ndarray           # (n, 3)


@dataclass
class Event:
    t_us: int                 # epoch microseconds (sensor clock) or host clock for 'key'
    device_id: int            # 0 for host-generated events
    kind: str                 # 'button' | 'sync_in' | 'key' | 'status'
    text: str = ''
    host_time: float = field(default_factory=time.time)


def to_body(block: SampleBlock, period: float, orientation: Optional[int] = None):
    """(W rad/sample, A m/s^2) in the body frame compute_position expects."""
    W, A, _ = _apply_orientation(block.gyro, block.accel, block.mag, period, orientation)
    return W, A


def _dev_num(sensor_id: str) -> int:
    """'XI-021789' -> 21789 (the number APDM uses as 'Sensor ID')."""
    return int(sensor_id.split('-')[-1])


class ReplaySource:
    """Replay [start_s, start_s + duration_s) of a session file in real time.

    ``patterns`` selects sensors by label substring (default: every foot);
    the 'event' Opal's Annotations are replayed as ``Event(kind='button')``.
    ``speed`` > 1 plays faster than real time.
    """

    def __init__(self, path: str, start_s: float = 0.0, duration_s: float = 60.0,
                 patterns: Sequence[str] = ('foot',), speed: float = 1.0,
                 block_s: float = 0.02):
        self.path = path
        self.start_s = start_s
        self.duration_s = duration_s
        self.patterns = tuple(patterns)
        self.speed = speed
        self.block_s = block_s
        self.devices: Dict[int, str] = {}
        self._data = {}
        self._events: List[Event] = []

    def start(self):
        with h5py.File(self.path, 'r') as f:
            sensors = f['Sensors']
            sid0 = next(iter(sensors))
            self.period = 1.0 / float(sensors[sid0]['Configuration'].attrs['Sample Rate'])
            t_all = sensors[sid0]['Time']
            i0 = int(self.start_s / self.period)
            i1 = min(i0 + int(self.duration_s / self.period), t_all.shape[0])
            for sid in sensors:
                lab = sensors[sid]['Configuration'].attrs.get('Label 0', b'')
                lab = lab.decode() if isinstance(lab, bytes) else str(lab)
                if not any(p in lab for p in self.patterns):
                    continue
                g = sensors[sid]
                dev = _dev_num(sid)
                self.devices[dev] = lab
                self._data[dev] = (g['Time'][i0:i1], g['Gyroscope'][i0:i1],
                                   g['Accelerometer'][i0:i1], g['Magnetometer'][i0:i1])
            t_lo, t_hi = int(t_all[i0]), int(t_all[i1 - 1])
            if 'Annotations' in f:
                for row in f['Annotations'][:]:
                    if t_lo <= row['Time'] <= t_hi:
                        self._events.append(Event(int(row['Time']), int(row['Sensor ID']),
                                                  'button', row['Annotation'].decode()))
        self._events.sort(key=lambda e: e.t_us)
        self.n = i1 - i0
        self.t0_us = t_lo
        self._pos = 0
        self._wall0 = time.monotonic()

    @property
    def finished(self) -> bool:
        return self._pos >= self.n

    def poll(self) -> Tuple[List[SampleBlock], List[Event]]:
        elapsed = (time.monotonic() - self._wall0) * self.speed
        upto = min(self.n, int(elapsed / self.period))
        if upto <= self._pos:
            return [], []
        lo, hi = self._pos, upto
        self._pos = upto
        blocks = [SampleBlock(dev, t[lo:hi], w[lo:hi], a[lo:hi], m[lo:hi])
                  for dev, (t, w, a, m) in self._data.items()]
        t_hi = int(blocks[0].t_us[-1])
        evs = []
        while self._events and self._events[0].t_us <= t_hi:
            evs.append(self._events.pop(0))
        return blocks, evs

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Hardware: the Java bridge
# ---------------------------------------------------------------------------
MOTION_STUDIO = '/Applications/MotionStudio.app/Contents/Resources'
SDK_DIR = os.path.join(MOTION_STUDIO, 'configuration/org.eclipse.osgi/6/0/.cp/apdm_sdk')
BUNDLED_JAVA = os.path.join(MOTION_STUDIO, 'jre/Contents/Home/jre/bin/java')
BRIDGE_CLASSES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bridge', 'build')


def bridge_command(mode: str, *args: str, sdk_dir: str = SDK_DIR,
                   java: str = BUNDLED_JAVA) -> List[str]:
    """The java command line for bridge/ApdmBridge (mode: 'probe',
    'configure' or 'stream'). libapdm.dylib is x86_64 only, so the JVM must be
    x86_64 too — Motion Studio's bundled JRE is, and runs under Rosetta."""
    cp = os.pathsep.join([BRIDGE_CLASSES, os.path.join(sdk_dir, 'java', 'apdm.jar')])
    lib = os.path.join(sdk_dir, 'libs', 'MacOSX', 'x64')
    return [java, f'-Djava.library.path={lib}', '-cp', cp, 'ApdmBridge', mode, *args]


class BridgeSource:
    """Live Opals through the access point (Java bridge child process).

    Line protocol on the bridge's stdout (one record per line, space-separated):
        M <json>                                  metadata: {"devices": {id: label}, "rate": hz}
        S <dev> <t_us> ax ay az gx gy gz mx my mz <button>
        B <dev> <t_us> <event_code> <text>        Opal button event
        X <ap> <sync> <pin> <value>               sync-box input edge
        I <text>                                  info / status
        E <text>                                  error
    Commands on its stdin: 'gpio <pin> <0|1>', 'quit'.
    """

    def __init__(self, latency_ms: int = 250, command: Optional[List[str]] = None):
        self.latency_ms = latency_ms
        self.command = command or bridge_command('stream', str(latency_ms))
        self.devices: Dict[int, str] = {}
        self.period: Optional[float] = None
        self.messages: List[str] = []
        self.stderr_tail: 'deque[str]' = deque(maxlen=200)
        self._q: 'queue.Queue[str]' = queue.Queue()

    def start(self, timeout_s: float = 20.0):
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()
        t_end = time.monotonic() + timeout_s
        while self.period is None:
            if time.monotonic() > t_end or self.proc.poll() is not None:
                raise RuntimeError('bridge did not report its devices:\n'
                                   + '\n'.join(self.messages[-20:] + list(self.stderr_tail)[-10:]))
            try:
                self._handle_meta(self._q.get(timeout=0.1))
            except queue.Empty:
                pass

    def _reader(self):
        for line in self.proc.stdout:
            self._q.put(line.rstrip('\n'))

    def _stderr_reader(self):
        # libapdm's own log lines; kept (last 200) for diagnosing a failed start
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip('\n'))

    def _handle_meta(self, line: str):
        if line.startswith('M '):
            meta = json.loads(line[2:])
            self.devices = {int(k): v for k, v in meta['devices'].items()}
            self.period = 1.0 / float(meta['rate'])
        else:
            self.messages.append(line)

    def poll(self):
        rows: Dict[int, list] = {}
        events: List[Event] = []
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                break
            tag = line[:1]
            if tag == 'S':
                p = line.split()
                rows.setdefault(int(p[1]), []).append(p[2:])
            elif tag == 'B':
                p = line.split(maxsplit=4)
                events.append(Event(int(p[2]), int(p[1]), 'button', p[4] if len(p) > 4 else p[3]))
            elif tag == 'X':
                p = line.split()
                events.append(Event(int(p[2]), int(p[1]), 'sync_in', f'pin {p[3]} = {p[4]}'))
            elif tag in 'IE':
                self.messages.append(line)
                events.append(Event(int(time.time() * 1e6), 0, 'status', line))
            else:
                self._handle_meta(line)
        blocks = []
        for dev, r in rows.items():
            arr = np.array(r, dtype=float)
            blocks.append(SampleBlock(dev, arr[:, 0].astype(np.uint64), arr[:, 4:7],
                                      arr[:, 1:4], arr[:, 7:10]))
        return blocks, events

    def set_output(self, pin: int, value: int):
        """Drive an access-point / sync-box output line (e.g. to mark an event
        on a force plate or mocap system)."""
        self.proc.stdin.write(f'gpio {pin} {value}\n')
        self.proc.stdin.flush()

    def stop(self):
        if getattr(self, 'proc', None) and self.proc.poll() is None:
            try:
                self.proc.stdin.write('quit\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        for pipe in (getattr(self, 'proc', None) and (self.proc.stdin, self.proc.stdout, self.proc.stderr)) or ():
            try:
                pipe.close()
            except Exception:
                pass
