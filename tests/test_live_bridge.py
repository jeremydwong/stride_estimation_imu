"""BridgeSource line protocol + HdfRecorder round trip, without hardware.

A tiny Python stand-in for bridge/ApdmBridge.java prints the same protocol
(M/S/B/X/I lines) and echoes stdin 'gpio' commands, so the parser, the event
plumbing and the recorder are exercised end to end.
"""
import os
import sys
import tempfile
import textwrap
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import stride_imu as imu  # noqa: E402
from stride_imu.live import BridgeSource, HdfRecorder, LivePipeline  # noqa: E402

FAKE = textwrap.dedent('''
    import sys, math
    print('M {"devices": {"101": "left_foot_a", "102": "right_foot_a"}, "rate": 128}', flush=True)
    print('I fake bridge', file=sys.stderr, flush=True)
    t0 = 1783517928158000
    for i in range(640):
        for dev in (101, 102):
            g = 0.5 * math.sin(i / 20.0)
            print(f"S {dev} {t0 + i * 7812} 0.1 0.2 9.8 {g} 0 0 10 20 30 0")
        if i == 300:
            print(f"B 101 {t0 + i * 7812} 1 Start")
            print(f"X 21710 {t0 + i * 7812} 0 1")
    sys.stdout.flush()
    for line in sys.stdin:
        if line.startswith('gpio'):
            print('I ' + line.strip(), flush=True)
        if line.startswith('quit'):
            break
''')


class BridgeProtocolTests(unittest.TestCase):
    def test_parse_record_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, 'fake_bridge.py')
            with open(fake, 'w') as f:
                f.write(FAKE)
            src = BridgeSource(command=[sys.executable, fake])
            src.start()
            self.assertEqual(src.devices, {101: 'left_foot_a', 102: 'right_foot_a'})
            self.assertAlmostEqual(src.period, 1 / 128)
            out = os.path.join(d, 'rec.h5')
            rec = HdfRecorder(out, src.devices, src.period)
            pipe = LivePipeline(src, rec)
            pipe.setup()
            t_end = time.monotonic() + 10
            while time.monotonic() < t_end and min(f.n for f in pipe.feet.values()) < 640:
                pipe.step()
                time.sleep(0.01)
            src.set_output(0, 1)
            time.sleep(0.3)
            pipe.step()
            src.stop()
            rec.close()

            kinds = [(e.kind, e.text) for e in pipe.events]
            self.assertIn(('button', 'Start'), kinds)
            self.assertIn(('sync_in', 'pin 0 = 1'), kinds)
            self.assertTrue(any(k == 'status' and 'gpio 0 1' in t for k, t in kinds))
            self.assertIn('I fake bridge', list(src.stderr_tail))

            # the recording opens with the normal loader, sensor frame preserved
            sensors = imu.list_sensors(out)
            self.assertEqual(sensors, {'XI-000101': 'left_foot_a', 'XI-000102': 'right_foot_a'})
            r = imu.load_imu_recording(out, sensor_id='XI-000101')
            self.assertEqual(len(r), 640)
            self.assertEqual(r.raw_time[1] - r.raw_time[0], 7812)
            # LED_UP_RIGHT_FRWD mapping: body X = sensor Y, Z = -sensor Z
            np.testing.assert_allclose(r.Ab[0], [0.2, 0.1, -9.8])


if __name__ == '__main__':
    unittest.main()
