"""Offscreen smoke test of the live experiment window (skipped without PySide6)."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from PySide6 import QtCore, QtWidgets
    HAVE_QT = True
except ImportError:                                  # pragma: no cover
    HAVE_QT = False


class SyntheticSource:
    """Four still feet, 128 Hz, paced to the wall clock like real hardware."""
    period = 1 / 128

    def __init__(self):
        self.devices = {1: 'left_foot_a', 2: 'right_foot_a', 3: 'left_foot_b', 4: 'right_foot_b'}
        self.t_us = 1_783_517_928_000_000
        self.rng = np.random.default_rng(0)

    def start(self):
        self.wall0 = time.monotonic()
        self.sent = 0

    def poll(self):
        from stride_imu.live import SampleBlock
        n = int((time.monotonic() - self.wall0) / self.period) - self.sent
        if n <= 0:
            return [], []
        self.sent += n
        t = self.t_us + np.arange(n) * 7812
        self.t_us = int(t[-1]) + 7812
        blocks = [SampleBlock(d, t.astype(np.uint64), self.rng.normal(0, 0.01, (n, 3)),
                              np.tile([0, 0, -9.803], (n, 1)), np.zeros((n, 3))) for d in self.devices]
        return blocks, []

    def stop(self):
        pass


@unittest.skipUnless(HAVE_QT, 'PySide6 not installed (uv sync --group live)')
class ExperimentWindowSmokeTest(unittest.TestCase):
    def test_runs_emits_and_logs(self):
        from stride_imu.live import LivePipeline
        from stride_imu.live.app import ExperimentWindow
        from stride_imu.live.schedule import BrockTiming, build_brock_schedule

        cond = pd.DataFrame({'trial': [1, 2], 'distance_m': [2.5, 5.0],
                             'package': ['ring', 'small'], 'pose': ['ground', 'waist']})
        timing = BrockTiming(start_offset_s=0.2, announce_to_ready_s=0.2, ready_to_go_s=0.2,
                             walk_speed_ms=10, handoff_s=0.1, stop_to_next_ready_s=0.1,
                             between_trials_s=0.3)
        sch = build_brock_schedule(cond, reps=[1], timing=timing)
        src = SyntheticSource()
        src.start()
        pipe = LivePipeline(src)
        pipe.setup()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as d:
            w = ExperimentWindow(pipe, sch, cond, out_dir=d, mute=True)
            w.show()
            w.conductor.play()
            QtCore.QTimer.singleShot(int((sch.end_s + 1.5) * 1000), app.quit)
            app.exec()
            w.close()
            ev = pd.read_csv(os.path.join(d, 'events.csv'))
            self.assertTrue(os.path.exists(os.path.join(d, 'roles.json')))
        self.assertEqual(list(ev.loc[ev['event'].notna(), 'event']), ['Start', 'Stop'] * 4)
        cue_events = [e for e in pipe.events if e.kind == 'cue']
        self.assertEqual(len(cue_events), 8)
        # stamped in the (synthetic) Opal clock, which is ~83 days behind the host
        # clock here: every cue must land inside the streamed sample times
        t_first = 1_783_517_928_000_000
        self.assertTrue(all(t_first <= e.t_us <= src.t_us + 1e5 for e in cue_events))
        self.assertEqual(w.roles[1], 'A left')


if __name__ == '__main__':
    unittest.main()
