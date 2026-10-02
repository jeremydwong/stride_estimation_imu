"""Cue schedule + conductor transport, and the host->sensor clock."""
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stride_imu.live.clock import SensorClock  # noqa: E402
from stride_imu.live.schedule import (BrockTiming, Conductor, Schedule,  # noqa: E402
                                      build_brock_schedule)

COND = pd.DataFrame({'trial': [1, 2, 3], 'distance_m': [5.0, 12.0, 2.5],
                     'package': ['ring', 'medium', 'small'],
                     'pose': ['ground', 'held out', 'waist']})


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ScheduleTests(unittest.TestCase):
    def test_protocol_walkers_and_events(self):
        s = build_brock_schedule(COND)
        self.assertEqual(len(s.blocks), 6)                     # 3 trials x 2 reps
        goes = [c for c in s.cues if c.kind == 'go']
        self.assertTrue(all(c.event == 'Start' for c in goes))
        self.assertEqual(sum(c.event == 'Stop' for c in s.cues), 12)
        # rep 1: a walks bout 1; rep 2 flipped
        w = {(c.trial, c.rep, c.bout): c.walker for c in goes}
        self.assertEqual(w[(1, 1, 1)], 'a')
        self.assertEqual(w[(1, 1, 2)], 'b')
        self.assertEqual(w[(1, 2, 1)], 'b')
        # stop timed from distance
        T = BrockTiming()
        go = next(c for c in s.cues if (c.trial, c.rep, c.bout, c.kind) == (2, 1, 1, 'go'))
        st = next(c for c in s.cues if (c.trial, c.rep, c.bout, c.kind) == (2, 1, 1, 'stop'))
        self.assertAlmostEqual(st.t_s - go.t_s, 12.0 / T.walk_speed_ms + T.handoff_s)

    def test_csv_roundtrip(self):
        s = build_brock_schedule(COND)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 's.csv')
            s.save_csv(p)
            s2 = Schedule.load_csv(p)
        self.assertEqual([(c.t_s, c.text, c.event, c.trial) for c in s.cues],
                         [(c.t_s, c.text, c.event, c.trial) for c in s2.cues])


class ConductorTests(unittest.TestCase):
    def setUp(self):
        self.s = build_brock_schedule(COND)
        self.clk = FakeClock()
        self.c = Conductor(self.s, clock=self.clk)

    def run_to(self, t):
        out = []
        while self.clk.t < t:
            self.clk.t += 0.05
            out += self.c.tick()
        return out

    def test_emits_in_order_and_pauses(self):
        self.c.play()
        em = self.run_to(self.s.blocks[0].t_end + 0.1)
        kinds = [e.cue.kind for e in em]
        self.assertEqual(kinds, ['announce', 'ready', 'go', 'stop', 'ready', 'go', 'stop'])
        self.c.pause()
        t_paused = self.c.exp_t
        self.assertEqual(self.run_to(self.clk.t + 30), [])      # nothing while paused
        self.assertAlmostEqual(self.c.exp_t, t_paused)
        self.c.play()
        self.assertTrue(self.run_to(self.clk.t + 30))

    def test_back_trial_replays_with_new_attempt(self):
        self.c.play()
        self.run_to(self.s.blocks[1].t_start + 2)                # into trial 2
        self.c.back_trial()                                       # -> trial 1
        self.assertLess(self.c.exp_t, self.s.blocks[0].t_start)
        em = self.run_to(self.clk.t + 3)
        self.assertEqual(em[0].cue.kind, 'announce')
        self.assertEqual(em[0].cue.trial, 1)
        self.assertEqual(em[0].attempt, 2)

    def test_restart_and_next(self):
        self.c.play()
        self.run_to(self.s.blocks[0].t_start + 6)
        self.c.restart_trial()
        self.assertAlmostEqual(self.c.exp_t, self.s.blocks[0].t_start - self.c.preroll_s)
        self.c.next_trial()
        self.assertAlmostEqual(self.c.exp_t, self.s.blocks[1].t_start - self.c.preroll_s)


class SensorClockTests(unittest.TestCase):
    def test_min_latency_offset(self):
        clk = SensorClock()
        # sensor clock is host - 5 s; packets arrive with 20-200 ms latency
        for k, lat in enumerate([0.2, 0.05, 0.02, 0.15]):
            host = 100.0 + k
            clk.observe(int((host - 5.0 - lat) * 1e6), host_s=host)
        self.assertAlmostEqual(clk.sensor_now(200.0) / 1e6, 195.0, delta=0.021)


if __name__ == '__main__':
    unittest.main()
