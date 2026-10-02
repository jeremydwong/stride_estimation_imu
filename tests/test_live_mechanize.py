"""Live (causal) mechanization must reproduce offline compute_position exactly."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import stride_imu as imu  # noqa: E402
from stride_imu.live import LiveFoot, run_live  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PILOT = os.path.join(DATA, '20251029-154305_LF_Pilot_Ch_Oct29.h5')


def _synthetic(period=1 / 128, n_strides=6):
    """Stance/swing pattern: 0.5 s still, 0.6 s of pitching swing."""
    rng = np.random.default_rng(0)
    W, A = [], []
    for _ in range(n_strides):
        ns, nw = int(0.5 / period), int(0.6 / period)
        W.append(rng.normal(0, 1e-4, (ns, 3)))
        A.append(np.tile([0, 0, 9.80297286843], (ns, 1)) + rng.normal(0, 0.02, (ns, 3)))
        t = np.linspace(0, np.pi, nw)
        w = np.zeros((nw, 3))
        w[:, 1] = 4 * np.sin(2 * t) * period
        a = np.tile([0, 0, 9.8], (nw, 1)) + np.column_stack([8 * np.sin(2 * t), 0 * t, 3 * np.sin(t)])
        W.append(w)
        A.append(a)
    return np.vstack(W), np.vstack(A), period


class LiveEqualsOfflineTests(unittest.TestCase):
    def _check(self, W, A, period):
        off = imu.compute_position(W, A, period)
        live = run_live(W, A, period)
        np.testing.assert_array_equal(off.FF, live.FF)
        np.testing.assert_array_equal(off.stationary_periods, live.stationary_periods)
        for k in ('quaternion', 'An', 'Anz', 'V', 'P'):
            np.testing.assert_array_equal(getattr(off, k), getattr(live, k), err_msg=k)
        # the emitted blocks themselves (not just the reassembled arrays)
        foot = LiveFoot(period)
        blocks = []
        for i in range(len(W)):
            foot.push(W[i], A[i])
            blocks += foot.pop_blocks()
        foot.finish()
        blocks += foot.pop_blocks()
        self.assertGreater(len(blocks), 2)
        for b in blocks:
            np.testing.assert_array_equal(b.V, off.V[b.start:b.stop + 1])
            np.testing.assert_array_equal(b.P, off.P[b.start:b.stop + 1])
        return foot, blocks

    def test_synthetic(self):
        self._check(*_synthetic())

    @unittest.skipUnless(os.path.exists(PILOT), 'pilot data not present')
    def test_pilot_recording(self):
        rec = imu.load_imu_recording(PILOT)[20000:26000]
        self._check(rec.Wb, rec.Ab, rec.period)

    def test_blocks_arrive_causally(self):
        """A stride's block must be emitted before the stream ends, and
        never before its footfall sample has been pushed."""
        W, A, period = _synthetic()
        foot = LiveFoot(period)
        emitted_at = []
        for i in range(len(W)):
            foot.push(W[i], A[i])
            for b in foot.pop_blocks():
                emitted_at.append((b.stop, i))
        self.assertTrue(emitted_at)
        for stop, i in emitted_at:
            self.assertLessEqual(stop, i)
            # decided within one plateau + T_FF (plateaus here are 0.5 s)
            self.assertLess((i - stop) * period, 1.0)


if __name__ == '__main__':
    unittest.main()
