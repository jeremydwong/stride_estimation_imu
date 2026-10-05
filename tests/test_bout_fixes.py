"""Fix A (walk-back cut detector) and Fix B (chain-back start), 2026-10-03."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import brock_functions as bf


def fake_res(fwd_left, fwd_right, t0=0, end=None, period=0.01, step=50):
    """A pipeline result whose feet land every `step` samples at the given
    forward positions (metres along +x)."""
    res = {'slice': (0, 0), 't_ref': t0, 't0_abs': t0, 'sides': {}}
    n = (max(len(fwd_left), len(fwd_right)) + 2) * step
    for side, fwd in (('left', fwd_left), ('right', fwd_right)):
        ff = np.arange(len(fwd)) * step + (0 if side == 'left' else step // 2)
        P = np.zeros((n, 3))
        P[ff, 0] = fwd
        res['sides'][side] = {'ff_idx': ff}
        res[f'{side}_info'] = NS(P=P)
    res['end_abs'] = end if end is not None else n - 1
    return res


class WalkbackTests(unittest.TestCase):
    def test_straight_walk_is_not_a_walkback(self):
        res = fake_res([0, 1.4, 2.8, 4.2], [0.7, 2.1, 3.5, 4.9])
        hit, back, turn, _ = bf.walkback_info(res, 0.01)
        self.assertFalse(hit)
        self.assertEqual(back, 0)

    def test_walk_back_is_detected_and_turn_is_the_farthest_footfall(self):
        # out to 4.2 m, then three footfalls back
        res = fake_res([0, 1.4, 2.8, 4.2, 3.0, 1.6], [0.7, 2.1, 3.5, 4.0, 2.4])
        hit, back, turn, _ = bf.walkback_info(res, 0.01)
        self.assertTrue(hit)
        self.assertGreaterEqual(back, 2)
        self.assertEqual(turn, 3 * 50 + 25)     # later of the feet's farthest (right, 4.0 m)

    def test_duration_test_needs_the_target(self):
        # straight, but 20 s of "gait" for a 2.5 m walk
        res = fake_res([0, 1.4], [0.7, 2.1], end=2000)
        self.assertFalse(bf.walkback_info(res, 0.01)[0])
        hit, _, _, ratio = bf.walkback_info(res, 0.01, expected_m=2.5)
        self.assertTrue(hit)
        self.assertGreater(ratio, bf.WALKBACK_TIME_RATIO)


class ChainBackTests(unittest.TestCase):
    def speeds(self, segments, n):
        v = np.zeros(n)
        for a, b in segments:
            v[a:b] = 2.0
        return {'left': v, 'right': np.zeros(n)}

    def test_chain_of_steps_into_the_start_is_followed_back(self):
        # gait start t0 = 600 (abs); swings at 380-430 and 480-560 chain into it
        t0, i0, p = 600, 600, 0.01
        a = max(0, i0 - int((bf.CHAIN_MAX_S + 1) / p))
        V = self.speeds([(380 - a, 430 - a), (480 - a, 560 - a)], t0 + 1 - a)
        with patch.object(bf, '_foot_speeds', return_value=V):
            self.assertEqual(bf.chain_back_start({}, 's1', i0, t0, p), 380)

    def test_isolated_earlier_movement_is_ignored(self):
        # one swing 2 s before the start, then quiet: not a chain
        t0, i0, p = 600, 600, 0.01
        a = max(0, i0 - int((bf.CHAIN_MAX_S + 1) / p))
        V = self.speeds([(350 - a, 400 - a)], t0 + 1 - a)
        with patch.object(bf, '_foot_speeds', return_value=V):
            self.assertIsNone(bf.chain_back_start({}, 's1', i0, t0, p))

    def test_unmechanizable_window_returns_none(self):
        with patch.object(bf, '_foot_speeds', return_value=None):
            self.assertIsNone(bf.chain_back_start({}, 's1', 600, 600, 0.01))


if __name__ == '__main__':
    unittest.main()


class StepSpeedTests(unittest.TestCase):
    """Step speed = midpoint-between-the-feet progression over each step
    (touchdown of one foot -> touchdown of the other), never stride speed."""
    def test_hand_built_walk(self):
        from stride_imu import inertial as I
        n, p = 400, 0.01
        def foot(touch, moves):
            x = np.zeros(n)
            for a, b, x0, x1 in moves:                     # linear swings
                x[a:b] = np.linspace(x0, x1, b - a)
                x[b:] = x1
            stat = np.zeros(n, bool)
            ffw = np.zeros(n, bool)
            for t in touch:
                stat[t:t + 15] = True
                ffw[t] = True
            return NS(P=np.c_[x, np.zeros(n), np.zeros(n)], Vm=np.zeros(n),
                      FF_walking=ffw, stationary_periods=stat, euler=np.zeros((n, 3)))
        # right steps first: 0 -> 0.6 m (step 1), left 0 -> 1.2 (step 2),
        # right 0.6 -> 1.8 (step 3); each step 1 s
        L = foot([0, 200], [(120, 200, 0.0, 1.2)])
        R = foot([0, 100, 300], [(20, 100, 0.0, 0.6), (220, 300, 0.6, 1.8)])
        frame = lambda Li, Ri, *a, **k: (np.c_[np.zeros(n), Li.P[:, 0], np.zeros(n)],
                                         np.c_[np.zeros(n), Ri.P[:, 0], np.zeros(n)], 0, 0.0)
        with patch.object(I, '_common_frame', side_effect=frame):
            st = I.steps_from_footfalls(L, R, p, force_snap=0)
        self.assertEqual(list(st['leading_foot']), ['right', 'left', 'right'])
        # midpoint moves half the swinging foot's travel per 1 s step
        np.testing.assert_allclose(st['frwd_speed'], [0.3, 0.6, 0.6], atol=1e-9)
        np.testing.assert_allclose(st['time'], [1.0, 1.0, 1.0])

    def test_no_step_ends_before_the_gait_start(self):
        from stride_imu import inertial as I
        n, p = 400, 0.01
        def foot(touch):
            stat = np.zeros(n, bool); ffw = np.zeros(n, bool)
            for t in touch:
                stat[t:t + 15] = True; ffw[t] = True
            return NS(P=np.zeros((n, 3)), Vm=np.zeros(n), FF_walking=ffw,
                      stationary_periods=stat, euler=np.zeros((n, 3)))
        L, R = foot([0, 60, 200]), foot([30, 100, 300])     # a shuffle before g
        frame = lambda Li, Ri, *a, **k: (np.zeros((n, 3)), np.zeros((n, 3)), 0, 0.0)
        with patch.object(I, '_common_frame', side_effect=frame):
            st = I.steps_from_footfalls(L, R, p, force_snap=80)
        self.assertTrue((np.asarray(st['end_idx']) > 80).all())
        self.assertTrue((np.asarray(st['start_idx']) >= 80).all())
