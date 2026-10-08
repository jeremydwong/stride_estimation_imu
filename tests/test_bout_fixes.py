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
    """chain_back_start reads foot speed from compute_position_two_imus over
    [t0 - CHAIN_MAX_S - 1 s, t0 + END_LOOKAHEAD_S); the engine is faked."""
    t0, i0, p = 600, 600, 0.01

    def run_with(self, segments, floor=None):
        a = max(0, self.i0 - int((bf.CHAIN_MAX_S + 1) / self.p))
        n = self.t0 + int(round(bf.END_LOOKAHEAD_S / self.p)) - a
        v = np.zeros(n)
        for s0, e0 in segments:
            v[s0 - a:e0 - a] = 2.0
        feet = {('s1', side): NS(Wb=np.zeros((5000, 3)),
                                               Ab=np.zeros((5000, 3)))
                for side in ('left', 'right')}
        engine = lambda *args: (NS(Vm=v),
                                NS(Vm=np.zeros(n)))
        with patch.object(bf.imu, 'compute_position_two_imus', engine):
            return bf.chain_back_start(feet, 's1', self.i0, self.t0, self.p,
                                       floor=floor)

    def test_chain_of_steps_into_the_start_is_followed_back(self):
        # swings at 380-430 and 480-560 chain into the gait start at 600
        self.assertEqual(self.run_with([(380, 430), (480, 560)]), 380)

    def test_swing_in_progress_at_the_start_is_followed_back(self):
        # the search floor (595) cut a swing that began at 560: the gait
        # start (600) falls inside it
        self.assertEqual(self.run_with([(500, 540), (560, 640)], floor=595), 500)

    def test_swing_in_progress_away_from_the_floor_is_ignored(self):
        # same in-progress swing, but the start is far from the floor: only
        # swings that END by the start count, and the 450-490 one is 1.1 s back
        self.assertIsNone(self.run_with([(450, 490), (560, 640)], floor=450))

    def test_isolated_earlier_movement_is_ignored(self):
        # one swing 2 s before the start, then quiet: not a chain
        self.assertIsNone(self.run_with([(350, 400)]))

    def test_unmechanizable_window_returns_none(self):
        def boom(*args):
            raise ValueError('no stance')
        feet = {('s1', side): NS(Wb=np.zeros((5000, 3)),
                                               Ab=np.zeros((5000, 3)))
                for side in ('left', 'right')}
        with patch.object(bf.imu, 'compute_position_two_imus', boom):
            self.assertIsNone(bf.chain_back_start(feet, 's1', 600, 600, 0.01))


class EndChainTests(unittest.TestCase):
    """snug_end(start=, max_gap=): shuffles after a pause don't extend the gait."""
    def speeds(self, segments, n=200):
        v = np.zeros(n)
        for s0, e0 in segments:
            v[s0:e0] = np.r_[np.linspace(0.3, 2.0, (e0 - s0) // 2),
                             np.linspace(2.0, 0.3, e0 - s0 - (e0 - s0) // 2)]
        return NS(Vm=v)

    def test_shuffle_after_a_pause_is_not_the_end(self):
        from stride_imu.inertial import snug_end
        L = self.speeds([(10, 30), (50, 70), (140, 150)])   # shuffle at 140
        R = self.speeds([(30, 50), (70, 90)])
        end, foot = snug_end(L, R, limit=199, start=5, max_gap=20)
        self.assertEqual(foot, 'right')
        self.assertTrue(89 <= end < 100)
        self.assertEqual(snug_end(L, R, limit=199)[1], 'left')   # no chain rule

    def test_closing_step_right_after_the_walk_counts(self):
        from stride_imu.inertial import snug_end
        L = self.speeds([(10, 30), (50, 70), (95, 105)])     # closing step 5 later
        R = self.speeds([(30, 50), (70, 90)])
        end, foot = snug_end(L, R, limit=199, start=5, max_gap=20)
        self.assertEqual(foot, 'left')
        self.assertTrue(end >= 104)

    def test_pause_after_a_lone_first_step_does_not_end_the_gait(self):
        from stride_imu.inertial import snug_end
        L = self.speeds([(5, 15), (60, 80), (100, 120)])     # lone step, pause
        R = self.speeds([(80, 100), (120, 140)])
        end, foot = snug_end(L, R, limit=199, start=0, max_gap=20)
        self.assertEqual(foot, 'right')
        self.assertTrue(end >= 139)


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
