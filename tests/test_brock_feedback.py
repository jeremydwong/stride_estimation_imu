import unittest
from types import SimpleNamespace
import numpy as np
from stride_imu.inertial import snug_start, snug_end
from stride_imu.plotting import EndSampleLocator


class BoutBoundaryTests(unittest.TestCase):
    def test_end_is_start_run_backwards(self):
        left=np.array([0.,.1,.6,1.5,.7,.1,0.,.15,.3])
        right=np.array([0.,0.,.2,.6,1.8,.9,.3,.15,.1])
        end,foot=snug_end(SimpleNamespace(Vm=left),SimpleNamespace(Vm=right))
        start,reverse_foot=snug_start(SimpleNamespace(Vm=left[::-1]),SimpleNamespace(Vm=right[::-1]))
        self.assertEqual((end,foot),(len(left)-1-start,reverse_foot))
        self.assertEqual((end,foot),(7,'right'))

    def test_end_stops_at_valley_and_keeps_last_real_swing(self):
        left=SimpleNamespace(Vm=np.array([0.,1.4,.8,.3,.4,.2]))
        right=SimpleNamespace(Vm=np.zeros(6))
        self.assertEqual(snug_end(left,right),(3,'left'))
        self.assertEqual(snug_end(right,right),(None,None))
        self.assertEqual(snug_end(SimpleNamespace(Vm=np.array([0.,1.5,1.6])),right),(2,'left'))

    def test_only_round_ticks_inside_view_even_after_zoom(self):
        locator=EndSampleLocator()
        np.testing.assert_array_equal(locator.tick_values(128031,129998),[128100,129900])
        np.testing.assert_array_equal(locator.tick_values(129998,128031),[128100,129900])
        np.testing.assert_array_equal(locator.tick_values(100,500),[100,500])
        np.testing.assert_array_equal(locator.tick_values(121,169),[130,160])
        np.testing.assert_array_equal(locator.tick_values(121.2,122.8),[122])
        self.assertEqual(len(locator.tick_values(121.2,121.8)),0)


if __name__=='__main__':unittest.main()


class SnugEndLimitTests(unittest.TestCase):
    """limit=: the gait ends at the last landing COMPLETED inside the window."""
    def test_swing_in_progress_at_the_window_edge_is_not_a_landing(self):
        from types import SimpleNamespace as NS
        # left: swing (3) landing at 3, swing (6-7) landing at 8; the window
        # ends at 6, mid-way through the second swing
        left = NS(Vm=np.array([0., 2., 2., 0.1, 0., 0., 2., 2., 0.1, 0.]))
        right = NS(Vm=np.zeros(10))
        self.assertEqual(snug_end(left, right, limit=6), (3, 'left'))
        self.assertEqual(snug_end(left, right, limit=9), (8, 'left'))
        self.assertEqual(snug_end(left, right), (8, 'left'))      # historic rule
        self.assertEqual(snug_end(left, right, limit=1), (None, None))

    def test_latest_completed_landing_across_both_feet(self):
        from types import SimpleNamespace as NS
        left = NS(Vm=np.array([2., 0.1, 0., 0., 0., 0., 0., 2., 2., 2.]))
        right = NS(Vm=np.array([0., 0., 0., 2., 2., 0.1, 0., 0., 0., 0.]))
        self.assertEqual(snug_end(left, right, limit=8), (5, 'right'))


class StopAfterTests(unittest.TestCase):
    """stop_after: both feet raw-still for STOP_HOLD_S soon after the end."""
    def feet(self, left_w, right_w, period=0.01):
        from types import SimpleNamespace as NS
        mk = lambda w: NS(Wb=np.c_[np.asarray(w, float) * period,
                                   np.zeros(len(w)), np.zeros(len(w))])
        return {('s1', 'left'): mk(left_w), ('s1', 'right'): mk(right_w)}, period

    def test_still_after_a_brief_closing_step_is_a_stop(self):
        import brock_functions as bf
        # 0.3 s of right-foot motion (closing step), then both still
        left, right = [0.0] * 200, [3.0] * 30 + [0.0] * 170
        feet, p = self.feet(left, right)
        self.assertTrue(bf.stop_after(feet, 's1', 0, p))

    def test_continuous_walking_is_not_a_stop(self):
        import brock_functions as bf
        # alternating swings: one foot is always moving
        left = ([3.0] * 20 + [0.0] * 20) * 6
        right = ([0.0] * 20 + [3.0] * 20) * 6
        feet, p = self.feet(left, right)
        self.assertFalse(bf.stop_after(feet, 's1', 0, p))

    def test_stop_too_late_does_not_count(self):
        import brock_functions as bf
        # moving for 2 s (> STOP_SEARCH_S), only then still
        left = [3.0] * 200 + [0.0] * 100
        feet, p = self.feet(left, left)
        self.assertFalse(bf.stop_after(feet, 's1', 0, p))
