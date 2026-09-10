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
