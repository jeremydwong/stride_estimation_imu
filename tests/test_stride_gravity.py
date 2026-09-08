"""Validate physical contact constraints against known synthetic motion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from track_research_models import synthetic
from debug_track_foot_difference import align_initial
from stride_imu.experimental import compute_position_stride_gravity


class StrideGravityTests(unittest.TestCase):
    def test_known_tight_track_with_asymmetric_swings(self):
        # Use 8 m corners, different from the 12 m research figure. Do not fit
        # a stadium, force closure, or use the other foot anywhere in estimation.
        for side in ['left','right']:
            t,w,a,truth,bias,a0=synthetic(side,radius=8.)
            r=compute_position_stride_gravity(w,a,1/128,gyro_bias_rad_s=bias,initial_gravity=a0)
            predicted,_=align_initial(r.P,t);expected,_=align_initial(truth,t)
            rmse=np.sqrt(np.mean(np.sum((predicted-expected)**2,axis=1)))
            self.assertLess(rmse,.06)
            np.testing.assert_allclose(r.V[r.FF],0,atol=1e-10)
            np.testing.assert_allclose(np.diff(r.P,axis=0),r.V[1:]/128,atol=1e-12)
            np.testing.assert_allclose(Rotation.from_quat(r.quaternion[:,[1,2,3,0]]).apply(a),r.An,atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(r.quaternion,axis=1),1,atol=1e-12)

    def test_default_preserves_real_rise_and_floor_prior_is_explicit(self):
        t,w,a,truth,bias,a0=synthetic('right',radius=20.,rise=4.)
        r=compute_position_stride_gravity(w,a,1/128,gyro_bias_rad_s=bias,initial_gravity=a0)
        self.assertAlmostEqual(r.P[-1,2],4.,delta=.03)
        flat=compute_position_stride_gravity(w,a,1/128,gyro_bias_rad_s=bias,initial_gravity=a0,assume_level_contacts=True)
        np.testing.assert_allclose(flat.P[:,:2],r.P[:,:2],atol=1e-12)
        np.testing.assert_allclose(flat.P[flat.FF,2],0,atol=1e-10)
        np.testing.assert_allclose(flat.V[flat.FF,2],0,atol=1e-10)

    def test_input_validation(self):
        w=np.zeros((128,3));a=np.tile([0.,0.,-9.80297286843],(128,1))
        with self.assertRaisesRegex(ValueError,'gravity_gain'):compute_position_stride_gravity(w,a,1/128,gravity_gain=2)
        with self.assertRaisesRegex(ValueError,'initial_gravity'):compute_position_stride_gravity(w,a,1/128,initial_gravity=[0,0,0])
        with self.assertRaisesRegex(ValueError,'including the final'):compute_position_stride_gravity(w,a,1/128,FF=np.zeros(128,dtype=bool))


if __name__=='__main__':unittest.main()
