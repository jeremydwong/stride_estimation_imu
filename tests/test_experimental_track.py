"""Physical invariants and compatibility for the opt-in turn estimator.

Run: .venv/bin/python -m unittest discover -s tests -v
"""
import unittest
import numpy as np
from stride_imu.inertial import GRAVITY, compute_position, eul2qua, qua2rot
from stride_imu.experimental import compute_position_experimental, kalman_filter_gravity


class ExperimentalTrackTests(unittest.TestCase):
    def test_gravity_update_reduces_tilt_without_absolute_yaw_observation(self):
        # A level stationary sensor with arbitrary initial roll/pitch and yaw.
        A = np.array([0.,0.,-GRAVITY])
        rotations=[]
        for yaw in [0.,1.2]:
            q=eul2qua(np.array([[.3,-.4,yaw]]))[0]
            initial=np.linalg.norm(qua2rot(q)@A-A)
            q,P=kalman_filter_gravity(q,A,.1)
            self.assertLess(np.linalg.norm(qua2rot(q)@A-A),initial)
            self.assertAlmostEqual(np.linalg.norm(q),1.)
            rotations.append(qua2rot(q))
        yaw_rotation=qua2rot(eul2qua(np.array([[0.,0.,1.2]]))[0])
        np.testing.assert_allclose(rotations[1],yaw_rotation@rotations[0],atol=1e-14)

    def test_explicit_bias_removes_known_stationary_yaw_drift(self):
        dt=1/128; A=np.tile([0.,0.,-GRAVITY],(256,1))
        bias=np.array([0.,0.,.02]); W=np.tile(bias*dt,(256,1))
        r=compute_position_experimental(W,A,dt,gyro_bias_rad_s=bias)
        np.testing.assert_allclose(r.quaternion,np.tile([1.,0.,0.,0.],(256,1)),atol=1e-14)
        np.testing.assert_allclose(r.P,0,atol=1e-12)
        np.testing.assert_array_equal(r.W,W)
        uncalibrated=compute_position_experimental(W,A,dt)
        self.assertGreater(abs(uncalibrated.euler[-1,2]),.03)

    def test_legacy_ablation_matches_existing_mechanization(self):
        rng=np.random.default_rng(17);dt=1/128
        W=rng.normal(0,.001,(512,3));A=rng.normal(0,.04,(512,3))+[.4,-.2,-GRAVITY]
        old=compute_position(W,A,dt)
        new=compute_position_experimental(W,A,dt,tilt_method='legacy')
        for name in vars(old):
            np.testing.assert_allclose(getattr(old,name),getattr(new,name),atol=1e-12,rtol=1e-12,err_msg=name)
        # Document the legacy no-op switch; don't silently change existing runs.
        np.testing.assert_array_equal(old.P,compute_position(W,A,dt,USE_KF=0).P)

    def test_supplied_contacts_and_real_off_switch(self):
        W=np.zeros((128,3));A=np.tile([1.,0.,-np.sqrt(GRAVITY**2-1)],(128,1))
        ff=np.zeros(128,dtype=bool);ff[::32]=True;ff[-1]=True
        off=compute_position_experimental(W,A,1/128,FF=ff,tilt_method='off')
        on=compute_position_experimental(W,A,1/128,FF=ff)
        np.testing.assert_array_equal(off.FF,ff)
        np.testing.assert_array_equal(off.quaternion,np.tile([1.,0.,0.,0.],(128,1)))
        self.assertGreater(np.linalg.norm(on.quaternion[-1]-off.quaternion[-1]),.01)

    def test_invalid_inputs_and_no_stance_fail_clearly(self):
        W=np.ones((128,3));A=np.tile([0.,0.,-GRAVITY],(128,1))
        with self.assertRaisesRegex(ValueError,'No footfall'):compute_position_experimental(W,A,1/128)
        with self.assertRaisesRegex(ValueError,'finite three-vector'):compute_position_experimental(W,A,1/128,gyro_bias_rad_s=[1,2])
        with self.assertRaisesRegex(ValueError,'positive and finite'):compute_position_experimental(W,A,0)


if __name__=='__main__':unittest.main()
