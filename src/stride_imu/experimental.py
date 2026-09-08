"""Opt-in mechanization experiments; existing inertial entry points are unchanged."""
import numpy as np

from .inertial import (
    GRAVITY, FootTrajectory, acc_tilt, detect_walking_section, foot_fall,
    kalman_filter_tilt, qua2eul, qua2rot, qua_est, zero_velocity_updates,
)


def kalman_filter_gravity(quaternion, acceleration, covariance, process_noise=1e-5,
                          measurement_noise=0.1, is_stationary=True):
    """Correct tilt with a gravity-vector innovation instead of Euler differences.

    Experimental alternative to ``inertial.kalman_filter_tilt``. Uses its scalar
    covariance/gain model, but left-multiplies a navigation-frame rotation about
    a horizontal axis. Gravity supplies no absolute heading observation. This
    avoids Euler branch cuts and the implicit heading coupling of holding Euler
    yaw fixed while changing roll/pitch. Acceleration follows this package's
    convention: stationary navigation acceleration points toward -Z.
    """
    q = np.asarray(quaternion, dtype=float)
    covariance += process_noise
    if is_stationary:
        a = np.asarray(acceleration, dtype=float)
        norm = np.linalg.norm(a)
        if norm > 0 and np.isfinite(norm):
            direction = qua2rot(q) @ (a / norm)
            target = np.array([0., 0., -1.])
            axis = np.cross(direction, target)
            sine = np.linalg.norm(axis)
            cosine = np.clip(direction @ target, -1., 1.)
            gain = covariance / (covariance + measurement_noise)
            if sine > 1e-12:
                axis /= sine
            elif cosine < 0:
                # Antiparallel gravity: tilt axis is ambiguous; choose horizontal X.
                axis = np.array([1., 0., 0.])
            else:
                axis = np.zeros(3)
            half_angle = gain * np.arctan2(sine, cosine) / 2
            scalar = np.cos(half_angle)
            vector = axis * np.sin(half_angle)
            q = np.r_[scalar*q[0] - vector @ q[1:],
                      scalar*q[1:] + q[0]*vector + np.cross(vector, q[1:])]
            covariance = (1-gain)**2*covariance + gain**2*measurement_noise
    return q / np.linalg.norm(q), covariance


def compute_position_experimental(W, A, period, *, tilt_method='gravity',
                                  gyro_bias_rad_s=None, tilt_w_deg_s=30.,
                                  tilt_a_m_s2=1., tilt_process_noise=1e-5,
                                  tilt_measurement_noise=0.1, W_FF=30., A_FF=1.,
                                  T_FF=0.4, MAX_T_FF=None, FF=None,
                                  initial_gravity=None, integration='right',
                                  rotate_after_tilt=False):
    """Opt-in alternative to ``inertial.compute_position`` for turn experiments.

    Keeps the legacy integration/ZUPT order and identity initialization to make
    controlled comparisons possible. Existing functions do not call this one.
    ``tilt_method`` is 'gravity', 'legacy', or 'off'; 'legacy' with default
    parameters reproduces compute_position (including pre-correction An).
    Unlike legacy USE_KF, 'off' really disables tilt updates.

    W is Nx3 rad/sample; A is Nx3 m/s². Optional gyro_bias_rad_s is a BODY-frame
    three-vector estimated from a separate stationary interval, never from the
    moving lap. Bias subtraction affects integration only: footfalls and tilt
    gates use original measurements, isolating the orientation experiment.
    Tilt thresholds are independent of W_FF/A_FF (deg/s and m/s²). Noise values
    are per-sample scalar variances, as in the legacy filter, not continuous-time
    spectral densities. FF may override contacts without disabling tilt gating.
    Returns the same FootTrajectory container; W retains the original input.
    This is experimental and does not impose track geometry or loop closure.
    ``initial_gravity`` optionally supplies a stationary body-frame acceleration
    vector; ``integration`` is 'right' (legacy) or 'trapezoid';
    ``rotate_after_tilt`` tests rotating acceleration after the tilt update.
    See ``compute_position_stride_gravity`` below for a whole-stride gravity
    observation that avoids treating low-rate swing samples as static gravity.
    """
    W, A = np.asarray(W, dtype=float), np.asarray(A, dtype=float)
    if W.ndim != 2 or W.shape[1] != 3 or A.shape != W.shape or len(W) < 2:
        raise ValueError('W and A must be matching Nx3 arrays with N >= 2')
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(A)):
        raise ValueError('W and A must be finite')
    values = [period, tilt_w_deg_s, tilt_a_m_s2, tilt_measurement_noise, W_FF, A_FF, T_FF]
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError('Period, thresholds, duration and measurement noise must be positive and finite')
    if not np.isfinite(tilt_process_noise) or tilt_process_noise < 0:
        raise ValueError('tilt_process_noise must be finite and nonnegative')
    if MAX_T_FF is not None and (not np.isfinite(MAX_T_FF) or MAX_T_FF < period):
        raise ValueError('MAX_T_FF must be finite and at least one sample')
    if T_FF < period:
        raise ValueError('T_FF must be at least one sample')
    if tilt_method not in ('legacy', 'gravity', 'off'):
        raise ValueError('Unknown tilt_method')
    bias = np.zeros(3) if gyro_bias_rad_s is None else np.asarray(gyro_bias_rad_s, dtype=float)
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise ValueError('gyro_bias_rad_s must be a finite three-vector in rad/s')
    corrected_W = W - bias * period
    if integration not in ('right', 'trapezoid'):
        raise ValueError("integration must be 'right' or 'trapezoid'")
    if integration == 'trapezoid':
        corrected_W[1:] = (corrected_W[:-1] + corrected_W[1:]) / 2
    rates = np.linalg.norm(W, axis=1) * 180 / np.pi / period
    residual = np.abs(np.linalg.norm(A, axis=1) - GRAVITY)
    stationary = (rates < W_FF) & (residual < A_FF)
    tilt_mask = (rates < tilt_w_deg_s) & (residual < tilt_a_m_s2)
    if FF is None:
        if not np.any(stationary):
            raise ValueError('No footfall candidates; relax thresholds or supply FF')
        FF, stationary = foot_fall(W, A, period, W_FF, A_FF, T_FF, MAX_T_FF)
    else:
        FF = np.asarray(FF)
        if FF.shape != (len(W),) or FF.dtype != np.bool_ or not FF[-1]:
            raise ValueError('FF must be an N-sample boolean mask including the final sample')
        FF = FF.copy()
    phi, theta = acc_tilt(A)
    q = np.zeros((len(W), 4)); q[0] = [1., 0., 0., 0.]
    if initial_gravity is not None:
        q[0] = _initial_gravity_quaternion(initial_gravity)
    An = np.zeros_like(A); An[0] = qua2rot(q[0]) @ A[0]
    Anz = np.zeros_like(A)
    covariance = 0
    last_footfall = 0
    for i in range(1, len(W)):
        q[i] = qua_est(corrected_W[i], q[i-1])
        # Same order as legacy compute_position, for clean ablations.
        An[i] = qua2rot(q[i]) @ A[i]
        if tilt_method == 'legacy':
            q[i], covariance, _, _ = kalman_filter_tilt(
                period, q[i], covariance, tilt_process_noise, tilt_measurement_noise,
                theta[i], phi[i], tilt_mask[i])
        elif tilt_method == 'gravity':
            q[i], covariance = kalman_filter_gravity(
                q[i], A[i], covariance, tilt_process_noise,
                tilt_measurement_noise, tilt_mask[i])
        if rotate_after_tilt:
            An[i] = qua2rot(q[i]) @ A[i]
        last_footfall = zero_velocity_updates(i, FF, An, Anz, last_footfall)
    V = np.cumsum(Anz, axis=0) * period
    Vm = np.linalg.norm(V[:, :2], axis=1)
    try:
        walking = detect_walking_section({'Vm': Vm, 'FF': FF})
    except Exception:
        walking = FF.copy()
    # qua2eul's arcsin can round outside [-1, 1] at exact pitch singularities.
    euler = np.column_stack((
        np.arctan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]), q[:,0]**2-q[:,1]**2-q[:,2]**2+q[:,3]**2),
        np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]), -1., 1.)),
        np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), q[:,0]**2+q[:,1]**2-q[:,2]**2-q[:,3]**2)))
    return FootTrajectory(FF=FF, FF_walking=walking, stationary_periods=stationary,
                          P=np.cumsum(V, axis=0)*period, V=V, Vm=Vm, euler=euler,
                          quaternion=q, An=An, Anz=Anz, A=A, W=W)


def _initial_gravity_quaternion(acceleration):
    acceleration = np.asarray(acceleration, dtype=float)
    if (acceleration.shape != (3,) or not np.all(np.isfinite(acceleration))
            or np.linalg.norm(acceleration) < 1e-6):
        raise ValueError('initial_gravity must be a finite nonzero three-vector')
    q, _ = kalman_filter_gravity([1., 0., 0., 0.], acceleration, 1e12)
    return q


def compute_position_stride_gravity(W, A, period, *, gyro_bias_rad_s=None,
                                    initial_gravity=None, integration='right',
                                    gravity_gain=1., assume_level_contacts=False,
                                    W_FF=30., A_FF=1., T_FF=.4,
                                    MAX_T_FF=None, FF=None):
    """Offline whole-stride tilt alternative to compute_position_experimental.

    Legacy ``inertial.compute_position`` and the sample-gated experimental
    function above are preserved. This variant uses the zero net velocity
    between contacts to estimate gravity from the entire stride. In a provisional
    navigation frame, mean rotated acceleration should be gravity. A single
    minimum rotation aligns that mean with -Z; apply it to EVERY orientation
    and acceleration in the stride, then carry corrected attitude forward.
    This avoids interpreting low angular velocity during swing as static tilt,
    and uses the same physical contact constraint for tilt and velocity.

    W: Nx3 rad/sample, A: Nx3 m/s², period: seconds. gyro_bias_rad_s is a constant
    body-frame rad/s three-vector calibrated separately. initial_gravity is a
    stationary body-frame acceleration vector (default A[0]). Inputs must start
    and end at rest; sample 0 is an implicit contact and the final sample is
    included in detected FF, following legacy endpoint assumptions. Consecutive
    contacts delimit complete strides. No yaw observation or track geometry is
    supplied. It is offline: a stride's correction requires its next contact.

    gravity_gain in [0,1] controls the segment correction; 1 applies the full
    zero-velocity gravity observation. integration='right' matches the legacy
    gyro sample convention; 'trapezoid' averages adjacent angular increments.
    Footfall thresholds retain legacy units/meaning. A supplied FF mask must
    include the last sample. A finite nonzero mean gravity is required per segment.

    assume_level_contacts=False leaves elevation unconstrained. When explicitly
    True, remove each stride's vertical displacement with a smooth velocity bump
    that vanishes at both contacts. This is a flat-floor PRIOR, not measured
    elevation accuracy; it changes only Z and must not be used on stairs/slopes.
    Returns FootTrajectory with consistent P/V/Anz, and original W/A inputs.
    """
    from scipy.spatial.transform import Rotation

    W, A = np.asarray(W, dtype=float), np.asarray(A, dtype=float)
    if W.ndim != 2 or W.shape[1] != 3 or A.shape != W.shape or len(W) < 2:
        raise ValueError('W and A must be matching Nx3 arrays with N >= 2')
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(A)):
        raise ValueError('W and A must be finite')
    values = [period, W_FF, A_FF, T_FF]
    if not np.all(np.isfinite(values)) or min(values) <= 0 or T_FF < period:
        raise ValueError('Positive finite period/thresholds and T_FF >= period required')
    if not np.isfinite(gravity_gain) or not 0 <= gravity_gain <= 1:
        raise ValueError('gravity_gain must be in [0,1]')
    if integration not in ('right', 'trapezoid'):
        raise ValueError("integration must be 'right' or 'trapezoid'")
    if MAX_T_FF is not None and (not np.isfinite(MAX_T_FF) or MAX_T_FF < period):
        raise ValueError('MAX_T_FF must be finite and at least one sample')
    bias = np.zeros(3) if gyro_bias_rad_s is None else np.asarray(gyro_bias_rad_s, dtype=float)
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise ValueError('gyro_bias_rad_s must be a finite three-vector in rad/s')
    rates = np.rad2deg(np.linalg.norm(W, axis=1) / period)
    stationary = (rates < W_FF) & (np.abs(np.linalg.norm(A, axis=1)-GRAVITY) < A_FF)
    if FF is None:
        if not np.any(stationary):
            raise ValueError('No footfall candidates; relax thresholds or supply FF')
        FF, stationary = foot_fall(W, A, period, W_FF, A_FF, T_FF, MAX_T_FF)
    else:
        FF = np.asarray(FF)
        if FF.shape != (len(W),) or FF.dtype != np.bool_ or not FF[-1]:
            raise ValueError('FF must be an N-sample boolean mask including the final sample')
        FF = FF.copy()
    q0 = _initial_gravity_quaternion(A[0] if initial_gravity is None else initial_gravity)
    rotation = Rotation.from_quat(q0[[1, 2, 3, 0]])
    increments = W - bias * period
    if integration == 'trapezoid':
        increments[1:] = (increments[:-1] + increments[1:]) / 2
    q = np.zeros((len(W), 4)); q[0] = q0[[1, 2, 3, 0]]
    An = np.zeros_like(A); An[0] = rotation.apply(A[0])
    Anz = np.zeros_like(A)
    contacts = np.unique(np.r_[0, np.flatnonzero(FF)])
    for start, end in zip(contacts[:-1], contacts[1:]):
        rotations = []
        for i in range(start+1, end+1):
            rotation = rotation * Rotation.from_rotvec(increments[i])
            rotations.append(rotation.as_quat())
        provisional = Rotation.from_quat(rotations)
        acceleration = provisional.apply(A[start+1:end+1])
        mean = acceleration.mean(axis=0)
        if np.linalg.norm(mean) < 1e-6:
            raise ValueError('Degenerate stride gravity; check contacts and acceleration')
        # Full-gain gravity update expressed as a navigation-frame quaternion.
        correction_q = _initial_gravity_quaternion(mean)
        correction = Rotation.from_quat(correction_q[[1, 2, 3, 0]])
        correction = Rotation.from_rotvec(correction.as_rotvec() * gravity_gain)
        corrected = correction * provisional
        rotation = corrected[-1]
        q[start+1:end+1] = corrected.as_quat()
        An[start+1:end+1] = correction.apply(acceleration)
        Anz[start+1:end+1] = An[start+1:end+1] - An[start+1:end+1].mean(axis=0)
        if assume_level_contacts and end-start > 1:
            h = np.arange(1, end-start+1) / (end-start)
            bump = h * (1-h)
            velocity = np.cumsum(Anz[start+1:end+1, 2]) * period
            velocity -= bump * (velocity.sum() / bump.sum())
            Anz[start+1:end+1, 2] = np.diff(np.r_[0., velocity]) / period
    V = np.cumsum(Anz, axis=0) * period
    Vm = np.linalg.norm(V[:, :2], axis=1)
    walking = detect_walking_section({'Vm': Vm, 'FF': FF})
    return FootTrajectory(FF=FF, FF_walking=walking, stationary_periods=stationary,
                          P=np.cumsum(V, axis=0)*period, V=V, Vm=Vm,
                          euler=Rotation.from_quat(q).as_euler('xyz'),
                          quaternion=q[:, [3, 0, 1, 2]], An=An, Anz=Anz, A=A, W=W)
