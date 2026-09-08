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
                                  T_FF=0.4, MAX_T_FF=None, FF=None):
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
