import numpy as np
from typing import Tuple, Optional, Any, Dict
from scipy.signal import find_peaks

GRAVITY = 9.80297286843

# --- Quaternion and rotation helpers ---
def eul2qua(att: np.ndarray) -> np.ndarray:
    phi, the, psi = att[:, 0], att[:, 1], att[:, 2]
    cos_phi = np.cos(phi / 2.0)
    sin_phi = np.sin(phi / 2.0)
    cos_the = np.cos(the / 2.0)
    sin_the = np.sin(the / 2.0)
    cos_psi = np.cos(psi / 2.0)
    sin_psi = np.sin(psi / 2.0)
    a = cos_phi * cos_the * cos_psi + sin_phi * sin_the * sin_psi
    b = sin_phi * cos_the * cos_psi - cos_phi * sin_the * sin_psi
    c = cos_phi * sin_the * cos_psi + sin_phi * cos_the * sin_psi
    d = cos_phi * cos_the * sin_psi - sin_phi * sin_the * cos_psi
    return np.column_stack([a, b, c, d])

def qua2eul(quaternion: np.ndarray) -> np.ndarray:
    a, b, c, d = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    phi = np.arctan2(2 * (a * b + c * d), a ** 2 - b ** 2 - c ** 2 + d ** 2)
    the = np.arcsin(2 * (a * c - d * b))
    psi = np.arctan2(2 * (a * d + b * c), a ** 2 + b ** 2 - c ** 2 - d ** 2)
    return np.column_stack([phi, the, psi])

def qua2rot(quaternion: np.ndarray) -> np.ndarray:
    a, b, c, d = quaternion
    R = np.zeros((3, 3))
    R[0, 0] = a ** 2 + b ** 2 - c ** 2 - d ** 2
    R[0, 1] = 2 * (b * c - a * d)
    R[0, 2] = 2 * (b * d + a * c)
    R[1, 0] = 2 * (b * c + a * d)
    R[1, 1] = a ** 2 - b ** 2 + c ** 2 - d ** 2
    R[1, 2] = 2 * (c * d - a * b)
    R[2, 0] = 2 * (b * d - a * c)
    R[2, 1] = 2 * (c * d + a * b)
    R[2, 2] = a ** 2 - b ** 2 - c ** 2 + d ** 2
    return R

def rotate_angle(X: np.ndarray, Y: np.ndarray, ang: float) -> Tuple[np.ndarray, np.ndarray]:
    Xr = X * np.cos(ang) - Y * np.sin(ang)
    Yr = X * np.sin(ang) + Y * np.cos(ang)
    return Xr, Yr

# --- Tilt and footfall detection ---
def acc_tilt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    AX, AY = A[:, 0], A[:, 1]
    accel_theta = AX / GRAVITY
    valid_angles = np.abs(accel_theta) <= 1
    accel_theta[valid_angles] = np.arcsin(accel_theta[valid_angles])
    accel_phi = AY / (np.cos(accel_theta) * GRAVITY)
    valid_angles_phi = np.abs(accel_phi) <= 1
    accel_phi[valid_angles_phi] = -np.arcsin(accel_phi[valid_angles_phi])
    return accel_phi, accel_theta

def foot_fall(W: np.ndarray, A: np.ndarray, period: float, W_FF: Optional[float] = None, A_FF: Optional[float] = None, T_FF: Optional[float] = None, MAX_T_FF: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if W_FF is None:
        W_FF = 30
    if A_FF is None:
        A_FF = 1
    if T_FF is None:
        T_FF = int(0.4 / period)
    else:
        T_FF = int(T_FF / period)
    if MAX_T_FF is None:
        MAX_T_FF = T_FF * 3
    else:
        MAX_T_FF = int(MAX_T_FF / period)
    Wm = np.sqrt(np.sum(W ** 2, axis=1)) * 180 / np.pi / period
    Am = np.sqrt(np.sum(A ** 2, axis=1)) - GRAVITY
    low_motion = np.where((Wm < W_FF) & (np.abs(Am) < A_FF))[0]
    N = W.shape[0]
    stationary_periods = np.zeros(N, dtype=bool)
    stationary_periods[low_motion] = True
    diff_low_motion = np.diff(low_motion)
    valid_FF = np.where(diff_low_motion > T_FF)[0]
    start_FF = low_motion[np.concatenate(([0], valid_FF + 1))]
    end_FF = low_motion[np.concatenate((valid_FF, [len(low_motion) - 1]))]
    FF_index = []
    for s, e in zip(start_FF, end_FF):
        Wm_cut = Wm[s:e + 1]
        N_cut = Wm_cut.size
        R_seg = N_cut // MAX_T_FF if MAX_T_FF > 0 else 1
        if R_seg > 0 and N_cut > 0:
            Wm_seg = Wm_cut[:R_seg * MAX_T_FF].reshape((MAX_T_FF, R_seg), order='F')
            if Wm_seg.size > 0:
                min_idx = np.argmin(Wm_seg, axis=0)
                FF_cut = min_idx + np.arange(R_seg) * MAX_T_FF + s
                FF_index.extend(FF_cut.tolist())
        if R_seg * MAX_T_FF < N_cut:
            remaining_cut = Wm_cut[R_seg * MAX_T_FF:]
            if remaining_cut.size > 0:
                min_idx2 = np.argmin(remaining_cut)
                FF_index.append(s + R_seg * MAX_T_FF + min_idx2)
    FF_index.append(N - 1)
    FF = np.zeros(N, dtype=bool)
    FF[FF_index] = True
    return FF, stationary_periods

def kf_tilt(period: float, quaternion: Optional[np.ndarray] = None, accel_theta: Optional[float] = None, accel_phi: Optional[float] = None, is_stationary: Optional[bool] = None) -> np.ndarray:
    # Simple complementary filter for tilt correction
    if quaternion is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if is_stationary:
        euler = qua2eul(quaternion[np.newaxis, :])[0]
        # Kalman gain (fixed for simplicity)
        K = 0.01
        euler[0] = euler[0] - (euler[0] - accel_phi) * K
        euler[1] = euler[1] - (euler[1] - accel_theta) * K
        quaternion = eul2qua(euler[np.newaxis, :])[0]
    return quaternion

def qua_est(W: np.ndarray, quaternion_prev: np.ndarray) -> np.ndarray:
    mag = np.sqrt(np.sum(W ** 2))
    if mag != 0:
        sin_mag = np.sin(mag / 2.0) / mag
    else:
        sin_mag = 0.5
    rotation = np.concatenate(([np.cos(mag / 2.0)], sin_mag * W))
    a, b, c, d = quaternion_prev
    quaternion_sqw = np.array([
        [a, -b, -c, -d],
        [b, a, -d, c],
        [c, d, a, -b],
        [d, -c, b, a],
    ])
    return quaternion_sqw @ rotation

def zupts(i: int, FF: np.ndarray, An: np.ndarray, Anz: np.ndarray, last_footfall: int) -> int:
    if i == 1:
        last_footfall = 0
    if FF[i]:
        step_range = np.arange(last_footfall, i + 1)
        if step_range.size < 2:
            return last_footfall
        step_samples = step_range.size
        velocity_error = np.sum(An[step_range, :], axis=0)
        acceleration_error = velocity_error / step_samples
        Anz[step_range, :] = An[step_range, :] - acceleration_error
        last_footfall = i
    return last_footfall

def compute_pos(W: np.ndarray, A: np.ndarray, period: float, USE_KF: int = 1, W_FF: Optional[float] = None, A_FF: Optional[float] = None, T_FF: Optional[float] = None, MAX_T_FF: Optional[float] = None, FF: Optional[np.ndarray] = None) -> Dict[str, Any]:
    N = W.shape[0]
    t = np.arange(N) * period
    accel_phi, accel_theta = acc_tilt(A)
    if FF is None:
        FF, stationary_periods = foot_fall(W, A, period, W_FF, A_FF, T_FF, MAX_T_FF)
    result = {'FF': FF}
    quaternion = np.zeros((N, 4))
    An = np.zeros((N, 3))
    Anz = np.zeros((N, 3))
    quaternion[0, :] = kf_tilt(period)
    last_footfall = 0
    for i in range(1, N):
        quaternion[i, :] = qua_est(W[i, :], quaternion[i - 1, :])
        rotation_matrix = qua2rot(quaternion[i, :])
        An[i, :] = rotation_matrix @ A[i, :]
        if USE_KF:
            quaternion[i, :] = kf_tilt(period, quaternion[i, :], accel_theta[i], accel_phi[i], stationary_periods[i])
        last_footfall = zupts(i, FF, An, Anz, last_footfall)
    result['Anz'] = Anz
    result['An'] = An
    result['A'] = A
    result['W'] = W
    result['quaternion'] = quaternion
    euler = qua2eul(quaternion)
    result['euler'] = euler
    V = np.cumsum(Anz, axis=0) * period
    Vm = np.sqrt(np.sum(V[:, :2] ** 2, axis=1))
    result['V'] = V
    result['Vm'] = Vm
    try:
        result['FF_walking'] = detect_walking_section(result)
    except Exception:
        result['FF_walking'] = result['FF']
    P = np.cumsum(V, axis=0) * period
    result['P'] = P
    return result

def detect_walking_section(walk_info: Dict[str, Any], MIN_WALK_SPEED: float = 2.0) -> np.ndarray:
    FF = np.where(walk_info['FF'])[0]
    Vm = walk_info['Vm']
    peaks_idx, _ = find_peaks(Vm, height=MIN_WALK_SPEED)
    FF_max_speed = np.zeros_like(Vm, dtype=bool)
    FF_max_speed[peaks_idx] = True
    median_vel = np.median(Vm[peaks_idx]) if peaks_idx.size > 0 else 0
    likely_walk_sections = np.where(Vm > median_vel * 0.90)[0]
    if likely_walk_sections.size == 0:
        return walk_info['FF']
    footfall_index = np.where((FF > likely_walk_sections[0]) & (FF < likely_walk_sections[-1]))[0]
    FF_walking = np.zeros_like(Vm, dtype=bool)
    if footfall_index.size > 0:
        FF_walking[FF[footfall_index[0]:footfall_index[-1] + 1]] = True
    return FF_walking

def stride_segmentation(walk_info: Dict[str, Any], period: float, FILTER: int = 0, OUTLIER_SECTION_SECONDS: Optional[Any] = None) -> Dict[str, Any]:
    if OUTLIER_SECTION_SECONDS is not None:
        number_of_sections = np.atleast_2d(OUTLIER_SECTION_SECONDS).shape[0]
        OUTLIER_SECTION_SAMPLES = np.concatenate([
            np.arange(int(np.floor(start / period)), int(np.floor(end / period)) + 1)
            for start, end in np.atleast_2d(OUTLIER_SECTION_SECONDS)
        ])
    else:
        OUTLIER_SECTION_SAMPLES = np.array([])
    MAX_FF_TIME = int(2 / period)
    FF = np.where(walk_info['FF_walking'])[0]
    swing_start = FF[:-1]
    swing_finish = FF[1:]
    swing_time = np.diff(FF)
    too_long = np.where(swing_time > MAX_FF_TIME)[0]
    swing_start = np.delete(swing_start, too_long)
    swing_finish = np.delete(swing_finish, too_long)
    stepData = get_steps(swing_start, swing_finish, walk_info, period, FILTER, OUTLIER_SECTION_SAMPLES)
    return stepData

def get_steps(step_start, step_end, walk_info, period, FILTER, OUTLIER_SECTION):
    # This is a simplified version, for brevity. Full implementation would port all MATLAB logic.
    # Returns a dict with keys: ltrl, frwd, elev, etc.
    P = walk_info['P']
    euler = walk_info['euler']
    number_of_steps = len(step_start)
    
    # Handle empty step arrays
    if number_of_steps == 0:
        return {'ltrl': np.array([]), 'frwd': np.array([]), 'elev': np.array([])}
    
    longest_step = np.max(step_end - step_start) + 1
    ltrl = np.zeros((number_of_steps, longest_step))
    frwd = np.zeros((number_of_steps, longest_step))
    elev = np.zeros((number_of_steps, longest_step))
    for i in range(number_of_steps):
        idx = np.arange(step_start[i], step_end[i] + 1)
        direction = np.arctan2(P[step_end[i], 1] - P[step_start[i], 1], P[step_end[i], 0] - P[step_start[i], 0])
        frwd[i, :len(idx)], ltrl[i, :len(idx)] = rotate_angle(P[idx, 0], P[idx, 1], -direction)
        elev[i, :len(idx)] = P[idx, 2]
    return {'ltrl': ltrl, 'frwd': frwd, 'elev': elev} 