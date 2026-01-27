import numpy as np
from typing import Tuple, Optional, Any, Dict, List
from scipy.signal import find_peaks
from dataclasses import dataclass

GRAVITY = 9.80297286843


@dataclass
class WalkingBout:
    """A detected walking bout with start/end indices and duration."""
    start_idx: int
    end_idx: int
    duration_seconds: float
    quiet_before_idx: int  # Start of quiet period before this bout
    quiet_after_idx: int   # End of quiet period after this bout


def detect_quiet_periods(W: np.ndarray, A: np.ndarray, period: float,
                         W_threshold: float = 30.0,
                         A_threshold: float = 1.0,
                         min_quiet_seconds: float = 1.0) -> List[Tuple[int, int]]:
    """
    Detect contiguous quiet (stationary) periods in IMU data.

    Parameters:
    -----------
    W : np.ndarray
        Angular velocity (Nx3, rad/sample)
    A : np.ndarray
        Acceleration (Nx3, m/s^2)
    period : float
        Sampling period in seconds
    W_threshold : float
        Max angular velocity magnitude (deg/s) to be considered quiet
    A_threshold : float
        Max acceleration deviation from gravity (m/s^2) to be considered quiet
    min_quiet_seconds : float
        Minimum duration (seconds) for a quiet period to count

    Returns:
    --------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) for each quiet period
    """
    min_quiet_samples = int(min_quiet_seconds / period)

    # Compute magnitudes
    Wm = np.sqrt(np.sum(W ** 2, axis=1)) * 180 / np.pi / period  # deg/s
    Am = np.sqrt(np.sum(A ** 2, axis=1))
    Am_deviation = np.abs(Am - GRAVITY)

    # Find low-motion samples
    is_quiet = (Wm < W_threshold) & (Am_deviation < A_threshold)

    # Find contiguous quiet regions
    quiet_periods = []
    in_quiet = False
    start_idx = 0

    for i in range(len(is_quiet)):
        if is_quiet[i] and not in_quiet:
            # Start of quiet period
            in_quiet = True
            start_idx = i
        elif not is_quiet[i] and in_quiet:
            # End of quiet period
            in_quiet = False
            if i - start_idx >= min_quiet_samples:
                quiet_periods.append((start_idx, i))

    # Handle case where recording ends during quiet period
    if in_quiet and len(is_quiet) - start_idx >= min_quiet_samples:
        quiet_periods.append((start_idx, len(is_quiet)))

    return quiet_periods


def detect_walking_bouts(W: np.ndarray, A: np.ndarray, period: float,
                         W_threshold: float = 30.0,
                         A_threshold: float = 1.0,
                         min_quiet_seconds: float = 1.5,
                         min_walk_seconds: float = 3.0) -> List[WalkingBout]:
    """
    Detect walking bouts as active periods between quiet periods.

    A walking bout is an active period bounded by quiet periods on both sides.

    Parameters:
    -----------
    W : np.ndarray
        Angular velocity (Nx3, rad/sample)
    A : np.ndarray
        Acceleration (Nx3, m/s^2)
    period : float
        Sampling period in seconds
    W_threshold : float
        Max angular velocity (deg/s) for quiet detection
    A_threshold : float
        Max acceleration deviation from gravity (m/s^2) for quiet detection
    min_quiet_seconds : float
        Minimum quiet period duration to count as a boundary
    min_walk_seconds : float
        Minimum walking bout duration to include

    Returns:
    --------
    List[WalkingBout]
        List of detected walking bouts
    """
    min_walk_samples = int(min_walk_seconds / period)

    quiet_periods = detect_quiet_periods(W, A, period, W_threshold, A_threshold, min_quiet_seconds)

    if len(quiet_periods) < 2:
        # Need at least 2 quiet periods to bound a walk
        return []

    bouts = []
    for i in range(len(quiet_periods) - 1):
        quiet_end = quiet_periods[i][1]      # End of quiet period before walk
        quiet_start = quiet_periods[i + 1][0]  # Start of quiet period after walk

        walk_start = quiet_end
        walk_end = quiet_start
        walk_samples = walk_end - walk_start

        if walk_samples >= min_walk_samples:
            bouts.append(WalkingBout(
                start_idx=walk_start,
                end_idx=walk_end,
                duration_seconds=walk_samples * period,
                quiet_before_idx=quiet_periods[i][0],
                quiet_after_idx=quiet_periods[i + 1][1]
            ))

    return bouts


def find_bouts_near_time(bouts: List[WalkingBout],
                         time_datetime: np.ndarray,
                         target_time,
                         window_seconds: float = 60.0) -> List[WalkingBout]:
    """
    Filter walking bouts to those near a target time.

    Parameters:
    -----------
    bouts : List[WalkingBout]
        List of detected walking bouts
    time_datetime : np.ndarray
        Array of datetime objects for each sample
    target_time : datetime.time
        Target time of day to search around
    window_seconds : float
        Search window in seconds (centered on target_time)

    Returns:
    --------
    List[WalkingBout]
        Bouts that overlap with the time window
    """
    half_window = window_seconds / 2
    target_secs = target_time.hour * 3600 + target_time.minute * 60 + target_time.second

    matching_bouts = []
    for bout in bouts:
        # Get time at bout midpoint
        mid_idx = (bout.start_idx + bout.end_idx) // 2
        if mid_idx < len(time_datetime):
            bout_time = time_datetime[mid_idx].time()
            bout_secs = bout_time.hour * 3600 + bout_time.minute * 60 + bout_time.second + bout_time.microsecond / 1e6

            if abs(bout_secs - target_secs) <= half_window:
                matching_bouts.append(bout)

    return matching_bouts


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
        MAX_T_FF = T_FF * 10
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

def kalman_filter_tilt(period: float, quaternion, P, Q, R, accel_theta: Optional[float] = None, accel_phi: Optional[float] = None, is_stationary: Optional[bool] = None) -> tuple[np.ndarray, float, float, float]:
    # Simple complementary filter for tilt correction
    if quaternion is None:
        return np.array([1.0, 0.0, 0.0, 0.0]), 0, 1e-5,1e-1
    
    P = P+Q
    if is_stationary:
        # compute Kalman gain. 
        K = P/(P+R)
        ang = qua2eul(quaternion[np.newaxis, :])[0]
        ang[0] = ang[0] - (ang[0] - accel_phi) * K
        ang[1] = ang[1] - (ang[1] - accel_theta) * K
        quaternion = eul2qua(ang[np.newaxis, :])[0]
        P = (1-K)*P*(1-K) + K*R*K
    return quaternion, P, Q, R

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

def zero_velocity_updates(i: int, FF: np.ndarray, An: np.ndarray, Anz: np.ndarray, last_footfall: int) -> int:
    if i == 1:
        last_footfall = -1
    if FF[i]:
        step_range = np.arange(last_footfall+1, i+1)
        if step_range.size < 2:
            return last_footfall
        step_samples = step_range.size
        velocity_error = np.sum(An[step_range, :], axis=0)
        acceleration_error = velocity_error / step_samples
        Anz[step_range, :] = An[step_range, :] - acceleration_error
        last_footfall = i
    return last_footfall


def compute_position(W: np.ndarray, A: np.ndarray, period: float, USE_KF: int = 1, W_FF: Optional[float] = None, A_FF: Optional[float] = None, T_FF: Optional[float] = None, MAX_T_FF: Optional[float] = None, FF: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Perform inertial mechanization on a single foot IMU recording.

    Integrates angular velocity to track orientation (as quaternions),
    transforms body-frame accelerations to the navigation frame, detects
    footfalls (stance phases), and applies zero-velocity updates (ZUPT) to
    correct drift. Returns a dict containing position trajectory, velocity,
    orientation, and footfall arrays.

    Parameters:
    -----------
    W : np.ndarray
        Angular velocity in body frame (Nx3, rad/sample)
    A : np.ndarray
        Acceleration in body frame (Nx3, m/s^2)
    period : float
        Sampling period in seconds (e.g. 1/128)
    USE_KF : int
        Use Kalman filter for tilt correction (default: 1, always on)
    W_FF : float, optional
        Angular velocity threshold for footfall detection (deg/s)
    A_FF : float, optional
        Acceleration threshold for footfall detection (m/s^2)
    T_FF : float, optional
        Minimum time between footfalls (seconds)
    MAX_T_FF : float, optional
        Maximum footfall segment duration (seconds)
    FF : np.ndarray, optional
        Pre-computed footfall boolean array (skips detection if provided)

    Returns:
    --------
    dict
        Keys: 'P' (position Nx3), 'V' (velocity Nx3), 'Vm' (speed magnitude),
        'FF' (footfall bool array), 'FF_walking' (walking-only footfalls),
        'euler' (orientation Nx3), 'quaternion' (Nx4),
        'An' (nav-frame accel), 'Anz' (ZUPT-corrected accel), 'A', 'W'
    """
    
    N = W.shape[0]
    t = np.arange(N) * period
    accel_phi, accel_theta = acc_tilt(A)
    if FF is None:
        FF, stationary_periods = foot_fall(W, A, period, W_FF, A_FF, T_FF, MAX_T_FF)
    result = {'FF': FF}
    quaternion = np.zeros((N, 4))
    An = np.zeros((N, 3))
    Anz = np.zeros((N, 3))

    # initialize default last footfall happened at time 0.
    last_footfall = 0

    #kalman filter init
    quaternion[0, :],P,Q,R = kalman_filter_tilt(period,None, None ,None, None)
    
    for i in range(1, N):
        quaternion[i, :] = qua_est(W[i, :], quaternion[i - 1, :])
        rotation_matrix = qua2rot(quaternion[i, :])
        An[i, :] = (rotation_matrix @ A[i:i+1, :].T).T
        # we always USE_KF
        quaternion[i, :],P,Q,R = kalman_filter_tilt(period, quaternion[i, :], P, Q, R, accel_theta[i], accel_phi[i], stationary_periods[i])
        
        last_footfall = zero_velocity_updates(i, FF, An, Anz, last_footfall)
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
    """Filter footfalls to only the walking portion of a recording.

    Uses velocity magnitude peaks to identify the region of sustained walking,
    excluding stationary periods at the start and end. Called internally by
    compute_position() to populate the 'FF_walking' key.

    Parameters:
    -----------
    walk_info : dict
        Output of compute_position() (must contain 'FF' and 'Vm' keys)
    MIN_WALK_SPEED : float
        Minimum peak velocity to count as walking (default: 2.0 m/s)

    Returns:
    --------
    np.ndarray
        Boolean footfall array for the walking section only
    """   
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
    """Segment individual strides from a processed walking recording.

    Takes the output of compute_position() and identifies individual stride
    cycles using the FF_walking footfall array. Each stride runs from one
    footfall to the next. Strides are rotated to a local forward/lateral
    coordinate frame and metrics (speed, length, duration) are computed.

    Parameters:
    -----------
    walk_info : dict
        Output of compute_position()
    period : float
        Sampling period in seconds
    FILTER : int
        If 1, apply outlier filtering to remove abnormal strides (default: 0)
    OUTLIER_SECTION_SECONDS : optional
        Time ranges (seconds) to exclude as outliers

    Returns:
    --------
    dict
        Keys: 'frwd' (forward trajectory per stride), 'ltrl' (lateral),
        'elev' (elevation), 'frwd_speed', 'time', 'step_samples',
        'foot_heading', 'frwd_swing', 'ltrl_swing', 'abs_frwd', 'abs_ltrl',
        'theta', 'start_end', 'diff_foot_heading', 'frwd_speed_compensated'
    """
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
    # display stepData
    print("Detected {} steps after segmentation.".format(len(swing_start)))
    return stepData

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def rotate_angle(X: np.ndarray, Y: np.ndarray, ang: float) -> Tuple[np.ndarray, np.ndarray]:
    Xr = X * np.cos(ang) - Y * np.sin(ang)
    Yr = X * np.sin(ang) + Y * np.cos(ang)
    return Xr, Yr

def get_steps(step_start, step_end, walk_info, PERIOD, FILTER, OUTLIER_SECTION, verbose=False):
    """
    get_steps(step_start, step_end,walk_info,PERIOD,FILTER,OUTLIER_SECTION,verbose=Fals)

    Parameters:
    -----------
    step_start : array-like
        Start indices of steps
    step_end : array-like
        End indices of steps
    walk_info : dict
        Dictionary containing 'P', 'euler', and 'Vm' arrays
    PERIOD : float
        Sampling period
    FILTER : bool
        Whether to apply filtering
    OUTLIER_SECTION : array-like
        Indices of outlier sections to remove
    verbose : bool, optional
        Whether to plot details (default: False)

    Returns:
    --------
    dict
        Dictionary containing stride metrics. If no steps are detected, returns
        a dict with empty arrays for all fields.
    """
    PLOT_DETAILS = verbose

    # Handle case when no steps are detected
    step_start = np.array(step_start).reshape(-1, 1).flatten()
    step_end = np.array(step_end).reshape(-1, 1).flatten()

    if len(step_start) == 0 or len(step_end) == 0:
        # Return empty result structure
        return {
            'frwd_swing': np.array([]).reshape(0, 0),
            'ltrl_swing': np.array([]).reshape(0, 0),
            'frwd': np.array([]).reshape(0, 0),
            'ltrl': np.array([]).reshape(0, 0),
            'abs_ltrl': np.array([]).reshape(0, 0),
            'abs_frwd': np.array([]).reshape(0, 0),
            'elev': np.array([]).reshape(0, 0),
            'theta': np.array([]).reshape(0, 0),
            'start_end': np.array([]).reshape(0, 0),
            'foot_heading': np.array([]),
            'diff_foot_heading': np.array([]),
            'step_samples': np.array([]),
            'time': np.array([]),
            'frwd_speed_compensated': np.array([]),
            'frwd_speed': np.array([])
        }

    # DIRECTION_STEPS, determine the number of steps before and after current one, used to define a straight segment
    DIRECTION_STEPS = 3  # default 3
    # mean_step_direction = 0  # when DIRECTION_STEPS is 0, uncomment this
    
    EXTRA_FRWD_CORRECTION = 1  # default is 1. This will straighten the paths perfectly
    # EXTRA_FRWD_CORRECTION = 0  # for hand reaching Oct 2023

    number_of_steps = len(step_start)
    longest_step = np.max(step_end - step_start) + 1
    
    P = walk_info['P']
    euler = walk_info['euler']
    Vm = walk_info['Vm']
    
    # Create matrices to store results
    ltrl_swing = np.zeros((number_of_steps, longest_step))
    frwd_swing = np.zeros((number_of_steps, longest_step))
    ltrl = np.zeros((number_of_steps, longest_step))
    frwd = np.zeros((number_of_steps, longest_step))
    ltrl_straighten = np.zeros((number_of_steps, longest_step))
    frwd_straighten = np.zeros((number_of_steps, longest_step))
    abs_ltrl = np.zeros((number_of_steps, longest_step))
    abs_frwd = np.zeros((number_of_steps, longest_step))
    elev = np.zeros((number_of_steps, longest_step))
    theta = np.zeros((number_of_steps, longest_step))
    foot_heading = np.zeros(number_of_steps)
    diff_foot_heading = np.zeros(number_of_steps)
    start_end = np.zeros((number_of_steps, 2))
    
    # Compute individual step direction
    direction = np.arctan2(P[step_end, 1] - P[step_start, 1], 
                           P[step_end, 0] - P[step_start, 0])
    
    if PLOT_DETAILS:
        # This is the average walk direction that is used to rotate the trajectory
        Px = P[step_start, 0]
        Py = P[step_start, 1]
        # Linear fit (polyfit with degree 1)
        coeffs = np.polyfit(Px, Py, 1)
        pol = np.poly1d(coeffs)
        # Use the atan2 to determine the right grid quadrant
        overall_step_direction = np.arctan2(pol(Px[-1]) - pol(Px[0]), Px[-1] - Px[0])
        frwd_pol_rot, ltrl_pol_rot = rotate_angle(Px, Py, -overall_step_direction)
        
        PATH_FIG = plt.figure()
        plt.plot(frwd_pol_rot, ltrl_pol_rot, 'k')
        plt.grid(True)
    
    # Unwrap the euler in order to eliminate discontinuities
    walk_foot_heading = np.unwrap(euler[step_end, 2])
    
    # Perform a default line fit correction for heading
    x = np.arange(1, len(walk_foot_heading) + 1)
    y = walk_foot_heading
    coeffs = np.polyfit(x, y, 1)
    pol = np.poly1d(coeffs)
    heading_correction = pol(x)
    corrected_heading = y - heading_correction
    
    if PLOT_DETAILS:
        ANG_FIG = plt.figure()
        plt.plot(walk_foot_heading - coeffs[1])
        plt.hold = True
        plt.grid(True)
        plt.plot(heading_correction - coeffs[1], 'r')
        plt.plot(y - heading_correction, 'g')
    
    step_samples = np.zeros(number_of_steps)
    
    for i in range(number_of_steps):
        step_len = step_end[i] - step_start[i] + 1
        
        # Rotate for swing
        frwd_swing[i, :step_len] = (P[step_start[i]:step_end[i]+1, 0] * np.cos(-direction[i]) - 
                                    P[step_start[i]:step_end[i]+1, 1] * np.sin(-direction[i]))
        frwd_swing[i, step_len:] = frwd_swing[i, step_len-1]
        
        ltrl_swing[i, :step_len] = (P[step_start[i]:step_end[i]+1, 0] * np.sin(-direction[i]) + 
                                    P[step_start[i]:step_end[i]+1, 1] * np.cos(-direction[i]))
        ltrl_swing[i, step_len:] = ltrl_swing[i, step_len-1]
        
        # Uses the nearby steps to determine the angle, is less sensitive to gyro drift
        if DIRECTION_STEPS:
            # Select the steps +/- DIRECTION_STEPS
            if i > DIRECTION_STEPS and number_of_steps - i > DIRECTION_STEPS:
                nearby_steps_index = np.arange(i - DIRECTION_STEPS, i + DIRECTION_STEPS + 1)
            elif i <= DIRECTION_STEPS:
                nearby_steps_index = np.arange(0, min(i + DIRECTION_STEPS + 1, number_of_steps))
            else:
                nearby_steps_index = np.arange(i - DIRECTION_STEPS, number_of_steps)
            
            # Find local direction of travel
            nearby_steps = step_start[nearby_steps_index]
            x_local = P[nearby_steps, 0]
            y_local = P[nearby_steps, 1]
            coeffs_local = np.polyfit(x_local, y_local, 1)
            pol_local = np.poly1d(coeffs_local)
            mean_step_direction = np.arctan2(pol_local(x_local[-1]) - pol_local(x_local[0]), 
                                            x_local[-1] - x_local[0])
            
            # Find a local heading correction
            y_heading = walk_foot_heading[nearby_steps_index]
            x_heading = nearby_steps_index
            coeffs_heading = np.polyfit(x_heading, y_heading, 1)
            pol_heading = np.poly1d(coeffs_heading)
            heading_correction = pol_heading(x_heading)
            current_index = np.where(x_heading == i)[0][0]
            corrected_heading[i] = y_heading[current_index] - heading_correction[current_index]
        else:
            mean_step_direction = 0  # If DIRECTION_STEPS is 0
        
        frwd_rot, ltrl_rot = rotate_angle(P[step_start[i]:step_end[i]+1, 0], 
                                          P[step_start[i]:step_end[i]+1, 1], 
                                          -mean_step_direction)
        frwd[i, :step_len] = frwd_rot
        frwd[i, step_len:] = frwd[i, step_len-1]
        
        ltrl[i, :step_len] = ltrl_rot
        ltrl[i, step_len:] = ltrl[i, step_len-1]
        
        if i > 0:
            ltrl[i, :] = ltrl[i, :] - ltrl[i, 0] + ltrl[i-1, -1]
        else:
            ltrl[i, :] = ltrl[i, :] - ltrl[i, 0]
        
        # Store elevation information
        elev[i, :step_len] = P[step_start[i]:step_end[i]+1, 2]
        elev[i, step_len:] = elev[i, step_len-1]
        
        # Store pitch angle
        theta[i, :step_len] = euler[step_start[i]:step_end[i]+1, 1]
        theta[i, step_len:] = theta[i, step_len-1]
        
        step_samples[i] = step_end[i] - step_start[i]
        
        # Store heading angle
        foot_heading[i] = corrected_heading[i]
        diff_foot_heading[i] = euler[step_end[i], 2] - euler[step_start[i], 2]
        start_end[i, :] = [step_start[i], step_end[i]]
    
    ltrl_end = ltrl[:, -1]
    frwd_end = frwd[:, -1]
    coeffs = np.polyfit(frwd_end, ltrl_end, 1)
    pol = np.poly1d(coeffs)
    frwd_pol = np.linspace(np.min(frwd_end), np.max(frwd_end), 100)
    ltrl_pol = pol(frwd_pol)
    
    if PLOT_DETAILS:
        plt.figure(ANG_FIG.number)
        plt.plot(foot_heading, 'k')
        plt.xlabel('Step #')
        plt.ylabel('Ang [rad]')
        plt.legend(['Org', 'Linear Fit', 'Line-Fit Correction', 'Piecewise Correction'])
        
        plt.figure(PATH_FIG.number)
        plt.plot(frwd[:, -1], ltrl[:, -1], 'b')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.plot(frwd_end, ltrl_end, '*')
        plt.plot(frwd_pol, ltrl_pol, 'b')
        plt.legend(['Line-Fit Correction', 'Piecewise Correction', '', ''])
    
    if EXTRA_FRWD_CORRECTION:
        mean_step_direction = np.arctan(coeffs[0])
        frwd_pol_rot, ltrl_pol_rot = rotate_angle(frwd_pol, ltrl_pol, -mean_step_direction)
        ltrl_pol_rot = ltrl_pol_rot - np.mean(ltrl_pol_rot)
        frwd_end, ltrl_endr = rotate_angle(frwd_end, ltrl_end, -mean_step_direction)
        center_ltrl_end = np.mean(ltrl_endr)
        ltrl_endr = ltrl_endr - center_ltrl_end
        
        for i in range(number_of_steps):
            frwd_straighten[i, :], ltrl_straighten[i, :] = rotate_angle(frwd[i, :], ltrl[i, :], 
                                                                        -mean_step_direction)
            ltrl_straighten[i, :] = ltrl_straighten[i, :] - center_ltrl_end
        
        ltrl = ltrl_straighten
        frwd = frwd_straighten
        
        if PLOT_DETAILS:
            plt.plot(frwd_pol_rot, ltrl_pol_rot, 'g')
            plt.plot(frwd_end, ltrl_endr, 'g*')
            plt.plot(frwd_straighten[:, -1], ltrl_straighten[:, -1], 'g')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.legend(['Line-Fit Correction', 'Piecewise Correction', '', '', 'Best correction'])
    
    # Translate foot fall location to the origin
    frwd_swing = frwd_swing - frwd_swing[:, 0:1]
    abs_frwd = frwd.copy()
    frwd = frwd - frwd[:, 0:1]
    ltrl_swing = ltrl_swing - ltrl_swing[:, 0:1]
    abs_ltrl = ltrl.copy()
    ltrl = ltrl - ltrl[:, 0:1]
    elev = elev - elev[:, 0:1]
    # theta = theta - theta[:, 0:1]  # commented in original
    
    # Check for negative values in frwd
    if np.any(frwd[:, -1] < 0):
        print('WARNING: NEGATIVE IN FORWARD DIRECTION DETECTED IN STEPS!')
        frwd = np.abs(frwd)
    
    # Compute step speed
    step_length = frwd[:, -1]
    time = step_samples * PERIOD
    step_speed = step_length / time
    coeffs = np.polyfit(step_speed, step_length, 1)
    pol = np.poly1d(coeffs)
    step_length_fit = pol(step_speed)
    frwd_speed_compensated = step_length - step_length_fit
    frwd_speed = step_speed
    
    # Compute first order statistics and assemble result structure
    result = {
        'frwd_swing': frwd_swing.T,
        'ltrl_swing': ltrl_swing.T,
        'frwd': frwd.T,
        'ltrl': ltrl.T,
        'abs_ltrl': abs_ltrl.T,
        'abs_frwd': abs_frwd.T,
        'elev': elev.T,
        'theta': theta.T,
        'start_end': start_end.T,
        'foot_heading': foot_heading,
        'diff_foot_heading': diff_foot_heading,
        'step_samples': step_samples,
        'time': time,
        'frwd_speed_compensated': frwd_speed_compensated,
        'frwd_speed': frwd_speed
    }
    
    # Eliminate user defined outliers
    out_of_bound = []
    for i in range(number_of_steps):
        if step_start[i] in OUTLIER_SECTION or step_end[i] in OUTLIER_SECTION:
            out_of_bound.append(i)
    
    if out_of_bound:
        print('User defined outliers')
        result, number_of_steps = cut_step_section(result, out_of_bound, number_of_steps)
    
    if PLOT_DETAILS:
        t = np.arange(len(Vm)) * PERIOD
        plt.figure()
        plt.plot(t, Vm)
        if len(OUTLIER_SECTION) > 0:
            plt.plot(t[OUTLIER_SECTION], Vm[OUTLIER_SECTION], '.y')
        plt.plot(t[step_start[0]], Vm[step_start[0]], '*g')
        plt.plot(t[step_end[-1]], Vm[step_end[-1]], '*r')
        plt.xlabel('Sample #')
        plt.ylabel('Speed [m/sec]')
        plt.legend(['Vm', 'Start', 'End'])
        plt.grid(True)
        plt.show()
    
    if FILTER:
        result, number_of_steps = filter_steps(result, number_of_steps)
    
    return result


def cut_step_section(stride, out_of_bound, number_of_steps):
    """
    Remove outlier steps from the stride data
    
    Parameters:
    -----------
    stride : dict
        Dictionary containing all step data
    out_of_bound : list
        Indices of steps to remove
    number_of_steps : int
        Current number of steps
    
    Returns:
    --------
    stride : dict
        Updated stride dictionary with outliers removed
    number_of_steps : int
        New number of steps after removal
    """
    if out_of_bound:
        # Delete columns for 2D arrays (note: in Python, we need to use np.delete)
        stride['frwd_swing'] = np.delete(stride['frwd_swing'], out_of_bound, axis=1)
        stride['ltrl_swing'] = np.delete(stride['ltrl_swing'], out_of_bound, axis=1)
        stride['frwd'] = np.delete(stride['frwd'], out_of_bound, axis=1)
        stride['ltrl'] = np.delete(stride['ltrl'], out_of_bound, axis=1)
        stride['abs_ltrl'] = np.delete(stride['abs_ltrl'], out_of_bound, axis=1)
        stride['abs_frwd'] = np.delete(stride['abs_frwd'], out_of_bound, axis=1)
        stride['elev'] = np.delete(stride['elev'], out_of_bound, axis=1)
        stride['theta'] = np.delete(stride['theta'], out_of_bound, axis=1)
        stride['start_end'] = np.delete(stride['start_end'], out_of_bound, axis=1)
        
        # Delete elements for 1D arrays
        stride['foot_heading'] = np.delete(stride['foot_heading'], out_of_bound)
        stride['diff_foot_heading'] = np.delete(stride['diff_foot_heading'], out_of_bound)
        stride['step_samples'] = np.delete(stride['step_samples'], out_of_bound)
        stride['time'] = np.delete(stride['time'], out_of_bound)
        stride['frwd_speed_compensated'] = np.delete(stride['frwd_speed_compensated'], out_of_bound)
        stride['frwd_speed'] = np.delete(stride['frwd_speed'], out_of_bound)
        
        new_number_of_steps = number_of_steps - len(out_of_bound)
        print(f'New number of steps {new_number_of_steps} out of {number_of_steps}')
        number_of_steps = new_number_of_steps
    
    return stride, number_of_steps


def filter_steps(stride, number_of_steps):
    """
    Filter out steps that are not within known specifications
    
    Parameters:
    -----------
    stride : dict
        Dictionary containing all step data
    number_of_steps : int
        Current number of steps
        
    Returns:
    --------
    stride : dict
        Filtered stride dictionary
    number_of_steps : int
        New number of steps after filtering
    """
    # Filter parameters
    MAX_STEP_LENGTH = 1.8  # Used to eliminate very long steps likely caused by non-detected footfalls
    MIN_STEP_LENGTH = 0.5  # Used to eliminate very short steps likely caused by non-detected footfalls
    MAX_VAR = 2  # Eliminates outliers based on the variance from the median value
    FILTER_ELEVATION = 0
    
    # Some of the filters use a double STD based filtering process
    
    # Eliminate long steps above the maximum limit
    frwd = stride['frwd'].T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = frwd[:, -1] > MAX_STEP_LENGTH
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd LONG')
        stride, number_of_steps = cut_step_section(stride, outlier, number_of_steps)
    
    # Eliminate very short steps
    frwd = stride['frwd'].T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = frwd[:, -1] < MIN_STEP_LENGTH
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd SHORT')
        stride, number_of_steps = cut_step_section(stride, outlier, number_of_steps)
    
    # Eliminate steps away from the length median value
    frwd = stride['frwd'].T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = np.abs(frwd[:, -1] - median_pos_frwd) > std_pos_frwd * MAX_VAR
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd +2 VAR')
        stride, number_of_steps = cut_step_section(stride, outlier, number_of_steps)
    
    # Eliminate steps that have too much side deviation
    ltrl = stride['ltrl'].T
    std_pos_ltrl = np.std(ltrl[:, -1])
    outlier = np.abs(ltrl[:, -1]) > std_pos_ltrl * MAX_VAR
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_ltrl +2 VAR')
        stride, number_of_steps = cut_step_section(stride, outlier, number_of_steps)
    
    if FILTER_ELEVATION:
        # Eliminate steps that have too much vertical deviation
        elev = stride['elev'].T
        median_pos_elev = np.median(elev[:, -1])
        std_pos_elev = np.std(elev[:, -1])
        outlier = np.abs(elev[:, -1] - median_pos_elev) > std_pos_elev * MAX_VAR
        outlier = np.where(outlier)[0]
        if len(outlier) > 0:
            print('_elev +2 VAR steps')
            stride, number_of_steps = cut_step_section(stride, outlier, number_of_steps)

    return stride, number_of_steps


def foot_fall_opposite_velocity(Vm: np.ndarray, FF_orig: np.ndarray, period: float,
                                  merge_mode: str = 'MAX_SPEED_OR_ORIG',
                                  plot_details: bool = False, WALK_SPEED_PERCENTAGE = 0.8,MIN_WALK_SPEED = 2,ACCEL_SLOW_STEPS = 1,STEP_DURATION_VARIABILITY = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect foot falls using the velocity of the opposite foot (temporally, should be near max other-foot speed)

    Parameters:
    -----------
    Vm : np.ndarray
        Velocity magnitude array
    FF_org : np.ndarray
        Original footfall detection boolean array
    period : float
        Sampling period
    merge_mode : str, optional
        Merge strategy (default: 'MAX_SPEED_OR_ORIG')
    plot_details : bool, optional
        Whether to plot debugging information (default: False)

    WALK_SPEED_PERCENTAGE = 0.8  # as a percentage of median speed; For normal walk use .8, for varying speed use .5
    MIN_WALK_SPEED = 2  # Use 2 for normal walk, 1.2 for varying speeds
    ACCEL_SLOW_STEPS = 1  # Number of steps used during acceleration and slowing down phase
    STEP_DURATION_VARIABILITY = 0.5  # Use .5 for normal walk and 1.9+ for varying speed

        
    Returns:
    --------
    tuple
        (FF, FF_walking, FF_max_speed) - footfall arrays
    """

    MIN_FF_SEPARATION = int(np.floor((1/period) / 2.5))
    
    # Find the location of the maximum velocities for the opposite foot
    peaks_idx, _ = find_peaks(Vm, height=MIN_WALK_SPEED, distance=MIN_FF_SEPARATION)

    # Designate max velocity points as likely footfalls for opposite foot
    FF_max_speed = np.zeros(len(Vm), dtype=bool)
    FF_max_speed[peaks_idx] = True

    # Determine the longest section with continuous motion
    walking_ff_time = np.diff(peaks_idx)
    median_ff_time = np.median(walking_ff_time)
    walk_section = (walking_ff_time < median_ff_time * (1 + STEP_DURATION_VARIABILITY))

    # Add padding to find start/end of walks
    B = np.concatenate(([0], walk_section, [0]))
    start_walks = np.where(np.diff(B.astype(int)) == 1)[0]
    end_walks = np.where(np.diff(B.astype(int)) == -1)[0]
    walk_sizes = end_walks - start_walks
    idx_walk = np.argmax(walk_sizes)
    start_walk = start_walks[idx_walk]
    end_walk = end_walks[idx_walk]

    # Eliminate first and last step, which may correspond to acceleration and slowing down
    start_walk = start_walk + ACCEL_SLOW_STEPS
    end_walk = end_walk - ACCEL_SLOW_STEPS

    if plot_details:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(walking_ff_time)
        plt.plot(walk_section * median_ff_time, 'ok')
        plt.plot(start_walk, walking_ff_time[start_walk], '.g')
        plt.plot(end_walk, walking_ff_time[end_walk], '.r')
        plt.legend(['Time between steps', 'Median walk time', 'Beginning of Walk', 'End of Walk'])
        plt.title('Stride duration')

        plt.figure()
        plt.plot(Vm)
        plt.plot(peaks_idx, Vm[peaks_idx], '.k')
        plt.plot(peaks_idx[start_walk], Vm[peaks_idx[start_walk]], '*g')
        plt.plot(peaks_idx[end_walk], Vm[peaks_idx[end_walk]], '*r')
        plt.legend(['Speed', 'Max Speed', 'Beginning of Walk', 'End of Walk'])
        plt.title('Increase STEP_DURATION_VARIABILITY until it includes all the walking area')

    # Walking portion is defined at the point that the speed reaches WALK_SPEED_PERCENTAGE of the median speed value
    median_vel = np.mean(Vm[FF_max_speed])
    likely_walk_sections = np.where(Vm > median_vel * WALK_SPEED_PERCENTAGE)[0]
    FF_stand_still_mask = np.zeros(len(FF_orig), dtype=bool)
    FF_stand_still_mask[:likely_walk_sections[0]] = True
    FF_stand_still_mask[likely_walk_sections[-1]:] = True

    FF_walking = np.zeros(len(FF_max_speed), dtype=bool)
    FF_walking[peaks_idx[start_walk]:peaks_idx[end_walk]+1] = FF_max_speed[peaks_idx[start_walk]:peaks_idx[end_walk]+1]
    FF_walking = FF_walking & ~FF_stand_still_mask

    # Footfall detection based on velocities of the opposite shoe
    # Compute a footfall region
    if merge_mode == 'LARGE_SPEED_AND_ORIG':
        # Uses large speed and original solutions combined
        FF = FF_orig | FF_max_speed
    elif merge_mode == 'MAX_SPEED_AND_ORIG':
        # Uses maximum speed and original solutions combined
        FF = FF_orig | FF_max_speed
    elif merge_mode == 'MAX_SPEED_OR_ORIG':
        # Uses maximum speed as the first option, if it is not available, it will use the original solution
        # Use original FFs when the person stands still even in the middle of the trial
        # force a FF at the beginning and end of trial
        FF = (FF_stand_still_mask & FF_orig) | FF_max_speed
    elif merge_mode == 'ORIG_UNCOUPLED':
        # This method will use the same results as foot_fall
        FF = FF_orig
    else:
        raise ValueError(f"Unknown merge mode: {merge_mode}")

    if plot_details:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(Vm)
        plt.plot(np.where(FF)[0], Vm[FF], 'o', markersize=12,
                markerfacecolor='k', markeredgecolor='k', label='Foot-fall')
        plt.plot(np.where(FF_max_speed)[0], Vm[FF_max_speed], 'o', markersize=9,
                markerfacecolor='g', markeredgecolor='g', label='FF max speed')
        plt.plot(np.where(FF_orig)[0], Vm[FF_orig], 'o', markersize=6,
                markerfacecolor='r', markeredgecolor='r', label='FF org')
        plt.plot(np.where(FF_walking)[0], Vm[FF_walking], 'o', markersize=4,
                markerfacecolor='y', markeredgecolor='m', label='FF walk')
        plt.legend()

    return FF, FF_walking, FF_max_speed


def compute_position_two_imus(leftWb: np.ndarray, leftAb: np.ndarray,
                         rightWb: np.ndarray, rightAb: np.ndarray,
                         period: float,
                         USE_KF: int = 1,
                         W_FF: Optional[float] = None,
                         A_FF: Optional[float] = None,
                         FFL: Optional[np.ndarray] = None,
                         FFR: Optional[np.ndarray] = None,
                         plot_details: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process two synchronized foot IMUs together.

    Runs compute_position() on each foot independently, then re-detects
    footfalls with tuned parameters and recomputes ZUPT-corrected velocities
    and positions. Produces diagnostic plots of both feet.

    Parameters:
    -----------
    leftWb : np.ndarray
        Left foot angular velocity (body frame)
    leftAb : np.ndarray
        Left foot acceleration (body frame)
    rightWb : np.ndarray
        Right foot angular velocity (body frame)
    rightAb : np.ndarray
        Right foot acceleration (body frame)
    period : float
        Sampling period
    USE_KF : int, optional
        Whether to use Kalman filter (default: 1)
    W_FF : float, optional
        Angular velocity threshold for footfall detection
    A_FF : float, optional
        Acceleration threshold for footfall detection
    FFL : np.ndarray, optional
        Pre-computed footfall array for left foot
    FFR : np.ndarray, optional
        Pre-computed footfall array for right foot
    plot_details : bool, optional
        Whether to create debugging plots (default: False)

    Returns:
    --------
    tuple
        (left_walk_info, right_walk_info) - dictionaries containing walking information
    """
    # Compute individual foot paths
    # Process each foot individually
    # Left foot
    left_walk_info = compute_position(leftWb, leftAb, period, USE_KF, W_FF, A_FF, FF=FFL)

    # Right foot
    right_walk_info = compute_position(rightWb, rightAb, period, USE_KF, W_FF, A_FF, FF=FFR)

    # Verify that both IMUs collected the same amount of information
    left_samples = len(left_walk_info['FF'])
    right_samples = len(right_walk_info['FF'])
    if left_samples != right_samples:
        raise ValueError(f'Files may not be synced: L = {left_samples}, R = {right_samples}, define a SECTION')

    SAMPLES = left_samples
    t = np.arange(SAMPLES) * period

    # Merge/Combine previous FF detection with new one based on cross-velocity
    # Notice that the footfall depends on information of the opposite foot
    # left_walk_info['FF'], left_walk_info['FF_walking'], left_walk_info['FF_max_speed'] = \
    #     foot_fall_opposite_velocity(right_walk_info['Vm'], left_walk_info['FF'], period)

    # right_walk_info['FF'], right_walk_info['FF_walking'], right_walk_info['FF_max_speed'] = \
    #     foot_fall_opposite_velocity(left_walk_info['Vm'], right_walk_info['FF'], period)
    left_walk_info['FF'],left_walk_info['stationary_periods'] = foot_fall(leftWb, leftAb, period, W_FF=30, A_FF=1, T_FF=.4, MAX_T_FF=10)
    # duplicate for FF_walking
    left_walk_info['FF_walking'] = left_walk_info['FF'].copy()
    right_walk_info['FF'], right_walk_info['stationary_periods'] = foot_fall(rightWb, rightAb, period, W_FF=30, A_FF=1, T_FF=.4, MAX_T_FF=10)
    # duplicate for FF_walking
    right_walk_info['FF_walking'] = right_walk_info['FF'].copy()

    MAX_NUMBER_STEP_DIFF = 2  # Default 2
    left_ff_count = np.sum(left_walk_info['FF_walking'])
    right_ff_count = np.sum(right_walk_info['FF_walking'])
    if abs(left_ff_count - right_ff_count) > MAX_NUMBER_STEP_DIFF:
        print('Warning! incompatible number of steps, this needs to be fixed.')
        print('Check MIN_WALK_SPEED in foot_fall_opposite_velocity')
        print(f'Left: {left_ff_count}, Right: {right_ff_count}')

    # Recompute accelerations (ZUPT) using combined FFs
    # Left foot
    An = left_walk_info['An']
    FF = left_walk_info['FF']
    Anz = np.zeros((SAMPLES, 3))
    last_footfall = 0
    for i in range(1, SAMPLES):
        last_footfall = zero_velocity_updates(i, FF, An, Anz, last_footfall)

    left_walk_info['Anz'] = Anz
    left_walk_info['V'] = np.cumsum(Anz, axis=0) * period
    left_walk_info['P'] = np.cumsum(left_walk_info['V'], axis=0) * period
    left_walk_info['Vm'] = np.sqrt(np.sum(left_walk_info['V'] ** 2, axis=1))

    # Right foot
    An = right_walk_info['An']
    FF = right_walk_info['FF']
    Anz = np.zeros((SAMPLES, 3))
    last_footfall = 0
    for i in range(1, SAMPLES):
        last_footfall = zero_velocity_updates(i, FF, An, Anz, last_footfall)

    right_walk_info['Anz'] = Anz
    right_walk_info['V'] = np.cumsum(Anz, axis=0) * period
    right_walk_info['P'] = np.cumsum(right_walk_info['V'], axis=0) * period
    right_walk_info['Vm'] = np.sqrt(np.sum(right_walk_info['V'] ** 2, axis=1))

    # Make Plots
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(left_walk_info['P'][left_walk_info['FF'], 0],
            left_walk_info['P'][left_walk_info['FF'], 1],
            left_walk_info['P'][left_walk_info['FF'], 2],
            '.g', label='left')
    ax.plot(right_walk_info['P'][right_walk_info['FF'], 0],
            right_walk_info['P'][right_walk_info['FF'], 1],
            right_walk_info['P'][right_walk_info['FF'], 2],
            '.r', label='right')
    ax.plot(left_walk_info['P'][:, 0],
            left_walk_info['P'][:, 1],
            left_walk_info['P'][:, 2], 'b-')
    ax.plot(right_walk_info['P'][:, 0],
            right_walk_info['P'][:, 1],
            right_walk_info['P'][:, 2], 'r-')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.grid(True)
    plt.axis('equal')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, left_walk_info['Vm'])
    axes[0].plot(t[left_walk_info['FF']], left_walk_info['Vm'][left_walk_info['FF']], 'g*')
    axes[0].plot(t[left_walk_info['FF_walking']], left_walk_info['Vm'][left_walk_info['FF_walking']], 'k.')
    axes[0].grid(True)
    axes[0].set_ylabel('Left |V| [m/s]')
    axes[0].set_xlabel('time [s]')
    axes[0].legend(['|V|', 'FFs', 'Walking'])
    axes[0].set_title('Foot Fall detection using opposite shoe speed')

    axes[1].plot(t, right_walk_info['Vm'])
    axes[1].plot(t[right_walk_info['FF']], right_walk_info['Vm'][right_walk_info['FF']], 'g*')
    axes[1].plot(t[right_walk_info['FF_walking']], right_walk_info['Vm'][right_walk_info['FF_walking']], 'k.')
    axes[1].grid(True)
    axes[1].set_ylabel('Right |V| [m/s]')
    axes[1].set_xlabel('time [s]')

    if plot_details:
        left_az_mag = np.sqrt(np.sum(left_walk_info['Anz'] ** 2, axis=1))
        right_az_mag = np.sqrt(np.sum(right_walk_info['Anz'] ** 2, axis=1))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(t, left_walk_info['Anz'], t, left_az_mag, 'k')
        axes[0].plot(t[left_walk_info['FF']], left_walk_info['Vm'][left_walk_info['FF']], 'g.')
        axes[0].plot(t[left_walk_info['FF_walking']], left_walk_info['Vm'][left_walk_info['FF_walking']], 'yo')
        axes[0].grid(True)
        axes[0].set_ylabel('left A [m/s^2]')
        axes[0].set_xlabel('time [s]')

        axes[1].plot(t, right_walk_info['Anz'], t, right_az_mag, 'k')
        axes[1].plot(t[right_walk_info['FF']], right_walk_info['Vm'][right_walk_info['FF']], 'g.')
        axes[1].plot(t[right_walk_info['FF_walking']], right_walk_info['Vm'][right_walk_info['FF_walking']], 'yo')
        axes[1].grid(True)
        axes[1].set_ylabel('right A [m/s^2]')
        axes[1].set_xlabel('time [s]')

    return left_walk_info, right_walk_info