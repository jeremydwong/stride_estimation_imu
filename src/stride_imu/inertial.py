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
        stride_range = np.arange(last_footfall+1, i+1)
        if stride_range.size < 2:
            return last_footfall
        stride_samples = stride_range.size
        velocity_error = np.sum(An[stride_range, :], axis=0)
        acceleration_error = velocity_error / stride_samples
        Anz[stride_range, :] = An[stride_range, :] - acceleration_error
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
        'elev' (elevation), 'frwd_speed', 'time', 'stride_samples',
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
    strides = get_strides(swing_start, swing_finish, walk_info, period, FILTER, OUTLIER_SECTION_SAMPLES)
    # display strides
    print("Detected {} strides after segmentation.".format(len(swing_start)))
    return strides

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def rotate_angle(X: np.ndarray, Y: np.ndarray, ang: float) -> Tuple[np.ndarray, np.ndarray]:
    Xr = X * np.cos(ang) - Y * np.sin(ang)
    Yr = X * np.sin(ang) + Y * np.cos(ang)
    return Xr, Yr

def get_strides(stride_start, stride_end, walk_info, PERIOD, FILTER, OUTLIER_SECTION, verbose=False):
    """
    get_strides(stride_start, stride_end,walk_info,PERIOD,FILTER,OUTLIER_SECTION,verbose=Fals)

    Parameters:
    -----------
    stride_start : array-like
        Start indices of strides
    stride_end : array-like
        End indices of strides
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
        Dictionary containing stride metrics. If no strides are detected, returns
        a dict with empty arrays for all fields.
    """
    PLOT_DETAILS = verbose

    # Handle case when no strides are detected
    stride_start = np.array(stride_start).reshape(-1, 1).flatten()
    stride_end = np.array(stride_end).reshape(-1, 1).flatten()

    if len(stride_start) == 0 or len(stride_end) == 0:
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
            'stride_samples': np.array([]),
            'time': np.array([]),
            'frwd_speed_compensated': np.array([]),
            'frwd_speed': np.array([])
        }

    # DIRECTION_STRIDES, determine the number of strides before and after current one, used to define a straight segment
    DIRECTION_STRIDES = 3  # default 3
    # mean_stride_direction = 0  # when DIRECTION_STRIDES is 0, uncomment this
    
    EXTRA_FRWD_CORRECTION = 1  # default is 1. This will straighten the paths perfectly
    # EXTRA_FRWD_CORRECTION = 0  # for hand reaching Oct 2023

    number_of_strides = len(stride_start)
    longest_stride = np.max(stride_end - stride_start) + 1
    
    P = walk_info['P']
    euler = walk_info['euler']
    Vm = walk_info['Vm']
    
    # Create matrices to store results
    ltrl_swing = np.zeros((number_of_strides, longest_stride))
    frwd_swing = np.zeros((number_of_strides, longest_stride))
    ltrl = np.zeros((number_of_strides, longest_stride))
    frwd = np.zeros((number_of_strides, longest_stride))
    ltrl_straighten = np.zeros((number_of_strides, longest_stride))
    frwd_straighten = np.zeros((number_of_strides, longest_stride))
    abs_ltrl = np.zeros((number_of_strides, longest_stride))
    abs_frwd = np.zeros((number_of_strides, longest_stride))
    elev = np.zeros((number_of_strides, longest_stride))
    theta = np.zeros((number_of_strides, longest_stride))
    foot_heading = np.zeros(number_of_strides)
    diff_foot_heading = np.zeros(number_of_strides)
    start_end = np.zeros((number_of_strides, 2))
    
    # Compute individual stride direction
    direction = np.arctan2(P[stride_end, 1] - P[stride_start, 1], 
                           P[stride_end, 0] - P[stride_start, 0])
    
    if PLOT_DETAILS:
        # This is the average walk direction that is used to rotate the trajectory
        Px = P[stride_start, 0]
        Py = P[stride_start, 1]
        # Linear fit (polyfit with degree 1)
        coeffs = np.polyfit(Px, Py, 1)
        pol = np.poly1d(coeffs)
        # Use the atan2 to determine the right grid quadrant
        overall_stride_direction = np.arctan2(pol(Px[-1]) - pol(Px[0]), Px[-1] - Px[0])
        frwd_pol_rot, ltrl_pol_rot = rotate_angle(Px, Py, -overall_stride_direction)
        
        PATH_FIG = plt.figure()
        plt.plot(frwd_pol_rot, ltrl_pol_rot, 'k')
        plt.grid(True)
    
    # Unwrap the euler in order to eliminate discontinuities
    walk_foot_heading = np.unwrap(euler[stride_end, 2])
    
    # Perform a default line fit correction for heading
    # (a line fit needs at least 2 strides; with 1 the correction is just y itself)
    x = np.arange(1, len(walk_foot_heading) + 1)
    y = walk_foot_heading
    if len(y) >= 2:
        coeffs = np.polyfit(x, y, 1)
    else:
        coeffs = np.array([0.0, y[0]])
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
    
    stride_samples = np.zeros(number_of_strides)
    
    for i in range(number_of_strides):
        stride_len = stride_end[i] - stride_start[i] + 1
        
        # Rotate for swing
        frwd_swing[i, :stride_len] = (P[stride_start[i]:stride_end[i]+1, 0] * np.cos(-direction[i]) - 
                                    P[stride_start[i]:stride_end[i]+1, 1] * np.sin(-direction[i]))
        frwd_swing[i, stride_len:] = frwd_swing[i, stride_len-1]
        
        ltrl_swing[i, :stride_len] = (P[stride_start[i]:stride_end[i]+1, 0] * np.sin(-direction[i]) + 
                                    P[stride_start[i]:stride_end[i]+1, 1] * np.cos(-direction[i]))
        ltrl_swing[i, stride_len:] = ltrl_swing[i, stride_len-1]
        
        # Uses the nearby strides to determine the angle, is less sensitive to gyro drift
        if DIRECTION_STRIDES:
            # Select the strides +/- DIRECTION_STRIDES
            if i > DIRECTION_STRIDES and number_of_strides - i > DIRECTION_STRIDES:
                nearby_strides_index = np.arange(i - DIRECTION_STRIDES, i + DIRECTION_STRIDES + 1)
            elif i <= DIRECTION_STRIDES:
                nearby_strides_index = np.arange(0, min(i + DIRECTION_STRIDES + 1, number_of_strides))
            else:
                nearby_strides_index = np.arange(i - DIRECTION_STRIDES, number_of_strides)
            
            # Find local direction of travel
            # (with a single stride there is no neighborhood to fit: fall back
            # to the stride's own direction and no heading correction)
            nearby_strides = stride_start[nearby_strides_index]
            if len(nearby_strides) >= 2:
                x_local = P[nearby_strides, 0]
                y_local = P[nearby_strides, 1]
                coeffs_local = np.polyfit(x_local, y_local, 1)
                pol_local = np.poly1d(coeffs_local)
                mean_stride_direction = np.arctan2(pol_local(x_local[-1]) - pol_local(x_local[0]),
                                                x_local[-1] - x_local[0])

                # Find a local heading correction
                y_heading = walk_foot_heading[nearby_strides_index]
                x_heading = nearby_strides_index
                coeffs_heading = np.polyfit(x_heading, y_heading, 1)
                pol_heading = np.poly1d(coeffs_heading)
                heading_correction = pol_heading(x_heading)
                current_index = np.where(x_heading == i)[0][0]
                corrected_heading[i] = y_heading[current_index] - heading_correction[current_index]
            else:
                mean_stride_direction = direction[i]
                corrected_heading[i] = 0.0
        else:
            mean_stride_direction = 0  # If DIRECTION_STRIDES is 0
        
        frwd_rot, ltrl_rot = rotate_angle(P[stride_start[i]:stride_end[i]+1, 0], 
                                          P[stride_start[i]:stride_end[i]+1, 1], 
                                          -mean_stride_direction)
        frwd[i, :stride_len] = frwd_rot
        frwd[i, stride_len:] = frwd[i, stride_len-1]
        
        ltrl[i, :stride_len] = ltrl_rot
        ltrl[i, stride_len:] = ltrl[i, stride_len-1]
        
        if i > 0:
            ltrl[i, :] = ltrl[i, :] - ltrl[i, 0] + ltrl[i-1, -1]
        else:
            ltrl[i, :] = ltrl[i, :] - ltrl[i, 0]
        
        # Store elevation information
        elev[i, :stride_len] = P[stride_start[i]:stride_end[i]+1, 2]
        elev[i, stride_len:] = elev[i, stride_len-1]
        
        # Store pitch angle
        theta[i, :stride_len] = euler[stride_start[i]:stride_end[i]+1, 1]
        theta[i, stride_len:] = theta[i, stride_len-1]
        
        stride_samples[i] = stride_end[i] - stride_start[i]
        
        # Store heading angle
        foot_heading[i] = corrected_heading[i]
        diff_foot_heading[i] = euler[stride_end[i], 2] - euler[stride_start[i], 2]
        start_end[i, :] = [stride_start[i], stride_end[i]]
    
    ltrl_end = ltrl[:, -1]
    frwd_end = frwd[:, -1]
    coeffs = np.polyfit(frwd_end, ltrl_end, 1)
    pol = np.poly1d(coeffs)
    frwd_pol = np.linspace(np.min(frwd_end), np.max(frwd_end), 100)
    ltrl_pol = pol(frwd_pol)
    
    if PLOT_DETAILS:
        plt.figure(ANG_FIG.number)
        plt.plot(foot_heading, 'k')
        plt.xlabel('Stride #')
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
        mean_stride_direction = np.arctan(coeffs[0])
        frwd_pol_rot, ltrl_pol_rot = rotate_angle(frwd_pol, ltrl_pol, -mean_stride_direction)
        ltrl_pol_rot = ltrl_pol_rot - np.mean(ltrl_pol_rot)
        frwd_end, ltrl_endr = rotate_angle(frwd_end, ltrl_end, -mean_stride_direction)
        center_ltrl_end = np.mean(ltrl_endr)
        ltrl_endr = ltrl_endr - center_ltrl_end
        
        for i in range(number_of_strides):
            frwd_straighten[i, :], ltrl_straighten[i, :] = rotate_angle(frwd[i, :], ltrl[i, :], 
                                                                        -mean_stride_direction)
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
        print('WARNING: NEGATIVE IN FORWARD DIRECTION DETECTED IN STRIDES!')
        frwd = np.abs(frwd)
    
    # Compute stride speed
    stride_length = frwd[:, -1]
    time = stride_samples * PERIOD
    stride_speed = stride_length / time
    coeffs = np.polyfit(stride_speed, stride_length, 1)
    pol = np.poly1d(coeffs)
    stride_length_fit = pol(stride_speed)
    frwd_speed_compensated = stride_length - stride_length_fit
    frwd_speed = stride_speed
    
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
        'stride_samples': stride_samples,
        'time': time,
        'frwd_speed_compensated': frwd_speed_compensated,
        'frwd_speed': frwd_speed
    }
    
    # Eliminate user defined outliers
    out_of_bound = []
    for i in range(number_of_strides):
        if stride_start[i] in OUTLIER_SECTION or stride_end[i] in OUTLIER_SECTION:
            out_of_bound.append(i)
    
    if out_of_bound:
        print('User defined outliers')
        result, number_of_strides = cut_stride_section(result, out_of_bound, number_of_strides)
    
    if PLOT_DETAILS:
        t = np.arange(len(Vm)) * PERIOD
        plt.figure()
        plt.plot(t, Vm)
        if len(OUTLIER_SECTION) > 0:
            plt.plot(t[OUTLIER_SECTION], Vm[OUTLIER_SECTION], '.y')
        plt.plot(t[stride_start[0]], Vm[stride_start[0]], '*g')
        plt.plot(t[stride_end[-1]], Vm[stride_end[-1]], '*r')
        plt.xlabel('Sample #')
        plt.ylabel('Speed [m/sec]')
        plt.legend(['Vm', 'Start', 'End'])
        plt.grid(True)
        plt.show()
    
    if FILTER:
        result, number_of_strides = filter_strides(result, number_of_strides)
    
    return result


def cut_stride_section(stride, out_of_bound, number_of_strides):
    """
    Remove outlier strides from the stride data
    
    Parameters:
    -----------
    stride : dict
        Dictionary containing all stride data
    out_of_bound : list
        Indices of strides to remove
    number_of_strides : int
        Current number of strides
    
    Returns:
    --------
    stride : dict
        Updated stride dictionary with outliers removed
    number_of_strides : int
        New number of strides after removal
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
        stride['stride_samples'] = np.delete(stride['stride_samples'], out_of_bound)
        stride['time'] = np.delete(stride['time'], out_of_bound)
        stride['frwd_speed_compensated'] = np.delete(stride['frwd_speed_compensated'], out_of_bound)
        stride['frwd_speed'] = np.delete(stride['frwd_speed'], out_of_bound)
        
        new_number_of_strides = number_of_strides - len(out_of_bound)
        print(f'New number of strides {new_number_of_strides} out of {number_of_strides}')
        number_of_strides = new_number_of_strides
    
    return stride, number_of_strides


def filter_strides(stride, number_of_strides):
    """
    Filter out strides that are not within known specifications
    
    Parameters:
    -----------
    stride : dict
        Dictionary containing all stride data
    number_of_strides : int
        Current number of strides
        
    Returns:
    --------
    stride : dict
        Filtered stride dictionary
    number_of_strides : int
        New number of strides after filtering
    """
    # Filter parameters
    MAX_STRIDE_LENGTH = 1.8  # Used to eliminate very long strides likely caused by non-detected footfalls
    MIN_STRIDE_LENGTH = 0.5  # Used to eliminate very short strides likely caused by non-detected footfalls
    MAX_VAR = 2  # Eliminates outliers based on the variance from the median value
    FILTER_ELEVATION = 0
    
    # Some of the filters use a double STD based filtering process
    
    # Eliminate long strides above the maximum limit
    frwd = stride['frwd'].T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = frwd[:, -1] > MAX_STRIDE_LENGTH
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd LONG')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    # Eliminate very short strides
    frwd = stride['frwd'].T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = frwd[:, -1] < MIN_STRIDE_LENGTH
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd SHORT')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    # Eliminate strides away from the length median value
    frwd = stride['frwd'].T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = np.abs(frwd[:, -1] - median_pos_frwd) > std_pos_frwd * MAX_VAR
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd +2 VAR')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    # Eliminate strides that have too much side deviation
    ltrl = stride['ltrl'].T
    std_pos_ltrl = np.std(ltrl[:, -1])
    outlier = np.abs(ltrl[:, -1]) > std_pos_ltrl * MAX_VAR
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_ltrl +2 VAR')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    if FILTER_ELEVATION:
        # Eliminate strides that have too much vertical deviation
        elev = stride['elev'].T
        median_pos_elev = np.median(elev[:, -1])
        std_pos_elev = np.std(elev[:, -1])
        outlier = np.abs(elev[:, -1] - median_pos_elev) > std_pos_elev * MAX_VAR
        outlier = np.where(outlier)[0]
        if len(outlier) > 0:
            print('_elev +2 VAR strides')
            stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)

    return stride, number_of_strides


def foot_fall_opposite_velocity(Vm: np.ndarray, FF_orig: np.ndarray, period: float,
                                  merge_mode: str = 'MAX_SPEED_OR_ORIG',
                                  plot_details: bool = False, WALK_SPEED_PERCENTAGE = 0.8,MIN_WALK_SPEED = 2,ACCEL_SLOW_STRIDES = 1,STRIDE_DURATION_VARIABILITY = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    ACCEL_SLOW_STRIDES = 1  # Number of strides used during acceleration and slowing down phase
    STRIDE_DURATION_VARIABILITY = 0.5  # Use .5 for normal walk and 1.9+ for varying speed

        
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
    walk_section = (walking_ff_time < median_ff_time * (1 + STRIDE_DURATION_VARIABILITY))

    # Add padding to find start/end of walks
    B = np.concatenate(([0], walk_section, [0]))
    start_walks = np.where(np.diff(B.astype(int)) == 1)[0]
    end_walks = np.where(np.diff(B.astype(int)) == -1)[0]
    walk_sizes = end_walks - start_walks
    idx_walk = np.argmax(walk_sizes)
    start_walk = start_walks[idx_walk]
    end_walk = end_walks[idx_walk]

    # Eliminate first and last stride, which may correspond to acceleration and slowing down
    start_walk = start_walk + ACCEL_SLOW_STRIDES
    end_walk = end_walk - ACCEL_SLOW_STRIDES

    if plot_details:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(walking_ff_time)
        plt.plot(walk_section * median_ff_time, 'ok')
        plt.plot(start_walk, walking_ff_time[start_walk], '.g')
        plt.plot(end_walk, walking_ff_time[end_walk], '.r')
        plt.legend(['Time between strides', 'Median walk time', 'Beginning of Walk', 'End of Walk'])
        plt.title('Stride duration')

        plt.figure()
        plt.plot(Vm)
        plt.plot(peaks_idx, Vm[peaks_idx], '.k')
        plt.plot(peaks_idx[start_walk], Vm[peaks_idx[start_walk]], '*g')
        plt.plot(peaks_idx[end_walk], Vm[peaks_idx[end_walk]], '*r')
        plt.legend(['Speed', 'Max Speed', 'Beginning of Walk', 'End of Walk'])
        plt.title('Increase STRIDE_DURATION_VARIABILITY until it includes all the walking area')

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

    MAX_NUMBER_FOOTFALL_DIFF = 2  # Default 2
    left_ff_count = np.sum(left_walk_info['FF_walking'])
    right_ff_count = np.sum(right_walk_info['FF_walking'])
    if abs(left_ff_count - right_ff_count) > MAX_NUMBER_FOOTFALL_DIFF:
        print('Warning! incompatible number of footfalls between feet, this needs to be fixed.')
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

    if plot_details:
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


def _empty_steps() -> Dict[str, Any]:
    return {
        'leading_foot': np.array([], dtype='<U5'),
        'start_idx': np.array([], dtype=int),
        'end_idx': np.array([], dtype=int),
        'time': np.array([]),
        'length': np.array([]),
        'width': np.array([]),
        'left_xy': np.zeros((0, 2)),
        'right_xy': np.zeros((0, 2)),
        'anchor': None,
        'anchor_quality': np.inf,
        'n_same_foot_skips': 0,
        'n_too_slow': 0,
    }


def _moving_contacts(P: np.ndarray, ff: np.ndarray, threshold: float) -> np.ndarray:
    """Mark footfalls whose foot travelled more than threshold since its own
    previous footfall (i.e. contacts that end a real swing, not standing)."""
    moving = np.zeros(len(ff), dtype=bool)
    moving[1:] = np.linalg.norm(np.diff(P[ff, :2], axis=0), axis=1) > threshold
    # foot_fall() unconditionally appends a footfall at the last sample; it is
    # not a real contact, so never let it count as a swing landing
    moving[ff == len(P) - 1] = False
    return moving


def step_segmentation(left_info: Dict[str, Any], right_info: Dict[str, Any],
                      period: float,
                      initial_separation: float = 0.3,
                      max_step_seconds: float = 2.0,
                      min_stride_displacement: float = 0.2) -> Dict[str, Any]:
    """Segment steps (opposite-foot footfall to footfall) from a two-IMU bout.

    A step runs from a footfall of one foot to the next footfall of the
    *other* foot — half a gait cycle. This is the cross-foot complement of
    stride_segmentation(), which treats each foot independently. Both feet
    must be sample-synchronized (as produced by compute_position_two_imus).

    Step *time* needs only the two footfall trains and is the most reliable
    output. Step *length* and *width* require both feet in a common spatial
    frame, which foot-mounted IMUs cannot observe directly; they are computed
    under two stated assumptions:

    - Each foot's trajectory is rotated so its travel heading (first to last
      moving contact) points +x (valid for straight walking bouts).
    - At the anchor — a standing moment with footfalls of both feet close
      together in time and the feet facing the same way — the feet are side
      by side, `initial_separation` metres apart across the line of facing
      (the facing comes from the feet's yaw, so a stance after the subject
      has turned still anchors correctly). The anchor is chosen as close in
      time to the walking block as possible, because position drift
      accumulates in un-ZUPTed gaps between standing and walking. The *mean*
      step width inherits the separation assumption wholesale (changing the
      argument shifts every width by the same amount); width *variability*
      across steps is real signal. The left/right *split* of step length
      inherits the forward part of the anchor error; the mean of a
      consecutive L+R step pair equals the stride length and is anchor-free.

    A step is recorded for each swing landing (a contact whose foot moved
    more than min_stride_displacement since its own previous contact), paired
    with the opposite foot's most recent contact — which may be a standing
    contact, so gait-initiation steps are included. Two same-foot swing
    landings in a row (a missed contact on the other side) and pairs further
    apart than max_step_seconds are counted in the diagnostics instead.
    Standing footfalls never form steps themselves; they serve as trailing
    contacts and anchor candidates.

    Parameters:
    -----------
    left_info, right_info : dict
        Outputs of compute_position_two_imus() (need 'FF_walking' and 'P')
    period : float
        Sampling period in seconds
    initial_separation : float
        Assumed lateral distance between the feet when standing (default 0.3 m)
    max_step_seconds : float
        Footfall pairs further apart than this do not form a step
    min_stride_displacement : float
        A contact counts as a swing landing when its foot moved at least this
        far (m) since its own previous contact

    Returns:
    --------
    dict
        'leading_foot' : 'left'/'right' per step (the foot that lands)
        'start_idx', 'end_idx' : sample indices of the trailing and leading
            contacts
        'time' : step duration in seconds
        'length' : forward distance between the successive foot placements
        'width' : lateral separation of the placements, signed left minus
            right (negative indicates crossover)
        'left_xy', 'right_xy' : both trajectories in the common frame
            (forward, lateral)
        'anchor' : (left_idx, right_idx) of the standing pair that anchors
            the common frame, or None if none was found (frames are then
            anchored at the first contacts, and lengths/widths are suspect)
        'anchor_quality' : the longest un-ZUPTed interval (s) separating the
            anchor from the walking block — position drift grows with it, so
            treat the length split and the mean width as unreliable when this
            exceeds a normal stride time (say > 2 s)
        'n_same_foot_skips', 'n_too_slow' : counts of swing contacts that did
            not form steps
    """
    left_ff = np.where(left_info['FF_walking'])[0]
    right_ff = np.where(right_info['FF_walking'])[0]
    if len(left_ff) == 0 or len(right_ff) == 0:
        return _empty_steps()

    P_left, P_right = left_info['P'], right_info['P']
    left_moving = _moving_contacts(P_left, left_ff, min_stride_displacement)
    right_moving = _moving_contacts(P_right, right_ff, min_stride_displacement)

    # Interleave the two footfall trains on the shared timeline
    events = np.concatenate([left_ff, right_ff])
    is_left = np.concatenate([np.ones(len(left_ff), bool), np.zeros(len(right_ff), bool)])
    moving = np.concatenate([left_moving, right_moving])
    order = np.argsort(events, kind='stable')
    events, is_left, moving = events[order], is_left[order], moving[order]

    steps = _empty_steps()

    # Pair each swing landing with the opposite foot's most recent contact
    max_step_samples = int(max_step_seconds / period)
    pairs = []  # (trail event position, lead event position)
    last_contact = {True: None, False: None}   # per foot, by is_left
    last_swing_is_left = None
    for k in range(len(events)):
        if moving[k]:
            trail = last_contact[not is_left[k]]
            if last_swing_is_left == is_left[k]:
                steps['n_same_foot_skips'] += 1
            elif trail is not None and events[k] - events[trail] > max_step_samples:
                steps['n_too_slow'] += 1
            elif trail is not None:
                pairs.append((trail, k))
            last_swing_is_left = is_left[k]
        last_contact[is_left[k]] = k
    if not pairs:
        return steps

    # Rotate each foot's trajectory by its travel heading (first to last
    # swing contact) so forward is +x for both, and derive the foot's facing
    # in that frame. The sensor is mounted at an arbitrary angle about
    # vertical, so its yaw is calibrated against the swing contacts, where
    # the foot faces the direction of travel (facing 0 in the common frame).
    def to_common(info, ff, moving_mask):
        P = info['P']
        swing = ff[moving_mask]
        a, b = (swing[0], swing[-1]) if len(swing) >= 2 else (0, len(P) - 1)
        heading = np.arctan2(P[b, 1] - P[a, 1], P[b, 0] - P[a, 0])
        XY = np.column_stack(rotate_angle(P[:, 0], P[:, 1], -heading))
        yaw = np.exp(1j * info['euler'][:, 2])
        mounting = np.angle(np.mean(yaw[swing])) if len(swing) else heading
        facing = np.angle(yaw * np.exp(-1j * mounting))
        return XY, facing

    left_xy, left_yaw = to_common(left_info, left_ff, left_moving)
    right_xy, right_yaw = to_common(right_info, right_ff, right_moving)

    # Anchor the common frame at a standing pair: a standing footfall of each
    # foot with the feet facing the same way (both settled — the subject may
    # be turned away from the travel direction, the separation vector is
    # oriented by the feet's facing). Position error accumulates with the
    # duration of un-ZUPTed intervals between contacts, so score each
    # candidate by the longest inter-contact interval on the chain linking it
    # to the walking block (plus the pair's own time separation) and take the
    # best.
    MAX_ANCHOR_FOOT_YAW_DIFF = 0.5  # rad between the two feet at the anchor

    def drift_to_walk(ff, moving_mask):
        """Longest un-ZUPTed interval (s) between each contact and the
        walking block of its foot."""
        out = np.full(len(ff), np.inf)
        swing_pos = np.where(moving_mask)[0]
        if len(swing_pos) == 0:
            return out
        lo, hi = swing_pos[0], swing_pos[-1]
        for i in range(len(ff)):
            if lo <= i <= hi:
                out[i] = 0.0
            elif i < lo:
                out[i] = np.max(np.diff(ff[i:lo + 1])) * period
            else:
                out[i] = np.max(np.diff(ff[hi:i + 1])) * period
        return out

    left_drift = drift_to_walk(left_ff, left_moving)
    right_drift = drift_to_walk(right_ff, right_moving)
    anchor = None
    best = np.inf
    for il, dl in zip(left_ff[~left_moving], left_drift[~left_moving]):
        for ir, dr in zip(right_ff[~right_moving], right_drift[~right_moving]):
            if np.abs(np.angle(np.exp(1j * (left_yaw[il] - right_yaw[ir])))) > \
                    MAX_ANCHOR_FOOT_YAW_DIFF:
                continue
            score = max(dl, dr) + abs(il - ir) * period
            if score < best:
                best, anchor = score, (int(il), int(ir))
    steps['anchor'] = anchor
    steps['anchor_quality'] = float(best)
    anchor_left, anchor_right = anchor if anchor else (left_ff[0], right_ff[0])

    # At the anchor the feet are side by side across the line of facing:
    # left foot a half-separation to the facing's left, right foot to its
    # right. (When the anchor stance faces the travel direction this reduces
    # to equal forward position and a +/- lateral offset.)
    facing = np.angle(np.exp(1j * left_yaw[anchor_left]) +
                      np.exp(1j * right_yaw[anchor_right]))
    leftward = np.array([-np.sin(facing), np.cos(facing)])
    left_xy = left_xy - left_xy[anchor_left] + leftward * initial_separation / 2
    right_xy = right_xy - right_xy[anchor_right] - leftward * initial_separation / 2

    records = []
    for k0, k1 in pairs:
        i0, i1 = events[k0], events[k1]
        lead_is_left = is_left[k1]
        lead_xy, trail_xy = (left_xy, right_xy) if lead_is_left else (right_xy, left_xy)
        iL, iR = (i1, i0) if lead_is_left else (i0, i1)
        records.append(('left' if lead_is_left else 'right', i0, i1,
                        (i1 - i0) * period,
                        lead_xy[i1, 0] - trail_xy[i0, 0],
                        left_xy[iL, 1] - right_xy[iR, 1]))

    foot, start_idx, end_idx, time, length, width = zip(*records)
    steps.update(leading_foot=np.array(foot),
                 start_idx=np.array(start_idx), end_idx=np.array(end_idx),
                 time=np.array(time), length=np.array(length),
                 width=np.array(width),
                 left_xy=left_xy, right_xy=right_xy)
    return steps