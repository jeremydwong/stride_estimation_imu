import numpy as np
from typing import Tuple, Optional, Any, Dict, List
from scipy.signal import find_peaks
from dataclasses import dataclass

GRAVITY = 9.80297286843

# Default foot-speed thresholds for snug_start (m/s): the first step of a bout is
# snugged to the velocity valley before the first swing that crosses *_HIGH,
# stopping at a local minimum or when |V| falls to *_LOW.
DEFAULT_START_FOOTSPEED_HIGH = 1.25
DEFAULT_START_FOOTSPEED_LOW = 0.2


@dataclass
class WalkingBout:
    """A detected walking bout with start/end indices and duration."""
    start_idx: int
    end_idx: int
    duration_seconds: float
    quiet_before_idx: int  # Start of quiet period before this bout
    quiet_after_idx: int   # End of quiet period after this bout


@dataclass
class FootTrajectory:
    """Per-foot output of inertial mechanization (``compute_position`` /
    ``compute_position_two_imus``): the full time-series state of one foot over
    a recording of N samples. Pure data container — no methods, on purpose.

    A note on the names: they are inherited verbatim from the original MATLAB
    port and several are opaque and frankly bad (``FF``, ``An``, ``Anz``,
    ``Vm`` ...). They are kept as-is so existing code and people familiar with
    the previous codebase keep reading the same names; rely on the documented
    meanings below rather than the names themselves. Renaming is a separate,
    later job.

    Footfall vs. stance — two DIFFERENT arrays, routinely confused:
      * ``FF`` / ``FF_walking`` are *sparse*: a single ``True`` at the one
        instant the foot is judged to plant (the arg-min angular-rate sample
        within a stance). One ``True`` per contact.
      * ``stationary_periods`` is *dense*: ``True`` for every low-motion sample,
        i.e. a contiguous run of ``True`` spanning each whole stance plateau —
        the "block of 1s per stance" that ``FF`` is often mistaken for.

    Fields (all indexed along axis 0 by sample; N = number of samples):
        FF : (N,) bool
            Footfall flags. Exactly one True per detected contact (the
            arg-min angular-rate sample inside a stance). NOT a run.
        FF_walking : (N,) bool
            FF restricted to the detected walking section (between the first
            and last walking footfall). In ``compute_position_two_imus`` this
            is currently a plain copy of FF.
        stationary_periods : (N,) bool, or None
            True over every low-motion (stance) sample → one True-run per
            stance plateau. None only when a precomputed FF was passed to
            ``compute_position`` so stance was never detected.
        P : (N, 3) float
            Position in the navigation frame [m]. [:, :2] is horizontal (x, y);
            [:, 2] is elevation. Begins at the origin.
        V : (N, 3) float
            Velocity in the navigation frame [m/s] (ZUPT-corrected).
        Vm : (N,) float
            Speed magnitude [m/s]. Horizontal sqrt(Vx²+Vy²) from
            ``compute_position``; full 3-D speed from
            ``compute_position_two_imus``.
        euler : (N, 3) float
            Orientation [roll, pitch, yaw] in radians.
        quaternion : (N, 4) float
            Orientation quaternion [w, x, y, z].
        An : (N, 3) float
            Body acceleration rotated into the navigation frame [m/s²]
            (gravity not removed).
        Anz : (N, 3) float
            ``An`` after zero-velocity-update drift correction [m/s²]
            (integrates to ~0 velocity across each stance). V and P derive
            from this.
        A : (N, 3) float
            Raw body-frame acceleration fed in [m/s²].
        W : (N, 3) float
            Raw body-frame angular velocity fed in [rad/sample].
    """
    FF: np.ndarray
    FF_walking: np.ndarray
    stationary_periods: Optional[np.ndarray]
    P: np.ndarray
    V: np.ndarray
    Vm: np.ndarray
    euler: np.ndarray
    quaternion: np.ndarray
    An: np.ndarray
    Anz: np.ndarray
    A: np.ndarray
    W: np.ndarray


@dataclass
class Strides:
    """Per-foot stride-segmentation output (``stride_segmentation`` /
    ``get_strides``): one entry per detected stride, footfall to the next
    footfall of the SAME foot. Pure data container — no methods, on purpose.

    The three trajectory arrays are (n_samples, n_strides): column j is stride
    j, rotated into the stride's local forward/lateral frame, re-origined so the
    stride starts at 0, then held flat (zero-velocity pad) after its last sample
    out to the longest stride's length. The scalar arrays are (n_strides,).

    Fields:
        frwd : (n_samples, n_strides) float
            Forward position along the stride's local walking axis [m].
        ltrl : (n_samples, n_strides) float
            Lateral position [m] (carried across strides to trace the foot).
        elev : (n_samples, n_strides) float
            Elevation [m], re-origined per stride.
        start_end : (2, n_strides) float
            [start; end] sample index of each stride into the source recording.
        time : (n_strides,) float
            Stride duration [s].
        frwd_speed : (n_strides,) float
            Forward speed = stride length / duration [m/s].
    """
    frwd: np.ndarray
    ltrl: np.ndarray
    elev: np.ndarray
    start_end: np.ndarray
    time: np.ndarray
    frwd_speed: np.ndarray


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
    # Legacy Euler correction is retained for compatibility. See experimental.
    # kalman_filter_gravity for an opt-in gravity-vector tilt correction.
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


def compute_position(W: np.ndarray, A: np.ndarray, period: float, USE_KF: int = 1, W_FF: Optional[float] = None, A_FF: Optional[float] = None, T_FF: Optional[float] = None, MAX_T_FF: Optional[float] = None, FF: Optional[np.ndarray] = None) -> 'FootTrajectory':
    """Perform inertial mechanization on a single foot IMU recording.

    Integrates angular velocity to track orientation (as quaternions),
    transforms body-frame accelerations to the navigation frame, detects
    footfalls (stance phases), and applies zero-velocity updates (ZUPT) to
    correct drift. Returns a dict containing position trajectory, velocity,
    orientation, and footfall arrays.

    Compatibility note: this legacy path is unchanged. The opt-in alternative
    ``experimental.compute_position_experimental`` adds gravity-vector tilt,
    independent tilt gates, and explicit gyro-bias calibration for turn studies.

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
    stationary_periods = None
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
    # The loop below starts at i=1, so without this An[0] stays [0,0,0]. Over the
    # short first-stride ZUPT window (often ~8-12 samples) that single zero biases
    # the gravity estimate ~10% low (|g|: 9.803*(n-1)/n) and injects a spurious
    # -g impulse at Anz[0]. Rotate A[0] by the init orientation like every other
    # sample so the first stride averages real stance.
    An[0, :] = (qua2rot(quaternion[0, :]) @ A[0:1, :].T).T

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
    result['stationary_periods'] = stationary_periods
    return FootTrajectory(**result)

def detect_walking_section(walk_info: Dict[str, Any], MIN_WALK_SPEED: float = 2.0) -> np.ndarray:
    """Filter footfalls to only the walking portion of a recording.

    Uses velocity magnitude peaks to identify the region of sustained walking,
    excluding stationary periods at the start and end. Called internally by
    compute_position() to populate the 'FF_walking' key.

    Parameters:
    -----------
    walk_info : FootTrajectory
        Output of compute_position() (must contain 'FF' and 'Vm' keys)
    MIN_WALK_SPEED : float
        Minimum peak velocity to count as walking (default: 2.0 m/s)

    Returns:
    --------
    np.ndarray
        Boolean footfall array for the walking section only
    """   
    # accept either the raw result dict (used internally, mid-construction) or a
    # finished FootTrajectory
    FF_arr = walk_info['FF'] if isinstance(walk_info, dict) else walk_info.FF
    Vm = walk_info['Vm'] if isinstance(walk_info, dict) else walk_info.Vm
    FF = np.where(FF_arr)[0]
    peaks_idx, _ = find_peaks(Vm, height=MIN_WALK_SPEED)
    FF_max_speed = np.zeros_like(Vm, dtype=bool)
    FF_max_speed[peaks_idx] = True
    median_vel = np.median(Vm[peaks_idx]) if peaks_idx.size > 0 else 0
    likely_walk_sections = np.where(Vm > median_vel * 0.90)[0]
    if likely_walk_sections.size == 0:
        return FF_arr
    footfall_index = np.where((FF > likely_walk_sections[0]) & (FF < likely_walk_sections[-1]))[0]
    FF_walking = np.zeros_like(Vm, dtype=bool)
    if footfall_index.size > 0:
        FF_walking[FF[footfall_index[0]:footfall_index[-1] + 1]] = True
        return FF_walking

def stride_segmentation(walk_info: 'FootTrajectory', period: float, FILTER: int = 0, OUTLIER_SECTION_SECONDS: Optional[Any] = None) -> 'Strides':
    """Segment individual strides from a processed walking recording.

    Takes the output of compute_position() and identifies individual stride
    cycles using the FF_walking footfall array. Each stride runs from one
    footfall to the next. Strides are rotated to a local forward/lateral
    coordinate frame and metrics (speed, length, duration) are computed.

    Parameters:
    -----------
    walk_info : FootTrajectory
        Output of compute_position()
    period : float
        Sampling period in seconds
    FILTER : int
        If 1, apply outlier filtering to remove abnormal strides (default: 0)
    OUTLIER_SECTION_SECONDS : optional
        Time ranges (seconds) to exclude as outliers

    Returns:
    --------
    Strides
        Per-stride trajectories (frwd/ltrl/elev), indices (start_end) and
        metrics (time, frwd_speed). See the Strides dataclass.
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
    FF = np.where(walk_info.FF_walking)[0]
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
    walk_info : FootTrajectory
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
        empty2 = np.array([]).reshape(0, 0)
        empty1 = np.array([])
        return Strides(frwd=empty2, ltrl=empty2, elev=empty2,
                       start_end=empty2, time=empty1, frwd_speed=empty1)

    # DIRECTION_STRIDES, determine the number of strides before and after current one, used to define a straight segment
    DIRECTION_STRIDES = 3  # default 3
    # mean_stride_direction = 0  # when DIRECTION_STRIDES is 0, uncomment this
    
    EXTRA_FRWD_CORRECTION = 1  # default is 1. This will straighten the paths perfectly
    # EXTRA_FRWD_CORRECTION = 0  # for hand reaching Oct 2023

    number_of_strides = len(stride_start)
    longest_stride = np.max(stride_end - stride_start) + 1
    
    P = walk_info.P
    euler = walk_info.euler
    Vm = walk_info.Vm

    # Create matrices to store results
    ltrl = np.zeros((number_of_strides, longest_stride))
    frwd = np.zeros((number_of_strides, longest_stride))
    ltrl_straighten = np.zeros((number_of_strides, longest_stride))
    frwd_straighten = np.zeros((number_of_strides, longest_stride))
    elev = np.zeros((number_of_strides, longest_stride))
    foot_heading = np.zeros(number_of_strides)  # kept: feeds the PLOT_DETAILS heading plot
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

        stride_samples[i] = stride_end[i] - stride_start[i]

        # Store heading angle (feeds the PLOT_DETAILS heading plot)
        foot_heading[i] = corrected_heading[i]
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
    frwd = frwd - frwd[:, 0:1]
    ltrl = ltrl - ltrl[:, 0:1]
    elev = elev - elev[:, 0:1]
    
    # Check for negative values in frwd
    if np.any(frwd[:, -1] < 0):
        print('WARNING: NEGATIVE IN FORWARD DIRECTION DETECTED IN STRIDES!')
        frwd = np.abs(frwd)
    
    # Compute stride speed
    stride_length = frwd[:, -1]
    time = stride_samples * PERIOD
    frwd_speed = stride_length / time

    # Assemble result (trajectories stored as (samples, n_strides))
    result = Strides(
        frwd=frwd.T,
        ltrl=ltrl.T,
        elev=elev.T,
        start_end=start_end.T,
        time=time,
        frwd_speed=frwd_speed,
    )
    
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
    stride : Strides
        Strides dataclass of stride data
    out_of_bound : list
        Indices of strides to remove
    number_of_strides : int
        Current number of strides
    
    Returns:
    --------
    stride : Strides
        Strides with outliers removed
    number_of_strides : int
        New number of strides after removal
    """
    if out_of_bound:
        # Delete columns for 2D arrays (note: in Python, we need to use np.delete)
        stride.frwd = np.delete(stride.frwd, out_of_bound, axis=1)
        stride.ltrl = np.delete(stride.ltrl, out_of_bound, axis=1)
        stride.elev = np.delete(stride.elev, out_of_bound, axis=1)
        stride.start_end = np.delete(stride.start_end, out_of_bound, axis=1)

        # Delete elements for 1D arrays
        stride.time = np.delete(stride.time, out_of_bound)
        stride.frwd_speed = np.delete(stride.frwd_speed, out_of_bound)

        new_number_of_strides = number_of_strides - len(out_of_bound)
        print(f'New number of strides {new_number_of_strides} out of {number_of_strides}')
        number_of_strides = new_number_of_strides
    
    return stride, number_of_strides


def filter_strides(stride, number_of_strides):
    """
    Filter out strides that are not within known specifications
    
    Parameters:
    -----------
    stride : Strides
        Strides dataclass of stride data
    number_of_strides : int
        Current number of strides
        
    Returns:
    --------
    stride : Strides
        Filtered Strides
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
    frwd = stride.frwd.T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = frwd[:, -1] > MAX_STRIDE_LENGTH
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd LONG')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    # Eliminate very short strides
    frwd = stride.frwd.T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = frwd[:, -1] < MIN_STRIDE_LENGTH
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd SHORT')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    # Eliminate strides away from the length median value
    frwd = stride.frwd.T
    median_pos_frwd = np.median(frwd[:, -1])
    std_pos_frwd = np.std(frwd[:, -1])
    outlier = np.abs(frwd[:, -1] - median_pos_frwd) > std_pos_frwd * MAX_VAR
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_frwd +2 VAR')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    # Eliminate strides that have too much side deviation
    ltrl = stride.ltrl.T
    std_pos_ltrl = np.std(ltrl[:, -1])
    outlier = np.abs(ltrl[:, -1]) > std_pos_ltrl * MAX_VAR
    outlier = np.where(outlier)[0]
    if len(outlier) > 0:
        print('_ltrl +2 VAR')
        stride, number_of_strides = cut_stride_section(stride, outlier, number_of_strides)
    
    if FILTER_ELEVATION:
        # Eliminate strides that have too much vertical deviation
        elev = stride.elev.T
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
                         plot_details: bool = False) -> Tuple['FootTrajectory', 'FootTrajectory']:
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
    left_samples = len(left_walk_info.FF)
    right_samples = len(right_walk_info.FF)
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
    left_walk_info.FF, left_walk_info.stationary_periods = foot_fall(leftWb, leftAb, period, W_FF=30, A_FF=1, T_FF=.4, MAX_T_FF=10)
    # duplicate for FF_walking
    left_walk_info.FF_walking = left_walk_info.FF.copy()
    right_walk_info.FF, right_walk_info.stationary_periods = foot_fall(rightWb, rightAb, period, W_FF=30, A_FF=1, T_FF=.4, MAX_T_FF=10)
    # duplicate for FF_walking
    right_walk_info.FF_walking = right_walk_info.FF.copy()

    MAX_NUMBER_FOOTFALL_DIFF = 2  # Default 2
    left_ff_count = np.sum(left_walk_info.FF_walking)
    right_ff_count = np.sum(right_walk_info.FF_walking)
    if abs(left_ff_count - right_ff_count) > MAX_NUMBER_FOOTFALL_DIFF:
        print('Warning! incompatible number of footfalls between feet, this needs to be fixed.')
        print('Check MIN_WALK_SPEED in foot_fall_opposite_velocity')
        print(f'Left: {left_ff_count}, Right: {right_ff_count}')

    # Recompute accelerations (ZUPT) using combined FFs
    # Left foot
    An = left_walk_info.An
    FF = left_walk_info.FF
    Anz = np.zeros((SAMPLES, 3))
    last_footfall = 0
    for i in range(1, SAMPLES):
        last_footfall = zero_velocity_updates(i, FF, An, Anz, last_footfall)

    left_walk_info.Anz = Anz
    left_walk_info.V = np.cumsum(Anz, axis=0) * period
    left_walk_info.P = np.cumsum(left_walk_info.V, axis=0) * period
    left_walk_info.Vm = np.sqrt(np.sum(left_walk_info.V ** 2, axis=1))

    # Right foot
    An = right_walk_info.An
    FF = right_walk_info.FF
    Anz = np.zeros((SAMPLES, 3))
    last_footfall = 0
    for i in range(1, SAMPLES):
        last_footfall = zero_velocity_updates(i, FF, An, Anz, last_footfall)

    right_walk_info.Anz = Anz
    right_walk_info.V = np.cumsum(Anz, axis=0) * period
    right_walk_info.P = np.cumsum(right_walk_info.V, axis=0) * period
    right_walk_info.Vm = np.sqrt(np.sum(right_walk_info.V ** 2, axis=1))

    # Make Plots
    import matplotlib.pyplot as plt

    if plot_details:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(left_walk_info.P[left_walk_info.FF, 0],
                left_walk_info.P[left_walk_info.FF, 1],
                left_walk_info.P[left_walk_info.FF, 2],
                '.g', label='left')
        ax.plot(right_walk_info.P[right_walk_info.FF, 0],
                right_walk_info.P[right_walk_info.FF, 1],
                right_walk_info.P[right_walk_info.FF, 2],
                '.r', label='right')
        ax.plot(left_walk_info.P[:, 0],
                left_walk_info.P[:, 1],
                left_walk_info.P[:, 2], 'b-')
        ax.plot(right_walk_info.P[:, 0],
                right_walk_info.P[:, 1],
                right_walk_info.P[:, 2], 'r-')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        ax.grid(True)
        plt.axis('equal')

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(t, left_walk_info.Vm)
        axes[0].plot(t[left_walk_info.FF], left_walk_info.Vm[left_walk_info.FF], 'g*')
        axes[0].plot(t[left_walk_info.FF_walking], left_walk_info.Vm[left_walk_info.FF_walking], 'k.')
        axes[0].grid(True)
        axes[0].set_ylabel('Left |V| [m/s]')
        axes[0].set_xlabel('time [s]')
        axes[0].legend(['|V|', 'FFs', 'Walking'])
        axes[0].set_title('Foot Fall detection using opposite shoe speed')

        axes[1].plot(t, right_walk_info.Vm)
        axes[1].plot(t[right_walk_info.FF], right_walk_info.Vm[right_walk_info.FF], 'g*')
        axes[1].plot(t[right_walk_info.FF_walking], right_walk_info.Vm[right_walk_info.FF_walking], 'k.')
        axes[1].grid(True)
        axes[1].set_ylabel('Right |V| [m/s]')
        axes[1].set_xlabel('time [s]')

    if plot_details:
        left_az_mag = np.sqrt(np.sum(left_walk_info.Anz ** 2, axis=1))
        right_az_mag = np.sqrt(np.sum(right_walk_info.Anz ** 2, axis=1))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(t, left_walk_info.Anz, t, left_az_mag, 'k')
        axes[0].plot(t[left_walk_info.FF], left_walk_info.Vm[left_walk_info.FF], 'g.')
        axes[0].plot(t[left_walk_info.FF_walking], left_walk_info.Vm[left_walk_info.FF_walking], 'yo')
        axes[0].grid(True)
        axes[0].set_ylabel('left A [m/s^2]')
        axes[0].set_xlabel('time [s]')

        axes[1].plot(t, right_walk_info.Anz, t, right_az_mag, 'k')
        axes[1].plot(t[right_walk_info.FF], right_walk_info.Vm[right_walk_info.FF], 'g.')
        axes[1].plot(t[right_walk_info.FF_walking], right_walk_info.Vm[right_walk_info.FF_walking], 'yo')
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
        'frwd_speed': np.array([]),
        'length': np.array([]),
        'width': np.array([]),
        'left_xyz': np.zeros((0, 3)),
        'right_xyz': np.zeros((0, 3)),
        'anchor': None,
        'anchor_drift_seconds': np.inf,
        'n_same_foot_skips': 0,
        'n_too_slow': 0,
        'start_snap': None,
        'start_snap_foot': None,
        'start_high': DEFAULT_START_FOOTSPEED_HIGH,
        'start_low': DEFAULT_START_FOOTSPEED_LOW,
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


def touchdown_map(stationary_periods: np.ndarray, period: Optional[float] = None,
                  bridge_s: float = 0.15) -> np.ndarray:
    """Map each sample to the START of the stationary run it belongs to (its
    touchdown). Non-stationary samples map to themselves. foot_fall() picks the
    footfall as the arg-min-gyro sample, which wanders within the stance
    plateau; snapping to the run start gives a consistent foot-on-ground
    instant and stabilizes step/stride timing.

    When `period` is given, non-stationary blips shorter than `bridge_s` are
    bridged first, so plateaus chained by micro-shuffles count as ONE stance
    starting at the true landing. Without this, the walker's FINAL landing —
    whose stance runs straight into the terminal standing, broken only by
    ~0.1 s weight-shift blips — got its footfall snapped to a later plateau
    and the last step vanished from the train (s07_s08 trial 1 rep 2: left
    lands at 4.50 s, touchdown reported 5.16 s). Real swings are >=0.3 s, so
    0.15 s bridges only blips. period=None keeps the historic behaviour."""
    stat = np.asarray(stationary_periods, bool).copy()
    if period is not None and bridge_s > 0:
        n = max(1, int(bridge_s / period))
        padded = np.r_[False, stat, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for gap_a, gap_b in zip(edges[1::2], edges[2::2]):
            if gap_b - gap_a <= n:
                stat[gap_a:gap_b] = True
    rs = np.arange(len(stat))
    for i in range(1, len(stat)):
        if stat[i] and stat[i - 1]:
            rs[i] = rs[i - 1]
    return rs


def snug_start(left_info: 'FootTrajectory', right_info: 'FootTrajectory',
               high: float = DEFAULT_START_FOOTSPEED_HIGH,
               low: float = DEFAULT_START_FOOTSPEED_LOW) -> Tuple[Optional[int], Optional[str]]:
    """Find the true walk-start sample to snug the first step onto.

    Bouts are often snipped loosely, so the opening is un-ZUPTed
    standing/box-handling whose drift inflates the first step. To recover the
    real start we take the FIRST foot whose speed |V| crosses `high` (the first
    committed swing), then walk backwards along that foot's |V| into the valley
    before the swing, stopping at the first local minimum (|V| starts rising
    again as we step further back in time) OR when |V| falls to `low`.

    Returns (sample_idx, 'left'|'right'), or (None, None) if neither foot ever
    reaches `high` (no clear swing — nothing to snug).
    """
    Vmap = {'left': left_info.Vm, 'right': right_info.Vm}
    first = None
    for foot, Vm in Vmap.items():
        hi = np.where(Vm >= high)[0]
        if hi.size and (first is None or hi[0] < first[0]):
            first = (int(hi[0]), foot)
    if first is None:
        return None, None
    h, foot = first
    Vm = Vmap[foot]
    i = h
    while i > 0:
        if Vm[i] <= low:            # reached the low threshold
            break
        if Vm[i - 1] > Vm[i]:       # local min: going further back rises again
            break
        i -= 1
    return i, foot


def _common_frame(left_info: 'FootTrajectory', right_info: 'FootTrajectory', period: float,
                  initial_separation: float = 0.3,
                  min_stride_displacement: float = 0.2,
                  anchor_mode: str = 'auto'):
    """Place both feet's trajectories in ONE frame: walking axis +Y, lateral X
    (+X = the walker's right), gravity Z. Returns (left_xyz, right_xyz, anchor,
    anchor_drift_seconds), each xyz an (N, 3) array with columns
    [X lateral (right +), Y forward, Z up].

    Foot-mounted IMUs cannot observe the lateral offset between the feet, so it
    is supplied by assuming the feet are `initial_separation` apart, side by
    side, at an anchor stance — a standing footfall of each foot facing the same
    way, chosen as close to the walking block as possible (drift accumulates in
    un-ZUPTed gaps). The *mean* step width inherits this separation wholesale
    (changing it shifts every width equally); width *variability* and the L/R
    *split* of length carry the anchor error, while a consecutive L+R length
    mean equals the stride length and is anchor-free. Treat the split / mean
    width as unreliable when anchor_drift_seconds exceeds a normal stride time.
    """
    if anchor_mode not in ('auto', 'firstonly'):
        raise ValueError(f"anchor_mode must be 'auto' or 'firstonly', got {anchor_mode!r}")
    NL, NR = len(left_info.P), len(right_info.P)
    left_ff = np.where(left_info.FF_walking)[0]
    right_ff = np.where(right_info.FF_walking)[0]
    if len(left_ff) == 0 or len(right_ff) == 0:
        return np.zeros((NL, 3)), np.zeros((NR, 3)), None, np.inf

    left_moving = _moving_contacts(left_info.P, left_ff, min_stride_displacement)
    right_moving = _moving_contacts(right_info.P, right_ff, min_stride_displacement)

    # Rotate each foot's trajectory by its travel heading (first to last swing
    # contact) so forward is +x' internally, and derive the foot's facing. The
    # sensor is mounted at an arbitrary yaw, so its yaw is calibrated against the
    # swing contacts, where the foot faces the direction of travel.
    def to_common(info, ff, moving_mask):
        P = info.P
        swing = ff[moving_mask]
        a, b = (swing[0], swing[-1]) if len(swing) >= 2 else (0, len(P) - 1)
        heading = np.arctan2(P[b, 1] - P[a, 1], P[b, 0] - P[a, 0])
        fwd, lat = rotate_angle(P[:, 0], P[:, 1], -heading)   # forward->+x', lateral->y'
        yaw = np.exp(1j * info.euler[:, 2])
        mounting = np.angle(np.mean(yaw[swing])) if len(swing) else heading
        facing = np.angle(yaw * np.exp(-1j * mounting))
        return fwd, lat, facing

    lf, ll, l_face = to_common(left_info, left_ff, left_moving)
    rf, rl, r_face = to_common(right_info, right_ff, right_moving)

    MAX_ANCHOR_FOOT_YAW_DIFF = 0.5  # rad between the two feet at the anchor

    def drift_to_walk(ff, moving_mask):
        """Longest un-ZUPTed interval (s) between each contact and the walking
        block of its foot."""
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
    if anchor_mode == 'firstonly':
        anchor = (int(left_ff[0]), int(right_ff[0]))
        best = max(left_drift[0], right_drift[0]) + \
            abs(int(left_ff[0]) - int(right_ff[0])) * period
    else:
        for il, dl in zip(left_ff[~left_moving], left_drift[~left_moving]):
            for ir, dr in zip(right_ff[~right_moving], right_drift[~right_moving]):
                if np.abs(np.angle(np.exp(1j * (l_face[il] - r_face[ir])))) > \
                        MAX_ANCHOR_FOOT_YAW_DIFF:
                    continue
                score = max(dl, dr) + abs(il - ir) * period
                if score < best:
                    best, anchor = score, (int(il), int(ir))
    aL, aR = anchor if anchor else (int(left_ff[0]), int(right_ff[0]))

    # At the anchor the feet are side by side across the line of facing: left a
    # half-separation to the facing's left, right to its right. leftward is in
    # (forward, lateral) components.
    facing = np.angle(np.exp(1j * l_face[aL]) + np.exp(1j * r_face[aR]))
    leftward = np.array([-np.sin(facing), np.cos(facing)])
    left_F = lf - lf[aL] + leftward[0] * initial_separation / 2
    left_L = ll - ll[aL] + leftward[1] * initial_separation / 2
    right_F = rf - rf[aR] - leftward[0] * initial_separation / 2
    right_L = rl - rl[aR] - leftward[1] * initial_separation / 2

    # output frame: X = lateral (+X = the walker's RIGHT), Y = forward (walking
    # axis), Z = up (gravity). left_L/right_L are left-of-travel (right-handed
    # rotation), so negate to make +X point right — then a top-down plot with
    # forward up shows the left foot on the left.
    left_xyz = np.column_stack([-left_L, left_F, left_info.P[:, 2]])
    right_xyz = np.column_stack([-right_L, right_F, right_info.P[:, 2]])
    return left_xyz, right_xyz, anchor, float(best)


def steps_from_strides(left_strides: 'Strides', right_strides: 'Strides',
                       left_info: 'FootTrajectory', right_info: 'FootTrajectory', period: float,
                       initial_separation: float = 0.3,
                       min_stride_displacement: float = 0.2,
                       anchor_mode: str = 'auto',
                       start_high: float = DEFAULT_START_FOOTSPEED_HIGH,
                       start_low: float = DEFAULT_START_FOOTSPEED_LOW,
                       force_snap: Optional[int] = None) -> Dict[str, Any]:
    """Segment steps from the merged foot-CONTACT train of the two feet, taking
    speed from the validated per-foot strides and placing both feet in one frame.

    A *step* is a crossing: a touchdown of one foot followed by the next
    touchdown of the OTHER foot (half a gait cycle). The real per-foot events
    are the walking touchdowns (FF_walking), each snapped to the start of its
    stationary plateau (touchdown_map) so timing is not jittered by the
    arg-min-gyro footfall placement inside the plateau.

    Steps are built from the touchdowns, NOT from the stride list, because
    stride_segmentation drops any stride spanning > 2 s — e.g. a foot that
    stands through gait initiation while the other foot steps out first. That
    dropped stride still corresponds to a genuine contact that should cross into
    a step; deriving steps from contacts keeps it.

    Per step:
      * frwd_speed — the leading foot's just-completed stride speed when a stride
        ended at that touchdown; when the inbound stride was dropped (gait
        initiation) the speed is recovered from the mechanized trajectory P (net
        horizontal displacement of the leading foot from its previous contact,
        over the elapsed time). Every step has a finite speed; there are no nans.
      * length, width — from the common frame (walking axis +Y, lateral X with
        +X = the walker's right, gravity Z; see _common_frame): length is the
        forward gap between the two successive placements, width the lateral gap
        (signed; positive = normal stance, negative = crossover). These carry the
        anchor assumption (see _common_frame and anchor_drift_seconds).

    Gait initiation: when both feet registered the shared pre-walk stance (two
    contacts at/before the snug start), the opening of whichever foot swings
    first is dropped so the contact train stays alternating. When the snip
    begins at walk onset the swinging foot has no stance plateau in-slice —
    the train already alternates and nothing is dropped. The FIRST step is
    then "snugged" (snug_start): its start/time/speed AND length/width are moved
    to the velocity valley before the first committed swing, removing the bogus
    long, fast first step that pre-walk drift produces. Since drift before the
    snug is ignored, that step's length is the leading foot's forward travel from
    the snug start (its position there subtracted) and its width is the assumed
    initial side-by-side separation. start_snap/start_snap_foot report the sample.

    Two same-foot touchdowns in a row mean the other foot missed a contact — a
    rare fault, counted in n_same_foot_skips rather than faked.

    Parameters
    ----------
    left_strides, right_strides : Strides
        Per-foot stride_segmentation() outputs (uses start_end, frwd_speed).
    left_info, right_info : FootTrajectory
        compute_position_two_imus() outputs (need FF_walking, stationary_periods,
        P, Vm, euler).
    period : float
        Sampling period (s).
    initial_separation, min_stride_displacement, anchor_mode :
        Passed to _common_frame for the spatial (length/width) computation.
    start_high, start_low : float
        Foot-speed thresholds (m/s) for snug_start.

    Returns
    -------
    dict with keys: leading_foot, start_idx, end_idx, time, frwd_speed, length,
        width, left_xyz, right_xyz, anchor, anchor_drift_seconds,
        n_same_foot_skips, n_too_slow, start_snap, start_snap_foot, start_high,
        start_low. (n_too_slow is always 0 here — kept for compatibility; this
        segmenter does not gate steps on a max duration.)
    """
    def foot_events(strides, info):
        """(touchdowns, speed_at, prev_td) for one foot. touchdowns: sorted-unique
        snapped FF_walking indices. speed_at: {snapped stride-end index -> forward
        speed} for completed strides (first wins). prev_td: {touchdown -> previous
        touchdown}, used to recover a step speed from P when a stride was dropped."""
        td = touchdown_map(info.stationary_periods, period)
        ff = np.where(info.FF_walking)[0]
        touchdowns = np.unique(td[ff]).astype(int) if ff.size else np.array([], int)
        speed_at = {}
        se = strides.start_end
        if se.size:
            spd = np.asarray(strides.frwd_speed, float)
            for e, s in zip(se[1], spd):
                speed_at.setdefault(int(td[int(e)]), float(s))
        prev_td = {int(c): int(p) for p, c in zip(touchdowns[:-1], touchdowns[1:])}
        return touchdowns, speed_at, prev_td

    L_td, L_spd, L_prev = foot_events(left_strides, left_info)
    R_td, R_spd, R_prev = foot_events(right_strides, right_info)
    if len(L_td) == 0 or len(R_td) == 0:
        return _empty_steps()
    spd_map = {'left': L_spd, 'right': R_spd}
    prev_map = {'left': L_prev, 'right': R_prev}
    P_map = {'left': left_info.P, 'right': right_info.P}

    # both feet in one frame for the spatial measures (walking +Y, lateral X, Z up)
    left_xyz, right_xyz, anchor, anchor_drift = _common_frame(
        left_info, right_info, period, initial_separation,
        min_stride_displacement, anchor_mode)
    fwd = {'left': left_xyz[:, 1], 'right': right_xyz[:, 1]}
    lat = {'left': left_xyz[:, 0], 'right': right_xyz[:, 0]}

    contacts = [(int(i), 'left') for i in L_td] + [(int(i), 'right') for i in R_td]
    contacts.sort(key=lambda c: c[0])

    # snug sample g = velocity valley before the first committed swing; needed
    # here to recognize opening stances, reported/used for step 1 further down.
    # force_snap (slice-relative sample) overrides the detected snug — used by
    # the interactive inspector when the user drags the gait-start marker.
    g, gfoot = snug_start(left_info, right_info, start_high, start_low)
    if force_snap is not None:
        g = int(np.clip(force_snap, 0, len(left_info.Vm) - 1))

    # Drop the first-swinging foot's opening stance contact (see docstring) —
    # but ONLY when both feet actually registered their shared pre-walk stance
    # (two contacts at/before the snug start g). When the snip begins at walk
    # onset, the first-swinging foot never stands long enough in-slice to get a
    # stance plateau: the train already alternates, and the unconditional drop
    # used to delete the STANDING foot's genuine stance — erasing the first
    # step entirely (its landing had no predecessor; seen as s07_s08 trial 5
    # rep 2, both bouts, where step 1 silently spanned two real steps).
    # The margin: the standing foot's plateau often RESTARTS a few samples
    # after g (weight shift as the other foot swings out), while a genuine
    # first landing is never before ~0.3 s after g — 0.15 s separates them.
    OPENING_MARGIN_S = 0.15
    g_lim = np.inf if g is None else g + int(OPENING_MARGIN_S / period)
    opening = [c for c in contacts[:2] if c[0] <= g_lim]
    if len(opening) == 2:
        l_next = L_td[1] if len(L_td) > 1 else np.inf
        r_next = R_td[1] if len(R_td) > 1 else np.inf
        first_swing = 'left' if l_next <= r_next else 'right'
        contacts.remove((int(L_td[0] if first_swing == 'left' else R_td[0]),
                         first_swing))

    out = {'leading_foot': [], 'end_idx': [], 'start_idx': [], 'time': [],
           'frwd_speed': [], 'length': [], 'width': []}
    n_same_foot_skips = 0
    for (i0, f0), (i1, f1) in zip(contacts, contacts[1:]):
        if f0 == f1 or i0 == i1:        # same foot in a row = a missed contact
            if f0 == f1:
                n_same_foot_skips += 1
            continue
        spd = spd_map[f1].get(i1, np.nan)
        prev = prev_map[f1].get(i1, i0)
        if not np.isfinite(spd) or (g is not None and prev < g):
            # Recover speed from the mechanized trajectory when the inbound
            # stride was dropped by the >2 s cap (gait initiation), OR when it
            # is clocked from a PRE-WALK stance (prev contact before the snug
            # start g): a stride-time that includes standing makes the early
            # steps read absurdly slow (e.g. step 2 at 0.37 m/s while step 1,
            # snug-clocked, reads 1.0 — the "first step faster than second"
            # artifact). No step's clock starts before g.
            lo = prev if g is None else max(prev, g)
            P = P_map[f1]
            dist = float(np.linalg.norm(P[i1, :2] - P[lo, :2]))
            spd = dist / max((i1 - lo) * period, period)
        iL, iR = (i1, i0) if f1 == 'left' else (i0, i1)
        out['leading_foot'].append(f1)
        out['start_idx'].append(i0)
        out['end_idx'].append(i1)
        out['time'].append((i1 - i0) * period)
        out['frwd_speed'].append(spd)
        out['length'].append(fwd[f1][i1] - fwd[f0][i0])
        # +X is the walker's right, so right-minus-left keeps normal stance
        # positive and crossover negative (value identical to the old left-minus-
        # right on the pre-flip lateral).
        out['width'].append(lat['right'][iR] - lat['left'][iL])

    result: Dict[str, Any] = {k: np.array(v) for k, v in out.items()}
    result['left_xyz'] = left_xyz
    result['right_xyz'] = right_xyz
    result['anchor'] = anchor
    result['anchor_drift_seconds'] = anchor_drift
    result['n_same_foot_skips'] = n_same_foot_skips
    result['n_too_slow'] = 0
    result['start_high'] = start_high
    result['start_low'] = start_low

    # Snug the FIRST step to the true walk start: back the first high-footspeed
    # swing into the velocity valley before it, then recompute that step's
    # start/time/speed/length/width from the leading foot integrated over the
    # snugged window only. Drift before the snug start is ignored, so length is
    # the leading foot's forward travel since g (subtracting its position at g)
    # and width is taken as the assumed initial side-by-side separation.
    result['start_snap'] = g            # computed above, before the contact train
    result['start_snap_foot'] = gfoot
    if g is not None and len(result['time']):
        lead = result['leading_foot'][0]
        end0 = int(result['end_idx'][0])
        if g < end0:
            P = P_map[lead]
            new_t = max((end0 - g) * period, period)
            result['start_idx'][0] = g
            result['time'][0] = new_t
            result['frwd_speed'][0] = float(
                np.linalg.norm(P[end0, :2] - P[g, :2])) / new_t
            result['length'][0] = float(fwd[lead][end0] - fwd[lead][g])
            result['width'][0] = float(initial_separation)
    return result
