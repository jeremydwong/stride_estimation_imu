import numpy as np
import h5py
from typing import Tuple, Optional, Any, Union, List
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone, timedelta
import pandas as pd

GRAVITY = 9.80297286843


def apdm_time_to_datetime(apdm_time: np.ndarray, utc_offset_hours: Optional[float] = None) -> np.ndarray:
    """
    Convert APDM uint64 timestamps to datetime objects.

    APDM stores time as microseconds since Unix epoch (Jan 1, 1970).
    If utc_offset_hours is provided, datetimes are converted to the recording's
    local timezone so results are consistent regardless of server timezone.

    Parameters:
    -----------
    apdm_time : np.ndarray
        Array of uint64 timestamps from APDM sensor
    utc_offset_hours : float, optional
        UTC offset of the recording timezone (e.g., -7 for Mountain Time).
        If provided, returns timezone-aware datetimes in that timezone.
        If None, falls back to system local time (legacy behavior).

    Returns:
    --------
    np.ndarray
        Array of datetime objects
    """
    timestamps_seconds = apdm_time / 1_000_000.0
    if utc_offset_hours is not None:
        tz = timezone(timedelta(hours=utc_offset_hours))
        return np.array([datetime.fromtimestamp(ts, tz=tz) for ts in timestamps_seconds])
    else:
        return np.array([datetime.fromtimestamp(ts) for ts in timestamps_seconds])


@dataclass
class ImuRecording:
    """
    Bundle of IMU recording data with time information.

    Attributes:
        Wb: Angular velocity in body frame (Nx3 array, rad/sample)
        Ab: Acceleration in body frame (Nx3 array, m/s^2)
        time_datetime: Array of datetime objects for each sample
        time_elapsed_samples: Array of elapsed time in samples (0, 1, 2, ...)
        period: Sampling period in seconds
        Mb: Optional magnetometer data (Nx3 array)
        raw_time: Optional raw APDM timestamps (microseconds since epoch) for sync
        file_path: Optional source file path

    Supports slicing: recording[100:200] returns a new ImuRecording with samples 100-199.
    """
    Wb: np.ndarray
    Ab: np.ndarray
    time_datetime: np.ndarray  # Array of datetime objects
    time_elapsed_samples: np.ndarray  # Array of sample indices (0, 1, 2, ...)
    period: float
    Mb: Optional[np.ndarray] = None
    raw_time: Optional[np.ndarray] = None  # Raw APDM timestamps for precise sync
    file_path: Optional[str] = None
    tz_offset_hours: Optional[float] = None  # UTC offset from APDM sensor config

    def __len__(self) -> int:
        return len(self.Wb)

    def __getitem__(self, key):
        """Slice the recording. Supports integer indices and slices."""
        if isinstance(key, int):
            # Single index - convert to slice for consistency
            if key < 0:
                key = len(self) + key
            key = slice(key, key + 1)

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise ValueError("ImuRecording slicing does not support step != 1")

            return ImuRecording(
                Wb=self.Wb[start:stop, :],
                Ab=self.Ab[start:stop, :],
                time_datetime=self.time_datetime[start:stop],
                time_elapsed_samples=np.arange(stop - start),
                period=self.period,
                Mb=self.Mb[start:stop, :] if self.Mb is not None else None,
                raw_time=self.raw_time[start:stop] if self.raw_time is not None else None,
                file_path=self.file_path,
                tz_offset_hours=self.tz_offset_hours,
            )

        raise TypeError(f"ImuRecording indices must be integers or slices, not {type(key).__name__}")


def indices_from_time_array(time_datetime: np.ndarray,
                            start_time: dt_time,
                            end_time: dt_time) -> np.ndarray:
    """
    Get indices from a time array for a given time range (ignoring date).

    Parameters:
    -----------
    time_datetime : np.ndarray
        Array of datetime objects
    start_time : datetime.time
        Start time (hour, minute, second) - date is ignored
    end_time : datetime.time
        End time (hour, minute, second) - date is ignored

    Returns:
    --------
    np.ndarray
        Array of indices within the specified time range
    """
    # Extract time-of-day from each datetime
    times_of_day = np.array([dt.time() if isinstance(dt, datetime) else dt for dt in time_datetime])

    # Find indices where time is within range
    mask = np.array([start_time <= t <= end_time for t in times_of_day])
    indices = np.where(mask)[0]

    return indices


def indices_from_time_recording(recording: ImuRecording,
                                start_time: dt_time,
                                end_time: dt_time) -> np.ndarray:
    """
    Get indices from an ImuRecording for a given time range (ignoring date).

    Parameters:
    -----------
    recording : ImuRecording
        IMU recording with time information
    start_time : datetime.time
        Start time (hour, minute, second) - date is ignored
    end_time : datetime.time
        End time (hour, minute, second) - date is ignored

    Returns:
    --------
    np.ndarray
        Array of indices within the specified time range
    """
    return indices_from_time_array(recording.time_datetime, start_time, end_time)


def slice_from_time_recording(recording: ImuRecording,
                              start_time: dt_time,
                              end_time: dt_time) -> ImuRecording:
    """
    Slice an ImuRecording to a specific time range (ignoring date).

    Parameters:
    -----------
    recording : ImuRecording
        IMU recording to slice
    start_time : datetime.time
        Start time (hour, minute, second) - date is ignored
    end_time : datetime.time
        End time (hour, minute, second) - date is ignored

    Returns:
    --------
    ImuRecording
        New ImuRecording containing only data within the time range
    """
    indices = indices_from_time_recording(recording, start_time, end_time)

    return ImuRecording(
        Wb=recording.Wb[indices, :],
        Ab=recording.Ab[indices, :],
        time_datetime=recording.time_datetime[indices],
        time_elapsed_samples=np.arange(len(indices)),  # Reset to 0-based
        period=recording.period,
        Mb=recording.Mb[indices, :] if recording.Mb is not None else None,
        raw_time=recording.raw_time[indices] if recording.raw_time is not None else None,
        file_path=recording.file_path,
        tz_offset_hours=recording.tz_offset_hours,
    )


def slice_from_time_arrays(arrays: List[np.ndarray],
                           time_datetime: np.ndarray,
                           start_time: dt_time,
                           end_time: dt_time) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Slice multiple arrays to a specific time range (ignoring date).

    All arrays must have the same length as time_datetime along axis 0.

    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of arrays to slice (all must have same length along axis 0)
    time_datetime : np.ndarray
        Array of datetime objects corresponding to the arrays
    start_time : datetime.time
        Start time (hour, minute, second) - date is ignored
    end_time : datetime.time
        End time (hour, minute, second) - date is ignored

    Returns:
    --------
    Tuple[List[np.ndarray], np.ndarray]
        (sliced_arrays, sliced_time_datetime)
    """
    # Check all arrays have same length
    n_samples = len(time_datetime)
    for i, arr in enumerate(arrays):
        assert arr.shape[0] == n_samples, \
            f"Array {i} has length {arr.shape[0]}, expected {n_samples}"

    indices = indices_from_time_array(time_datetime, start_time, end_time)

    sliced_arrays = []
    for arr in arrays:
        if arr.ndim == 1:
            sliced_arrays.append(arr[indices])
        else:
            sliced_arrays.append(arr[indices, :])

    return sliced_arrays, time_datetime[indices]


def hdf5read(file_path: str, dataset_name: str) -> Union[np.ndarray, Any]:
    """
    Python implementation of MATLAB's hdf5read function.
    Reads data from an HDF5 dataset.
    """
    with h5py.File(file_path, 'r') as f:
        # Navigate to the dataset
        if dataset_name.startswith('/'):
            dataset_name = dataset_name[1:]  # Remove leading slash
        
        # Handle nested paths
        path_parts = dataset_name.split('/')
        current = f
        
        for part in path_parts:
            if part in current:
                current = current[part]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in file")
        
        # Read the data
        data = current[()]
        
        # For CaseIdList, MATLAB returns a structure-like object
        # We need to handle this specially
        if dataset_name.endswith('CaseIdList'):
            # In MATLAB, this returns a structure with .data field
            # We'll simulate this by creating a simple object
            class CaseIdItem:
                def __init__(self, data):
                    self.data = data
            
            # Convert bytes to string if needed
            if isinstance(data, np.ndarray) and data.dtype.char == 'S':
                # Handle string arrays from HDF5
                case_ids = []
                for item in data:
                    if isinstance(item, (bytes, np.bytes_)):
                        case_ids.append(CaseIdItem(item.decode('utf-8')))
                    else:
                        case_ids.append(CaseIdItem(str(item)))
                return case_ids
            elif isinstance(data, bytes):
                return [CaseIdItem(data.decode('utf-8'))]
            else:
                return [CaseIdItem(str(data))]
        
        return data


def _apply_orientation(w: np.ndarray, a: np.ndarray, m: np.ndarray,
                       period: float, orientation: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply APDM sensor orientation transform.

    Parameters:
    -----------
    w : np.ndarray
        Raw gyroscope data (Nx3)
    a : np.ndarray
        Raw accelerometer data (Nx3)
    m : np.ndarray
        Raw magnetometer data (Nx3)
    period : float
        Sampling period in seconds (used to scale angular velocity to rad/sample)
    orientation : int, optional
        Orientation code (default: LED_UP_RIGHT_FRWD = 1)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (W, A, M) — angular velocity (rad/sample), acceleration (m/s^2), magnetometer
    """
    # Orientation constants (matching MATLAB exactly)
    ORIGINAL = 0
    LED_UP_RIGHT_FRWD = 1  # This one is normally used on feet
    LED_UP_LEFT_FRWD = 2
    LED_LOW_RIGHT_FRWD = 3
    LED_LOW_DOWN_FRWD = 4
    LED_UP_RIGHT_BACK = 5
    LED_LOW_LEFT_BACK = 6

    if orientation is None:
        orientation = LED_UP_RIGHT_FRWD

    if orientation == ORIGINAL:
        WX, WY, WZ = w[:, 0], w[:, 1], w[:, 2]
        AX, AY, AZ = a[:, 0], a[:, 1], a[:, 2]
    elif orientation == LED_UP_RIGHT_FRWD:
        WX, WY, WZ = w[:, 1], w[:, 0], -w[:, 2]
        AX, AY, AZ = a[:, 1], a[:, 0], -a[:, 2]
    elif orientation == LED_UP_LEFT_FRWD:
        WX, WY, WZ = w[:, 0], -w[:, 1], -w[:, 2]
        AX, AY, AZ = a[:, 0], -a[:, 1], -a[:, 2]
    elif orientation == LED_LOW_DOWN_FRWD:
        WX, WY, WZ = -w[:, 1], -w[:, 0], -w[:, 2]
        AX, AY, AZ = -a[:, 1], -a[:, 0], -a[:, 2]
    elif orientation == LED_LOW_RIGHT_FRWD:
        WX, WY, WZ = -w[:, 0], w[:, 1], -w[:, 2]
        AX, AY, AZ = -a[:, 0], a[:, 1], -a[:, 2]
    elif orientation == LED_UP_RIGHT_BACK:
        WX, WY, WZ = w[:, 2], -w[:, 0], w[:, 1]
        AX, AY, AZ = a[:, 2], -a[:, 0], a[:, 1]
    elif orientation == LED_LOW_LEFT_BACK:
        WX, WY, WZ = w[:, 2], w[:, 0], -w[:, 1]
        AX, AY, AZ = a[:, 2], a[:, 0], -a[:, 1]
    else:
        raise ValueError('Unknown orientation')

    W = np.column_stack([WX, WY, WZ]) * period
    A = np.column_stack([AX, AY, AZ])
    M = m

    return W, A, M


def getdata_apdm(file_path: str, orientation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """
    Load IMU data from APDM .h5 file and apply orientation transformation.

    Returns:
    --------
    Tuple containing:
        W : np.ndarray
            Angular velocity (Nx3, rad/sample)
        A : np.ndarray
            Acceleration (Nx3, m/s^2)
        period : float
            Sampling period in seconds
        M : np.ndarray
            Magnetometer data (Nx3)
        time_datetime : np.ndarray
            Array of datetime objects for each sample
        time_elapsed_samples : np.ndarray
            Array of elapsed time in samples (0, 1, 2, ...)
        tz_offset_hours : float or None
            UTC offset from sensor config (e.g., -6.0 for MDT), or None if not found
    """

    with h5py.File(file_path, 'r') as f:
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        l_time = f[l1][l2_sensorid]['Time'][()]
        freq = float(f[l1][l2_sensorid]['Configuration'].attrs['Sample Rate'])
        period = 1.0 / freq

        # Read timezone offset from sensor configuration if available
        config = f[l1][l2_sensorid]['Configuration']
        if 'Timezone Offset' in config.attrs:
            tz_offset_hours = float(config.attrs['Timezone Offset'])
        else:
            tz_offset_hours = None
        # Read sensor data (transposed to make Nx3 like MATLAB)
        accel_path = f'{l1}/{l2_sensorid}/Accelerometer'
        gyro_path = f'{l1}/{l2_sensorid}/Gyroscope'
        mag_path = f'{l1}/{l2_sensorid}/Magnetometer'

        # Read accelerometer data
        if accel_path in f:
            a = f[accel_path][()]
        else:
            raise ValueError(f"Accelerometer data not found at {accel_path}")

        # Read gyroscope data
        if gyro_path in f:
            w = f[gyro_path][()]
        else:
            raise ValueError(f"Gyroscope data not found at {gyro_path}")

        # Read magnetometer data
        if mag_path in f:
            m = f[mag_path][()]
        else:
            raise ValueError(f"Magnetometer data not found at {mag_path}")

    # Convert APDM timestamps to datetime (using recording timezone if available)
    time_datetime = apdm_time_to_datetime(l_time, tz_offset_hours)
    time_elapsed_samples = np.arange(len(l_time))

    # Apply orientation transformation
    W, A, M = _apply_orientation(w, a, m, period, orientation)

    return W, A, period, M, time_datetime, time_elapsed_samples, tz_offset_hours

def detect_quiet_time(W: np.ndarray, period: float) -> np.ndarray:
    """
    Auto-detect static period in angular velocity data.
    Returns indices of static period.
    """
    MAXIMUM_STATIC_RATE = 2 * np.pi / 180
    MINIMUN_STATIC_SAMPLES = int(1.5 / period)
    CONTINIOUS_SAMPLE_SPACE = 1
    MAXIMUN_BIAS_SIGNAL_VAR = 2
    Wm = np.sqrt(np.sum(W ** 2, axis=1)) / period
    dWm = np.diff(Wm)
    low_dynamic_periods = np.where(np.abs(dWm) < MAXIMUM_STATIC_RATE)[0]
    diff_low_dynamic_periods = np.diff(low_dynamic_periods)
    not_static = np.array([0] + list(np.where(diff_low_dynamic_periods > CONTINIOUS_SAMPLE_SPACE)[0]) + [len(diff_low_dynamic_periods)-1])
    diff_not_static = np.diff(not_static)
    most_likely_static_sections = np.where(diff_not_static > MINIMUN_STATIC_SAMPLES)[0]
    if most_likely_static_sections.size == 0:
        most_likely_static_sections = np.where(diff_not_static > MINIMUN_STATIC_SAMPLES // 10)[0]
    start_static_section = low_dynamic_periods[not_static[most_likely_static_sections]]
    end_static_section = low_dynamic_periods[not_static[most_likely_static_sections+1]]
    if len(start_static_section) > 0:
        static_period = np.concatenate([np.arange(start, end + 1) for start, end in zip(start_static_section, end_static_section)])
        bias_var = np.std(Wm[static_period])
        bias_mean = np.median(Wm[static_period])
        best_static_signal = np.where(np.abs(Wm[static_period] - bias_mean) < bias_var * MAXIMUN_BIAS_SIGNAL_VAR)[0]
        static_period = static_period[best_static_signal]
    else:
        # If no static period found, use first few samples as fallback
        static_period = np.arange(min(100, len(Wm)))
    return static_period

def getdata(Win: np.ndarray, A: np.ndarray, period: float, section_seconds: Optional[Any] = None,
            bias: Optional[float] = None, M: Optional[np.ndarray] = None,
            time_datetime: Optional[np.ndarray] = None,
            time_elapsed_samples: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Section data, apply bias correction, normalize accelerometer, and plot signals.

    Returns:
    --------
    Tuple containing:
        W : np.ndarray
            Bias-corrected angular velocity (Nx3)
        A : np.ndarray
            Gravity-normalized acceleration (Nx3)
        static_period : np.ndarray
            Indices of detected static period
        M : np.ndarray or None
            Magnetometer data (Nx3) if provided
        time_datetime : np.ndarray or None
            Sliced datetime array if provided
        time_elapsed_samples : np.ndarray or None
            Reset elapsed samples (0, 1, 2, ...) if time was provided
    """
    W = Win.copy()

    # If section_seconds is None, use the entire data range
    if section_seconds is None:
        SECTION_SAMPLES = list(range(W.shape[0]))
    else:
        # Handle single tuple case - wrap it in a list
        if isinstance(section_seconds, tuple):
            section_seconds = [section_seconds]
        elif not isinstance(section_seconds, list):
            raise TypeError("section_seconds must be a list of tuples or a single tuple")
        elif len(section_seconds) > 0 and not isinstance(section_seconds[0], tuple):
            # If it's a list but not of tuples, try to convert
            section_seconds = [tuple(section_seconds)]

        # Now build SECTION_SAMPLES
        SECTION_SAMPLES = []
        for section in section_seconds:
            start_sec, end_sec = section
            SECTION_SAMPLES.extend(range(
                int(np.floor(start_sec / period))-1,
                int(np.floor(end_sec / period))
            ))

    W = W[SECTION_SAMPLES, :]
    A = A[SECTION_SAMPLES, :]
    if M is not None:
        M = M[SECTION_SAMPLES, :]
    if time_datetime is not None:
        time_datetime = time_datetime[SECTION_SAMPLES]
        time_elapsed_samples = np.arange(len(SECTION_SAMPLES))
    if bias is not None:
        static_period = np.arange(int(bias / period))
    else:
        static_period = detect_quiet_time(W, period)
    W = W - np.mean(W[static_period, :], axis=0)
    static_acceleration = np.sqrt(np.sum(A[static_period, :] ** 2, axis=1))
    gravity_measurement = np.mean(static_acceleration)
    A = A * GRAVITY / gravity_measurement
    
    # Create plots showing the inertial signals (as in MATLAB version)
    import matplotlib.pyplot as plt
    t = np.arange(W.shape[0]) * period
    
    # Determine if we have magnetometer data
    include_magnetometer = M is not None and M.size > 0
    
    fig, axes = plt.subplots(2 + (1 if include_magnetometer else 0), 1, figsize=(10, 8))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot gyroscope data
    axes[0].plot(t[static_period], W[static_period, :] / period * 180 / np.pi, '.k', label='Static time')
    axes[0].plot(t, W / period * 180 / np.pi)
    axes[0].grid(True)
    axes[0].set_ylabel('W [deg/s]')
    axes[0].set_title('Body referenced inertial signals')
    axes[0].legend()
    
    # Plot accelerometer data
    axes[1].plot(t, A)
    axes[1].grid(True)
    axes[1].set_ylabel('A [m/s^2]')
    
    # Plot magnetometer data if available
    if include_magnetometer:
        axes[2].plot(t, M)
        axes[2].grid(True)
        axes[2].set_ylabel('Mag')
    
    # Link x-axes and add time label
    for ax in axes:
        ax.set_xlim(t[0], t[-1])
    axes[-1].set_xlabel('time [s]')
    
    plt.tight_layout()
    plt.show()

    return W, A, static_period, M, time_datetime, time_elapsed_samples


def load_imu_recording(file_path: str, orientation: Optional[int] = None) -> ImuRecording:
    """Primary API for loading APDM sensor data from an HDF5 file.

    Opens the file once, reads all sensor channels (accelerometer, gyroscope,
    magnetometer), timestamps, sampling rate, and timezone offset, applies the
    orientation transform, and returns a fully populated ImuRecording.

    Parameters:
    -----------
    file_path : str
        Path to APDM HDF5 file
    orientation : int, optional
        Orientation code (default: LED_UP_RIGHT_FRWD = 1)

    Returns:
    --------
    ImuRecording
        IMU recording with all sensor data and metadata
    """
    with h5py.File(file_path, 'r') as f:
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        raw_time = np.asarray(f[l1][l2_sensorid]['Time'][()])  # type: ignore[index]
        freq = float(f[l1][l2_sensorid]['Configuration'].attrs['Sample Rate'])
        period = 1.0 / freq

        # Read timezone offset from sensor configuration if available
        config = f[l1][l2_sensorid]['Configuration']
        if 'Timezone Offset' in config.attrs:
            tz_offset_hours = float(config.attrs['Timezone Offset'])
        else:
            tz_offset_hours = None

        # Read sensor data
        accel_path = f'{l1}/{l2_sensorid}/Accelerometer'
        gyro_path = f'{l1}/{l2_sensorid}/Gyroscope'
        mag_path = f'{l1}/{l2_sensorid}/Magnetometer'

        if accel_path in f:
            a = f[accel_path][()]
        else:
            raise ValueError(f"Accelerometer data not found at {accel_path}")

        if gyro_path in f:
            w = f[gyro_path][()]
        else:
            raise ValueError(f"Gyroscope data not found at {gyro_path}")

        if mag_path in f:
            m = f[mag_path][()]
        else:
            raise ValueError(f"Magnetometer data not found at {mag_path}")

    # Apply orientation transform
    W, A, M = _apply_orientation(w, a, m, period, orientation)

    # Convert timestamps
    time_datetime = apdm_time_to_datetime(raw_time, tz_offset_hours)
    time_elapsed_samples = np.arange(len(raw_time))

    return ImuRecording(
        Wb=W,
        Ab=A,
        time_datetime=time_datetime,
        time_elapsed_samples=time_elapsed_samples,
        period=period,
        Mb=M,
        raw_time=raw_time,
        file_path=file_path,
        tz_offset_hours=tz_offset_hours,
    )


def slice_recording_by_indices(recording: ImuRecording, start_idx: int, end_idx: int) -> ImuRecording:
    """
    Slice an ImuRecording by sample indices.

    Parameters:
    -----------
    recording : ImuRecording
        IMU recording to slice
    start_idx : int
        Start index (inclusive)
    end_idx : int
        End index (exclusive)

    Returns:
    --------
    ImuRecording
        Sliced recording
    """
    return ImuRecording(
        Wb=recording.Wb[start_idx:end_idx, :],
        Ab=recording.Ab[start_idx:end_idx, :],
        time_datetime=recording.time_datetime[start_idx:end_idx],
        time_elapsed_samples=np.arange(end_idx - start_idx),
        period=recording.period,
        Mb=recording.Mb[start_idx:end_idx, :] if recording.Mb is not None else None,
        raw_time=recording.raw_time[start_idx:end_idx] if recording.raw_time is not None else None,
        file_path=recording.file_path,
        tz_offset_hours=recording.tz_offset_hours,
    )


def find_overlapping_recordings(recordings: List[ImuRecording]) -> List[ImuRecording]:
    """
    Find the overlapping time region across multiple recordings and return sliced copies.

    Parameters:
    -----------
    recordings : List[ImuRecording]
        List of ImuRecording objects (must have raw_time populated)

    Returns:
    --------
    List[ImuRecording]
        New ImuRecording objects sliced to the overlapping time region
    """
    if len(recordings) == 0:
        return []

    if len(recordings) == 1:
        return [recordings[0]]

    # Ensure all recordings have raw_time
    for rec in recordings:
        if rec.raw_time is None:
            raise ValueError(f"Recording {rec.file_path} missing raw_time - needed for sync")

    # Find the global start time (latest start among all sensors)
    global_start_time = max(rec.raw_time[0] for rec in recordings)  # type: ignore[index]

    # Find the global end time (earliest end among all sensors)
    global_end_time = min(rec.raw_time[-1] for rec in recordings)  # type: ignore[index]

    if global_start_time >= global_end_time:
        raise ValueError("No overlapping time period found among recordings")

    # For each recording, find the indices corresponding to the overlap region
    result = []
    for rec in recordings:
        # Find first index >= global_start_time
        start_indices = np.where(rec.raw_time >= global_start_time)[0]  # type: ignore[union-attr]
        start_idx = int(start_indices[0])

        # Find last index <= global_end_time
        end_indices = np.where(rec.raw_time <= global_end_time)[0]  # type: ignore[union-attr]
        end_idx = int(end_indices[-1]) + 1  # +1 for exclusive end

        result.append(slice_recording_by_indices(rec, start_idx, end_idx))

    # Print summary
    overlap_samples = min(len(rec.Wb) for rec in result)
    overlap_seconds = overlap_samples * result[0].period
    print(f"Found overlapping region across {len(recordings)} recordings:")
    print(f"  Overlapping samples: ~{overlap_samples} ({overlap_seconds:.2f} seconds)")
    for rec in result:
        print(f"  {rec.file_path}: {len(rec.Wb)} samples")

    return result


def load_overlapping_recordings(file_paths: List[str],
                                 orientation: Optional[int] = None) -> List[ImuRecording]:
    """
    Load multiple APDM sensor files and return only the overlapping time region.

    Parameters:
    -----------
    file_paths : List[str]
        List of paths to APDM HDF5 files
    orientation : int, optional
        Orientation code for all sensors (default: LED_UP_RIGHT_FRWD = 1)

    Returns:
    --------
    List[ImuRecording]
        List of ImuRecording objects, all sliced to the overlapping time region
    """
    # Load all recordings
    recordings = [load_imu_recording(fp, orientation) for fp in file_paths]

    # Find and return overlapping region
    return find_overlapping_recordings(recordings)


def sync_apdm(l_file: str, r_file: str, sync: bool = True, force_sync_value: int = 0) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synchronize two APDM sensor files (left and right IMUs).

    Parameters:
    -----------
    l_file : str
        Path to left IMU HDF5 file
    r_file : str
        Path to right IMU HDF5 file
    sync : bool, optional
        Whether to synchronize the data (default: True)
    force_sync_value : int, optional
        Force a specific sync shift value in samples (default: 0)

    Returns:
    --------
    tuple
        (LeftWb, LeftAb, RightWb, RightAb, PERIOD, LeftMb, RightMb, static_period,
         left_time_datetime, left_time_elapsed_samples,
         right_time_datetime, right_time_elapsed_samples)
    """
    import matplotlib.pyplot as plt

    # Read timestamps from both files
    # COMMENTED OUT - using direct h5py access instead of hdf5read()
    # with h5py.File(l_file, 'r') as f:
    #     case_id_list = hdf5read(l_file, '/CaseIdList')
    #     group_name = case_id_list[0].data
    #     l_time = hdf5read(l_file, f'{group_name}/Time')
    #     freq = hdf5read(l_file, f'{group_name}/SampleRate')
    #     if isinstance(freq, np.ndarray):
    #         freq = float(freq.flat[0])
    #     else:
    #         freq = float(freq)
    #     period = 1.0 / freq
    #
    # with h5py.File(r_file, 'r') as f:
    #     case_id_list = hdf5read(r_file, '/CaseIdList')
    #     group_name = case_id_list[0].data
    #     r_time = hdf5read(r_file, f'{group_name}/Time')

    # Direct h5py.File reading (adapted from getdata_apdm)
    # Read left file
    with h5py.File(l_file, 'r') as f:
        # Find group name (same logic as getdata_apdm lines 89-130)
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        l3 = 'Time'
        l_time = np.asarray(f[l1][l2_sensorid][l3][()])  # type: ignore[index]
        freq = float(f[l1][l2_sensorid]['Configuration'].attrs['Sample Rate'])  # type: ignore[index]
        period = 1.0 / freq

    with h5py.File(r_file, 'r') as f:
        # Find group name (same logic as getdata_apdm lines 89-130)
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        l3 = 'Time'
        r_time = np.asarray(f[l1][l2_sensorid][l3][()])  # type: ignore[index]
        # freq = float(f[l1][l2_sensorid]['Configuration'].attrs['Sample Rate'])    # well, it would be very bad if it were different than l_file
        # period = 1.0 / freq                                                       # well, it would be very bad if it were different than l_file

    # Define initial sections (full files)
    l_section = [1 * period, len(l_time) * period]
    r_section = [1 * period, len(r_time) * period]

    if sync:
        if force_sync_value != 0:
            # If a known value of the time shift is known, use it
            print('Warning: Sync with user defined value')
            shift = force_sync_value
        else:
            print('Sync with sensor timer')
            if l_time[0] <= r_time[0]:
                # Find first index in left that is >= right's first time
                shift = -(np.where(l_time >= r_time[0])[0][0]+1) #0 to 1 indexing matlab. Converting indices to time.
            else:
                # Find first index in right that is >= left's first time
                shift = np.where(r_time >= l_time[0])[0][0]+1    #0 to 1 as above.

        if shift < 0:
            print(f'Shift Left IMU signals [{-shift}]')
            l_start = -shift
            r_start = 0
        else:
            print(f'Shift Right IMU signals [{shift}]')
            l_start = 0
            r_start = shift

        # Calculate the overlapping region
        # After alignment, find how many samples overlap
        l_remaining = len(l_time) - l_start
        r_remaining = len(r_time) - r_start
        overlap_samples = min(l_remaining, r_remaining)

        # Set sections to the overlapping region
        l_section = [l_start * period, (l_start + overlap_samples) * period]
        r_section = [r_start * period, (r_start + overlap_samples) * period]
        print(f'Overlapping samples: {overlap_samples} ({overlap_samples * period:.2f} seconds)')
    else:
        print('Warning: Not Syncing')

    # Load left IMU data
    Wb, Ab, period, Mb, time_dt, time_samp, _ = getdata_apdm(l_file, 1)
    LeftWb, LeftAb, static_period, LeftMb, left_time_datetime, left_time_elapsed_samples = getdata(
        Wb, Ab, period, [tuple(l_section)], M=Mb, time_datetime=time_dt, time_elapsed_samples=time_samp)
    if plt.get_fignums():
        plt.gcf().canvas.manager.set_window_title('Left IMU')

    # Load right IMU data
    Wb, Ab, period, Mb, time_dt, time_samp, _ = getdata_apdm(r_file, 1)
    RightWb, RightAb, _, RightMb, right_time_datetime, right_time_elapsed_samples = getdata(
        Wb, Ab, period, [tuple(r_section)], M=Mb, time_datetime=time_dt, time_elapsed_samples=time_samp)
    if plt.get_fignums():
        plt.gcf().canvas.manager.set_window_title('Right IMU')

    return (LeftWb, LeftAb, RightWb, RightAb, period, LeftMb, RightMb, static_period,
            left_time_datetime, left_time_elapsed_samples,
            right_time_datetime, right_time_elapsed_samples)