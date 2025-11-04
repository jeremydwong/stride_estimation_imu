import numpy as np
import h5py
from typing import Tuple, Optional, Any, Union

GRAVITY = 9.80297286843


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


import h5py
import numpy as np
from typing import Optional, Tuple

import h5py
import numpy as np
from typing import Optional, Tuple

def getdata_apdm(file_path: str, orientation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Load IMU data from APDM .h5 file and apply orientation transformation.
    Returns: (W, A, PERIOD, M)
    """
    
    with h5py.File(file_path, 'r') as f:
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        l_time = f[l1][l2_sensorid]['Time'][()]
        freq = float(f[l1][l2_sensorid]['Configuration'].attrs['Sample Rate'])
        period = 1.0 / freq
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
    
    # Apply orientation transformation (exactly matching MATLAB switch statement)
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
    
    # Assemble final outputs (matching MATLAB exactly)
    W = np.column_stack([WX, WY, WZ]) * period
    A = np.column_stack([AX, AY, AZ])
    M = m
    
    return W, A, period, M

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

def getdata(Win: np.ndarray, A: np.ndarray, period: float, section_seconds: Optional[Any] = None, bias: Optional[float] = None, M: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Section data, apply bias correction, normalize accelerometer, and plot signals.
    """
    W = Win.copy()
    if not isinstance(section_seconds, list):
        raise TypeError("section_seconds must be a list of tuples")

    # Handle single tuple case - wrap it in a list
    if isinstance(section_seconds, tuple):
        section_seconds = [section_seconds]
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
    
    return W, A, static_period, M


def sync_apdm(l_file: str, r_file: str, sync: bool = True, force_sync_value: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
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
        (LeftWb, LeftAb, RightWb, RightAb, PERIOD, LeftMb, RightMb, static_period)
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
    l_time = 0
    r_time = 0
    with h5py.File(l_file, 'r') as f:
        # Find group name (same logic as getdata_apdm lines 89-130)
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        l3 = 'Time'
        l_time = f[l1][l2_sensorid][l3][()]
        freq = float(f[l1][l2_sensorid]['Configuration'].attrs['Sample Rate'])
        period = 1.0 / freq

    with h5py.File(l_file, 'r') as f:
        # Find group name (same logic as getdata_apdm lines 89-130)
        l1 = 'Sensors'
        l2_sensorid = list(f[l1].keys())[0]
        l3 = 'Time'
        r_time = f[l1][l2_sensorid][l3][()]
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
            l_section = [-shift * period, len(l_time) * period]
        else:
            print(f'Shift Right IMU signals [{shift}]')
            r_section = [shift * period, len(r_time) * period]
    else:
        print('Warning: Not Syncing')

    # Load left IMU data
    Wb, Ab, period, Mb = getdata_apdm(l_file, 1)
    LeftWb, LeftAb, static_period, LeftMb = getdata(Wb, Ab, period, [tuple(l_section)], M=Mb)
    if plt.get_fignums():
        plt.gcf().canvas.manager.set_window_title('Left IMU')

    # Load right IMU data
    Wb, Ab, period, Mb = getdata_apdm(r_file, 1)
    RightWb, RightAb, _, RightMb = getdata(Wb, Ab, period, [tuple(r_section)], M=Mb)
    if plt.get_fignums():
        plt.gcf().canvas.manager.set_window_title('Right IMU')

    return LeftWb, LeftAb, RightWb, RightAb, period, LeftMb, RightMb, static_period