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
        # First, let's explore the file structure to understand it better
        print("Exploring file structure...")
        
        def print_all_paths(name, obj):
            print(f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Group):
                for attr_name, attr_value in obj.attrs.items():
                    print(f"  Attribute '{attr_name}': {attr_value}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
                for attr_name, attr_value in obj.attrs.items():
                    print(f"  Attribute '{attr_name}': {attr_value}")
        
        # Quick exploration (comment out after debugging)
        f.visititems(print_all_paths) if hasattr(f, 'visititems') else f.visit(lambda name: print_all_paths(name, f[name]))
        
        # Read CaseIdList - this might be stored as an attribute or dataset
        group_name = None
        
        # Method 1: Try as a dataset
        if 'CaseIdList' in f:
            case_id_obj = f['CaseIdList']
            if isinstance(case_id_obj, h5py.Dataset):
                case_id_data = case_id_obj[()]
                # Handle string encoding
                if case_id_data.dtype.kind in ['S', 'U', 'O']:
                    if case_id_data.ndim == 0:
                        group_name = case_id_data.decode() if hasattr(case_id_data, 'decode') else str(case_id_data)
                    else:
                        # Take first element if it's an array
                        first_item = case_id_data.flat[0]
                        group_name = first_item.decode() if hasattr(first_item, 'decode') else str(first_item)
        
        # Method 2: Try as an attribute
        if group_name is None and 'CaseIdList' in f.attrs:
            case_id_attr = f.attrs['CaseIdList']
            if isinstance(case_id_attr, bytes):
                group_name = case_id_attr.decode()
            elif isinstance(case_id_attr, np.ndarray):
                group_name = case_id_attr[0].decode() if hasattr(case_id_attr[0], 'decode') else str(case_id_attr[0])
            else:
                group_name = str(case_id_attr)
        
        # Method 3: If still not found, check if it's the first/only group
        if group_name is None:
            # Sometimes APDM files have the case ID as the first group name
            root_keys = list(f.keys())
            if len(root_keys) == 1:
                group_name = root_keys[0]
            else:
                # Look for a group that's not a standard HDF5 name
                for key in root_keys:
                    if not key.startswith('#') and isinstance(f[key], h5py.Group):
                        group_name = key
                        break
        
        if group_name is None:
            raise ValueError("Could not find CaseIdList or determine group name")
        
        # Clean up group name (remove leading/trailing slashes if present)
        group_name = group_name.strip('/')
        
        # Now find the sample rate - it could be in various places
        freq = None
        
        # Method 1: As a dataset at the expected path
        sample_rate_path = f'{group_name}/SampleRate'
        if sample_rate_path in f:
            freq_data = f[sample_rate_path][()]
            if isinstance(freq_data, np.ndarray):
                freq = float(freq_data.flat[0])
            else:
                freq = float(freq_data)
        
        # Method 2: As an attribute of the group
        if freq is None and group_name in f:
            group = f[group_name]
            if 'SampleRate' in group.attrs:
                freq = float(group.attrs['SampleRate'])
            elif 'sampleRate' in group.attrs:  # Try different case
                freq = float(group.attrs['sampleRate'])
            elif 'Fs' in group.attrs:  # Common abbreviation
                freq = float(group.attrs['Fs'])
        
        # Method 3: Check in Calibrated subgroup
        if freq is None:
            calib_path = f'{group_name}/Calibrated'
            if calib_path in f:
                calib_group = f[calib_path]
                if 'SampleRate' in calib_group.attrs:
                    freq = float(calib_group.attrs['SampleRate'])
                elif 'sampleRate' in calib_group.attrs:
                    freq = float(calib_group.attrs['sampleRate'])
        
        # Method 4: Look for it anywhere in the file with a sample rate name
        if freq is None:
            def find_sample_rate(name, obj):
                nonlocal freq
                if freq is not None:
                    return
                
                # Check if this is a sample rate dataset
                if any(sr in name.lower() for sr in ['samplerate', 'sample_rate', 'fs', 'frequency']):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            data = obj[()]
                            if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
                                freq = float(data.flat[0] if isinstance(data, np.ndarray) else data)
                                print(f"Found sample rate at: {name} = {freq}")
                        except:
                            pass
                
                # Check attributes
                if hasattr(obj, 'attrs'):
                    for attr_name in ['SampleRate', 'sampleRate', 'sample_rate', 'Fs', 'fs', 'frequency']:
                        if attr_name in obj.attrs:
                            try:
                                freq = float(obj.attrs[attr_name])
                                print(f"Found sample rate as attribute: {name}/{attr_name} = {freq}")
                                return
                            except:
                                pass
            
            if hasattr(f, 'visititems'):
                f.visititems(find_sample_rate)
            else:
                f.visit(lambda name: find_sample_rate(name, f[name]))
        
        if freq is None:
            # Last resort - common default sample rates for APDM
            print("Warning: Could not find sample rate, using default 128 Hz")
            freq = 128.0
        
        period = 1.0 / freq
        
        # Read sensor data (transposed to make Nx3 like MATLAB)
        accel_path = f'{group_name}/Calibrated/Accelerometers'
        gyro_path = f'{group_name}/Calibrated/Gyroscopes'
        mag_path = f'{group_name}/Calibrated/Magnetometers'
        
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

def detect_quite_time(W: np.ndarray, period: float) -> np.ndarray:
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
    not_static = np.array([0] + list(np.where(diff_low_dynamic_periods > CONTINIOUS_SAMPLE_SPACE)[0] + 1) + [len(diff_low_dynamic_periods)])
    diff_not_static = np.diff(not_static)
    most_likely_static_sections = np.where(diff_not_static > MINIMUN_STATIC_SAMPLES)[0]
    if most_likely_static_sections.size == 0:
        most_likely_static_sections = np.where(diff_not_static > MINIMUN_STATIC_SAMPLES // 10)[0]
    start_static_section = low_dynamic_periods[not_static[most_likely_static_sections]]
    end_static_section = low_dynamic_periods[not_static[most_likely_static_sections + 1]]
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

def getdata(W: np.ndarray, A: np.ndarray, period: float, section_seconds: Optional[Any] = None, bias: Optional[float] = None, M: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Section data, apply bias correction, normalize accelerometer, and plot signals.
    """
    if section_seconds is not None and len(section_seconds) > 0:
        section_samples = np.concatenate([
            np.arange(int(np.floor(start / period)), int(np.floor(end / period)) + 1)
            for start, end in np.atleast_2d(section_seconds)
        ])
        W = W[section_samples, :]
        A = A[section_samples, :]
        if M is not None:
            M = M[section_samples, :]
    if bias is not None:
        static_period = np.arange(int(bias / period))
    else:
        static_period = detect_quite_time(W, period)
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