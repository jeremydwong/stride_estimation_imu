import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.stride.apdm import sync_apdm, getdata
from src.stride.inertial import compute_pos_two_imus, stride_segmentation
from src.stride.plotting import plt_ltrl_frwd_strides, plt_frwd_elev_strides, plt_stride_var

# Path to the APDM .h5 files (update if needed)
# LEFT_H5_FILE = os.path.join(os.getcwd(),'matlab', '20120418-132855_sensor_data_monitor_472_label_Left.h5')
# RIGHT_H5_FILE = os.path.join(os.getcwd(),'matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')
# LEFT_H5_FILE = os.path.join('matlab', '20251008-153007_Pilot_Ch_LF_13087.h5')
# RIGHT_H5_FILE = os.path.join('matlab', '20251008-153009_Pilot_Ch_RF_13097.h5')
# RIGHT_H5_FILE = os.path.join('/Users/jeremy/Downloads/chanelsandbox/Pilot_Ch_Oct29','20251029-154310_RF_Pilot_Ch_Oct29.h5')
# LEFT_H5_FILE = os.path.join('/Users/jeremy/Downloads/chanelsandbox/Pilot_Ch_Oct29','20251029-154305_LF_Pilot_Ch_Oct29.h5')

MATLAB_FILE = 'matlab_demo6.mat'

def compare_arrays(python_arr, matlab_arr, name, rtol=1e-5, atol=1e-8):
    """Compare Python and MATLAB arrays and print statistics."""
    print(f"\n--- Comparing {name} ---")
    print(f"  Python shape: {python_arr.shape}, MATLAB shape: {matlab_arr.shape}")

    if python_arr.shape != matlab_arr.shape:
        print(f"  WARNING: Shape mismatch!")
        return

    diff = python_arr - matlab_arr
    max_abs_diff = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))

    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")

    if np.allclose(python_arr, matlab_arr, rtol=rtol, atol=atol):
        print(f"  ✓ Arrays match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"  ✗ Arrays differ beyond tolerance (rtol={rtol}, atol={atol})")
        # Show where differences are largest
        max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"  Largest diff at index {max_idx}: Python={python_arr[max_idx]:.6e}, MATLAB={matlab_arr[max_idx]:.6e}")

    return diff

def compare_structs(python_dict, matlab_struct, name, rtol=1e-5, atol=1e-8):
    """Compare Python dictionary with MATLAB struct."""
    print(f"\n{'='*60}")
    print(f"Comparing {name}")
    print(f"{'='*60}")

    # Get field names from MATLAB struct
    matlab_fields = set(matlab_struct.dtype.names)
    python_fields = set(python_dict.keys())

    print(f"Python fields: {sorted(python_fields)}")
    print(f"MATLAB fields: {sorted(matlab_fields)}")

    common_fields = python_fields & matlab_fields
    python_only = python_fields - matlab_fields
    matlab_only = matlab_fields - python_fields

    if python_only:
        print(f"\nFields only in Python: {python_only}")
    if matlab_only:
        print(f"Fields only in MATLAB: {matlab_only}")

    # Compare common fields
    for field in sorted(common_fields):
        python_val = python_dict[field]
        matlab_val = matlab_struct[field][0, 0]

        if isinstance(python_val, np.ndarray):
            compare_arrays(python_val, matlab_val, f"{name}.{field}", rtol, atol)
        else:
            print(f"\n--- {name}.{field} ---")
            print(f"  Python: {python_val}")
            print(f"  MATLAB: {matlab_val}")

def plot_comparison(python_arr, matlab_arr, name, ylabel='Value'):
    """Plot comparison between Python and MATLAB arrays."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    if len(python_arr.shape) == 1:
        python_arr = python_arr.reshape(-1, 1)
        matlab_arr = matlab_arr.reshape(-1, 1)

    n_cols = python_arr.shape[1]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    labels = ['X', 'Y', 'Z', 'W', 'Q1', 'Q2', 'Q3'] if n_cols <= 7 else [f'Col{i}' for i in range(n_cols)]

    # Plot Python data
    for i in range(n_cols):
        axes[0].plot(python_arr[:, i], color=colors[i % len(colors)], alpha=0.7, label=labels[i])
    axes[0].set_ylabel(f'Python {ylabel}')
    axes[0].set_title(f'{name} - Python')
    axes[0].grid(True)
    axes[0].legend()

    # Plot MATLAB data
    for i in range(n_cols):
        axes[1].plot(matlab_arr[:, i], color=colors[i % len(colors)], alpha=0.7, label=labels[i])
    axes[1].set_ylabel(f'MATLAB {ylabel}')
    axes[1].set_title(f'{name} - MATLAB')
    axes[1].grid(True)
    axes[1].legend()

    # Plot difference
    diff = python_arr - matlab_arr
    for i in range(n_cols):
        axes[2].plot(diff[:, i], color=colors[i % len(colors)], alpha=0.7, label=labels[i])
    axes[2].set_ylabel(f'Difference')
    axes[2].set_xlabel('Sample')
    axes[2].set_title(f'{name} - Difference (Python - MATLAB)')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("=== Loading and syncing two IMU data files ===")

    # Load IMU information from two files and sync them
    left_Wb, left_Ab, right_Wb, right_Ab, PERIOD, left_Mb, right_Mb, static_period = \
        sync_apdm(LEFT_H5_FILE, RIGHT_H5_FILE)

    print(f"\nData loaded and synced successfully:")
    print(f"  Left angular velocity shape: {left_Wb.shape}")
    print(f"  Left acceleration shape: {left_Ab.shape}")
    print(f"  Right angular velocity shape: {right_Wb.shape}")
    print(f"  Right acceleration shape: {right_Ab.shape}")
    print(f"  Sample period: {PERIOD}")
    print(f"  Sample rate: {1/PERIOD} Hz")

    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(240, 350)]
    left_Wb, left_Ab, _, _ = getdata(left_Wb, left_Ab, PERIOD, SECTION)
    right_Wb, right_Ab, _, _ = getdata(right_Wb, right_Ab, PERIOD, SECTION)

    print(f"\nSectioned data:")
    print(f"  Left angular velocity shape: {left_Wb.shape}")
    print(f"  Left acceleration shape: {left_Ab.shape}")
    print(f"  Right angular velocity shape: {right_Wb.shape}")
    print(f"  Right acceleration shape: {right_Ab.shape}")
    print(f"  Section duration: {left_Wb.shape[0] * PERIOD:.1f} seconds")

    # Process data from the two IMUs simultaneously
    print("\n=== Processing two IMUs simultaneously ===")
    left_walk_info, right_walk_info = compute_pos_two_imus(left_Wb, left_Ab, right_Wb, right_Ab, PERIOD)

    # Segment steps
    print("\n=== Segmenting strides ===")
    left_strides = stride_segmentation(left_walk_info, PERIOD)
    right_strides = stride_segmentation(right_walk_info, PERIOD)

    print(f"\nFound {left_strides['frwd'].shape[1] if 'frwd' in left_strides else 0} left strides")
    print(f"Found {right_strides['frwd'].shape[1] if 'frwd' in right_strides else 0} right strides")

    # Load MATLAB reference data if available and compare
    if os.path.exists(MATLAB_FILE):
        print("\n" + "="*60)
        print("LOADING MATLAB REFERENCE DATA FOR COMPARISON")
        print("="*60)
        matlab_data = scipy.io.loadmat(MATLAB_FILE)
        print(f"Loaded MATLAB data with keys: {[k for k in matlab_data.keys() if not k.startswith('__')]}")

        # Compare sectioned IMU data
        print("\n" + "="*60)
        print("COMPARING SECTIONED IMU DATA")
        print("="*60)
        if 'left_Wb' in matlab_data:
            compare_arrays(left_Wb, matlab_data['left_Wb'], 'left_Wb')
        if 'left_Ab' in matlab_data:
            compare_arrays(left_Ab, matlab_data['left_Ab'], 'left_Ab')
        if 'right_Wb' in matlab_data:
            compare_arrays(right_Wb, matlab_data['right_Wb'], 'right_Wb')
        if 'right_Ab' in matlab_data:
            compare_arrays(right_Ab, matlab_data['right_Ab'], 'right_Ab')

        # Compare walk_info
        if 'left_walk_info' in matlab_data:
            matlab_left_walk_info = matlab_data['left_walk_info']
            compare_structs(left_walk_info, matlab_left_walk_info, 'left_walk_info', rtol=1e-5, atol=1e-8)

        if 'right_walk_info' in matlab_data:
            matlab_right_walk_info = matlab_data['right_walk_info']
            compare_structs(right_walk_info, matlab_right_walk_info, 'right_walk_info', rtol=1e-5, atol=1e-8)

        # Plot comparisons
        print("\n=== Creating walk_info comparison plots ===")

        # Left foot comparisons
        if 'left_walk_info' in matlab_data:
            matlab_left = matlab_data['left_walk_info']
            if 'P' in left_walk_info and 'P' in matlab_left.dtype.names:
                fig = plot_comparison(left_walk_info['P'], matlab_left['P'][0, 0],
                                    'left_walk_info.P', ylabel='Position [m]')
                fig.suptitle('Left Foot Position Comparison', fontsize=14, y=1.00)

            if 'V' in left_walk_info and 'V' in matlab_left.dtype.names:
                fig = plot_comparison(left_walk_info['V'], matlab_left['V'][0, 0],
                                    'left_walk_info.V', ylabel='Velocity [m/s]')
                fig.suptitle('Left Foot Velocity Comparison', fontsize=14, y=1.00)

        # Right foot comparisons
        if 'right_walk_info' in matlab_data:
            matlab_right = matlab_data['right_walk_info']
            if 'P' in right_walk_info and 'P' in matlab_right.dtype.names:
                fig = plot_comparison(right_walk_info['P'], matlab_right['P'][0, 0],
                                    'right_walk_info.P', ylabel='Position [m]')
                fig.suptitle('Right Foot Position Comparison', fontsize=14, y=1.00)

            if 'V' in right_walk_info and 'V' in matlab_right.dtype.names:
                fig = plot_comparison(right_walk_info['V'], matlab_right['V'][0, 0],
                                    'right_walk_info.V', ylabel='Velocity [m/s]')
                fig.suptitle('Right Foot Velocity Comparison', fontsize=14, y=1.00)

    # Plot results for left and right foot
    print("\n=== Plotting stride results ===")

    # Create subplots for lateral vs forward strides
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    plt_ltrl_frwd_strides(left_strides, show=False)
    axes[0].set_title('Left Foot - Lateral vs Forward')

    plt.sca(axes[1])
    plt_ltrl_frwd_strides(right_strides, show=False)
    axes[1].set_title('Right Foot - Lateral vs Forward')
    plt.tight_layout()

    # Create subplots for forward vs elevation strides
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.sca(axes[0])
    plt_frwd_elev_strides(left_strides, show=False)
    axes[0].set_title('Left Foot - Forward vs Elevation')

    plt.sca(axes[1])
    plt_frwd_elev_strides(right_strides, show=False)
    axes[1].set_title('Right Foot - Forward vs Elevation')
    plt.tight_layout()

    # Create subplots for stride variability
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    plt_stride_var(left_strides, show=False)
    axes[0].set_title('Left Foot - Stride Variability')

    plt.sca(axes[1])
    plt_stride_var(right_strides, show=False)
    axes[1].set_title('Right Foot - Stride Variability')
    plt.tight_layout()

    plt.show()
