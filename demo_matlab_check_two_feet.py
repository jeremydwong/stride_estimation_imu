import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu
from matlab_compare import compare_arrays, compare_structs, plot_comparison

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the APDM .h5 files: the Oct 29 pilot recordings that matlab/demo6.m
# used to generate the matlab_demo6.mat reference (update if needed)
LEFT_H5_FILE = os.path.join(REPO_DIR, 'data', '20251029-154305_LF_Pilot_Ch_Oct29.h5')
RIGHT_H5_FILE = os.path.join(REPO_DIR, 'data', '20251029-154310_RF_Pilot_Ch_Oct29.h5')

MATLAB_FILE = os.path.join(REPO_DIR, 'matlab_demo6.mat')

if __name__ == '__main__':
    print("=== Loading and syncing two IMU data files ===")

    # Load IMU information from two files and sync them
    (left_Wb, left_Ab, right_Wb, right_Ab, PERIOD, left_Mb, right_Mb, static_period,
     left_time_datetime, left_time_elapsed_samples,
     right_time_datetime, right_time_elapsed_samples) = \
        imu.sync_apdm(LEFT_H5_FILE, RIGHT_H5_FILE)

    print(f"\nData loaded and synced successfully:")
    print(f"  Left angular velocity shape: {left_Wb.shape}")
    print(f"  Left acceleration shape: {left_Ab.shape}")
    print(f"  Right angular velocity shape: {right_Wb.shape}")
    print(f"  Right acceleration shape: {right_Ab.shape}")
    print(f"  Sample period: {PERIOD}")
    print(f"  Sample rate: {1/PERIOD} Hz")

    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(2200, 3500)]  # matches SECTION in matlab/demo6.m
    left_Wb, left_Ab, _, _, _, _ = imu.getdata(left_Wb, left_Ab, PERIOD, SECTION)
    right_Wb, right_Ab, _, _, _, _ = imu.getdata(right_Wb, right_Ab, PERIOD, SECTION)

    print(f"\nSectioned data:")
    print(f"  Left angular velocity shape: {left_Wb.shape}")
    print(f"  Left acceleration shape: {left_Ab.shape}")
    print(f"  Right angular velocity shape: {right_Wb.shape}")
    print(f"  Right acceleration shape: {right_Ab.shape}")
    print(f"  Section duration: {left_Wb.shape[0] * PERIOD:.1f} seconds")

    # Process data from the two IMUs simultaneously
    print("\n=== Processing two IMUs simultaneously ===")
    left_walk_info, right_walk_info = imu.compute_position_two_imus(left_Wb, left_Ab, right_Wb, right_Ab, PERIOD)

    # Segment steps
    print("\n=== Segmenting strides ===")
    left_strides = imu.stride_segmentation(left_walk_info, PERIOD)
    right_strides = imu.stride_segmentation(right_walk_info, PERIOD)

    print(f"\nFound {left_strides.frwd.shape[1]} left strides")
    print(f"Found {right_strides.frwd.shape[1]} right strides")

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
            if hasattr(left_walk_info, 'P') and 'P' in matlab_left.dtype.names:
                fig = plot_comparison(left_walk_info.P, matlab_left['P'][0, 0],
                                    'left_walk_info.P', ylabel='Position [m]')
                fig.suptitle('Left Foot Position Comparison', fontsize=14, y=1.00)

            if hasattr(left_walk_info, 'V') and 'V' in matlab_left.dtype.names:
                fig = plot_comparison(left_walk_info.V, matlab_left['V'][0, 0],
                                    'left_walk_info.V', ylabel='Velocity [m/s]')
                fig.suptitle('Left Foot Velocity Comparison', fontsize=14, y=1.00)

        # Right foot comparisons
        if 'right_walk_info' in matlab_data:
            matlab_right = matlab_data['right_walk_info']
            if hasattr(right_walk_info, 'P') and 'P' in matlab_right.dtype.names:
                fig = plot_comparison(right_walk_info.P, matlab_right['P'][0, 0],
                                    'right_walk_info.P', ylabel='Position [m]')
                fig.suptitle('Right Foot Position Comparison', fontsize=14, y=1.00)

            if hasattr(right_walk_info, 'V') and 'V' in matlab_right.dtype.names:
                fig = plot_comparison(right_walk_info.V, matlab_right['V'][0, 0],
                                    'right_walk_info.V', ylabel='Velocity [m/s]')
                fig.suptitle('Right Foot Velocity Comparison', fontsize=14, y=1.00)

    # Plot results for left and right foot
    print("\n=== Plotting stride results ===")

    # Create subplots for lateral vs forward strides
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    imu.plt_ltrl_frwd_strides(left_strides, show=False)
    axes[0].set_title('Left Foot - Lateral vs Forward')

    plt.sca(axes[1])
    imu.plt_ltrl_frwd_strides(right_strides, show=False)
    axes[1].set_title('Right Foot - Lateral vs Forward')
    plt.tight_layout()

    # Create subplots for forward vs elevation strides
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.sca(axes[0])
    imu.plt_frwd_elev_strides(left_strides, show=False)
    axes[0].set_title('Left Foot - Forward vs Elevation')

    plt.sca(axes[1])
    imu.plt_frwd_elev_strides(right_strides, show=False)
    axes[1].set_title('Right Foot - Forward vs Elevation')
    plt.tight_layout()

    # Create subplots for stride variability
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    imu.plt_stride_var(left_strides, show=False)
    axes[0].set_title('Left Foot - Stride Variability')

    plt.sca(axes[1])
    imu.plt_stride_var(right_strides, show=False)
    axes[1].set_title('Right Foot - Stride Variability')
    plt.tight_layout()

    plt.show()
