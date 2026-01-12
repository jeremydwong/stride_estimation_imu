import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu

# Path to the APDM .h5 files (update if needed)
LEFT_H5_FILE = os.path.join('matlab', '20120418-132855_sensor_data_monitor_472_label_Left.h5')
RIGHT_H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')
LEFT_H5_FILE = os.path.join('matlab', '20251008-153007_Pilot_Ch_LF_13087.h5')
RIGHT_H5_FILE = os.path.join('matlab', '20251008-153009_Pilot_Ch_RF_13097.h5')

RIGHT_H5_FILE = os.path.join('matlab','20251029-154310_RF_Pilot_Ch_Oct29.h5')
LEFT_H5_FILE = os.path.join('matlab','20251029-154305_LF_Pilot_Ch_Oct29.h5')
if __name__ == '__main__':
    # Load IMU information from two files and sync them
    (left_Wb, left_Ab, right_Wb, right_Ab, PERIOD, _, _, _,
     left_time_datetime, left_time_elapsed_samples,
     right_time_datetime, right_time_elapsed_samples) = \
        imu.sync_apdm(LEFT_H5_FILE, RIGHT_H5_FILE)

    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(240, 350)]
    left_Wb, left_Ab, _, _, left_time_datetime, left_time_elapsed_samples = imu.getdata(
        left_Wb, left_Ab, PERIOD, SECTION, time_datetime=left_time_datetime, time_elapsed_samples=left_time_elapsed_samples)
    right_Wb, right_Ab, _, _, right_time_datetime, right_time_elapsed_samples = imu.getdata(
        right_Wb, right_Ab, PERIOD, SECTION, time_datetime=right_time_datetime, time_elapsed_samples=right_time_elapsed_samples)

    # Process data from the two IMUs simultaneously
    left_walk_info, right_walk_info = imu.compute_position_two_imus(left_Wb, left_Ab, right_Wb, right_Ab, PERIOD)

    # Segment steps
    left_strides = imu.stride_segmentation(left_walk_info, PERIOD)
    right_strides = imu.stride_segmentation(right_walk_info, PERIOD)

    # Plot results for left and right foot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    plt_ltrl_frwd_strides(left_strides, show=False)
    axes[0].set_title('Left Foot')
    plt.sca(axes[1])
    plt_ltrl_frwd_strides(right_strides, show=False)
    axes[1].set_title('Right Foot')
    plt.tight_layout()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.sca(axes[0])
    plt_frwd_elev_strides(left_strides, show=False)
    axes[0].set_title('Left Foot')
    plt.sca(axes[1])
    plt_frwd_elev_strides(right_strides, show=False)
    axes[1].set_title('Right Foot')
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    plt_stride_var(left_strides, show=False)
    axes[0].set_title('Left Foot')
    plt.sca(axes[1])
    plt_stride_var(right_strides, show=False)
    axes[1].set_title('Right Foot')
    plt.tight_layout()

    plt.show()
