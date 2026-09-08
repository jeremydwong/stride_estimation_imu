import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import stride_imu as imu
# Path to the APDM .h5 file (update if needed)
# H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')
# H5_FILE = '/Users/jeremy/Downloads/January 5th Pilot Testing/20260105-141714_LeftFoot_013087.h5'
H5_FILE = '/Users/jeremy/Downloads/January 5th Pilot Testing/20260105-141717_RightFoot_013097.h5'
if __name__ == '__main__':
    # Load IMU information from a file
    recording = imu.load_imu_recording(H5_FILE)

    # Perform inertial mechanization
    walk_info = imu.compute_position(recording.Wb, recording.Ab, recording.period)

    # Segment strides for the walking period
    strides = imu.stride_segmentation(walk_info, recording.period)

    # Plot results
    imu.plt_walk_info_position(walk_info)
    imu.plt_ltrl_frwd_strides(strides)
