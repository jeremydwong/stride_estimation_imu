import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu
# Path to the APDM .h5 file (update if needed)
# H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')
H5_FILE = os.path.join('/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 CC Balance/Pilot testing/2025-10-29/20251029-154305_LF_Pilot_Ch_Oct29.h5')
H5_FILE = os.path.join('/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 CC Balance/Pilot testing/January 5th Pilot Testing/20260105-141714_LeftFoot_013087.h5')
H5_FILE = os.path.join('/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 CC Balance/Pilot testing/January 5th Pilot Testing/20260105-141717_RightFoot_013097.h5')
if __name__ == '__main__':
    # Load IMU information from a file
    Wb, Ab, PERIOD, _, time_datetime, time_elapsed_samples, _ = imu.getdata_apdm(H5_FILE)

    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    
    # Perform inertial mechanization
    walk_info = imu.compute_position(Wb, Ab, PERIOD)

    # Segment strides only for the walking period
    
    imu.plt_walk_info_position(walk_info)
    # # Plot results
    imu.plt_ltrl_frwd_strides(walk_info)
    # imu.plt_frwd_elev_strides(walk_info)
    # imu.plt_stride_var(walk_info) 
    
    # Example: compute date difference between two datetime
    
