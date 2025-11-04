import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.stride.apdm import getdata_apdm, getdata
from src.stride.inertial import compute_position, stride_segmentation
from src.stride.plotting import plt_ltrl_frwd_strides, plt_frwd_elev_strides, plt_stride_var,plt_walk_info_position

# Path to the APDM .h5 file (update if needed)
H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')
H5_FILE = os.path.join('/Users/jeremy/Downloads/chanelsandbox/Pilot_Ch_Oct29','20251029-154305_LF_Pilot_Ch_Oct29.h5')
if __name__ == '__main__':
    # Load IMU information from a file
    Wb, Ab, PERIOD, _ = getdata_apdm(H5_FILE)

    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(2200, 3500)]
    Wb, Ab, _, _ = getdata(Wb, Ab, PERIOD, SECTION)

    # Perform inertial mechanization
    walk_info = compute_position(Wb, Ab, PERIOD)

    # Segment strides only for the walking period
    strides = stride_segmentation(walk_info, PERIOD)

    # Plot results
    plt_ltrl_frwd_strides(strides)
    plt_frwd_elev_strides(strides)
    plt_stride_var(strides) 
    plt_walk_info_position(walk_info)