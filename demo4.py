import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from stride.apdm import getdata_apdm, getdata
from stride.inertial import compute_pos, stride_segmentation
from stride.plotting import plt_ltrl_frwd_strides, plt_frwd_elev_strides, plt_stride_var
from stride.compute_pos import ComputePos

# Path to the APDM .h5 file (update if needed)
H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')

if __name__ == '__main__':
    # Load IMU information from a file
    Wb, Ab, PERIOD, _ = getdata_apdm(H5_FILE)

    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(240, 350)]
    Wb, Ab, _, _ = getdata(Wb, Ab, PERIOD, SECTION)

    # Perform inertial mechanization
    processor = ComputePos()
    walk_info = processor.compute_pos(Wb, Ab, PERIOD)

    # Segment strides only for the walking period
    strides = stride_segmentation(walk_info, PERIOD)

    # Plot results
    plt_ltrl_frwd_strides(strides)
    plt_frwd_elev_strides(strides)
    plt_stride_var(strides) 