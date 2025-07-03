import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from stride.apdm import getdata_apdm, getdata
from stride.inertial import compute_pos, stride_segmentation
from stride.plotting import plt_ltrl_frwd_strides, plt_frwd_elev_strides, plt_stride_var

# Path to the APDM .h5 file (update if needed)
H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')

if __name__ == '__main__':
    print("=== Loading IMU data from h5 file ===")
    
    # Load IMU information from a file
    Wb, Ab, PERIOD, _ = getdata_apdm(H5_FILE)
    
    print(f"Data loaded successfully:")
    print(f"  Angular velocity shape: {Wb.shape}")
    print(f"  Acceleration shape: {Ab.shape}")
    print(f"  Sample period: {PERIOD}")
    print(f"  Sample rate: {1/PERIOD} Hz")
    
    # Calculate total duration
    total_duration = Wb.shape[0] * PERIOD
    print(f"  Total duration: {total_duration:.1f} seconds")
    
    # Check for any NaN or zero values
    print(f"  Angular velocity range: [{np.min(Wb):.3f}, {np.max(Wb):.3f}]")
    print(f"  Acceleration range: [{np.min(Ab):.3f}, {np.max(Ab):.3f}]")
    print(f"  Any NaN in Wb: {np.any(np.isnan(Wb))}")
    print(f"  Any NaN in Ab: {np.any(np.isnan(Ab))}")
    
    # Plot raw data first to see what we have
    print("\n=== Plotting raw data ===")
    t = np.arange(Wb.shape[0]) * PERIOD
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot angular velocity
    axes[0].plot(t, Wb[:, 0], 'r-', label='Wx', alpha=0.7)
    axes[0].plot(t, Wb[:, 1], 'g-', label='Wy', alpha=0.7)
    axes[0].plot(t, Wb[:, 2], 'b-', label='Wz', alpha=0.7)
    axes[0].set_ylabel('Angular velocity [rad/s]')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Raw Angular Velocity Data')
    
    # Plot acceleration
    axes[1].plot(t, Ab[:, 0], 'r-', label='Ax', alpha=0.7)
    axes[1].plot(t, Ab[:, 1], 'g-', label='Ay', alpha=0.7)
    axes[1].plot(t, Ab[:, 2], 'b-', label='Az', alpha=0.7)
    axes[1].set_ylabel('Acceleration [m/s²]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Raw Acceleration Data')
    
    plt.tight_layout()
    plt.show()
    
    # Plot signals to help determining the SECTION that corresponds to the experiment
    print("\n=== Running getdata() to detect quiet periods ===")
    getdata(Wb, Ab, PERIOD)
    
    print("\n=== Processing section 240-350s ===")
    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(240, 350)]
    Wb_section, Ab_section, _, _ = getdata(Wb, Ab, PERIOD, SECTION)
    
    print(f"Section data:")
    print(f"  Angular velocity shape: {Wb_section.shape}")
    print(f"  Acceleration shape: {Ab_section.shape}")
    print(f"  Section duration: {Wb_section.shape[0] * PERIOD:.1f} seconds")
    
    # Perform inertial mechanization
    print("\n=== Computing position ===")
    walk_info = compute_pos(Wb_section, Ab_section, PERIOD)
    
    print(f"Walk info computed with {len(walk_info)} data points")
    
    # Segment strides only for the walking period
    print("\n=== Segmenting strides ===")
    strides = stride_segmentation(walk_info, PERIOD)
    
    print(f"Found {len(strides)} strides")
    
    # Plot results
    print("\n=== Plotting results ===")
    plt_ltrl_frwd_strides(strides)
    plt_frwd_elev_strides(strides)
    plt_stride_var(strides)