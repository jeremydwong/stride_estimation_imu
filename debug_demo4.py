import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.stride.apdm import getdata_apdm, getdata
from src.stride.inertial import compute_position, stride_segmentation
from src.stride.plotting import plt_ltrl_frwd_strides, plt_frwd_elev_strides, plt_stride_var

# Path to the APDM .h5 file (update if needed)
H5_FILE = os.path.join('matlab', '20120418-132857_sensor_data_monitor_403_label_Right.h5')
MATLAB_FILE = 'matlab_demo4.mat'

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
    
    # Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
    SECTION = [(240, 350)]
    Wb_section, Ab_section, _, _ = getdata(Wb, Ab, PERIOD, SECTION)

    # Perform inertial mechanization
    print("\n=== Computing position ===")
    walk_info = compute_position(Wb_section, Ab_section, PERIOD)

    # Segment strides only for the walking period
    print("\n=== Segmenting strides ===")
    strides = stride_segmentation(walk_info, PERIOD)

    print(f"Found {len(strides)} strides")

    print(f"Section data:")
    print(f"  Angular velocity shape: {Wb_section.shape}")
    print(f"  Acceleration shape: {Ab_section.shape}")
    print(f"  Section duration: {Wb_section.shape[0] * PERIOD:.1f} seconds")

    # Load MATLAB reference data and compare section data
    print("\n" + "="*60)
    print("LOADING MATLAB REFERENCE DATA FOR COMPARISON")
    print("="*60)
    matlab_data = scipy.io.loadmat(MATLAB_FILE)
    print(f"Loaded MATLAB data with keys: {[k for k in matlab_data.keys() if not k.startswith('__')]}")

    # Compare Wb_section and Ab_section
    print("\n" + "="*60)
    print("COMPARING SECTIONED IMU DATA")
    print("="*60)
    compare_arrays(Wb_section, matlab_data['Wb_section'], 'Wb_section')
    compare_arrays(Ab_section, matlab_data['Ab_section'], 'Ab_section')

    # Plot comparison of sectioned data
    plot_comparison(Wb_section, matlab_data['Wb_section'], 'Wb_section', ylabel='Angular velocity [rad/s]')
    plot_comparison(Ab_section, matlab_data['Ab_section'], 'Ab_section', ylabel='Acceleration [m/s²]')
    

    print(f"Walk info computed with {len(walk_info)} data points")

    # Compare walk_info
    matlab_walk_info = matlab_data['walk_info']
    compare_structs(walk_info, matlab_walk_info, 'walk_info', rtol=1e-5, atol=1e-8)

    # Compare strides - note that strides is a dictionary with arrays
    matlab_strides = matlab_data['strides']
    print(f"\n{'='*60}")
    print(f"Comparing strides")
    print(f"{'='*60}")

    # Get number of strides from the first field
    if matlab_strides.shape[1] > 0:
        # MATLAB strides is a 1xN array of structs
        matlab_first_stride = matlab_strides[0, 0]
        matlab_fields = matlab_first_stride.dtype.names

        # Python strides is a dictionary with arrays
        python_fields = set(strides.keys())
        matlab_field_set = set(matlab_fields)

        print(f"Python fields: {sorted(python_fields)}")
        print(f"MATLAB fields: {sorted(matlab_field_set)}")

        common_fields = python_fields & matlab_field_set
        python_only = python_fields - matlab_field_set
        matlab_only = matlab_field_set - python_fields

        if python_only:
            print(f"\nFields only in Python: {python_only}")
        if matlab_only:
            print(f"Fields only in MATLAB: {matlab_only}")

        # Compare common fields
        for field in sorted(common_fields):
            python_val = strides[field]

            # Collect MATLAB values for this field from all strides
            matlab_vals = []
            for i in range(matlab_strides.shape[1]):
                stride = matlab_strides[0, i]
                val = stride[field]
                # Handle nested arrays
                if val.ndim > 0:
                    val = val.squeeze()
                matlab_vals.append(val)

            # Stack into array if possible
            try:
                if len(matlab_vals) > 0 and hasattr(matlab_vals[0], 'shape'):
                    if matlab_vals[0].ndim == 1:
                        matlab_val = np.vstack(matlab_vals)
                    else:
                        matlab_val = matlab_vals[0]
                else:
                    matlab_val = matlab_vals[0]

                print(f"\n--- strides.{field} ---")
                print(f"  Python shape: {python_val.shape}")
                print(f"  MATLAB shape: {matlab_val.shape}")

                if python_val.shape == matlab_val.shape:
                    diff = np.max(np.abs(python_val - matlab_val))
                    mean_diff = np.mean(np.abs(python_val - matlab_val))
                    match = "✓" if np.allclose(python_val, matlab_val, rtol=1e-5, atol=1e-8) else "✗"
                    print(f"  Max absolute difference: {diff:.6e}")
                    print(f"  Mean absolute difference: {mean_diff:.6e}")
                    print(f"  {match}")
                else:
                    print(f"  WARNING: Shape mismatch!")
            except Exception as e:
                print(f"\n--- strides.{field} ---")
                print(f"  Error comparing: {e}")
    
    # Create additional comparison plots for walk_info fields
    print("\n=== Creating walk_info comparison plots ===")

    # Plot quaternion comparison
    if 'quaternion' in walk_info and 'quaternion' in matlab_walk_info.dtype.names:
        fig = plot_comparison(walk_info['quaternion'], matlab_walk_info['quaternion'][0, 0],
                            'walk_info.quaternion', ylabel='Quaternion')
        fig.suptitle('Quaternion Comparison', fontsize=14, y=1.00)

    # Plot euler angles comparison
    if 'euler' in walk_info and 'euler' in matlab_walk_info.dtype.names:
        fig = plot_comparison(walk_info['euler'], matlab_walk_info['euler'][0, 0],
                            'walk_info.euler', ylabel='Euler angles [rad]')
        fig.suptitle('Euler Angles Comparison', fontsize=14, y=1.00)

    # Plot velocity comparison
    if 'V' in walk_info and 'V' in matlab_walk_info.dtype.names:
        fig = plot_comparison(walk_info['V'], matlab_walk_info['V'][0, 0],
                            'walk_info.V', ylabel='Velocity [m/s]')
        fig.suptitle('Velocity Comparison', fontsize=14, y=1.00)

    # Plot position comparison
    if 'P' in walk_info and 'P' in matlab_walk_info.dtype.names:
        fig = plot_comparison(walk_info['P'], matlab_walk_info['P'][0, 0],
                            'walk_info.P', ylabel='Position [m]')
        fig.suptitle('Position Comparison', fontsize=14, y=1.00)

    # Plot acceleration comparison
    if 'An' in walk_info and 'An' in matlab_walk_info.dtype.names:
        fig = plot_comparison(walk_info['An'], matlab_walk_info['An'][0, 0],
                            'walk_info.An', ylabel='Acceleration (nav) [m/s²]')
        fig.suptitle('Navigation Frame Acceleration Comparison', fontsize=14, y=1.00)

    # Create stride field comparison plots
    print("\n=== Creating strides comparison plots ===")

    # For strides, we need to handle the shape differences
    for field in ['frwd', 'ltrl', 'elev']:
        if field in strides:
            python_val = strides[field]

            # Collect MATLAB values
            matlab_vals = []
            for i in range(matlab_strides.shape[1]):
                stride = matlab_strides[0, i]
                val = stride[field]
                if val.ndim > 0:
                    val = val.squeeze()
                matlab_vals.append(val)

            try:
                # Try to create a comparable view
                if len(matlab_vals) > 0:
                    # MATLAB appears to be (1, n_samples, n_strides)
                    # Python is (n_strides, n_samples)
                    # Need to reshape for comparison
                    matlab_val = np.array(matlab_vals)  # (n_strides, n_samples)

                    if matlab_val.ndim == 3:
                        matlab_val = matlab_val.squeeze(0)  # Try to match Python shape

                    print(f"\nPlotting {field}:")
                    print(f"  Python shape: {python_val.shape}")
                    print(f"  MATLAB shape: {matlab_val.shape}")

                    # Plot the first few strides
                    n_plot = min(3, python_val.shape[0], matlab_val.shape[0])
                    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 4*n_plot))
                    if n_plot == 1:
                        axes = axes.reshape(1, -1)

                    for i in range(n_plot):
                        # Python stride
                        if python_val.ndim == 2:
                            axes[i, 0].plot(python_val[:,i], 'b-', linewidth=2)
                        else:
                            axes[i, 0].plot(python_val, 'b-', linewidth=2)
                        axes[i, 0].set_ylabel(f'Stride {i+1}')
                        axes[i, 0].grid(True)
                        if i == 0:
                            axes[i, 0].set_title(f'Python - {field}')

                        # MATLAB stride
                        if matlab_val.ndim == 2:
                            axes[i, 1].plot(matlab_val[:,i], 'r-', linewidth=2)
                        else:
                            axes[i, 1].plot(matlab_val, 'r-', linewidth=2)
                        axes[i, 1].grid(True)
                        if i == 0:
                            axes[i, 1].set_title(f'MATLAB - {field}')

                    axes[-1, 0].set_xlabel('Sample')
                    axes[-1, 1].set_xlabel('Sample')
                    plt.tight_layout()
            except Exception as e:
                print(f"  Error plotting {field}: {e}")

    # Plot results
    print("\n=== Plotting original stride results ===")
    plt_ltrl_frwd_strides(strides)
    plt_frwd_elev_strides(strides)
    plt_stride_var(strides)

    plt.show()