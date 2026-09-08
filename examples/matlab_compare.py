"""Shared helpers for comparing Python pipeline output against MATLAB reference .mat files.

Used by demo_matlab_check_one_foot.py and demo_matlab_check_two_feet.py.
"""
import numpy as np
import matplotlib.pyplot as plt


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
