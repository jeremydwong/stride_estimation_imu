"""
Demo: Two feet + head IMU analysis with walking bout extraction.

This script:
1. Loads and synchronizes 4 IMUs (left foot, right foot, head, hand)
2. Detects walking bouts by finding quiet periods that bound active walking
3. Shows full trajectory with bout locations highlighted
4. Interactive bout selection with accelerometer visualization
5. Computes stride speed (distance / duration) for each bout
6. Analyzes head motion during walking bouts
7. Caches results to pkl file
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time as dt_time, datetime
from typing import Optional, List, Dict, Any
import pickle
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu
from stride_imu.apdm import (
    load_imu_recording,
    find_overlapping_recordings,
    ImuRecording
)
from stride_imu.inertial import (
    detect_walking_bouts,
    find_bouts_near_time,
    WalkingBout
)


class BoutSelection:
    """Stores bout selection with precise timing info for caching."""
    def __init__(self, name: str, bout: WalkingBout, start_datetime: datetime,
                 end_datetime: datetime, confirmed: bool = False):
        self.name = name
        self.bout = bout
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.start_idx = bout.start_idx
        self.end_idx = bout.end_idx
        self.duration_seconds = bout.duration_seconds
        self.confirmed = confirmed

    def __repr__(self):
        return (f"BoutSelection('{self.name}', "
                f"start={self.start_datetime.strftime('%H:%M:%S')}, "
                f"duration={self.duration_seconds:.1f}s, "
                f"confirmed={self.confirmed})")


def compute_alignment_rotation(P: np.ndarray) -> np.ndarray:
    """
    Compute a 2D rotation matrix that aligns a straight-walk trajectory with the +Y axis.

    Assumes the trajectory is from a straight walk. The start-to-end displacement
    vector is used to determine the walking direction, which is then rotated to
    align with +Y (forward). After rotation, the trajectory should have:
    - Main motion along +Y (forward direction)
    - Minimal motion in X (lateral sway only)

    Parameters:
    -----------
    P : np.ndarray
        Position trajectory (Nx3 array) from a straight walk

    Returns:
    --------
    np.ndarray
        3x3 rotation matrix (rotates around Z axis to align walk with +Y)
    """
    # Vector from start to end (in XY plane) - this is the walking direction
    direction = P[-1, :2] - P[0, :2]

    # Angle of walking direction from +X axis (standard atan2 convention)
    theta = np.arctan2(direction[1], direction[0])

    # We want to rotate so that this direction becomes +Y (which is at angle pi/2)
    # So we need to rotate by (pi/2 - theta)
    rotation_angle = np.pi / 2 - theta

    # 2D rotation matrix around Z axis
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    return R


def apply_rotation_and_offset(P: np.ndarray, R: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Apply rotation matrix and offset to trajectory.

    Parameters:
    -----------
    P : np.ndarray
        Position trajectory (Nx3 array)
    R : np.ndarray
        3x3 rotation matrix
    offset : np.ndarray
        3-element offset vector

    Returns:
    --------
    np.ndarray
        Transformed trajectory
    """
    # Center at origin, rotate, then add offset
    P_centered = P - P[0, :]  # Start at origin
    P_rotated = (R @ P_centered.T).T
    P_offset = P_rotated + offset
    return P_offset


def compute_total_distance(P: np.ndarray) -> float:
    """
    Compute total distance travelled along a trajectory (path length).

    Parameters:
    -----------
    P : np.ndarray
        Position trajectory (Nx3 array)

    Returns:
    --------
    float
        Total path length in meters
    """
    # Sum of Euclidean distances between consecutive points
    diffs = np.diff(P, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.sum(distances))


def plot_rotation_corrected_trajectories(left_walk_info: dict, right_walk_info: dict,
                                          title: str = "",
                                          dx_offset: float = 0.3):
    """
    Plot left and right foot trajectories with rotation correction for straight walks.

    Assumes both trajectories are from straight walks. Each foot's trajectory is
    independently rotated so its start-to-end displacement aligns with +Y (forward).
    After rotation:
    - Main motion should be along +Y
    - X motion should be minimal (just lateral gait sway)

    A small X offset separates left and right feet for visualization.

    Parameters:
    -----------
    left_walk_info : dict
        Output from compute_position for left foot (straight walk)
    right_walk_info : dict
        Output from compute_position for right foot (straight walk)
    title : str
        Plot title
    dx_offset : float
        X offset between left and right trajectories (default 0.3m)
    """
    P_left = left_walk_info.P
    P_right = right_walk_info.P

    # Compute rotation matrices for each foot
    R_left = compute_alignment_rotation(P_left)
    R_right = compute_alignment_rotation(P_right)

    # Apply rotations with offsets (left foot at -dx/2, right at +dx/2)
    P_left_rot = apply_rotation_and_offset(P_left, R_left, np.array([-dx_offset/2, 0, 0]))
    P_right_rot = apply_rotation_and_offset(P_right, R_right, np.array([dx_offset/2, 0, 0]))

    # Create figure with 2D and 3D views
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'Rotation Corrected Trajectories - {title}', fontsize=12)

    # 2D XY view
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(P_left_rot[:, 0], P_left_rot[:, 1], 'b-', linewidth=1, label='Left Foot', alpha=0.8)
    ax1.plot(P_right_rot[:, 0], P_right_rot[:, 1], 'r-', linewidth=1, label='Right Foot', alpha=0.8)

    # Mark start and end points
    ax1.plot(P_left_rot[0, 0], P_left_rot[0, 1], 'bo', markersize=8, label='Start')
    ax1.plot(P_left_rot[-1, 0], P_left_rot[-1, 1], 'b^', markersize=8, label='End')
    ax1.plot(P_right_rot[0, 0], P_right_rot[0, 1], 'ro', markersize=8)
    ax1.plot(P_right_rot[-1, 0], P_right_rot[-1, 1], 'r^', markersize=8)

    ax1.set_xlabel('X (lateral) [m]')
    ax1.set_ylabel('Y (forward) [m]')
    ax1.set_title('XY View (aligned to forward direction)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 3D view
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(P_left_rot[:, 0], P_left_rot[:, 1], P_left_rot[:, 2],
             'b-', linewidth=1, label='Left Foot', alpha=0.8)
    ax2.plot(P_right_rot[:, 0], P_right_rot[:, 1], P_right_rot[:, 2],
             'r-', linewidth=1, label='Right Foot', alpha=0.8)

    # Mark start and end
    ax2.scatter(*P_left_rot[0, :], c='b', s=50, marker='o')
    ax2.scatter(*P_left_rot[-1, :], c='b', s=50, marker='^')
    ax2.scatter(*P_right_rot[0, :], c='r', s=50, marker='o')
    ax2.scatter(*P_right_rot[-1, :], c='r', s=50, marker='^')

    ax2.set_xlabel('X (lateral) [m]')
    ax2.set_ylabel('Y (forward) [m]')
    ax2.set_zlabel('Z (vertical) [m]')  # type: ignore[attr-defined]
    ax2.set_title('3D View')
    ax2.legend(loc='best', fontsize=8)

    plt.tight_layout()
    return fig


def compute_stride_metrics(strides: dict) -> dict:
    """Compute stride-level metrics from stride segmentation output.

    Handles empty strides (no strides detected) by returning zeros/empty arrays.
    """
    stride_speeds = strides.frwd_speed
    stride_durations = strides.time

    # Handle empty strides (no strides detected)
    if len(stride_speeds) == 0:
        return {
            'stride_lengths': np.array([]),
            'stride_durations': np.array([]),
            'stride_speeds': np.array([]),
            'mean_speed': 0.0,
            'std_speed': 0.0,
            'mean_length': 0.0,
            'std_length': 0.0,
            'mean_duration': 0.0,
            'std_duration': 0.0,
            'n_strides': 0
        }

    stride_lengths = strides.frwd[-1, :]

    return {
        'stride_lengths': stride_lengths,
        'stride_durations': stride_durations,
        'stride_speeds': stride_speeds,
        'mean_speed': np.mean(stride_speeds),
        'std_speed': np.std(stride_speeds),
        'mean_length': np.mean(stride_lengths),
        'std_length': np.std(stride_lengths),
        'mean_duration': np.mean(stride_durations),
        'std_duration': np.std(stride_durations),
        'n_strides': len(stride_speeds)
    }


def analyze_head_motion(head_recording: ImuRecording, period: float) -> dict:
    """Analyze head IMU motion characteristics."""
    Wm = np.sqrt(np.sum(head_recording.Wb ** 2, axis=1)) / period * 180 / np.pi
    Am = np.sqrt(np.sum(head_recording.Ab ** 2, axis=1))

    return {
        'ang_vel_mean': np.mean(Wm),
        'ang_vel_std': np.std(Wm),
        'ang_vel_max': np.max(Wm),
        'accel_mean': np.mean(Am),
        'accel_std': np.std(Am),
        'accel_max': np.max(Am),
        'Wm': Wm,
        'Am': Am
    }


def plot_full_trajectory_with_bouts(walk_info: dict, period: float,
                                     bout_selections: list,
                                     title: str = "Full Trajectory"):
    """
    Plot full compute_pos trajectory with bout sections highlighted in red.
    """
    P = walk_info.P
    t = np.arange(len(P)) * period

    # Create figure with trajectory and time series
    fig = plt.figure(figsize=(16, 10))

    # 2D trajectory (X-Y)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(P[:, 0], P[:, 1], 'b-', alpha=0.5, linewidth=0.5, label='Full trajectory')
    for sel in bout_selections:
        bout_P = P[sel.start_idx:sel.end_idx, :]
        ax1.plot(bout_P[:, 0], bout_P[:, 1], 'r-', linewidth=2, label=sel.name)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'{title} - XY Trajectory')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True)
    ax1.axis('equal')

    # 3D trajectory
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(P[:, 0], P[:, 1], P[:, 2], 'b-', alpha=0.3, linewidth=0.5)
    for sel in bout_selections:
        bout_P = P[sel.start_idx:sel.end_idx, :]
        ax2.plot(bout_P[:, 0], bout_P[:, 1], bout_P[:, 2], 'r-', linewidth=2)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Z [m]')
    ax2.set_title('3D Trajectory')

    # Time vs Position X, Y, Z
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(t, P[:, 0], 'b-', alpha=0.5, linewidth=0.5, label='X')
    ax3.plot(t, P[:, 1], 'g-', alpha=0.5, linewidth=0.5, label='Y')
    ax3.plot(t, P[:, 2], 'c-', alpha=0.5, linewidth=0.5, label='Z')

    # Overlay bout sections in red
    for sel in bout_selections:
        bout_t = t[sel.start_idx:sel.end_idx]
        bout_P = P[sel.start_idx:sel.end_idx, :]
        ax3.plot(bout_t, bout_P[:, 0], 'r-', linewidth=2)
        ax3.plot(bout_t, bout_P[:, 1], 'r-', linewidth=2)
        ax3.plot(bout_t, bout_P[:, 2], 'r-', linewidth=2)
        # Add vertical lines at bout boundaries
        ax3.axvline(bout_t[0], color='r', linestyle='--', alpha=0.5)
        ax3.axvline(bout_t[-1], color='r', linestyle='--', alpha=0.5)
        # Label
        mid_t = (bout_t[0] + bout_t[-1]) / 2
        ax3.text(mid_t, ax3.get_ylim()[1], sel.name, ha='center', va='bottom',
                 fontsize=8, color='red', rotation=45)

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Position [m]')
    ax3.set_title('Position vs Time (bouts in red)')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    plt.tight_layout()
    return fig


def plot_accelerometer_for_bout_selection(recordings: dict, period: float,
                                           bout: WalkingBout,
                                           time_datetime: np.ndarray,
                                           bout_name: str,
                                           context_seconds: float = 10.0):
    """
    Plot accelerometer norm from all sensors around a bout for interactive selection.
    Shows context before and after the bout.
    """
    context_samples = int(context_seconds / period)

    # Expand window with context
    start_with_context = max(0, bout.quiet_before_idx - context_samples)
    end_with_context = min(len(time_datetime), bout.quiet_after_idx + context_samples)

    t = np.arange(end_with_context - start_with_context) * period

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Accelerometer Signals for: {bout_name}\n'
                 f'Bout: {time_datetime[bout.start_idx].strftime("%H:%M:%S")} - '
                 f'{time_datetime[bout.end_idx-1].strftime("%H:%M:%S")} '
                 f'({bout.duration_seconds:.1f}s)', fontsize=12)

    sensor_names = ['Left Foot', 'Right Foot', 'Head', 'Hand']
    sensor_keys = ['left', 'right', 'head', 'hand']

    for ax, name, key in zip(axes, sensor_names, sensor_keys):
        rec = recordings[key]
        Ab_slice = rec.Ab[start_with_context:end_with_context, :]
        Am = np.sqrt(np.sum(Ab_slice ** 2, axis=1))

        ax.plot(t, Am, 'b-', linewidth=0.5, alpha=0.7)

        # Highlight the detected bout region
        bout_start_rel = bout.start_idx - start_with_context
        bout_end_rel = bout.end_idx - start_with_context
        quiet_before_rel = bout.quiet_before_idx - start_with_context
        quiet_after_rel = bout.quiet_after_idx - start_with_context

        # Shade quiet periods in green
        ax.axvspan(t[max(0, quiet_before_rel)], t[bout_start_rel],
                   alpha=0.2, color='green', label='Quiet')
        ax.axvspan(t[bout_end_rel], t[min(len(t)-1, quiet_after_rel)],
                   alpha=0.2, color='green')

        # Shade walking bout in red
        ax.axvspan(t[bout_start_rel], t[bout_end_rel],
                   alpha=0.2, color='red', label='Walking')

        ax.set_ylabel(f'{name}\n|A| [m/s²]')
        ax.grid(True, alpha=0.3)
        ax.axhline(9.81, color='gray', linestyle='--', alpha=0.5, label='Gravity')

        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time [s] (relative to context window)')

    plt.tight_layout()
    return fig


def interactive_bout_selection(recordings: dict, all_bouts: list,
                                time_datetime: np.ndarray, period: float,
                                bout_name: str, target_time: dt_time,
                                search_window: float) -> Optional[BoutSelection]:
    """
    Interactive selection of a walking bout with accelerometer visualization.
    Returns a BoutSelection with confirmed=True if user accepts.
    """
    # Find bouts near target time
    matching_bouts = find_bouts_near_time(all_bouts, time_datetime, target_time, search_window)

    if not matching_bouts:
        print(f"  No walking bouts found within {search_window}s of target time")
        return None

    print(f"\n  Found {len(matching_bouts)} bout(s) near {target_time.strftime('%H:%M:%S')}:")
    for i, bout in enumerate(matching_bouts):
        bout_start_time = time_datetime[bout.start_idx]
        print(f"    [{i}] {bout_start_time.strftime('%H:%M:%S')} - {bout.duration_seconds:.1f}s")

    # Default: longest bout
    default_idx = max(range(len(matching_bouts)),
                      key=lambda i: matching_bouts[i].duration_seconds)
    selected_bout = matching_bouts[default_idx]

    # Show accelerometer plot for selection
    plot_accelerometer_for_bout_selection(
        recordings, period, selected_bout, time_datetime, bout_name
    )
    plt.show(block=False)
    plt.pause(0.1)

    # Interactive selection
    while True:
        print(f"\n  Current selection: [{default_idx}] "
              f"{time_datetime[selected_bout.start_idx].strftime('%H:%M:%S')} "
              f"({selected_bout.duration_seconds:.1f}s)")
        print("  Options:")
        print("    [Enter] Accept current selection")
        print("    [0-N]   Select different bout by number")
        print("    [s]     Skip this bout entirely")
        print("    [m]     Manually enter start/end indices")

        user_input = input("  Choice: ").strip().lower()

        if user_input == '':
            # Accept current selection
            start_dt = time_datetime[selected_bout.start_idx]
            end_dt = time_datetime[selected_bout.end_idx - 1]
            return BoutSelection(bout_name, selected_bout, start_dt, end_dt, confirmed=True)

        elif user_input == 's':
            print("  Skipping this bout")
            return None

        elif user_input == 'm':
            # Manual entry
            try:
                start_str = input("    Enter start time (HH:MM:SS): ").strip()
                end_str = input("    Enter end time (HH:MM:SS): ").strip()
                start_parts = [int(x) for x in start_str.split(':')]
                end_parts = [int(x) for x in end_str.split(':')]
                manual_start = dt_time(*start_parts)
                manual_end = dt_time(*end_parts)

                # Find indices
                start_secs = manual_start.hour * 3600 + manual_start.minute * 60 + manual_start.second
                end_secs = manual_end.hour * 3600 + manual_end.minute * 60 + manual_end.second

                start_idx = None
                end_idx = None
                for i, dt in enumerate(time_datetime):
                    t_secs = dt.hour * 3600 + dt.minute * 60 + dt.second
                    if start_idx is None and t_secs >= start_secs:
                        start_idx = i
                    if t_secs <= end_secs:
                        end_idx = i

                if start_idx is not None and end_idx is not None and end_idx > start_idx:
                    duration = (end_idx - start_idx) * period
                    manual_bout = WalkingBout(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        duration_seconds=duration,
                        quiet_before_idx=max(0, start_idx - int(2/period)),
                        quiet_after_idx=min(len(time_datetime), end_idx + int(2/period))
                    )
                    return BoutSelection(bout_name, manual_bout,
                                         time_datetime[start_idx],
                                         time_datetime[end_idx],
                                         confirmed=True)
                else:
                    print("    Invalid time range")
            except Exception as e:
                print(f"    Error parsing time: {e}")

        else:
            # Try to parse as number
            try:
                idx = int(user_input)
                if 0 <= idx < len(matching_bouts):
                    selected_bout = matching_bouts[idx]
                    default_idx = idx
                    plt.close('all')
                    plot_accelerometer_for_bout_selection(
                        recordings, period, selected_bout, time_datetime, bout_name
                    )
                    plt.show(block=False)
                    plt.pause(0.1)
                else:
                    print(f"    Invalid index. Must be 0-{len(matching_bouts)-1}")
            except ValueError:
                print("    Invalid input")


def process_walking_bout(bout: WalkingBout,
                         left_rec: ImuRecording,
                         right_rec: ImuRecording,
                         head_rec: ImuRecording,
                         period: float) -> dict:
    """Process a single walking bout: slice recordings, compute strides, analyze."""
    left_bout = left_rec[bout.start_idx:bout.end_idx]
    right_bout = right_rec[bout.start_idx:bout.end_idx]
    head_bout = head_rec[bout.start_idx:bout.end_idx]

    left_walk_info, right_walk_info = imu.compute_position_two_imus(
        left_bout.Wb, left_bout.Ab,
        right_bout.Wb, right_bout.Ab,
        period
    )

    left_strides = imu.stride_segmentation(left_walk_info, period)
    right_strides = imu.stride_segmentation(right_walk_info, period)

    left_metrics = compute_stride_metrics(left_strides)
    right_metrics = compute_stride_metrics(right_strides)
    head_metrics = analyze_head_motion(head_bout, period)

    # Compute total distance travelled for each foot
    left_total_distance = compute_total_distance(left_walk_info.P)
    right_total_distance = compute_total_distance(right_walk_info.P)

    return {
        'left_metrics': left_metrics,
        'right_metrics': right_metrics,
        'head_metrics': head_metrics,
        'left_strides': left_strides,
        'right_strides': right_strides,
        'left_walk_info': left_walk_info,
        'right_walk_info': right_walk_info,
        'left_bout': left_bout,
        'right_bout': right_bout,
        'head_bout': head_bout,
        'bout': bout,
        'left_total_distance': left_total_distance,
        'right_total_distance': right_total_distance
    }


def save_cache(cache_file: Path, data: dict):
    """Save analysis results to pickle file."""
    print(f"\nSaving cache to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def load_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    """Load analysis results from pickle file."""
    if cache_file.exists():
        print(f"Loading cache from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def plot_head_motion(head_recording: ImuRecording, period: float, title: str = ''):
    """Plot head motion during a walking bout."""
    head_metrics = analyze_head_motion(head_recording, period)
    t = np.arange(len(head_recording.Wb)) * period

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(t, head_recording.Wb / period * 180 / np.pi)
    axes[0].set_ylabel('Angular Vel [deg/s]')
    axes[0].set_title(f'Head Motion - {title}')
    axes[0].legend(['Roll', 'Pitch', 'Yaw'])
    axes[0].grid(True)

    axes[1].plot(t, head_recording.Ab)
    axes[1].set_ylabel('Acceleration [m/s²]')
    axes[1].legend(['X', 'Y', 'Z'])
    axes[1].grid(True)

    axes[2].plot(t, head_metrics['Wm'], label='Ang Vel Mag')
    axes[2].set_ylabel('Angular Vel [deg/s]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True)

    ax2 = axes[2].twinx()
    ax2.plot(t, head_metrics['Am'], 'r-', alpha=0.7, label='Accel Mag')
    ax2.set_ylabel('Acceleration [m/s²]', color='r')

    plt.tight_layout()
    return fig


def plot_bout_summary(bout_name: str, left_metrics: dict, right_metrics: dict,
                      head_metrics: dict):
    """Create a summary plot for a walking bout."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Walking Bout: {bout_name}', fontsize=14)

    ax = axes[0, 0]
    x = np.arange(2)
    means = [left_metrics['mean_speed'], right_metrics['mean_speed']]
    stds = [left_metrics['std_speed'], right_metrics['std_speed']]
    ax.bar(x, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Left Foot', 'Right Foot'])
    ax.set_ylabel('Step Speed [m/s]')
    ax.set_title('Mean Step Speed')
    ax.grid(True, axis='y')

    ax = axes[0, 1]
    means = [left_metrics['mean_length'], right_metrics['mean_length']]
    stds = [left_metrics['std_length'], right_metrics['std_length']]
    ax.bar(x, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Left Foot', 'Right Foot'])
    ax.set_ylabel('Step Length [m]')
    ax.set_title('Mean Step Length')
    ax.grid(True, axis='y')

    ax = axes[1, 0]
    # Compute shared bin edges for comparable histograms
    all_speeds = []
    if left_metrics['n_strides'] > 0:
        all_speeds.extend(left_metrics['stride_speeds'])
    if right_metrics['n_strides'] > 0:
        all_speeds.extend(right_metrics['stride_speeds'])
    if all_speeds:
        bins = np.linspace(min(all_speeds), max(all_speeds), 46)  # 45 bins = 46 edges
        if left_metrics['n_strides'] > 0:
            ax.hist(left_metrics['stride_speeds'], bins=bins, alpha=0.5, label='Left', color='blue')
        if right_metrics['n_strides'] > 0:
            ax.hist(right_metrics['stride_speeds'], bins=bins, alpha=0.5, label='Right', color='red')
    ax.set_xlabel('Step Speed [m/s]')
    ax.set_ylabel('Count')
    ax.set_title('Step Speed Distribution')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    metrics_text = (
        f"Head Motion Summary:\n"
        f"Angular Velocity:\n"
        f"  Mean: {head_metrics['ang_vel_mean']:.1f} deg/s\n"
        f"  Std: {head_metrics['ang_vel_std']:.1f} deg/s\n"
        f"  Max: {head_metrics['ang_vel_max']:.1f} deg/s\n\n"
        f"Acceleration:\n"
        f"  Mean: {head_metrics['accel_mean']:.2f} m/s²\n"
        f"  Std: {head_metrics['accel_std']:.2f} m/s²\n"
        f"  Max: {head_metrics['accel_max']:.2f} m/s²\n\n"
        f"Strides: L={left_metrics['n_strides']}, R={right_metrics['n_strides']}"
    )
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Head Motion & Step Counts')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("=" * 60)
    print("Loading and synchronizing all IMU recordings...")
    print("=" * 60)

    # =============================================================================
    # File paths
    # =============================================================================
    # Data directory: override with the STRIDE_DATA_DIR environment variable
    # (e.g. on Colab, point it at the uploaded/mounted copy of the recordings)
    DATA_DIR = os.environ.get('STRIDE_DATA_DIR',
                              '/Users/jeremy/Downloads/January 5th Pilot Testing')
    HEAD_FILE = os.path.join(DATA_DIR, '20260105-141716_Head_013120.h5')
    LEFT_FILE = os.path.join(DATA_DIR, '20260105-141714_LeftFoot_013087.h5')
    RIGHT_FILE = os.path.join(DATA_DIR, '20260105-141717_RightFoot_013097.h5')
    HAND_FILE = os.path.join(DATA_DIR, '20260105-141719_Hand_013056.h5')
    # =============================================================================
    # Walking bout target times (January 5, 2026)
    # Each tuple: (name, target_time, search_window_seconds)
    # The script will find walking bouts (bounded by quiet periods) near these times
    # =============================================================================
    WALKING_BOUT_TARGETS = [
        # ('7m walk', dt_time(15, 0, 0), 120),      # 3:00 PM
        # ('5m walk', dt_time(14 , 59, 0), 120),     # 2:59 PM
        # ('20m walk', dt_time(15, 8, 0), 120),     # 3:08 PM
        # ('15m walk', dt_time(15, 9, 0), 120),     # 3:09 PM
        ('track 2 laps', dt_time(15, 13, 0), 240),  # 3:11 PM
    ]

    # LEFT_FILE = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 CC Balance/Pilot testing/2025-10-29/20251029-154305_LF_Pilot_Ch_Oct29.h5'
    # RIGHT_FILE = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 CC Balance/Pilot testing/2025-10-29/20251029-154310_RF_Pilot_Ch_Oct29.h5'
    # HEAD_FILE = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 CC Balance/Pilot testing/2025-10-29/20251029-154313_Head_Pilot_Ch_Oct29.h5'
    # HAND_FILE = HEAD_FILE
    # # =============================================================================
    # # Walking bout target times (January 5, 2026)
    # # Each tuple: (name, target_time, search_window_seconds)
    # # The script will find walking bouts (bounded by quiet periods) near these times
    # # =============================================================================
    # WALKING_BOUT_TARGETS = [
    #     # ('7m walk', dt_time(15, 0, 0), 120),      # 3:00 PM
    #     # ('5m walk', dt_time(14 , 59, 0), 120),     # 2:59 PM
    #     # ('20m walk', dt_time(15, 8, 0), 120),     # 3:08 PM
    #     # ('15m walk', dt_time(15, 9, 0), 120),     # 3:09 PM
    #     ('track 2 laps', dt_time(16, 34, 0), 240),  # 3:11 PM
    # ]

    # Cache file location (same directory as head file)
    CACHE_DIR = Path(HEAD_FILE).parent
    CACHE_FILE = CACHE_DIR / 'cached_analysis.pkl'

    # =============================================================================
    # Bout detection parameters
    # =============================================================================
    MIN_QUIET_SECONDS = 1.5   # Minimum quiet period to count as walk boundary
    MIN_WALK_SECONDS = 3.0    # Minimum walking duration to include
    W_THRESHOLD = 30.0        # Max angular velocity (deg/s) for quiet detection
    A_THRESHOLD = 1.0         # Max accel deviation from gravity (m/s^2) for quiet

    # ==========================================================================
    # END: User parameters (once we have a smooth pipeline, the above can all be inputs)
    # ==========================================================================

    # ===========================================================================
    # STEP 1/N: Load data and synchronize the IMUs, while checking for any previously cached analysis
    # ===========================================================================
    cached_data = load_cache(CACHE_FILE)
    use_cache = False

    if cached_data is not None:
        print("\nFound cached analysis data.")
        user_input = input("Use cached bout selections? [y/N]: ").strip().lower()
        use_cache = user_input == 'y'

    # Load all recordings: display the number of samples.
    print("\nLoading individual recordings...")
    left_rec = load_imu_recording(LEFT_FILE)
    right_rec = load_imu_recording(RIGHT_FILE)
    head_rec = load_imu_recording(HEAD_FILE)
    hand_rec = load_imu_recording(HAND_FILE)
    print(f"  Left foot: {len(left_rec)} samples")
    print(f"  Right foot: {len(right_rec)} samples")
    print(f"  Head: {len(head_rec)} samples")
    print(f"  Hand: {len(hand_rec)} samples")

    # Synchronize all recordings to overlapping region: display the period and frequency.
    print("\nFinding overlapping time region...")
    synced_recordings = find_overlapping_recordings([left_rec, right_rec, head_rec, hand_rec])
    left_synced, right_synced, head_synced, hand_synced = synced_recordings
    PERIOD = left_synced.period
    print(f"\nSampling period: {PERIOD:.6f} s ({1/PERIOD:.1f} Hz)")

    # Create recordings dict for convenience (this just groups the synced recordings together. access via recordings['left'], etc.)
    recordings = {
        'left': left_synced,
        'right': right_synced,
        'head': head_synced,
        'hand': hand_synced
    }

    # ==========================================================================
    # Step 2/N: Compute full trajectory for left foot
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Computing full trajectory for left foot...")
    print("=" * 60)

    left_full_walk_info = imu.compute_position(left_synced.Wb, left_synced.Ab, PERIOD)
    print(f"  Full trajectory computed: {len(left_full_walk_info.P)} samples")

    # ==========================================================================
    # Step 3/N: Detect all walking bouts using quiet period detection
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Detecting walking bouts from quiet periods...")
    print("=" * 60)

    all_bouts = detect_walking_bouts(
        left_synced.Wb, left_synced.Ab, PERIOD,
        W_threshold=W_THRESHOLD,
        A_threshold=A_THRESHOLD,
        min_quiet_seconds=MIN_QUIET_SECONDS,
        min_walk_seconds=MIN_WALK_SECONDS
    )

    print(f"\nDetected {len(all_bouts)} walking bouts total:")
    for i, bout in enumerate(all_bouts):
        bout_start_time = left_synced.time_datetime[bout.start_idx]
        print(f"  {i+1}. {bout_start_time.strftime('%H:%M:%S')} - {bout.duration_seconds:.1f}s")

    # ==========================================================================
    # Step 4/N: Interactive bout selection (or use cache)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Selecting walking bouts...")
    print("=" * 60)

    bout_selections = []

    if use_cache and cached_data is not None and 'bout_selections' in cached_data:
        print("Using cached bout selections")
        bout_selections = cached_data['bout_selections']
        for sel in bout_selections:
            print(f"  {sel}")
    else:
        for bout_name, target_time, search_window in WALKING_BOUT_TARGETS:
            print(f"\n--- {bout_name} (target: {target_time.strftime('%H:%M:%S')}) ---")

            selection = interactive_bout_selection(
                recordings, all_bouts, left_synced.time_datetime, PERIOD,
                bout_name, target_time, search_window
            )

            if selection is not None:
                bout_selections.append(selection)
                print(f"  Selected: {selection}")

        plt.close('all')

    # ==========================================================================
    # Step 5/N: Plot full trajectory with bout sections
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Plotting full trajectory with bout locations...")
    print("=" * 60)

    plot_full_trajectory_with_bouts(left_full_walk_info, PERIOD, bout_selections,
                                     "Left Foot Full Recording")

    # ==========================================================================
    # Step 6/N:Process each selected bout for left/right/head analysis
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Processing selected walking bouts...")
    print("=" * 60)

    bout_results = {}

    for sel in bout_selections:
        print(f"\n--- {sel.name} ---")

        try:
            result = process_walking_bout(
                sel.bout, left_synced, right_synced, head_synced, PERIOD
            )

            left_metrics = result['left_metrics']
            right_metrics = result['right_metrics']
            head_metrics = result['head_metrics']

            print(f"  Left strides: {left_metrics['n_strides']}, "
                  f"mean speed: {left_metrics['mean_speed']:.2f} m/s, "
                  f"mean length: {left_metrics['mean_length']:.2f} m, "
                  f"total dist: {result['left_total_distance']:.2f} m")
            print(f"  Right strides: {right_metrics['n_strides']}, "
                  f"mean speed: {right_metrics['mean_speed']:.2f} m/s, "
                  f"mean length: {right_metrics['mean_length']:.2f} m, "
                  f"total dist: {result['right_total_distance']:.2f} m")
            print(f"  Head ang vel: {head_metrics['ang_vel_mean']:.1f} ± {head_metrics['ang_vel_std']:.1f} deg/s")

            bout_results[sel.name] = {
                'selection': sel,
                **result
            }

            # Plot stride patterns for this bout (skip if no strides detected)
            if left_metrics['n_strides'] > 0 and right_metrics['n_strides'] > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                plt.sca(axes[0])
                imu.plt_ltrl_frwd_strides(result['left_strides'], show=False)
                axes[0].set_title(f'Left Foot - {sel.name}')
                plt.sca(axes[1])
                imu.plt_ltrl_frwd_strides(result['right_strides'], show=False)
                axes[1].set_title(f'Right Foot - {sel.name}')
                plt.tight_layout()
            else:
                print(f"  WARNING: No strides detected, skipping stride plots")

            # Plot rotation-corrected trajectories (both feet aligned to Y axis)
            plot_rotation_corrected_trajectories(
                result['left_walk_info'], result['right_walk_info'],
                title=sel.name
            )

            # Plot head motion
            plot_head_motion(result['head_bout'], PERIOD, sel.name)

            # Plot bout summary
            plot_bout_summary(sel.name, left_metrics, right_metrics, head_metrics)

        except Exception as e:
            print(f"  ERROR processing bout: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ==========================================================================
    # Step 7/N:Save cache
    # ==========================================================================
    cache_data = {
        'bout_selections': bout_selections,
        'bout_results': {name: {
            'selection': r['selection'],
            'left_metrics': r['left_metrics'],
            'right_metrics': r['right_metrics'],
            'head_metrics': {k: v for k, v in r['head_metrics'].items()
                            if k not in ['Wm', 'Am']},  # Don't save large arrays
        } for name, r in bout_results.items()},
        'analysis_timestamp': datetime.now().isoformat(),
    }
    save_cache(CACHE_FILE, cache_data)

    # ==========================================================================
    # Step 8/N: Summary table comparison across bouts
    # ==========================================================================
    if bout_results:
        print("\n" + "=" * 60)
        print("Summary across all bouts")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparison Across Walking Bouts', fontsize=14)

        bout_names = list(bout_results.keys())
        x = np.arange(len(bout_names))

        left_speeds = [bout_results[b]['left_metrics']['mean_speed'] for b in bout_names]
        left_speed_stds = [bout_results[b]['left_metrics']['std_speed'] for b in bout_names]
        right_speeds = [bout_results[b]['right_metrics']['mean_speed'] for b in bout_names]
        right_speed_stds = [bout_results[b]['right_metrics']['std_speed'] for b in bout_names]

        width = 0.35
        axes[0, 0].bar(x - width/2, left_speeds, width, yerr=left_speed_stds,
                       label='Left', capsize=3, alpha=0.7)
        axes[0, 0].bar(x + width/2, right_speeds, width, yerr=right_speed_stds,
                       label='Right', capsize=3, alpha=0.7)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(bout_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Step Speed [m/s]')
        axes[0, 0].set_title('Step Speed by Bout')
        axes[0, 0].legend()
        axes[0, 0].grid(True, axis='y')

        left_lengths = [bout_results[b]['left_metrics']['mean_length'] for b in bout_names]
        right_lengths = [bout_results[b]['right_metrics']['mean_length'] for b in bout_names]

        axes[0, 1].bar(x - width/2, left_lengths, width, label='Left', alpha=0.7)
        axes[0, 1].bar(x + width/2, right_lengths, width, label='Right', alpha=0.7)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(bout_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Step Length [m]')
        axes[0, 1].set_title('Step Length by Bout')
        axes[0, 1].legend()
        axes[0, 1].grid(True, axis='y')

        head_ang_vels = [bout_results[b]['head_metrics']['ang_vel_mean'] for b in bout_names]
        head_ang_vel_stds = [bout_results[b]['head_metrics']['ang_vel_std'] for b in bout_names]

        axes[1, 0].bar(x, head_ang_vels, yerr=head_ang_vel_stds, capsize=3,
                       color='green', alpha=0.7)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(bout_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Angular Velocity [deg/s]')
        axes[1, 0].set_title('Head Angular Velocity by Bout')
        axes[1, 0].grid(True, axis='y')

        left_counts = [bout_results[b]['left_metrics']['n_strides'] for b in bout_names]
        right_counts = [bout_results[b]['right_metrics']['n_strides'] for b in bout_names]

        axes[1, 1].bar(x - width/2, left_counts, width, label='Left', alpha=0.7)
        axes[1, 1].bar(x + width/2, right_counts, width, label='Right', alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(bout_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Number of Steps')
        axes[1, 1].set_title('Step Count by Bout')
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis='y')

        plt.tight_layout()

        # Print summary table with precise times and total distance
        print(f"\n{'Bout':<15} {'Start Time':<12} {'Duration':>10} {'L Steps':>8} {'R Steps':>8} {'L Speed':>10} {'R Speed':>10} {'L Dist':>10} {'R Dist':>10}")
        print("-" * 105)
        for name in bout_names:
            r = bout_results[name]
            sel = r['selection']
            print(f"{name:<15} {sel.start_datetime.strftime('%H:%M:%S'):<12} "
                  f"{sel.duration_seconds:>10.1f} "
                  f"{r['left_metrics']['n_strides']:>8} {r['right_metrics']['n_strides']:>8} "
                  f"{r['left_metrics']['mean_speed']:>10.2f} {r['right_metrics']['mean_speed']:>10.2f} "
                  f"{r['left_total_distance']:>10.2f} {r['right_total_distance']:>10.2f}")

    plt.show()
