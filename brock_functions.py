"""Shared helpers for the Brock 2025 two-subject session demos.

Data loading, trial-bound parsing, walker scheduling, and the per-bout stride /
velocity pipeline used by the demo_brock_* scripts. These live here (not in a
demo) so no demo imports from another demo. The session .h5 path is hardcoded
in each demo as H5_FILE and passed in; the functions here take paths/positions
as arguments.
"""
import io
import os
import sys
import contextlib
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu

# Foot-IMU role -> 'Label 0' string in the session .h5 (the full six-IMU map,
# incl. experimenter/box, lives in demo_brock_two_subjects).
ROLES = {
    's1_left_foot':  'left_foot_a',
    's1_right_foot': 'right_foot_a',
    's2_left_foot':  'left_foot_b',
    's2_right_foot': 'right_foot_b',
}

PAD_SECONDS = 1.0   # quiet padding around each scored bout for orientation init


def load_feet(file_path):
    """Load the four foot IMUs, keyed ('s1'|'s2', 'left'|'right')."""
    label_to_id = {label: sid for sid, label in imu.list_sensors(file_path).items()}
    feet = {}
    for subject in ['s1', 's2']:
        for side in ['left', 'right']:
            label = ROLES[f'{subject}_{side}_foot']
            feet[(subject, side)] = imu.load_imu_recording(
                file_path, sensor_id=label_to_id[label])
    return feet


def load_trial_bounds(mat_file):
    """Return (start_idx, end_idx) as (96, 2) 0-based sample indices at 100 Hz."""
    m = sio.loadmat(mat_file)
    start_idx = m['TrialStartPoint'].astype(int) - 1   # MATLAB 1-based
    end_idx = m['TrialEndPoint'].astype(int) - 1
    return start_idx, end_idx


def walker_for(trial, bout):
    """Scheduled walker ('s1'|'s2') for 0-based trial and bout indices.

    Trials 0-47: A (s1) holds box and walks second -> bout 0 = s2, bout 1 = s1.
    Trials 48-95: the reverse.
    """
    if trial < 48:
        return 's2' if bout == 0 else 's1'
    return 's1' if bout == 0 else 's2'


def process_bout(feet, subject, i0, i1, period, pad_seconds=PAD_SECONDS):
    """Slice both of the subject's feet (with padding), run the stride pipeline.

    Returns a dict with walk_info and strides for both feet plus summary
    metrics. Distances are net horizontal displacement of each foot trajectory
    over the scored bout (the padding is excluded by construction: the foot is
    stationary and ZUPT-pinned during the pads).

    pad_seconds: quiet lead-in/out added around the scored bout before slicing.
    Pass 0 to detect on exactly the scored window [i0, i1].
    """
    pad = int(pad_seconds / period)
    n = len(feet[(subject, 'left')])
    j0, j1 = max(0, i0 - pad), min(n, i1 + pad)

    left = feet[(subject, 'left')][j0:j1]
    right = feet[(subject, 'right')][j0:j1]

    # stride_segmentation prints per-call; keep the console usable over 192 bouts
    with contextlib.redirect_stdout(io.StringIO()):
        left_info, right_info = imu.compute_position_two_imus(
            left.Wb, left.Ab, right.Wb, right.Ab, period)
        left_strides = imu.stride_segmentation(left_info, period)
        right_strides = imu.stride_segmentation(right_info, period)

    def net_displacement(P):
        return float(np.linalg.norm(P[-1, :2] - P[0, :2]))

    speeds = np.r_[left_strides.frwd_speed, right_strides.frwd_speed]
    return {
        'left_info': left_info, 'right_info': right_info,
        'left_strides': left_strides, 'right_strides': right_strides,
        'slice': (j0, j1),
        'n_strides_left': len(left_strides.frwd_speed),
        'n_strides_right': len(right_strides.frwd_speed),
        'dist_left_m': net_displacement(left_info.P),
        'dist_right_m': net_displacement(right_info.P),
        'stride_speed_mps': float(np.mean(speeds)) if len(speeds) else np.nan,
        'stride_speed_std': float(np.std(speeds)) if len(speeds) else np.nan,
    }


def leading_static_block(W, period):
    """(start, onset) of the leading static block, slice-relative.

    `onset` is the first sample of motion (end of the leading static block + 1);
    `start` is the first sample of that block. detect_quiet_time() finds the
    static periods; we take the first contiguous run. Returns (0, 0) if none.
    """
    static = imu.detect_quiet_time(W, period)
    if len(static) == 0:
        return 0, 0
    gaps = np.where(np.diff(static) > 1)[0]
    lead_end = static[gaps[0]] if len(gaps) else static[-1]
    return int(static[0]), int(lead_end) + 1


def first_motion(W, period):
    """First sample where the foot leaves its initial stance (== onset)."""
    return leading_static_block(W, period)[1]


def bout_sync_strides_steps(feet, subject, i0, i1, period,
                  initial_separation=0.2, anchor_mode='firstonly',
                  gravity_seconds=0.0):
    """Walk-onset snip + stride pipeline for one bout; pull out the series the
    figure needs (|A| and |V| per foot, footfalls, steps, snug start).

    anchor_mode='firstonly' anchors the common frame at the first foot contacts
    so the assumed `initial_separation` lateral offset sits at the gait start
    (visible at the beginning of the overhead); 'auto' minimizes drift instead.

    gravity_seconds: (SHELVED, default 0 = off) keep up to this many seconds of
    the clean stationary block just before walk onset inside the slice, so the
    first stride's ZUPT averages gravity over real stance. It reduces first-step
    drift but perturbs the (anchor-fragile) spatial split, so it's left off by
    default pending a re-slice-free reimplementation; 0 = snip exactly at onset.
    The separate An[0] init bug fix (in compute_position) is always on.
    """
    blocks = {side: leading_static_block(feet[(subject, side)].Wb[i0:i1], period)
              for side in ('left', 'right')}
    onset = i0 + min(end for _, end in blocks.values())     # earliest foot motion
    latest_start = i0 + max(start for start, _ in blocks.values())
    grav = int(round(gravity_seconds / period))
    # lead-in window [snip, onset] must be quiet for BOTH feet, hence latest_start
    lead = max(0, min(grav, onset - latest_start))
    snip = max(0, onset - lead)
    result = process_bout(feet, subject, snip, i1, period, pad_seconds=0.0)
    steps = imu.steps_from_strides(result['left_strides'], result['right_strides'],
                                   result['left_info'], result['right_info'], period,
                                   initial_separation=initial_separation,
                                   anchor_mode=anchor_mode)
    j0, j1 = result['slice']
    # time axis referenced to the SNUG-UP sample (best estimate of gait start):
    # t=0 = snug start, the manually-clipped pre-walk sits at negative t. Fall back
    # to the detected onset if no snug was found.
    snap = steps.get('start_snap')
    onset_in_slice = onset - j0
    t_ref = snap if snap is not None else onset_in_slice   # slice-relative t=0
    t0_abs = j0 + t_ref                                    # absolute sample at t=0
    t = (np.arange(j1 - j0) - t_ref) * period

    sides, step_sides = {}, {}
    for side in ['left', 'right']:
        info = result[f'{side}_info']
        td = imu.touchdown_map(info.stationary_periods)
        ff = np.unique(td[np.where(info.FF_walking)[0]])
        sides[side] = {'Vm': info.Vm, 'Am': np.linalg.norm(info.A, axis=1),
                       'ff_idx': ff,
                       'ff_t': (ff - t_ref) * period, 'ff_v': info.Vm[ff]}
        sel = steps['leading_foot'] == side
        lead_idx = steps['end_idx'][sel].astype(int)
        step_sides[side] = {'t': (lead_idx - t_ref) * period,
                            'v': info.Vm[lead_idx]}

    n_strides = (len(result['left_strides'].time) +
                 len(result['right_strides'].time))
    # plot series (sides/step_sides/steps/t…) for the panel, PLUS the raw
    # per-foot objects + slice so a caller can build a table without re-running
    # the pipeline. This dict is the single per-bout result for both the batch
    # figures and the notebook table.
    return {'subject': subject, 'period': period, 'onset': onset,
            't_ref': t_ref, 't0_abs': t0_abs, 'slice': (j0, j1), 't': t,
            'sides': sides, 'step_sides': step_sides, 'steps': steps,
            'n_strides': n_strides, 'initial_separation': initial_separation,
            'left_info': result['left_info'], 'right_info': result['right_info'],
            'left_strides': result['left_strides'],
            'right_strides': result['right_strides']}
