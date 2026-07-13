"""Brock 2025: overhead step map for one trial, both subjects on a shared axis.

A trial has two walking bouts (one walker each). The two subjects stand on
the same walkway axis, facing each other, separated by the trial's start
distance (trialtable_20260507.csv, "Distance (m)"). s1 is placed at x = 0
walking +x; s2 at x = distance walking -x (their frame is rotated 180 deg
about vertical, which keeps left/right handedness correct in overhead view).

Per walker, steps_from_strides() provides both feet in a common frame plus
the step train; foot placements at footfalls are drawn red (right) / blue
(left). A second figure shows step speed (length/time) over the bout.

Outputs: brock_trial<N>_steps_overhead.png, brock_trial<N>_step_speeds.png
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu
from brock_functions import (load_feet, load_trial_bounds, walker_for,
                             process_bout)

H5_FILE = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s01_s02_20260507.h5'
TRIAL = int(os.environ.get('STRIDE_TRIAL', '1'))   # 1-based
TRIALTABLE_FILE = os.environ.get(
    'STRIDE_TRIALTABLE_FILE',
    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/trialtable_20260507.csv')
ANCHOR_MODE = os.environ.get('STRIDE_ANCHOR_MODE', 'auto')


def fwd_lat(steps, side):
    """The foot's common-frame trajectory as (forward, lateral). steps_from_strides
    returns left_xyz/right_xyz with columns [lateral X, forward Y, up Z]; the
    overhead map wants (walkway=forward, lateral), so reorder to columns [1, 0]."""
    return steps[f'{side}_xyz'][:, [1, 0]]


def walk_start_reference(steps, result):
    """Mean position of the two feet at their first contacts — the walker's
    home stance. steps_from_strides anchors its common frame at the chosen
    anchor stance (often the post-walk stance), so the walk does not start
    at the origin; this recovers the start so the walker can be placed at
    their scene position."""
    refs = []
    for side in ['left', 'right']:
        ff = np.where(result[f'{side}_info'].FF_walking)[0]
        refs.append(fwd_lat(steps, side)[ff[0]])
    return np.mean(refs, axis=0)


def scene_transform(xy, start_ref, walker_start, walks_positive):
    """Map a walker's common frame (forward +x, home stance at start_ref)
    onto the shared walkway axis.

    walks_positive=False rotates 180 deg about vertical (x -> start - x,
    y -> -y), preserving handedness for the subject walking the other way.
    """
    out = xy - start_ref
    if walks_positive:
        out[:, 0] += walker_start
    else:
        out[:, 0] = walker_start - out[:, 0]
        out[:, 1] = -out[:, 1]
    return out


if __name__ == '__main__':
    table = pd.read_csv(TRIALTABLE_FILE, encoding='utf-8-sig')
    distance = float(table['Distance (m)'].iloc[TRIAL - 1])
    print(f"Trial {TRIAL}: start separation {distance} m "
          f"({table['Package size'].iloc[TRIAL - 1]} box, "
          f"{table['Hand-off pose'].iloc[TRIAL - 1]} hand-off)")

    feet = load_feet(H5_FILE)
    period = feet[('s1', 'left')].period
    start_idx, end_idx = load_trial_bounds(
        os.environ.get('STRIDE_TRIALBOUNDS_FILE',
                       '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/'
                       's01_s02_ExpTrialBounds.mat'))

    # s1 stands at x=0 walking +x; s2 at x=distance walking -x
    placement = {'s1': (0.0, True), 's2': (distance, False)}
    side_color = {'left': 'tab:blue', 'right': 'tab:red'}
    subj_marker = {'s1': 'o', 's2': '^'}

    bouts = {}
    for bout in range(2):
        i0, i1 = start_idx[TRIAL - 1, bout], end_idx[TRIAL - 1, bout]
        subject = walker_for(TRIAL - 1, bout)
        result = process_bout(feet, subject, i0, i1, period)
        steps = imu.steps_from_strides(result['left_strides'], result['right_strides'],
                                       result['left_info'], result['right_info'],
                                       period, anchor_mode=ANCHOR_MODE)
        bouts[subject] = {'bout': bout + 1, 'result': result, 'steps': steps}
        print(f"  bout {bout + 1}: {subject} walks — {len(steps['time'])} steps, "
              f"anchor drift {steps['anchor_drift_seconds']:.2f} s, "
              f"{steps['n_same_foot_skips']} skips, {steps['n_too_slow']} slow")

    # ----- overhead step map -----
    fig, ax = plt.subplots(figsize=(14, 5))
    for subject, b in bouts.items():
        steps, info = b['steps'], b['result']
        start_x, positive = placement[subject]
        start_ref = walk_start_reference(steps, info)
        for side in ['left', 'right']:
            xy = scene_transform(fwd_lat(steps, side), start_ref, start_x, positive)
            ff = np.where(info[f'{side}_info'].FF_walking)[0]
            ax.plot(xy[:, 0], xy[:, 1], lw=0.6, alpha=0.35,
                    color=side_color[side])
            ax.plot(xy[ff, 0], xy[ff, 1], subj_marker[subject],
                    color=side_color[side], ms=7, mec='k', mew=0.4, alpha=0.85,
                    label=f'{subject} {side} (bout {b["bout"]})')
        ax.axvline(start_x, color='gray', lw=0.8, ls=':')
        ax.annotate(f'{subject} start', (start_x, 0.55), ha='center',
                    fontsize=9, color='gray')
    ax.set_xlabel('walkway position [m]')
    ax.set_ylabel('lateral [m]')
    ax.set_title(f'Trial {TRIAL} — footfall placements, both subjects on the '
                 f'shared axis (separation {distance} m, anchor_mode='
                 f'{ANCHOR_MODE})')
    ax.legend(fontsize=8, ncols=2, loc='upper center')
    ax.grid(alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    out1 = f'brock_trial{TRIAL}_steps_overhead_{ANCHOR_MODE}.png'
    fig.savefig(out1, dpi=150)
    print(f"Wrote {out1}")

    # ----- step speeds -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (subject, b) in zip(axes, bouts.items()):
        steps = b['steps']
        t_end = steps['end_idx'] * period
        speed = steps['length'] / steps['time']
        for side in ['left', 'right']:
            sel = steps['leading_foot'] == side
            ax.plot(t_end[sel], speed[sel], subj_marker[subject] + '-',
                    color=side_color[side], ms=6, lw=0.8,
                    label=f'{side} leads')
        ax.set_title(f'{subject} (bout {b["bout"]}) — '
                     f'drift {steps["anchor_drift_seconds"]:.2f} s')
        ax.set_xlabel('time in bout [s]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('step speed = length / time [m/s]')
    fig.suptitle(f'Trial {TRIAL} — step speed per step')
    plt.tight_layout()
    out2 = f'brock_trial{TRIAL}_step_speeds_{ANCHOR_MODE}.png'
    fig.savefig(out2, dpi=150)
    print(f"Wrote {out2}")
