"""Brock 2025: batch foot-speed figures over every trial.

Wraps demo_brock_velocity's per-bout logic in a loop. For each ExpTrialBounds
trial it writes one per-trial SVG: a 3-row x 2-column panel (rows = raw accel /
foot speed / step speed; columns = the two bouts) into the Dropbox figures
folder. It then builds one Distance x Package-size grid of step-speed-over-time
per walking subject, so each person's bouts are easy to scan side by side.

Y-axis ceilings (|A|, foot speed, step speed) are configured at the top of this
script so all trials compare directly. The rightward plots share a time axis
referenced to the snug-up gait start (t=0); pre-walk clipped time is negative.

Condition mapping: trialtable_20260507.csv has 48 rows = 16 (Distance x Package
size) combos x 3 reps, describing ExpTrialBounds trials 1-48. Trials 49-96
repeat the same 48 conditions (row = trial % 48) with the carry role swapped.
For each subject's grid we take the 48 trials where that subject is the active
walker:
  s2 -> trials 1-48,  s1 -> trials 49-96
(active bout picked by walker_for, which is gyro-verified upstream). Each
(Distance, Package) cell then holds that subject's 3 reps.

Outputs (in FIG_DIR):
  brock_trial<N>_velocity.svg           per trial: accel / |V| / step-speed x 2 bouts
  brock_subject_<s>_step_speed_grid.svg one per walking subject
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from stride_imu.plotting import make_bout_axes, draw_bout_block
from brock_functions import load_feet, load_trial_bounds, walker_for, bout_sync_strides_steps

H5_FILE = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s01_s02_20260507.h5'
FIG_DIR = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/figures'
TRIALTABLE = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/trialtable_20260507.csv'
TRIALBOUNDS = os.environ.get(
    'STRIDE_TRIALBOUNDS_FILE',
    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/s01_s02_ExpTrialBounds.mat')
DIST_COL = 'Distance (m)'
PKG_COL = 'Package size'

# Shared figure scales so every trial compares directly. Edit here.
ACCEL_MAX = 60.0        # m/s^2, |A| y-axis
FOOTSPEED_MAX = 4.5     # m/s, foot speed |V| y-axis
STEPSPEED_MAX = 1.75     # m/s, step speed y-axis
YMAX = (ACCEL_MAX, FOOTSPEED_MAX, STEPSPEED_MAX)
# Overhead axes are fixed in plotting.draw_overhead: X always [-1, 1] m, Y up to
# 1.10x the furthest forward point.

# Spatial-frame assumptions (drive step width and the overhead's foot separation).
INITIAL_SEPARATION = 0.2  # m, assumed lateral gap between the feet at the anchor
ANCHOR_MODE = 'firstonly'  # 'firstonly' puts the offset at the gait start (visible
#                            at the start); 'auto' minimizes accumulated drift

# (SHELVED) stance-gravity lead-in seconds. Reduces first-step drift but perturbs
# the anchor-fragile spatial split, so it's left off pending a re-slice-free
# reimplementation. 0 = snip at onset. (The An[0] init bug fix is always on.)
INITIAL_GRAVITY_SECONDS = 0.0

# Per subject, the 48-trial block (0-based) where that subject is the walker we
# pair with the condition table; condition row = trial % 48.
SUBJECT_BLOCK = {'s2': range(0, 48), 's1': range(48, 96)}


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    table = pd.read_csv(TRIALTABLE)         # rows 0-47 == Trial # 1-48
    feet = load_feet(H5_FILE)
    period = feet[('s1', 'left')].period
    start_idx, end_idx = load_trial_bounds(TRIALBOUNDS)
    n_trials = start_idx.shape[0]
    print(f"{n_trials} trials x 2 bouts; writing to {FIG_DIR}")

    cache = {}                              # (trial, bout) -> bout_sync_strides_steps dict
    for trial in range(n_trials):
        cond = table.iloc[trial % 48]
        dist, pkg = cond[DIST_COL], cond[PKG_COL]
        # the two bouts stacked; each = overhead foot map (left) + accel / foot
        # speed / step speed (right half)
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        outer = fig.add_gridspec(2, 1)
        for bout in range(2):
            subject = walker_for(trial, bout)
            i0, i1 = int(start_idx[trial, bout]), int(end_idx[trial, bout])
            axes4 = make_bout_axes(fig, outer[bout])
            axes4[0].set_title(f'bout {bout + 1}: {subject} walks',
                               loc='left', fontsize=10, fontweight='bold')
            try:
                data = bout_sync_strides_steps(feet, subject, i0, i1, period,
                                     initial_separation=INITIAL_SEPARATION,
                                     anchor_mode=ANCHOR_MODE,
                                     gravity_seconds=INITIAL_GRAVITY_SECONDS)
            except Exception as e:        # one bad bout shouldn't sink the batch
                cache[(trial, bout)] = None
                print(f"  SKIP trial {trial + 1} bout {bout + 1} ({subject}): "
                      f"{type(e).__name__}: {e}")
                for ax in axes4:
                    ax.text(0.5, 0.5, 'bout failed', transform=ax.transAxes,
                            ha='center', va='center', color='red')
                continue
            cache[(trial, bout)] = data
            draw_bout_block(axes4, data, ymax=YMAX)

        cond_txt = f'distance {dist} m, package {pkg}'
        fig.suptitle(f'Trial {trial + 1} ({cond_txt}) — active walker: overhead '
                     f'foot map + raw accel, foot speed, step speed (first step '
                     f'snugged)')
        fig.savefig(os.path.join(FIG_DIR, f'brock_trial{trial + 1}_velocity.svg'))
        plt.close(fig)

        if (trial + 1) % 16 == 0:
            print(f"  wrote per-trial figures {trial + 1}/{n_trials}")

    # ----- per-subject Distance x Package grid of |V| over time -----
    distances = sorted(table[DIST_COL].unique())     # ascending: top -> bottom
    packages = sorted(table[PKG_COL].unique())       # ascending: left -> right
    for subject, trials in SUBJECT_BLOCK.items():
        # group this subject's active-walker bouts by (distance, package)
        grouped = {(d, p): [] for d in distances for p in packages}
        for trial in trials:
            bout = 0 if walker_for(trial, 0) == subject else 1
            data = cache[(trial, bout)]
            if data is None:              # bout failed upstream; nothing to plot
                continue
            cond = table.iloc[trial % 48]
            grouped[(cond[DIST_COL], cond[PKG_COL])].append(data)

        nrow, ncol = len(distances), len(packages)
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow),
                                sharex=True, sharey=True, squeeze=False)
        for i, d in enumerate(distances):
            for j, p in enumerate(packages):
                ax = axs[i][j]
                reps = grouped[(d, p)]
                ax.axvline(0, color='k', lw=0.6, ls=':', alpha=0.5)
                for data in reps:
                    steps = data['steps']
                    t_step = (steps['end_idx'] - data['t_ref']) * period
                    ax.plot(np.r_[0.0, t_step], np.r_[0.0, steps['frwd_speed']],
                            'o-', color='tab:purple', lw=1.0, ms=4, mec='k',
                            mew=0.3, alpha=0.7)
                ax.set_ylim(0, STEPSPEED_MAX)
                ax.grid(alpha=0.3)
                ax.text(0.97, 0.92, f'n={len(reps)}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=7, color='gray')
                if i == 0:
                    ax.set_title(f'package {p}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'distance {d} m\nstep speed [m/s]', fontsize=9)
                if i == nrow - 1:
                    ax.set_xlabel('time from snug gait start [s]', fontsize=8)
        fig.suptitle(f'{subject} — step speed over time, by Distance (rows, '
                     f'increasing down) x Package size (cols, increasing right); '
                     f'reps overlaid', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(FIG_DIR, f'brock_subject_{subject}_step_speed_grid.svg')
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")

    print("Done.")
