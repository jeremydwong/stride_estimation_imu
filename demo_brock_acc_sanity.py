"""Brock 2025 sanity check: full-trial foot accelerometer for both subjects on
a shared time axis, with detected steps marked.

In this protocol the two subjects take turns within a trial: one stays (roughly)
still while the other walks the gap between them, then they swap for the second
bout. This plot is the eyeball test for that story. Each subject gets a subplot
showing both feet's accelerometer magnitude |a| over the entire trial window
(both bouts). You should see:

  - subject 1 quiet (|a| ~ g, ~9.8 m/s^2, flat) during the bout where subject 2
    walks, and bursting with footstep spikes during their own bout;
  - subject 2 the mirror image.

The two subplots share x and y axes so the two subjects are directly comparable.
Stars mark each segmented step (placed at the leading-foot contact, on that
foot's trace), so you can confirm the steps land inside the walking bursts and
not in the quiet stretch.

Output: brock_trial<N>_acc_sanity.png
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
ANCHOR_MODE = os.environ.get('STRIDE_ANCHOR_MODE', 'auto')
PAD_SECONDS = 2.0   # quiet context shown on each side of the trial (display only)
# detection pad: quiet lead-in/out given to the stride pipeline before slicing.
# 0 = detect on exactly the scored bound [i0, i1].
DETECT_PAD_SECONDS = float(os.environ.get('STRIDE_PAD_SECONDS', '0'))

side_color = {'left': 'tab:blue', 'right': 'tab:red'}


if __name__ == '__main__':
    feet = load_feet(H5_FILE)
    period = feet[('s1', 'left')].period
    start_idx, end_idx = load_trial_bounds(
        os.environ.get('STRIDE_TRIALBOUNDS_FILE',
                       '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/'
                       's01_s02_ExpTrialBounds.mat'))

    # full-trial window: from the start of bout 1 to the end of bout 2, padded
    n = len(feet[('s1', 'left')])
    pad = int(PAD_SECONDS / period)
    w0 = max(0, int(start_idx[TRIAL - 1, 0]) - pad)
    w1 = min(n, int(end_idx[TRIAL - 1, 1]) + pad)
    t = (np.arange(w0, w1) - w0) * period   # seconds from window start

    # who walks in which bout, and that bout's segmented steps
    walker = {bout: walker_for(TRIAL - 1, bout) for bout in range(2)}
    steps_by_subject = {}
    for bout in range(2):
        subject = walker[bout]
        i0, i1 = int(start_idx[TRIAL - 1, bout]), int(end_idx[TRIAL - 1, bout])
        result = process_bout(feet, subject, i0, i1, period,
                              pad_seconds=DETECT_PAD_SECONDS)
        steps = imu.steps_from_strides(result['left_strides'], result['right_strides'],
                                       result['left_info'], result['right_info'],
                                       period, anchor_mode=ANCHOR_MODE)
        j0, _ = result['slice']
        steps_by_subject[subject] = {'steps': steps, 'j0': j0, 'bout': bout + 1}
        print(f"Trial {TRIAL} bout {bout + 1}: {subject} walks — "
              f"{len(steps['time'])} steps")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, sharey=True)

    # second x-axis in ABSOLUTE sample index, so diagnostic numbers (e.g. a
    # footfall at sample 51815, or the snip start at 51782) read directly off
    # the plot. Bottom axis stays in seconds from the trial-window start.
    to_sample = lambda x: x / period + w0
    to_seconds = lambda s: (s - w0) * period
    secax = axes[0].secondary_xaxis('top', functions=(to_sample, to_seconds))
    secax.set_xlabel('absolute sample index')
    secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}'))

    for ax, subject in zip(axes, ['s1', 's2']):
        # both feet's accelerometer magnitude over the whole trial window
        acc = {}
        for side in ['left', 'right']:
            a = feet[(subject, side)].Ab[w0:w1]
            acc[side] = np.linalg.norm(a, axis=1)
            ax.plot(t, acc[side], lw=0.6, color=side_color[side], alpha=0.8,
                    label=f'{side} foot |a|')

        # mark this subject's segmented steps on the leading foot's trace
        sb = steps_by_subject[subject]
        steps, j0 = sb['steps'], sb['j0']
        for side in ['left', 'right']:
            sel = steps['leading_foot'] == side
            abs_samp = j0 + steps['end_idx'][sel].astype(int)
            in_win = (abs_samp >= w0) & (abs_samp < w1)
            abs_samp = abs_samp[in_win]
            rel = abs_samp - w0
            ax.plot(rel * period, acc[side][rel], '*', color=side_color[side],
                    ms=13, mec='k', mew=0.5,
                    label=f'{side} step', linestyle='None')

        # shade each bout window and label its walker
        for bout in range(2):
            b0 = (max(w0, int(start_idx[TRIAL - 1, bout])) - w0) * period
            b1 = (min(w1, int(end_idx[TRIAL - 1, bout])) - w0) * period
            walks_here = walker[bout] == subject
            ax.axvspan(b0, b1, color='tab:green' if walks_here else 'gray',
                       alpha=0.12 if walks_here else 0.06)
            ax.annotate(f"bout {bout + 1}: {walker[bout]} walks"
                        f"{'  <-- this subject' if walks_here else ''}",
                        (0.5 * (b0 + b1), 0.97), xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=8,
                        color='tab:green' if walks_here else 'gray')

        ax.axhline(imu.GRAVITY if hasattr(imu, 'GRAVITY') else 9.80297286843,
                   color='k', lw=0.5, ls=':', alpha=0.5)
        ax.set_title(f'{subject}', loc='left', fontsize=10, fontweight='bold')
        ax.set_ylabel('|acceleration| [m/s$^2$]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncols=2, loc='upper right')

    axes[-1].set_xlabel('time from trial start [s]')
    fig.suptitle(f'Trial {TRIAL} — foot accelerometer over the full trial, '
                 f'steps marked (anchor_mode={ANCHOR_MODE})')
    plt.tight_layout()
    out = f'brock_trial{TRIAL}_acc_sanity_{ANCHOR_MODE}.svg'
    fig.savefig(out)
    print(f"Wrote {out}")
