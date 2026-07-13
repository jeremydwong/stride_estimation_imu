"""Brock 2025: per-trial speed panel for the ACTIVE walker of each bout.

Each trial has two bouts; only one subject walks in each (the other stands).
This plots just the walker, one column per bout, three stacked rows:
  1. raw body-frame accelerometer magnitude |A| per foot (impact spikes mark
     contacts; pre-walk box-handling shows up as early activity),
  2. foot speed |V| per foot (ZUPT-pinned to ~0 at stance, peaking mid-swing)
     with footfall dots and step stars on the leading foot, and
  3. step speed between consecutive footfalls (one curve).

The scored bout bounds include ~1.5-2 s of pre-walk standing, so we detect the
first foot motion with detect_quiet_time() and snip from there. Even so the
opening is drift-prone, so the FIRST step is "snugged" to the true walk start
(see snug_start): the two foot-speed thresholds are drawn as horizontal lines
and the chosen start as a vertical line on the |V| row.

Output: brock_trial<N>_velocity.svg
"""
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from stride_imu.plotting import make_bout_axes, draw_bout_block
from brock_functions import (load_feet, load_trial_bounds, walker_for,
                             bout_sync_strides_steps)

H5_FILE = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s01_s02_20260507.h5'
TRIAL = int(os.environ.get('STRIDE_TRIAL', '1'))   # 1-based


if __name__ == '__main__':
    feet = load_feet(H5_FILE)
    period = feet[('s1', 'left')].period
    start_idx, end_idx = load_trial_bounds(
        os.environ.get('STRIDE_TRIALBOUNDS_FILE',
                       '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/'
                       's01_s02_ExpTrialBounds.mat'))

    # one figure, the two bouts stacked: each bout = overhead foot map (left,
    # walk up +Y) + accel / foot speed / step speed (right half)
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    outer = fig.add_gridspec(2, 1)
    for b, bout in enumerate(range(2)):
        subject = walker_for(TRIAL - 1, bout)
        i0, i1 = int(start_idx[TRIAL - 1, bout]), int(end_idx[TRIAL - 1, bout])
        data = bout_sync_strides_steps(feet, subject, i0, i1, period)
        axes4 = make_bout_axes(fig, outer[b])
        axes4[0].set_title(f'bout {bout + 1}: {subject} walks', loc='left',
                           fontsize=10, fontweight='bold')
        draw_bout_block(axes4, data)
        snap = data['steps'].get('start_snap')
        snap_txt = (f"sample {snap} ({snap * period:.2f}s)"
                    if snap is not None else 'n/a')
        print(f"Trial {TRIAL} bout {bout + 1}: {subject} walks — onset "
              f"sample {data['onset']} ({data['onset'] * period:.2f}s), "
              f"{data['n_strides']} strides (both feet) -> "
              f"{len(data['steps']['time'])} steps, snug start {snap_txt}")

    fig.suptitle(f'Trial {TRIAL} — active walker: overhead foot map + raw accel, '
                 f'foot speed, step speed (snipped at onset; first step snugged)')
    out = f'brock_trial{TRIAL}_velocity.svg'
    fig.savefig(out)
    print(f"Wrote {out}")
