"""Diagnose the missing right-foot stride/step in trial 1 bout 2 (s1)."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import stride_imu as imu
from stride_imu import side_color
from brock_functions import (load_feet, load_trial_bounds, walker_for, process_bout,
                             first_motion)

H5_FILE = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s01_s02_20260507.h5'

feet = load_feet(H5_FILE)
period = feet[('s1', 'left')].period
start_idx, end_idx = load_trial_bounds(
    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/s01_s02_ExpTrialBounds.mat')

TRIAL, BOUT = 0, 1
subject = walker_for(TRIAL, BOUT)
i0, i1 = int(start_idx[TRIAL, BOUT]), int(end_idx[TRIAL, BOUT])
onset = min(i0 + first_motion(feet[(subject, side)].Wb[i0:i1], period)
            for side in ['left', 'right'])
result = process_bout(feet, subject, max(0, onset), i1, period, pad_seconds=0.0)
j0, j1 = result['slice']
onset_in_slice = onset - j0

for side in ['left', 'right']:
    se = result[f'{side}_strides'].start_end
    spd = result[f'{side}_strides'].frwd_speed
    print(f"\n{side} strides start_end (abs):")
    for k in range(se.shape[1] if se.size else 0):
        print(f"   {int(se[0,k])+j0} -> {int(se[1,k])+j0}   spd={spd[k]:.2f}")

steps = imu.steps_from_strides(result['left_strides'], result['right_strides'],
                               result['left_info'], result['right_info'], period)
print("\nSTEPS (leading, start_abs, end_abs, t_end, speed):")
for k in range(len(steps['time'])):
    e = int(steps['end_idx'][k]); s = int(steps['start_idx'][k])
    print(f"   {steps['leading_foot'][k]:5s} {s+j0} -> {e+j0}  "
          f"t={(e-onset_in_slice)*period:5.2f}  v={steps['frwd_speed'][k]:.2f}")

# zoom render of bout 2, early window
fig, ax = plt.subplots(figsize=(11, 5))
t = (np.arange(j1 - j0) - onset_in_slice) * period
for side in ['left', 'right']:
    info = result[f'{side}_info']
    ax.plot(t, info.Vm, lw=0.8, color=side_color[side], label=f'{side} |V|')
    td = imu.touchdown_map(info.stationary_periods)
    ff = np.unique(td[np.where(info.FF_walking)[0]])
    ax.plot((ff - onset_in_slice) * period, info.Vm[ff], '.',
            color=side_color[side], ms=11)
for side in ['left', 'right']:
    sel = steps['leading_foot'] == side
    li = steps['end_idx'][sel].astype(int)
    ax.plot((li - onset_in_slice) * period, result[f'{side}_info'].Vm[li],
            '*', color=side_color[side], ms=18, mec='k', mew=0.7, linestyle='None')
ax.set_xlim(-0.3, 2.5)
ax.axvline(0, color='k', ls=':', lw=0.8)
ax.set_xlabel('time from onset [s]'); ax.set_ylabel('|V|'); ax.grid(alpha=0.3)
ax.legend()
fig.savefig('/tmp/bout2_zoom.svg')
print("\nwrote /tmp/bout2_zoom.svg")
