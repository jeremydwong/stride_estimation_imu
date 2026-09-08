"""Brock 2025 two-subject session: slice trials by the hand-scored bounds,
run stride estimation per walking bout, and build a per-bout summary table.

Trial bounds come from s01_s02_ExpTrialBounds.mat (manually scored, 96 trials
x 2 bouts, on the same 100 Hz / elapsed-seconds clock as the IMU recording;
sample points are MATLAB 1-based). Each trial has two walking bouts roughly
22 s apart. Per the researcher: trials 1-48 are "A holds box, walks second"
and trials 49-96 are "B holds box, walks second" (A = s01, B = s02) — this
is verified empirically per bout from foot gyro energy.

Outputs:
  - brock_trial_table.csv     one row per bout (192 rows)
  - brock_bout_trajectories.png   overlaid aligned bout trajectories per subject
  - brock_stride_speeds.png       stride speed per trial, by subject/condition
  - `walks` dict (in __main__): walks[subject] = list of per-bout dicts with
    walk_info and stride segmentation for both feet
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brock_functions import load_feet, load_trial_bounds, walker_for, process_bout

H5_FILE = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s01_s02_20260507.h5'
MAT_FILE = os.environ.get(
    'STRIDE_TRIALBOUNDS_FILE',
    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/s01_s02_ExpTrialBounds.mat')


def gyro_energy(rec, i0, i1):
    """Mean angular-velocity norm (rad/s) over a sample range."""
    return float(np.mean(np.linalg.norm(rec.Wb[i0:i1], axis=1)) / rec.period)


def align_to_forward(P):
    """Rotate a trajectory about Z so its net displacement points along +Y."""
    d = P[-1, :2] - P[0, :2]
    ang = np.pi / 2 - np.arctan2(d[1], d[0])
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (R @ (P[:, :2] - P[0, :2]).T).T


if __name__ == '__main__':
    print(f"Loading foot IMUs from {H5_FILE}")
    feet = load_feet(H5_FILE)
    period = feet[('s1', 'left')].period
    assert abs(1 / period - 100.0) < 0.01, "trial bounds were scored on a 100 Hz grid"

    start_idx, end_idx = load_trial_bounds(MAT_FILE)
    n_trials = start_idx.shape[0]
    print(f"{n_trials} trials x 2 bouts from {MAT_FILE}")

    rows = []
    walks = {'s1': [], 's2': []}
    for trial in range(n_trials):
        for bout in range(2):
            i0, i1 = start_idx[trial, bout], end_idx[trial, bout]
            subject = walker_for(trial, bout)

            # verify the scheduled walker is the one actually moving
            other = 's2' if subject == 's1' else 's1'
            e_walker = max(gyro_energy(feet[(subject, 'left')], i0, i1),
                           gyro_energy(feet[(subject, 'right')], i0, i1))
            e_other = max(gyro_energy(feet[(other, 'left')], i0, i1),
                          gyro_energy(feet[(other, 'right')], i0, i1))
            if e_walker <= e_other:
                print(f"  WARNING trial {trial+1} bout {bout+1}: scheduled walker "
                      f"{subject} (gyro {e_walker:.2f}) moves less than {other} "
                      f"({e_other:.2f})")

            result = process_bout(feet, subject, i0, i1, period)
            walks[subject].append({'trial': trial + 1, 'bout': bout + 1, **result})

            condition = 'A_holds_box_walks_second' if trial < 48 else 'B_holds_box_walks_second'
            rows.append({
                'trial': trial + 1,
                'bout': bout + 1,
                'subject': subject,
                'condition': condition,
                'start_idx': i0,
                'end_idx': i1,
                'duration_s': (i1 - i0) * period,
                'n_strides_left': result['n_strides_left'],
                'n_strides_right': result['n_strides_right'],
                'dist_left_m': result['dist_left_m'],
                'dist_right_m': result['dist_right_m'],
                'dist_m': 0.5 * (result['dist_left_m'] + result['dist_right_m']),
                'stride_speed_mps': result['stride_speed_mps'],
                'stride_speed_std': result['stride_speed_std'],
            })
        if (trial + 1) % 16 == 0:
            print(f"  processed {trial+1}/{n_trials} trials")

    table = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'brock_trial_table.csv')
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    with pd.option_context('display.width', 160, 'display.max_columns', 20):
        print("\n", table.head(8).to_string(index=False))
        print("\nPer subject x condition:")
        print(table.groupby(['subject', 'condition'])[
            ['duration_s', 'dist_m', 'stride_speed_mps']].agg(['mean', 'std']).round(2))

    # ----- plots -----
    cond_color = {'A_holds_box_walks_second': 'tab:blue',
                  'B_holds_box_walks_second': 'tab:red'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for ax, subject in zip(axes, ['s1', 's2']):
        for w in walks[subject]:
            cond = ('A_holds_box_walks_second' if w['trial'] <= 48
                    else 'B_holds_box_walks_second')
            XY = align_to_forward(w['left_info'].P)
            ax.plot(XY[:, 0], XY[:, 1], lw=0.5, alpha=0.4, color=cond_color[cond])
        ax.set_title(f'{subject} — left-foot bout trajectories (aligned to +Y)')
        ax.set_xlabel('lateral [m]')
        ax.grid(alpha=0.3)
        ax.axis('equal')
    axes[0].set_ylabel('forward [m]')
    handles = [plt.Line2D([], [], color=c, label=k) for k, c in cond_color.items()]
    axes[1].legend(handles=handles, fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig('brock_bout_trajectories.png', dpi=150)

    fig, ax = plt.subplots(figsize=(14, 5))
    for subject, marker in [('s1', 'o'), ('s2', 's')]:
        sel = table[table.subject == subject]
        ax.scatter(sel.trial, sel.stride_speed_mps, s=18, marker=marker,
                   c=[cond_color[c] for c in sel.condition],
                   label=subject, alpha=0.8)
    ax.axvline(48.5, color='gray', ls='--', lw=1)
    ax.set_xlabel('trial')
    ax.set_ylabel('mean stride speed [m/s]')
    ax.set_title('Stride speed per bout (blue: A holds box; red: B holds box; '
                 'circle: s1, square: s2)')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig('brock_stride_speeds.png', dpi=150)

    plt.show()
