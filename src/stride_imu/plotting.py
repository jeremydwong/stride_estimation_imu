import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict
from .inertial import GRAVITY, Strides

# Foot colours and shared y-axis ceilings for the per-bout speed panels
# (make_bout_axes / draw_bout_block below). These defaults cap every panel so
# figures compare directly across trials; callers can override per draw call
# (the batch sets them at the top of the script).
side_color = {'left': 'tab:blue', 'right': 'tab:red'}
ACCEL_YMAX = 50.0       # m/s^2, |A| ceiling
FOOTSPEED_YMAX = 3.0    # m/s, foot speed |V| ceiling
STEPSPEED_YMAX = 1.6    # m/s, step speed ceiling

def plt_ltrl_frwd_strides(strides: 'Strides', show: bool = True):
    """Plot lateral vs forward strides."""
    ltrl = strides.ltrl
    frwd = strides.frwd
    n = ltrl.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, ltrl.shape[1]))
    if show:
        plt.figure()
    for i in range(ltrl.shape[1]):
        plt.plot(ltrl[:,i], frwd[:,i], color=colors[i])

    plt.scatter(ltrl[-1,:], frwd[-1,:], s=20,marker='o')
    plt.grid(True)
    plt.ylabel('Frwd [m]')
    plt.xlabel('Ltrl [m]')
    if show:
        plt.show()

def plt_frwd_elev_strides(strides: 'Strides', show: bool = True):
    """Plot forward vs elevation strides."""
    frwd = strides.frwd
    elev = strides.elev
    n = frwd.shape[1]
    colors = plt.cm.jet(np.linspace(0, 1, n))
    if show:
        plt.figure()
    for i in range(n):
        elev_i = elev[:,i]
        delev = np.diff(elev_i)
        NN = np.where(delev != 0)[0]
        if NN.size > 0:
            NN = NN[-1] + 1
            elev_i = elev_i[:NN]
            err = elev_i[-1]
            elev_i = elev_i - np.linspace(0, err, NN)
            elev_i = np.pad(elev_i, (0, frwd.shape[0] - NN), 'constant')
        plt.plot(-elev_i, color=colors[i])
        plt.plot(frwd[-1, i], -elev_i[-1], 'ko', markerfacecolor='k', markersize=5)
    plt.grid(True)
    plt.ylabel('Elevation [m]')
    plt.xlabel('Forward [m]')
    if show:
        plt.show()

def plt_stride_var(strides: 'Strides', show: bool = True):
    """Plot stride variability ellipse and points."""
    ltrl = strides.ltrl
    frwd = strides.frwd

    # Handle empty strides
    if ltrl.size == 0 or frwd.size == 0:
        if show:
            plt.figure()
        plt.grid(True)
        plt.ylabel('Frwd [m]')
        plt.xlabel('Ltrl [m]')
        plt.title('No strides detected')
        if show:
            plt.show()
        return

    # Extract final position for each stride
    if ltrl.ndim == 2:
        ltrl = ltrl[-1,:]
        frwd = frwd[-1,:]

    cx, cy, ex, ey, covar = compute_cov(ltrl, frwd)
    colors = plt.cm.jet(np.linspace(0, 1, len(ltrl)))
    if show:
        plt.figure()
    for i in range(len(ltrl)):
        plt.plot(cx[i], cy[i], 'o', markeredgecolor='k', markerfacecolor=colors[i], markersize=5)
    plt.plot(ex, ey, color='r', linewidth=3)
    plt.grid(True)
    plt.ylabel('Frwd [m]')
    plt.xlabel('Ltrl [m]')
    plt.axis('equal')
    if show:
        plt.show()

def plt_walk_info_position(walk_info,s=5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points
    P = walk_info.P
    X = P[:,0]
    Y = P[:,1]
    Z = P[:,2]
    ff = walk_info.FF
    ax.scatter(X[ff], Y[ff], Z[ff], c='blue', marker='.', s=s)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D IMU Position')
    plt.show()
	
def compute_cov(x, y, shift_to_zero=True):
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.column_stack([x, y])
    data_center = np.mean(data, axis=0)
    data = data - data_center
    sigma = 1
    from scipy.stats import chi2, norm
    confidence_value = norm.cdf(sigma) - norm.cdf(-sigma)
    scale = chi2.ppf(confidence_value, 2)
    covar = np.cov(data, rowvar=False) * scale
    vals, vecs = np.linalg.eigh(covar)
    t = np.linspace(0, 2 * np.pi, 100)
    e = np.array([np.cos(t), np.sin(t)])
    VV = vecs @ np.diag(np.sqrt(vals))
    e = VV @ e
    ex = e[0, :]
    ey = e[1, :]
    cx = data[:, 0]
    cy = data[:, 1]
    if not shift_to_zero:
        ex = ex + np.mean(x)
        ey = ey + np.mean(y)
        cx = cx + np.mean(x)
        cy = cy + np.mean(y)
    return cx, cy, ex, ey, covar


# ---------------------------------------------------------------------------
# Per-bout speed panel (raw accel / foot speed / step speed), shared by the
# Brock velocity demo, the batch, and brock_analysis.ipynb. Each draw_* takes a
# matplotlib axis and a `data` dict (as built by brock_functions.bout_sync_strides_steps
# / the notebook): keys period, t, t_ref, t0_abs, sides[side]={Vm,Am,ff_idx,ff_t,
# ff_v}, step_sides[side]={t,v}, steps (steps_from_strides output). The time axis
# is referenced to the SNUG-UP sample (t=0 = best estimate of gait start); the
# manually-clipped pre-walk time sits at negative t, and |A| / |V| / step speed
# all share this axis.
# ---------------------------------------------------------------------------

def draw_accel(ax, data, ymax=ACCEL_YMAX):
    """Row 1: raw body-frame accelerometer magnitude |A| per foot. Impact spikes
    mark contacts; pre-walk box-handling shows up as activity left of t=0 (the
    snug-up gait start). Carries the absolute-sample-index secondary axis."""
    period, t0_abs = data['period'], data['t0_abs']
    ax.axvline(0, color='k', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(GRAVITY, color='gray', lw=0.6, ls='--', alpha=0.5)
    for side in ['left', 'right']:
        ax.plot(data['t'], data['sides'][side]['Am'], lw=0.6,
                color=side_color[side], alpha=0.85, label=f'{side} |A|')
    secax = ax.secondary_xaxis(
        'top', functions=(lambda x, o=t0_abs: x / period + o,
                          lambda s, o=t0_abs: (s - o) * period))
    secax.set_xlabel('absolute sample index', fontsize=8)
    secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}'))
    ax.set_ylabel(r'|A| [m/s$^2$]')
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncols=2)


def draw_velocity(ax, data, ymax=FOOTSPEED_YMAX):
    """Row 2: |V| per foot with footfall dots and step stars, plus the snug-start
    foot-speed thresholds (horizontal). t=0 is the snug-up gait start."""
    steps = data['steps']
    ax.axvline(0, color='k', lw=0.8, ls=':', alpha=0.6)   # t=0 = snug gait start
    ax.axhline(steps['start_high'], color='gray', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(steps['start_low'], color='gray', lw=0.7, ls=':', alpha=0.6)
    for side in ['left', 'right']:
        s = data['sides'][side]
        ax.plot(data['t'], s['Vm'], lw=0.7, color=side_color[side], alpha=0.85,
                label=f'{side} |V|')
        ax.plot(s['ff_t'], s['ff_v'], '.', color=side_color[side], ms=9,
                alpha=0.9)
    for side in ['left', 'right']:
        ss = data['step_sides'][side]
        ax.plot(ss['t'], ss['v'], '*', color=side_color[side], ms=15, mec='k',
                mew=0.6, linestyle='None',
                label=f'{side} step' if side == 'left' else None)
    ax.set_ylabel('foot speed |V| [m/s]')
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncols=3)


def draw_step_speed(ax, data, ymax=STEPSPEED_YMAX):
    """Row 3: step speed between consecutive footfalls (one curve), zero-anchored
    at t=0 (the snug-up gait start, walker at rest). First point is the snugged
    first step."""
    steps = data['steps']
    t_step = (steps['end_idx'] - data['t_ref']) * data['period']
    ax.plot(np.r_[0.0, t_step], np.r_[0.0, steps['frwd_speed']], 'o-',
            color='tab:purple', lw=1.2, ms=6, mec='k', mew=0.4)
    ax.axvline(0, color='k', lw=0.8, ls=':', alpha=0.6)
    ax.set_ylabel('step speed [m/s]')
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.3)


def draw_overhead(ax, data):
    """Left panel: top-down foot-position map. Each foot is rotated INDEPENDENTLY
    so its OWN direction of travel — from the snug gait start to its last walking
    contact — points straight up +Y. This is a final, visualization-only rotation
    (the spatial measures in the table come from the shared common frame and are
    untouched); rotating each foot by its own heading keeps both tracks vertical
    even when the two feet's net headings differ by a few degrees.

    Y=0 at the snug gait start. The two feet are drawn `initial_separation` apart
    at that start — left at -sep/2, right at +sep/2 — so +X = the walker's right
    (left foot on the left). Per foot: trajectory (thin), footfall placements
    (dots) and step landings (stars); the snug start is a green plus.

    Axes are fixed for cross-trial comparison: X always [-1, 1] m, Y from the
    start (~0) up to 1.10x the furthest forward point."""
    steps = data['steps']
    snap, sfoot = steps.get('start_snap'), steps.get('start_snap_foot')
    sep = float(data.get('initial_separation', 0.2))
    g = int(snap) if snap is not None else 0      # snug sample (synced: same for both)
    sign = {'left': -1.0, 'right': +1.0}          # left drawn left, right drawn right
    fy_all = []
    for side in ['left', 'right']:
        xyz = steps.get(f'{side}_xyz')
        if xyz is None or not len(xyz):
            continue
        xy = xyz[:, :2].astype(float)             # [lateral, forward] in common frame
        gi = g if 0 <= g < len(xy) else 0
        start = xy[gi].copy()
        # end of travel = this foot's last walking footfall (fall back: last sample)
        ff = data['sides'][side].get('ff_idx')
        end_i = int(ff[-1]) if (ff is not None and len(ff)) else len(xy) - 1
        d = xy[end_i] - start
        # rotate about the snug start so d (start->end) points +Y, then offset the
        # start to (+/- sep/2, 0): forward Y=0 at gait start, feet sep apart.
        alpha = np.pi / 2 - np.arctan2(d[1], d[0])   # d = (lateral, forward)
        c, s = np.cos(alpha), np.sin(alpha)
        rel = xy - start
        rx = rel[:, 0] * c - rel[:, 1] * s + sign[side] * sep / 2.0
        ry = rel[:, 0] * s + rel[:, 1] * c
        fy_all.append(ry)
        ax.plot(rx, ry, lw=0.6, color=side_color[side], alpha=0.5,
                label=f'{side} path')
        if ff is not None and len(ff):
            ffi = ff[ff < len(rx)]
            ax.plot(rx[ffi], ry[ffi], '.', color=side_color[side], ms=7, alpha=0.9)
        sel = steps['leading_foot'] == side
        li = steps['end_idx'][sel].astype(int)
        li = li[li < len(rx)]
        if len(li):
            ax.plot(rx[li], ry[li], '*', color=side_color[side], ms=12,
                    mec='k', mew=0.5, ls='None')
        if side == sfoot:
            ax.plot(rx[gi], ry[gi], 'P', color='tab:green', ms=10, mec='k',
                    mew=0.5, ls='None', label='snug start')
    ax.axvline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlim(-1.0, 1.0)                            # always [-1, 1] m laterally
    if fy_all:
        ymax = max(float(f.max()) for f in fy_all)
        ymin = min(float(f.min()) for f in fy_all)
        ax.set_ylim(min(0.0, ymin), ymax * 1.10)     # start (~0) up to 1.10x furthest
    ax.set_xlabel('lateral X [m]  (right +)', fontsize=8)
    ax.set_ylabel(r'forward Y [m]  (walk $\uparrow$, 0 = gait start)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc='lower right')


def make_bout_axes(fig, subspec):
    """Lay out one bout inside `subspec`: a tall overhead foot-position panel on
    the left and accel / foot speed / step speed stacked on the right half.
    Returns (ax_overhead, ax_accel, ax_vel, ax_step)."""
    gs = subspec.subgridspec(3, 2, width_ratios=[0.8, 1.0], wspace=0.28, hspace=0.12)
    ax_over = fig.add_subplot(gs[:, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_acc)
    ax_step = fig.add_subplot(gs[2, 1], sharex=ax_acc)
    return ax_over, ax_acc, ax_vel, ax_step


def draw_bout_block(axes4, data, ymax=(ACCEL_YMAX, FOOTSPEED_YMAX, STEPSPEED_YMAX)):
    """Fill one bout's axes (from make_bout_axes) — overhead + accel/|V|/step.
    ymax = (accel, foot speed, step speed) y-axis ceilings."""
    ax_over, ax_acc, ax_vel, ax_step = axes4
    draw_overhead(ax_over, data)
    draw_accel(ax_acc, data, ymax[0])
    draw_velocity(ax_vel, data, ymax[1])
    draw_step_speed(ax_step, data, ymax[2])
    ax_acc.tick_params(labelbottom=False)     # shared x: only label the bottom plot
    ax_vel.tick_params(labelbottom=False)
    ax_step.set_xlabel('time from snug gait start [s]')