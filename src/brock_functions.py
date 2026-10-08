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
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

    LEGACY: used only by the dataset-1 example scripts. The Brock bout path
    (bout_sync_strides_steps) does not use it - it calls
    compute_position_two_imus directly and computes no strides.

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


END_LOOKAHEAD_S = 1.0   # snug_end looks this far past the window (see below)

# A "true stop" after the gait end: BOTH feet raw-still (|gyro| < STILL_RAD_S,
# as for walk onset) for STOP_HOLD_S, starting within STOP_SEARCH_S of the
# snugged end (room for a slow closing step). Calibrated on s07_s08: during
# walking both feet are essentially never still together even 0.15 s (132/151
# mid-walk windows), while hand-off pauses are often only 0.25-0.5 s.
# Automatic bouts whose Stop-press window shows no stop are re-snugged with
# the window extended STOP_EXTEND_STEP_S at a time, up to STOP_EXTEND_MAX_S.
# Manual/dragged ends are never extended - only flagged (stop_found=False).
STILL_RAD_S = 0.35
STOP_HOLD_S = 0.25
STOP_SEARCH_S = 1.5
STOP_EXTEND_STEP_S = 1.0
STOP_EXTEND_MAX_S = 3.0

# Fix A (2026-10-03, Hallee's review: 22 walk-back bouts): an AUTOMATIC bout
# whose gait reverses direction (>= WALKBACK_MIN_BACK_STEPS footfalls stepping
# back along the foot's own direction of travel) or that lasts more than
# WALKBACK_TIME_RATIO x the time its target distance needs (target /
# NORMAL_WALK_MPS + 1 s) ends at the TURN - the farthest-forward footfall - and
# is never extended looking for a stop (the stop it finds is after the walk
# back). Manual/dragged ends are only flagged.
WALKBACK_CUT = True
WALKBACK_MIN_BACK_STEPS = 2
WALKBACK_TIME_RATIO = 1.4
NORMAL_WALK_MPS = 1.25

# Fix B (2026-10-03, missed first steps): the onset search never reached more
# than gravity_seconds (1 s) before the Start press, but walkers often start
# earlier (the press was not a go cue). For an AUTOMATIC bout, if committed
# swings (>= the snug threshold, either foot) run back-to-back into the
# detected gait start with gaps <= CHAIN_GAP_S, the search restarts at the
# first of them, at most CHAIN_MAX_S before that start. Not a plain threshold:
# a blanket 3 s search pulled 12/72 clean bouts onto pre-walk activity.
CHAIN_BACK = True
CHAIN_GAP_S = 0.6
CHAIN_MAX_S = 3.0

# End chain (2026-10-08, the mirror of Fix B): with the 0.9 m/s swing
# threshold the slow closing feet-together step counts, but so did shuffles
# after the hand-off. For an AUTOMATIC bout the gait ends at the last swing
# of the unbroken chain (gaps <= CHAIN_GAP_S) that starts at the gait start.
END_CHAIN = True

# The top plot also shows the NEIGHBOURING trials, so you can see which walk
# you are looking at: NEIGHBOR_S seconds either side of the shown bouts. In
# s07_s08 the gap from one trial's last Stop to the next trial's first Start
# is ~36 s (90% within 54 s), so 60 s usually brings in both neighbours.
NEIGHBOR_S = 60.0
NEIGHBOR_COLORS = ('#b35806', '#542788', '#1f78b4', '#c51b7d', '#8c510a',
                   '#2166ac')          # cycled by trial number


def chain_back_start(feet, subject, i0, t0, period, floor=None):
    """Fix B. Earliest start of an unbroken chain of committed swings that
    runs into the gait start t0 (absolute samples), or None. Swings = runs of
    either foot's speed >= DEFAULT_START_FOOTSPEED_HIGH; consecutive swings
    (and the last one to t0) are <= CHAIN_GAP_S apart; at most CHAIN_MAX_S
    before t0.

    Foot speed comes from compute_position_two_imus (the same engine call
    as everything else) run END_LOOKAHEAD_S past t0: ZUPT pins velocity
    only at footfalls, so a run that ENDS at t0 after seconds of standing
    reads the standing as moving and squashes the first step (12.1 b2: 1.05
    vs 1.61 m/s with the look-ahead).

    floor: the first sample the onset search could see (absolute). When t0
    sits at it (<= 0.15 s after), the floor cut a step in progress, so a
    swing still in progress at t0 also joins the chain; otherwise only
    swings that end by t0 do (in-progress pre-walk activity elsewhere
    chained bouts 2-3 s back onto box handling)."""
    a = max(0, min(i0, t0) - int(round((CHAIN_MAX_S + 1.0) / period)))
    lf, rf = feet[(subject, 'left')], feet[(subject, 'right')]
    b = min(len(lf.Wb), len(rf.Wb), t0 + int(round(END_LOOKAHEAD_S / period)))
    try:
        L, R = imu.compute_position_two_imus(lf.Wb[a:b], lf.Ab[a:b],
                                             rf.Wb[a:b], rf.Ab[a:b], period)
    except Exception:
        return None
    from stride_imu.inertial import DEFAULT_START_FOOTSPEED_HIGH as hi
    fast = (L.Vm >= hi) | (R.Vm >= hi)
    d = np.diff(np.r_[0, fast.astype(int), 0])
    at_floor = floor is not None and t0 - floor <= int(round(0.15 / period))
    runs = [(s0 + a, e0 + a) for s0, e0 in zip(np.flatnonzero(d == 1),
                                               np.flatnonzero(d == -1))
            if (s0 + a < t0 if at_floor else e0 + a <= t0)]
    gap, far = int(round(CHAIN_GAP_S / period)), int(round(CHAIN_MAX_S / period))
    start, last = None, t0
    for s0, e0 in reversed(runs):
        if last - e0 > gap or t0 - s0 > far:
            break
        start, last = s0, s0
    return start


def walkback_info(res, period, expected_m=None):
    """Fix A detector on one pipeline result. Returns (detected, back_steps,
    turn_abs, time_ratio). Per foot, footfalls after the gait start are
    projected on that foot's own initial direction of travel (first footfall
    -> first footfall > 0.5 m away; short 2.5 m walks give few footfalls);
    back steps = later footfalls that move back > 0.15 m. turn_abs = the later of the two feet's farthest-forward
    footfalls (absolute sample)."""
    j0, tr = res['slice'][0], res['t_ref']
    back, turns = 0, []
    for side in ('left', 'right'):
        ff = np.asarray(res['sides'][side]['ff_idx'], int)
        ff = ff[ff >= tr]
        if len(ff) < 2:
            continue
        P = res[f'{side}_info'].P[ff, :2]
        d0 = P - P[0]
        far = np.flatnonzero(np.linalg.norm(d0, axis=1) > 0.5)
        if not far.size:
            continue
        ref = d0[far[0]] / np.linalg.norm(d0[far[0]])
        proj = d0 @ ref
        k = int(np.argmax(proj))
        back += int(np.sum(np.diff(proj[k:]) < -0.15))
        turns.append(j0 + int(ff[k]))
    gait = (res['end_abs'] - res['t0_abs']) * period
    ratio = (gait / (expected_m / NORMAL_WALK_MPS + 1.0)
             if expected_m is not None and np.isfinite(expected_m) and expected_m > 0
             else np.nan)
    detected = bool(back >= WALKBACK_MIN_BACK_STEPS
                    or (np.isfinite(ratio) and ratio > WALKBACK_TIME_RATIO))
    return detected, back, (max(turns) if turns else None), ratio


def stop_after(feet, subject, end_abs, period):
    """Did the walker come to a TRUE stop after the gait end `end_abs`?

    True if both of the subject's feet are raw-still (|gyro| < STILL_RAD_S)
    for STOP_HOLD_S, starting within STOP_SEARCH_S after end_abs. Reads the
    raw gyro only (no mechanization), so it may look past any window.
    `feet` is keyed (subject, side) as for bout_sync_strides_steps.
    """
    hold = int(round(STOP_HOLD_S / period))
    a = int(end_abs)
    b = min(min(len(feet[(subject, s)].Wb) for s in ('left', 'right')),
            a + int(round(STOP_SEARCH_S / period)) + hold)
    if b - a < hold:
        return False
    still = np.ones(b - a, bool)
    for side in ('left', 'right'):
        still &= (np.linalg.norm(feet[(subject, side)].Wb[a:b], axis=1)
                  / period) < STILL_RAD_S
    # a run of >= hold still samples that starts within the search span
    run = np.convolve(still.astype(int), np.ones(hold, int), 'valid') == hold
    return bool(run[:int(round(STOP_SEARCH_S / period)) + 1].any())


def _onset_and_snip(feet, subject, i0, i1, period, gravity_seconds):
    """Walk onset + standing lead-in for a search window starting at i0.

    From the subject's RAW stillness (|gyro| < STILL_RAD_S on either foot =
    moving). onset: first moving sample at/after i0 - but if a foot is
    already mid-swing AT i0 (jumped the gun / late press), walk back to that
    motion run's start. lead: consecutive BOTH-feet-still samples just before
    onset, up to gravity_seconds, allowed to cross i0 (the true side-by-side
    stance often sits just before the Start press). The kept standing is
    pinned as stance by the mechanization's own footfall criterion, so the
    first stride's drift correction averages gravity over real standing.
    (detect_quiet_time is NOT used: it is a bias-window hunter that demands
    1.5 s runs, misses short genuine stances, and can even place "onset"
    after the first swing.) Returns (onset, snip), absolute samples."""
    grav = int(round(gravity_seconds / period))
    a0 = max(0, i0 - grav)                       # search floor
    b0 = min(i1, i0 + int(round(5.0 / period)))  # onset must be near i0
    moving = np.zeros(b0 - a0, bool)
    for side in ('left', 'right'):
        Wm = np.linalg.norm(feet[(subject, side)].Wb[a0:b0], axis=1) / period
        moving |= Wm >= STILL_RAD_S
    click = i0 - a0
    if moving[click]:                            # mid-swing at i0
        run = np.flatnonzero(~moving[:click][::-1])
        onset = i0 - (int(run[0]) if len(run) else click)
    else:
        after = np.flatnonzero(moving[click:])
        onset = i0 + (int(after[0]) if len(after) else b0 - i0)
    still_before = ~moving[:onset - a0]
    run = np.flatnonzero(~still_before[::-1])
    lead = min(grav, int(run[0]) if len(run) else len(still_before))
    return onset, max(0, onset - lead)


def find_straight_gait_window(feet, subject, i0, i1, period, manual=False,
                              resnug=True, snug_abs=None, end_overridden=False,
                              expected_m=None, gravity_seconds=1.0):
    """EVERY decision about where a bout starts and ends, in one place - and
    the window it returns is ONE STRAIGHT WALK: reversals are rejected.

    A bout is a single walk in one direction, start to stop. If the walker
    turns and walks back inside the window (a hand-off followed by the
    return), that reversal is not part of the gait: an automatic bout is cut
    at the turn - the farthest-forward footstep - and is never extended past
    it looking for a stop; a manual window is left as the scientist set it
    but flagged (walkback_steps). See walkback_info for how a reversal is
    detected (footsteps stepping back along each foot's own direction, or a
    duration far beyond what the target distance needs).

    Runs BEFORE the analysis (compute_position_two_imus + steps_from_footfalls,
    in bout_sync_strides_steps), which then runs once on the window decided
    here and never moves it. To decide, this looks at foot speeds and
    footstep positions from compute_position_two_imus runs on candidate
    windows (the same engine call the analysis makes).

    In order:
      1. onset + standing lead-in (_onset_and_snip) from raw gyro stillness;
      2. Fix B, CHAIN_BACK (automatic only): if real steps run back-to-back
         into the gait start from before the search window, restart the
         search at the first of them;
      3. gait START: snug_start - the foot-speed valley before the first
         committed swing (or pinned: resnug=False manual windows);
      4. gait END: snug_end - the last landing completed inside the window
         (looked for on a run END_LOOKAHEAD_S past the window, so a swing
         cut off at the edge is not mistaken for a landing);
      5. Fix A, WALKBACK_CUT: reversals rejected - an automatic bout that
         walks back after the turn ends at the turn, no extension (manual:
         flagged only);
      6. stop check: an automatic window with no true stop after the end is
         extended STOP_EXTEND_STEP_S at a time up to STOP_EXTEND_MAX_S; if no
         stop is found the original window is kept, flagged.
    Manual windows are never extended or cut, only flagged; resnug=True
    (default) treats a manual window like button presses (steps 3-4 apply).

    Returns a dict of absolute samples and flags: snip (first sample
    analysed, incl. the standing lead-in), onset, start (gait start = t=0),
    start_from ('snug' | 'pinned' | 'onset' when no committed swing was
    found), end (last gait sample, inclusive), end_foot, stop_found, end_extended_s,
    stop_searched_s, walkback_cut, walkback_steps, chain_back_s.
    """
    i0, i1 = int(i0), int(i1)
    force_start = None if resnug else (snug_abs if snug_abs is not None
                                       else (i0 if manual else None))
    snug_end_on = resnug or not (manual or end_overridden)
    automatic = not manual and not end_overridden and force_start is None
    may_extend = snug_end_on and not manual and not end_overridden
    n_rec = min(len(feet[(subject, s)].Wb) for s in ('left', 'right'))
    lookahead = int(round(END_LOOKAHEAD_S / period))
    lf, rf = feet[(subject, 'left')], feet[(subject, 'right')]

    def decide(stop, search_from):
        """Start/end of the gait for the search window [search_from, stop)."""
        onset, snip = _onset_and_snip(feet, subject, search_from, stop, period,
                                      gravity_seconds)
        L, R = imu.compute_position_two_imus(lf.Wb[snip:stop], lf.Ab[snip:stop],
                                             rf.Wb[snip:stop], rf.Ab[snip:stop],
                                             period)
        if force_start is not None:
            start, start_from = int(np.clip(force_start, snip, stop - 1)), 'pinned'
        else:
            g, _ = imu.snug_start(L, R)
            start, start_from = ((snip + g, 'snug') if g is not None
                                 else (onset, 'onset'))
        end, end_foot = stop - 1, None
        if snug_end_on:
            ext_stop = min(n_rec, stop + lookahead)
            eL, eR = ((L, R) if ext_stop <= stop else
                      imu.compute_position_two_imus(
                          lf.Wb[snip:ext_stop], lf.Ab[snip:ext_stop],
                          rf.Wb[snip:ext_stop], rf.Ab[snip:ext_stop], period))
            e, end_foot = imu.snug_end(
                eL, eR, limit=stop - 1 - snip, start=start - snip,
                max_gap=(int(round(CHAIN_GAP_S / period))
                         if END_CHAIN and automatic else None))
            if e is not None and snip + 2 <= snip + e + 1 < stop:
                end = snip + e
        return {'snip': snip, 'onset': onset, 'start': start, 'end': end,
                'start_from': start_from, 'end_foot': end_foot,
                'stop_found': stop_after(feet, subject, end, period),
                'end_extended_s': (stop - i1) * period, 'stop_searched_s': 0.0,
                'walkback_cut': False, 'walkback_steps': 0, 'chain_back_s': 0.0,
                '_infos': (L, R)}

    def walkback(w):
        """walkback_info on the GAIT window [snip, end] - the run the analysis
        will use (footfall detection near a window's edge depends on where
        the window ends, so the wider look-ahead run is not used here)."""
        L, R = w['_infos']
        a, b = w['snip'], w['end'] + 1
        if b < a + len(L.Vm):
            L, R = imu.compute_position_two_imus(lf.Wb[a:b], lf.Ab[a:b],
                                                 rf.Wb[a:b], rf.Ab[a:b], period)
        sides = {}
        for side, info in (('left', L), ('right', R)):
            td = imu.touchdown_map(info.stationary_periods, period)
            ff = np.unique(td[np.where(info.FF_walking)[0]])
            sides[side] = {'ff_idx': ff[ff <= w['end'] - w['snip']]}
        probe = {'slice': (w['snip'], w['snip'] + len(L.Vm)),
                 't_ref': w['start'] - w['snip'], 't0_abs': w['start'],
                 'end_abs': w['end'], 'sides': sides,
                 'left_info': L, 'right_info': R}
        return walkback_info(probe, period, expected_m)

    def done(w):
        w.pop('_infos', None)
        return w

    # 2. Fix B: the walker may already be stepping before the search window
    search_from, chain = i0, 0.0
    if CHAIN_BACK and automatic:
        w0 = decide(i1, i0)
        cs = chain_back_start(feet, subject, i0, w0['start'], period,
                              floor=w0['snip'])
        if cs is not None and cs < w0['start']:
            s2 = max(0, cs - int(round(0.1 / period)))
            w2 = decide(i1, s2)
            if w2['start'] < w0['start']:
                search_from, chain = s2, (w0['start'] - w2['start']) * period

    # 5-6. walk-back cut, stop check + extension
    step = int(round(STOP_EXTEND_STEP_S / period))
    n_steps = (int(round(STOP_EXTEND_MAX_S / STOP_EXTEND_STEP_S))
               if may_extend else 0)
    first = None
    for k in range(n_steps + 1):
        stop = min(n_rec, i1 + k * step)
        w = decide(stop, search_from)
        w['chain_back_s'] = chain
        first = w if first is None else first
        if WALKBACK_CUT:
            hit, back, turn, _ = walkback(w)
            if automatic and hit and turn is not None:
                cut_at = min(turn + int(round(0.4 / period)), stop)
                c = decide(cut_at, search_from)
                c.update(end_extended_s=0.0, walkback_cut=True,
                         walkback_steps=back, chain_back_s=chain)
                return done(c)
            if not automatic and hit:           # manual: flag only, never cut
                w['walkback_steps'] = back
        if w['stop_found']:
            return done(w)
        if stop >= n_rec:
            break
    # no stop even when extended: keep the ORIGINAL window, flagged
    # (searching further would only reach the next movement)
    first.update(end_extended_s=0.0,
                 stop_searched_s=n_steps * STOP_EXTEND_STEP_S)
    return done(first)


def bout_sync_strides_steps(feet, subject, i0, i1, period,
                  initial_separation=0.2, anchor_mode='firstonly',
                  gravity_seconds=1.0, manual=False, resnug=True,
                  snug_abs=None, end_overridden=False, expected_m=None):
    """THE per-bout pipeline, in two stages:

    1. find_straight_gait_window() decides where the gait starts and ends (all
       the snugging, walk-back, chain-back and stop logic - see its docstring);
    2. the bout is sliced to that window and compute_position_two_imus runs
       ONCE on it;
    3. steps_from_footfalls builds the steps from that run, with the gait
       start handed over (it does not re-decide it).
    No strides are computed: steps (and step speed) come from footfalls.

    Returns the per-bout dict the figures and tables use: the series for the
    panels (sides, step_sides, steps, t), the raw per-foot objects, slice,
    t_ref / t0_abs (gait start) / end_abs (gait end), and the window flags
    (stop_found, end_extended_s, stop_searched_s, walkback_cut,
    walkback_steps, chain_back_s). Arguments: see find_straight_gait_window; plus
    anchor_mode ('firstonly' anchors the common frame at the first foot
    contacts so the assumed `initial_separation` sits at the gait start;
    'auto' minimizes drift) for the step lengths/widths.
    """
    import warnings
    rank_warning = (   # np.RankWarning in numpy 1.x, moved in 2.x
        getattr(getattr(np, 'exceptions', None), 'RankWarning', None)
        or getattr(np, 'RankWarning', RuntimeWarning))
    # the engine prints a footfall-count warning per call; the look-ahead runs
    # make that noise, so silence it here once for the whole bout
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore', rank_warning)
        # 1. where the straight walk starts and ends
        win = find_straight_gait_window(feet, subject, i0, i1, period,
                                        manual=manual, resnug=resnug,
                                        snug_abs=snug_abs,
                                        end_overridden=end_overridden,
                                        expected_m=expected_m,
                                        gravity_seconds=gravity_seconds)
        # 2. slice the bout to that window, run the engine once
        j0, j1 = win['snip'], win['end'] + 1
        lf, rf = feet[(subject, 'left')], feet[(subject, 'right')]
        left_info, right_info = imu.compute_position_two_imus(
            lf.Wb[j0:j1], lf.Ab[j0:j1], rf.Wb[j0:j1], rf.Ab[j0:j1], period)
        t_ref = int(np.clip(win['start'] - j0, 0, j1 - j0 - 1))  # slice-relative t=0
        # 3. steps from footfalls; the gait start is handed over ('onset' =
        # no committed swing was found, i.e. no snug start exists)
        steps = imu.steps_from_footfalls(
            left_info, right_info, period,
            initial_separation=initial_separation, anchor_mode=anchor_mode,
            force_snap=None if win['start_from'] == 'onset' else t_ref)
    steps['end_snap'] = j1 - j0 - 1
    steps['end_snap_foot'] = win['end_foot']
    t0_abs = j0 + t_ref
    t = (np.arange(j1 - j0) - t_ref) * period

    sides, step_sides = {}, {}
    for side, info in (('left', left_info), ('right', right_info)):
        td = imu.touchdown_map(info.stationary_periods, period)
        ff = np.unique(td[np.where(info.FF_walking)[0]])
        sides[side] = {'Vm': info.Vm, 'Am': np.linalg.norm(info.A, axis=1),
                       'ff_idx': ff,
                       'ff_t': (ff - t_ref) * period, 'ff_v': info.Vm[ff]}
        sel = steps['leading_foot'] == side
        lead_idx = steps['end_idx'][sel].astype(int)
        step_sides[side] = {'t': (lead_idx - t_ref) * period,
                            'v': info.Vm[lead_idx]}

    out = {'subject': subject, 'period': period, 'onset': win['onset'],
           't_ref': t_ref, 't0_abs': t0_abs, 'end_abs': j1 - 1,
           'slice': (j0, j1), 't': t,
           'sides': sides, 'step_sides': step_sides, 'steps': steps,
           'n_steps': len(steps['time']), 'initial_separation': initial_separation,
           'left_info': left_info, 'right_info': right_info, 'window': win}
    for k in ('stop_found', 'end_extended_s', 'stop_searched_s',
              'walkback_cut', 'walkback_steps', 'chain_back_s'):
        out[k] = win[k]
    return out


# ---------------------------------------------------------------------------
# Event-scored sessions (dataset 2, 20260622+): the experimenter pressed the
# 'event' sensor button at the start and stop of every movement bout, so the
# trial bounds come from the .h5 Annotations instead of a hand-scored .mat.
# ---------------------------------------------------------------------------

PACKAGE_CODE = {'ring': 1, 'small': 2, 'medium': 3, 'large': 4}


def load_events(file_path):
    """Button-press annotations from a session .h5, sorted by time.

    Returns a dict:
      'label'   - 'Start' | 'Stop' per press
      'time_s'  - seconds relative to the first sensor sample (all sensors in
                  these files share one synced clock)
      'time_us' - raw epoch microseconds (matches Sensors/<id>/Time)
      't0_us'   - the recording's first-sample timestamp
    """
    with h5py.File(file_path, 'r') as f:
        ann = np.sort(f['Annotations'][:], order='Time')
        sid0 = list(f['Sensors'].keys())[0]
        t0 = int(f['Sensors'][sid0]['Time'][0])
    labels = np.array([a.decode().strip('\x00').strip() for a in ann['Annotation']])
    time_us = ann['Time'].astype(np.int64)
    return {'label': labels, 'time_s': (time_us - t0) / 1e6,
            'time_us': time_us, 't0_us': t0}


def infer_stop_from_quiet(data, start_s, limit_s, quiet_seconds=0.75,
                          min_bout_s=2.0, still_rad_s=0.35):
    """When the experimenter forgot the Stop click, find when walking ended.

    Per SUBJECT (feet grouped by label suffix: 'left_foot_a'/'right_foot_a' ->
    'a'), build a both-feet-moving mask (either foot |gyro| >= `still_rad_s`
    rad/s), then morphologically close it: still gaps shorter than
    `quiet_seconds` (within-gait double-stance, ~0.2-0.3 s) are absorbed, so a
    walk becomes ONE continuous motion run that ends exactly where the pair is
    still for >= `quiet_seconds` — the stance at the hand-off. The bout stop is
    the end of the first such run of at least `min_bout_s` by either subject
    (the bout opens with the walker leaving, so the first sustained run is the
    walker's outbound walk; the partner's box-handling shuffles are shorter).

    An earlier version required ALL FOUR feet still at once for 1.5 s. That
    moment structurally never happens at the true stop — the walker hands off
    and turns straight back while the partner handles the box — so inferred
    stops slid past the hand-off AND the walker's un-clicked return walk to
    the between-trial lull, giving 30-40 s "bouts" for 10 s walks (s07_s08).
    `quiet_seconds` must stay below the shortest hand-off stance (~0.9 s
    observed) and above any double-stance still (~0.3 s): 0.75 s.

    Returns the inferred stop time in seconds, or None if no sustained walk
    ends before `limit_s` (the Start is left unpaired rather than guessed —
    including a run still in progress at `limit_s`).
    """
    if data is None or not np.isfinite(limit_s) or limit_s <= start_s:
        return None
    period = next(iter(data.values())).period
    j0, j1 = int(start_s / period), int(limit_s / period)
    n = j1 - j0
    if n <= 0:
        return None

    moving_by_subject = {}
    for label, rec in data.items():
        subj = label.rsplit('_', 1)[-1]
        Wm = np.linalg.norm(rec.Wb[j0:j1], axis=1) / period   # rad/s
        mask = moving_by_subject.setdefault(subj, np.zeros(n, bool))
        mask |= Wm >= still_rad_s

    gap_max = max(1, int(quiet_seconds / period))
    min_run = max(1, int(min_bout_s / period))
    best = None
    for moving in moving_by_subject.values():
        padded = np.r_[False, moving, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = list(zip(edges[::2], edges[1::2]))
        # close still gaps shorter than quiet_seconds between motion runs
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] < gap_max:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if b - a >= min_run and b < n:   # sustained walk, settled in-window
                if best is None or a < best[0]:
                    best = (a, b)
                break                        # only the first per subject
    return start_s + best[1] * period if best else None


def automatically_score_movements_from_events(events, bounding_window_s=30.0,
                                              min_bout_s=1.5, data=None,
                                              infer_stops=False,
                                              quiet_seconds=0.75,
                                              infer_min_bout_s=2.0):
    """Pair Start/Stop button presses into scored movement bouts.

    Walks the annotation stream in time order holding at most one pending
    Start. A Stop within `bounding_window_s` of the pending Start closes a
    bout; a later Start before any Stop replaces the pending one (a restart /
    double-press); a Stop with no viable Start is an orphan. Bouts shorter
    than `min_bout_s` are kept but flagged (likely accidental presses).

    infer_stops: when True (and `data`, a label -> ImuRecording map, is given),
    a Start that would otherwise be dropped — superseded by the next Start, or
    left pending with no Stop in the window — is instead closed by
    `infer_stop_from_quiet()`: the bout ends where every foot settles into
    `quiet_seconds` of stance. This recovers bouts the experimenter started but
    forgot to end. The search never runs past the next button press, so an
    inferred bout cannot swallow the following one. Starts with no settle
    before that limit stay unpaired.

    Returns (snips, report):
      snips  - (n, 2) float array of [start_s, stop_s] per accepted bout
      report - dict of anomaly times: 'restarts' (superseded Starts still
               dropped), 'expired_starts' (no Stop within the window),
               'orphan_stops', 'short' (row indices into snips), and
               'inferred' (bool per snip row: was the Stop inferred?)
    """
    order = np.argsort(events['time_s'])
    t, lab = events['time_s'][order], events['label'][order]
    snips, inferred, restarts, expired, orphans = [], [], [], [], []
    pending = None

    def close_pending(start, limit):
        """Try to end an abandoned bout at the feet's settle; True if closed."""
        if not (infer_stops and data is not None):
            return False
        stop = infer_stop_from_quiet(data, start, limit,
                                     quiet_seconds=quiet_seconds,
                                     min_bout_s=infer_min_bout_s)
        if stop is None:
            return False
        snips.append((start, stop))
        inferred.append(True)
        return True

    for k, (ti, li) in enumerate(zip(t, lab)):
        if li == 'Start':
            if pending is not None and not close_pending(pending, ti):
                restarts.append(pending)
            pending = ti
        elif li == 'Stop':
            if pending is None:
                orphans.append(ti)
            elif ti - pending > bounding_window_s:
                # the Stop is too far to belong to this Start: close the bout at
                # the feet if we can, and treat the late press as an orphan.
                if not close_pending(pending, ti):
                    expired.append(pending)
                orphans.append(ti)
                pending = None
            else:
                snips.append((pending, ti))
                inferred.append(False)
                pending = None
    if pending is not None:
        limit = min(pending + bounding_window_s, float(t[-1]))
        if not close_pending(pending, limit):
            expired.append(pending)

    snips = np.array(snips) if snips else np.empty((0, 2))
    inferred = np.array(inferred, bool)
    # inferred bouts can be appended out of order relative to a later real pair
    if len(snips):
        srt = np.argsort(snips[:, 0])
        snips, inferred = snips[srt], inferred[srt]
    durations = snips[:, 1] - snips[:, 0] if len(snips) else np.empty(0)
    report = {'restarts': np.array(restarts), 'expired_starts': np.array(expired),
              'orphan_stops': np.array(orphans),
              'short': np.where(durations < min_bout_s)[0],
              'inferred': inferred, 'n_inferred': int(inferred.sum()),
              'n_start': int(np.sum(lab == 'Start')),
              'n_stop': int(np.sum(lab == 'Stop')), 'n_snips': len(snips)}
    return snips, report


def list_xlsx_sheets(xlsx_path):
    """Names of the session sheets in a SubInfo2-style workbook.

    Ask the workbook what it contains (don't memorize): returns e.g.
    ['s01_s02', 's03_s04', ...]. Non-session sheets (like 'Summary Stats')
    are included too — pick the one matching your SESSION_TAG.
    """
    import openpyxl
    return openpyxl.load_workbook(xlsx_path, read_only=True).sheetnames


def trialtable_from_xlsx(xlsx_path, sheet_name, out_dir):
    """Build trialtable_<date>.csv from one sheet of the SubInfo2 workbook.

    Each session sheet carries an 'Experiment Trials' block (Trial #, rand #,
    Distance (m), Package size, Hand-off pose, Rep1/Rep2 status, Notes). This
    extracts it and writes the CSV load_condition_table() reads, named by the
    sheet's Date row, into `out_dir`. Returns the CSV path.

    Typical use in a notebook (Colab: upload the .xlsx first):
        CONDITION_CSV = trialtable_from_xlsx(xlsx, SESSION_TAG, DATA_DIR)
    """
    import re
    import openpyxl
    ws = openpyxl.load_workbook(xlsx_path, data_only=True)[sheet_name]
    date = None
    header_pos = None
    for row in ws.iter_rows():
        for cell in row:
            if cell.value == 'Date':
                raw = ws.cell(row=cell.row, column=cell.column + 1).value
                if raw is not None:
                    date = re.sub(r'\D', '', str(raw)[:10])
            if cell.value == 'Trial #':
                header_pos = (cell.row, cell.column)
    if header_pos is None:
        raise ValueError(f'sheet {sheet_name!r}: no "Trial #" header found - '
                         f'is this a session sheet? (see list_xlsx_sheets)')
    columns = ['Trial #', 'rand #', 'Distance (m)', 'Package size',
               'Hand-off pose', 'Rep1 status', 'Rep2 status', 'Notes']
    r0, c0 = header_pos
    records = []
    for r in range(r0 + 1, ws.max_row + 1):
        vals = [ws.cell(row=r, column=c0 + k).value for k in range(len(columns))]
        if not isinstance(vals[0], (int, float)):
            break
        records.append(vals)
    df = pd.DataFrame(records, columns=columns)
    if not date or df.empty or df['Distance (m)'].isna().all():
        raise ValueError(f'sheet {sheet_name!r}: no date or empty trial block')
    df['Trial #'] = df['Trial #'].astype(int)
    out = os.path.join(out_dir, f'trialtable_{date}.csv')
    df.to_csv(out, index=False)
    return out


def load_condition_table(csv_path):
    """Processed condition table for the event-scored sessions.

    Reads the trialtable CSV (columns by position: trial #, rand #, distance
    (m), package size, hand-off pose, rep1 status, rep2 status), normalizes the
    names, and transcribes package size to a code (ring=1 small=2 medium=3
    large=4) in 'package_code'.
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df.rename(columns=dict(zip(df.columns[:7], [
        'trial', 'rand', 'distance_m', 'package', 'pose',
        'rep1_status', 'rep2_status'])))
    df['package'] = df['package'].str.strip().str.lower()
    df['package_code'] = df['package'].map(PACKAGE_CODE)
    for col in ('rep1_status', 'rep2_status'):
        df[col] = df[col].fillna('').str.strip()
    return df


def load_available_feet(file_path, patterns=('foot',)):
    """Load the foot IMUs of BOTH subjects from a session .h5, keyed by label.

    Returns a plain dict {label -> ImuRecording}: 'left_foot_a' and
    'right_foot_a' are subject a's feet, 'left_foot_b'/'right_foot_b' are
    subject b's. Each ImuRecording carries the raw gyro (Wb, rad/sample), the
    raw accelerometer (Ab, m/s²) and the sample period (s), time-synced
    across sensors; rec[a:b] slices it. This dict is the `feet` argument that
    inspect_snipped_trial(), save_trial_figures() and manual_correct() take.

    Unlike load_feet() this tolerates missing sensors (e.g. the 20260622
    session has no left_foot_a) — it loads every sensor whose label contains
    any of `patterns`. Pass patterns=('foot', 'box') to also treat the box
    IMU like a foot (its distance is ZUPT-anchored only while it rests, so
    expect drift over a carry).
    """
    return {label: imu.load_imu_recording(file_path, sensor_id=sid)
            for sid, label in imu.list_sensors(file_path).items()
            if any(p in label for p in patterns)}


OLD_COLUMN_NAMES = {        # pre-2026-10-02 names -> current (see README)
    'measured_m': 'align_max_m',
    'distance_error_m': 'align_error_m',
    'automatic_m': 'automatic_align_max_m',
    'change_m': 'align_change_m',
}


def _upgrade_columns(df):
    """Copy of a bout table with old column names mapped to the current ones.

    `measured_m` was renamed `align_max_m` (and distance_error_m ->
    align_error_m) because students read "measured" as the result; it is the
    alignment FEATURE (largest single-foot excursion over the padded press
    window), computed before any bout is identified. The result distance is
    `walked_m`. Old cached CSVs load unchanged through this.
    """
    df = df.copy(deep=True)
    ren = {o: n for o, n in OLD_COLUMN_NAMES.items()
           if o in df.columns and n not in df.columns}
    return df.rename(columns=ren) if ren else df


def snip_distances(data, snips, pad_seconds=1.0, inferred=None):
    """Per-snip walked distance from the foot IMUs.

    For each [start_s, stop_s] snip, runs single-foot inertial mechanization
    (compute_position) on every recording in `data` (label -> ImuRecording)
    over the padded snip window and takes the maximum horizontal excursion
    from the starting position. The bout distance is the max over feet (the
    walker's feet move the trial distance; the stander's stay ~0) and the
    moving foot's label identifies the walker.

    Returns a DataFrame with start_s/stop_s/duration_s, align_max_m, walker
    (label of the foot that moved farthest), and every per-foot distance.
    """
    rec0 = next(iter(data.values()))
    period = rec0.period
    snips = np.asarray(snips)
    if inferred is None:
        inferred = np.zeros(len(snips), bool)
    rows = []
    for (start_s, stop_s), is_inf in zip(snips, np.asarray(inferred, bool)):
        row = {'start_s': start_s, 'stop_s': stop_s,
               'duration_s': stop_s - start_s, 'inferred': bool(is_inf)}
        for label, rec in data.items():
            j0 = max(0, int((start_s - pad_seconds) / period))
            j1 = min(len(rec.Wb), int((stop_s + pad_seconds) / period))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    traj = imu.compute_position(rec.Wb[j0:j1], rec.Ab[j0:j1],
                                                period)
                excursion = np.linalg.norm(traj.P[:, :2] - traj.P[0, :2], axis=1)
                row[f'dist_{label}'] = float(excursion.max())
            except Exception:
                row[f'dist_{label}'] = np.nan
        dists = {label: row[f'dist_{label}'] for label in data}
        best = max(dists, key=lambda k: -1 if np.isnan(dists[k]) else dists[k])
        row['align_max_m'] = dists[best]
        row['walker'] = best
        rows.append(row)
    return pd.DataFrame(rows)


def expected_bouts(processed_trialtable, reps=2, walkers_per_rep=2,
                   order='rep_major'):
    """Expand the condition table into the expected in-order bout sequence.

    Each rep of a trial has `walkers_per_rep` movement bouts (both
    participants walk the trial distance). order='rep_major' (the 20260622
    protocol, confirmed against the measured distances): the whole 48-trial
    table is run once (rep 1), then again (rep 2). 'trial_major': both reps of
    a trial happen back-to-back. Returns a DataFrame with
    trial/rep/bout/distance_m plus the rep's status note.
    """
    rows = []
    rep_trial = [(rep, i) for rep in range(1, reps + 1)
                 for i in range(len(processed_trialtable))]
    if order == 'trial_major':
        rep_trial.sort(key=lambda rt: (rt[1], rt[0]))
    for rep, i in rep_trial:
        tr = processed_trialtable.iloc[i]
        for bout in range(1, walkers_per_rep + 1):
            rows.append({'trial': int(tr['trial']), 'rep': rep, 'bout': bout,
                         'distance_m': float(tr['distance_m']),
                         'package': tr['package'],
                         'package_code': tr['package_code'],
                         'status': tr[f'rep{rep}_status']})
    return pd.DataFrame(rows)


def align_snips_to_expected(measured, expected, gap_penalty=0.55, times=None,
                            same_trial=None, spacing=None, time_weight=0.6):
    """Needleman-Wunsch alignment of measured snip distances to the expected
    bout-distance sequence.

    Match cost = |measured - expected| / max(measured, expected) (0 = perfect);
    skipping either an expected bout (missed / not captured by button presses)
    or a measured snip (extra press pair) costs `gap_penalty`.

    **Time consistency.** Distance alone lets the alignment place the two bouts
    of one trial minutes apart, orphaning the real walks in between — the two
    walkers of a rep actually go ~13 s apart and trials ~40 s apart. When
    `times` (snip start times, seconds) and `same_trial` (per expected row: is
    it the same trial-rep as the previous row?) are given, a diagonal step also
    pays for the mismatch between the observed inter-snip gap and the gap that
    the expected pair implies, weighted by `time_weight`. Set time_weight=0 for
    the old distance-only behaviour.

    Returns (matches, missed_idx, extra_idx): matches is a list of
    (snip_i, expected_j) pairs; missed_idx indexes expected rows with no snip;
    extra_idx indexes snips with no expected bout.
    """
    obs = np.asarray(measured, float)
    exp = np.asarray(expected, float)
    n, m = len(obs), len(exp)
    use_time = (times is not None and same_trial is not None and time_weight > 0)
    if use_time:
        times = np.asarray(times, float)
        same_trial = np.asarray(same_trial, bool)
        spacing = spacing or {'within_trial_s': 13.0, 'between_trial_s': 40.0}

    def dcost(o, e):
        if np.isnan(o):
            return gap_penalty * 0.9   # unmeasurable snip: near-neutral match
        return abs(o - e) / max(o, e, 0.5)

    def tcost(i, j):
        """Penalty for the time step into a diagonal match at (i, j), 1-based.

        Charged ONLY when expected rows j-2 and j-1 are the two bouts of the
        same trial-rep: those walkers go one after the other (~13 s), so a match
        implying minutes between them is the specific pathology worth
        forbidding. Between-trial gaps are left free — the session's real
        transitions vary widely (breaks, resets) and penalising them makes the
        alignment drop good matches rather than fix bad ones.
        """
        if not use_time or i < 2 or j < 2 or not same_trial[j - 1]:
            return 0.0
        dt = times[i - 1] - times[i - 2]
        want = spacing['within_trial_s']
        if dt <= want:
            return 0.0
        return min((dt - want) / (want + 20.0), 6.0)

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    D[:, 0] = np.arange(n + 1) * gap_penalty
    D[0, :] = np.arange(m + 1) * gap_penalty
    diag = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag[i, j] = dcost(obs[i-1], exp[j-1]) + time_weight * tcost(i, j)
            D[i, j] = min(D[i-1, j-1] + diag[i, j],
                          D[i-1, j] + gap_penalty,
                          D[i, j-1] + gap_penalty)
    matches, missed, extra = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(D[i, j], D[i-1, j-1] + diag[i, j]):
            matches.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and np.isclose(D[i, j], D[i-1, j] + gap_penalty):
            extra.append(i - 1); i -= 1
        else:
            missed.append(j - 1); j -= 1
    return matches[::-1], sorted(missed), sorted(extra)


def explain_missed(aligned, events, data=None, spacing=None, slack=0.6):
    """Diagnose every missed bout: was there a click there, and is there room?

    A red X on the timeline only says "the aligner found no snip for this
    expected bout" — and since the aligner sees ONLY distance, a red X inside a
    run of equal-distance trials is not localized (see the notebook's
    identifiability note). This asks *why*, from two signals the alignment
    never sees:

    1. **The press stream** — how many Start/Stop presses fall in the gap
       between the neighbouring matched snips. Zero means the button was never
       pressed there; an odd count means a press whose partner is missing.
    2. **The clock** — `gap_s` against the session's own pacing
       (`bout_spacing`). A gap no longer than a normal between-trial
       transition has no room for an extra walk, so a red X there is more
       likely an alignment artifact than a real missing bout. Only a gap with
       unexplained EXTRA time can hide a bout.

    `walked_m` (max horizontal foot excursion over the gap, if `data` is given)
    is reported for context but is deliberately NOT used for the verdict: the
    participants walk back to the start between trials, so normal transition
    gaps contain just as much walking as the suspect ones (measured: median
    11.2 m in control gaps vs 9.1 m in missed-bout gaps). Foot motion cannot
    discriminate here.

    Returns one row per missed bout (rows sharing a gap repeat its stats) with
    a `verdict` column.
    """
    if spacing is None:
        spacing = bout_spacing(aligned)
    room = spacing['between_trial_s'] * (1 + slack)
    ok = np.isfinite(aligned['start_s'].to_numpy(float))
    stop_s, start_s = aligned['stop_s'].to_numpy(float), aligned['start_s'].to_numpy(float)
    period = next(iter(data.values())).period if data else None
    cache, rows = {}, []
    for i in np.where(~ok)[0]:
        prev = stop_s[:i][ok[:i]].max() if ok[:i].any() else np.nan
        nxt = start_s[i+1:][ok[i+1:]].min() if ok[i+1:].any() else np.nan
        key = (prev, nxt)
        if key not in cache:
            sel = (events['time_s'] > prev) & (events['time_s'] < nxt)
            walked = np.nan
            if data is not None and np.isfinite(prev) and np.isfinite(nxt):
                dists = []
                for rec in data.values():
                    j0, j1 = int(prev / period), int(nxt / period)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            traj = imu.compute_position(rec.Wb[j0:j1], rec.Ab[j0:j1], period)
                        dists.append(float(np.linalg.norm(
                            traj.P[:, :2] - traj.P[0, :2], axis=1).max()))
                    except Exception:
                        pass
                walked = max(dists) if dists else np.nan
            cache[key] = {'gap_s': nxt - prev, 'n_presses': int(sel.sum()),
                          'presses': ','.join(events['label'][sel]), 'walked_m': walked}
        info = cache[key]
        expect = float(aligned['distance_m'].iloc[i])
        extra = info['gap_s'] > room
        if info['n_presses'] % 2 == 1:
            verdict = 'unpaired press - half-recorded bout'
        elif info['n_presses'] > 0:
            verdict = 'presses present but rejected/unpaired'
        elif extra:
            verdict = 'no press + unexplained extra time - likely real miss'
        else:
            verdict = 'no press, no room in clock - likely alignment artifact'
        rows.append({'trial': int(aligned['trial'].iloc[i]),
                     'rep': int(aligned['rep'].iloc[i]),
                     'bout': int(aligned['bout'].iloc[i]),
                     'distance_m': expect, **info,
                     'room_for_bout': bool(extra), 'verdict': verdict})
    return pd.DataFrame(rows)


def bout_spacing(aligned):
    """Median within-trial-rep and between-trial bout gaps, in seconds.

    The session's own pacing: how long between the two walkers of one rep, and
    between trials. A missed-bout gap much longer than the between-trial median
    has unexplained time in it; one shorter has no room for an extra walk.
    """
    m = aligned.dropna(subset=['start_s'])
    within, between = [], []
    prev = None
    for _, r in m.iterrows():
        if prev is not None:
            gap = r['start_s'] - prev['stop_s']
            same = (r['trial'] == prev['trial']) and (r['rep'] == prev['rep'])
            (within if same else between).append(gap)
        prev = r
    return {'within_trial_s': float(np.median(within)),
            'between_trial_s': float(np.median(between))}


def estimate_reaction_s(feet, aligned, pre_s=2.0, post_s=6.0):
    """Per aligned row: the walker's REACTION time (Start press -> snug gait
    start) in seconds; negative = already walking at the press. NaN when the
    bout is unmatched or no committed swing is found near the press.

    Cheap version of what the inspection figure annotates: mechanizes only
    the walker's two feet over [start_s - pre_s, start_s + post_s] (a ~8 s
    window, fast) and runs the same snug_start() the pipeline uses — the
    velocity valley before the first swing that reaches the start_high
    foot-speed threshold. Raw gyro thresholds were tried first and cannot
    separate energetic box handling from actual walking (0.35 rad/s at the
    click flagged 105/189 bouts; a 2.5 rad/s swing threshold still flagged
    box fumbles): the snug needs foot SPEED, hence the mini-mechanization.

    `feet` is the label-keyed recordings dict from load_available_feet();
    `aligned` the per-bout table (uses start_s and walker).
    """
    period = next(iter(feet.values())).period
    out = np.full(len(aligned), np.nan)
    for k, (_, row) in enumerate(aligned.iterrows()):
        if not (pd.notna(row.get('start_s')) and pd.notna(row.get('walker'))
                and str(row['walker'])):
            continue
        suffix = str(row['walker'])[-1]
        L = feet.get(f'left_foot_{suffix}')
        R = feet.get(f'right_foot_{suffix}')
        if L is None or R is None:
            continue
        i0 = int(row['start_s'] / period)
        a = max(0, i0 - int(pre_s / period))
        b = min(len(L), i0 + int(post_s / period))
        # anchor the mini-window on a verified BOTH-feet-still stretch before
        # the click, so compute_position initializes on real stance — starting
        # mid-box-fumble corrupts Vm and drags the snug onto the fumble
        # (t4 r2b2 read -1.68 s vs the pipeline's +0.75 s without this)
        lo = max(0, i0 - int(6.0 / period))
        still = np.ones(i0 - lo, bool)
        for rec in (L, R):
            Wm = np.linalg.norm(rec.Wb[lo:i0], axis=1) / period
            still &= Wm < 0.35
        need = int(0.4 / period)
        run = 0
        for j in range(len(still) - 1, -1, -1):
            run = run + 1 if still[j] else 0
            if run >= need:
                a = lo + j          # start of the latest long-enough still run
                break
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                li = imu.compute_position(L.Wb[a:b], L.Ab[a:b], period)
                ri = imu.compute_position(R.Wb[a:b], R.Ab[a:b], period)
        except Exception:
            continue
        # snug_start latches onto the FIRST >=1.25 m/s crossing, but the
        # pre-click window can contain box-fumble spikes and even genuine
        # REPOSITIONING steps that stop again before the press. The trial's
        # gait start is the first sustained WALKING RUN (>=3 high crossings,
        # gaps <=2.5 s between swings) whose end reaches the click — i.e.
        # walking that continues through the press (a true jumped gun gives a
        # negative reaction) — or, failing that, the first run after the
        # press. Then walk back into the valley exactly as snug_start does.
        from stride_imu.inertial import (DEFAULT_START_FOOTSPEED_HIGH as high,
                                         DEFAULT_START_FOOTSPEED_LOW as low)
        edges = []                       # (sample, foot Vm) rising crossings
        for Vm in (li.Vm, ri.Vm):
            hi = (Vm >= high)
            e = np.flatnonzero(hi & ~np.r_[False, hi[:-1]])
            edges.extend((int(x), Vm) for x in e)
        edges.sort(key=lambda t: t[0])
        gap = int(2.5 / period)
        runs = []                        # [first_i, last_i] indices into edges
        for i, (e, _) in enumerate(edges):
            if runs and e - edges[runs[-1][1]][0] <= gap:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        click_rel = i0 - a
        g = None
        for r0, r1 in runs:
            if r1 - r0 + 1 < 3:
                continue                 # not sustained walking
            if edges[r1][0] < click_rel - gap:
                continue                 # stopped again before the press:
                                         # repositioning, not the trial walk
            e, Vm = edges[r0]
            j = e
            while j > 0:
                if Vm[j] <= low or Vm[j - 1] > Vm[j]:
                    break
                j -= 1
            g = j
            break
        if g is not None:
            out[k] = (a + g - i0) * period
    return out


def flag_jumped_gun(feet, aligned, reaction_s=None):
    """True per aligned row when the Start press was NOT used as a go cue —
    the walker's snug gait start precedes the press (negative reaction time).
    Thin wrapper over estimate_reaction_s(); pass a precomputed `reaction_s`
    array to skip the mini-mechanization. Unmatched bouts get False.
    """
    if reaction_s is None:
        reaction_s = estimate_reaction_s(feet, aligned)
    return np.asarray(reaction_s) < 0

def align_snips_to_trial_table(feet, snips, processed_trialtable, pad_seconds=1.0,
                      reps=2, walkers_per_rep=2, order='rep_major',
                      gap_penalty=0.55, measured=None, inferred=None,
                      time_weight=0.6, spacing=None):
    """Compare event-scored snips against the condition table's distances.

    `feet` is the label-keyed dict of both subjects' foot recordings from
    load_available_feet(). Measures each snip's walked distance from the
    foot IMUs (`snip_distances`),
    expands the table into the expected in-order bout sequence
    (`expected_bouts`), and aligns the two so missed / skipped trials fall out
    as expected bouts with no matching snip.

    `order` sets the expected sequence layout (see expected_bouts). Pass a
    precomputed `measured` DataFrame (from snip_distances) to skip the slow
    per-snip mechanization when re-aligning.

    `time_weight` / `spacing` feed the alignment's time-consistency term (see
    align_snips_to_expected): without it, distance-only matching will place the
    two bouts of one trial minutes apart and orphan the real walks between
    them. time_weight=0 restores the old behaviour.

    Returns a dict:
      'measured' - per-snip DataFrame (times, per-foot + best distance, walker)
      'expected' - expected bout DataFrame (trial/rep/bout/distance_m/status)
      'aligned'  - expected joined with its matched snip (NaNs where missed),
                   plus align_error_m
      'missed'   - the aligned rows with no matching snip
      'extra'    - measured rows matching no expected bout
    """
    if measured is None:
        measured = snip_distances(feet, snips, pad_seconds=pad_seconds,
                                  inferred=inferred)
    if spacing is None:
        spacing = {'within_trial_s': 13.0, 'between_trial_s': 40.0}
    expected = expected_bouts(processed_trialtable, reps=reps,
                              walkers_per_rep=walkers_per_rep, order=order)
    # same_trial[j]: does expected row j continue the previous row's trial-rep?
    key = list(zip(expected['trial'], expected['rep']))
    same_trial = np.array([False] + [key[j] == key[j-1] for j in range(1, len(key))])
    matches, missed_idx, extra_idx = align_snips_to_expected(
        measured['align_max_m'], expected['distance_m'], gap_penalty=gap_penalty,
        times=measured['start_s'], same_trial=same_trial, spacing=spacing,
        time_weight=time_weight)

    aligned = expected.copy()
    for col in ('snip', 'start_s', 'stop_s', 'duration_s', 'align_max_m'):
        aligned[col] = np.nan
    aligned['walker'] = ''
    aligned['inferred'] = False
    for si, ej in matches:
        aligned.loc[ej, ['snip', 'start_s', 'stop_s', 'duration_s']] = (
            si, *measured.loc[si, ['start_s', 'stop_s', 'duration_s']])
        aligned.loc[ej, 'align_max_m'] = measured.loc[si, 'align_max_m']
        aligned.loc[ej, 'walker'] = measured.loc[si, 'walker']
        if 'inferred' in measured.columns:
            aligned.loc[ej, 'inferred'] = bool(measured.loc[si, 'inferred'])
    aligned['align_error_m'] = aligned['align_max_m'] - aligned['distance_m']
    aligned['reaction_s'] = estimate_reaction_s(feet, aligned)
    aligned['jumped_gun'] = flag_jumped_gun(feet, aligned,
                                            reaction_s=aligned['reaction_s'])
    return {'measured': measured, 'expected': expected, 'aligned': aligned,
            'missed': aligned.iloc[missed_idx],
            'extra': measured.iloc[extra_idx]}


def assign_walkers_by_protocol(feet, result, first_walker='a'):
    """Fix walker assignment from the protocol's walking order.

    align_snips_to_trial_table() names each bout's walker after the single
    foot with the largest horizontal excursion. On short trials a STANDING
    foot's integration drift can out-travel a real 2.5 m walk, so both bouts
    of a trial-rep get pinned on one person (s07_s08: 20 trial-reps).

    The protocol fixes the order instead: in rep 1 (the first pass through
    the 48 trials) `first_walker` walks bout 1 and the other person bout 2;
    rep 2 flips it. This relabels every MATCHED bout accordingly and takes
    its distance from that person's farther-moving foot (the per-foot
    distances already in result['measured'] - nothing is re-mechanized).
    reaction_s / jumped_gun are recomputed, since they follow the walker's
    feet. The snip-to-trial matching itself is NOT redone.

    Returns a new result dict (the input is not modified) whose 'aligned'
    table gains 'walker_auto' (the excursion-based label) and
    'walker_changed' (True where the protocol disagreed). Missed bouts (no
    snip) are left untouched.
    """
    if first_walker not in ('a', 'b'):
        raise ValueError("first_walker must be 'a' or 'b'")
    other = {'a': 'b', 'b': 'a'}[first_walker]
    measured = result['measured']
    if 'align_max_m' not in measured.columns and 'distance_m' in measured.columns:
        measured = measured.rename(columns={'distance_m': 'align_max_m'})
    aligned = _upgrade_columns(result['aligned'])
    aligned['walker_auto'] = aligned['walker']
    aligned['walker_changed'] = False
    for i, row in aligned[aligned['snip'].notna()].iterrows():
        person = first_walker if (row['rep'] == 1) == (row['bout'] == 1) else other
        dists = {lab: measured.loc[int(row['snip']), f'dist_{lab}']
                 for lab in (f'left_foot_{person}', f'right_foot_{person}')
                 if f'dist_{lab}' in measured.columns}
        if not dists:
            continue                     # that person's feet aren't recorded
        foot = max(dists, key=lambda k: -1 if np.isnan(dists[k]) else dists[k])
        aligned.loc[i, 'walker_changed'] = (
            str(row['walker']).rsplit('_', 1)[-1] != person)
        aligned.loc[i, 'walker'] = foot
        aligned.loc[i, 'align_max_m'] = dists[foot]
    aligned['align_error_m'] = aligned['align_max_m'] - aligned['distance_m']
    aligned['reaction_s'] = estimate_reaction_s(feet, aligned)
    aligned['jumped_gun'] = flag_jumped_gun(feet, aligned,
                                            reaction_s=aligned['reaction_s'])
    out = dict(result)
    out['aligned'] = aligned
    out['missed'] = aligned.loc[result['missed'].index]
    return out


# ---------------------------------------------------------------------------
# Per-trial visualization figures + manual inspection/rescoring for the
# event-scored sessions.
# ---------------------------------------------------------------------------

def feet_pairs_from_labels(data):
    """Re-key label-keyed foot recordings for the stride pipeline.

    load_available_feet() returns {'left_foot_a': rec, ...}; the stride/step
    pipeline (bout_sync_strides_steps, process_bout) wants
    {('s1'|'s2', 'left'|'right'): rec} with suffix a -> s1, b -> s2. Feet
    absent from the file are simply absent from the result.
    """
    import re
    out = {}
    for label, rec in data.items():
        m = re.match(r'(left|right)_foot_([ab])$', label)
        if m:
            out[('s1' if m.group(2) == 'a' else 's2', m.group(1))] = rec
    return out


def add_gait_timing(feet, aligned, only=None, resnug=True, cache=None):
    """Add gait_start_s / gait_end_s / gait_duration_s from the snugged bouts,
    plus stop_found (a true stop follows the gait end), end_extended_s
    (how far an automatic window had to be extended to find it; 0 = none) and
    walked_m: the walked distance exactly as the inspection figures draw it
    (snugged gait, mean of both feet, gait start -> farthest footfall), and
    came_back (the window also contains a walk back after a turn). Compare
    walked_m with the target distance_m; align_max_m is the alignment's padded
    (also walkback_cut: Fix A ended the gait at the turn; chain_back_s: Fix
    B started it this many seconds earlier.)
    max excursion and runs larger.

    For every matched bout (or only the rows where boolean `only` is True),
    runs bout_sync_strides_steps() on its [start_s, stop_s] window with the walker's
    feet: the same snugging the figures show, so table timing and figures
    agree, for automatic and manual bouts alike. Session seconds; NaN where
    the bout is missed or the pipeline fails. `cache` (dict) memoizes by
    (walker, window, manual, resnug). Returns a copy.
    """
    out = _upgrade_columns(aligned)
    for col in ('gait_start_s', 'gait_end_s', 'gait_duration_s',
                'end_extended_s', 'walked_m', 'walked_error_m', 'chain_back_s'):
        if col not in out:
            out[col] = np.nan
    for col in ('stop_found', 'came_back', 'walkback_cut'):
        if col not in out:
            out[col] = pd.Series(pd.NA, index=out.index, dtype='boolean')
    pairs = feet_pairs_from_labels(feet)
    period = next(iter(feet.values())).period
    has_win = out['start_s'].notna() & out['stop_s'].notna()
    scope = (pd.Series(only, index=out.index).fillna(False).astype(bool)
             if only is not None else pd.Series(True, index=out.index))
    # in scope but no window (missed, or unmatched by hand): no gait
    for idx in out.index[scope & ~has_win]:
        out.loc[idx, ['gait_start_s', 'gait_end_s', 'gait_duration_s',
                      'end_extended_s', 'walked_m', 'walked_error_m']] = np.nan
        out.loc[idx, ['stop_found', 'came_back']] = pd.NA
    sel = has_win & scope
    for idx in out.index[sel]:
        row = out.loc[idx]
        person = str(row['walker']).rsplit('_', 1)[-1]
        manual = bool(row.get('manual', False) == True)
        key = ('gait3', person, float(row['start_s']), float(row['stop_s']),
               manual, resnug, WALKBACK_CUT, CHAIN_BACK)
        if cache is not None and key in cache:
            g0, g1, stop, ext, walked, came_back, wbc, chain = cache[key]
        else:
            g0 = g1 = ext = walked = chain = np.nan
            came_back = wbc = pd.NA
            stop = pd.NA
            subject = {'a': 's1', 'b': 's2'}.get(person)
            if subject and (subject, 'left') in pairs and (subject, 'right') in pairs:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        r = bout_sync_strides_steps(pairs, subject,
                                         int(row['start_s'] / period),
                                         int(row['stop_s'] / period), period,
                                         manual=manual, resnug=resnug,
                                         expected_m=float(row['distance_m']))
                    g0, g1 = r['t0_abs'] * period, r['end_abs'] * period
                    stop, ext = bool(r['stop_found']), float(r['end_extended_s'])
                    travel = imu.overhead_travel(r)
                    walked, came_back = travel['mean'], travel['came_back']
                    wbc = bool(r.get('walkback_cut', False))
                    chain = float(r.get('chain_back_s', 0.0))
                except Exception:
                    pass
            if cache is not None:
                cache[key] = (g0, g1, stop, ext, walked, came_back, wbc, chain)
        out.loc[idx, ['gait_start_s', 'gait_end_s', 'gait_duration_s',
                      'end_extended_s', 'walked_m', 'walked_error_m']] = (
            g0, g1, g1 - g0, ext, walked, walked - float(row['distance_m']))
        out.loc[idx, 'came_back'] = came_back
        out.loc[idx, 'stop_found'] = stop
        out.loc[idx, 'walkback_cut'] = wbc
        out.loc[idx, 'chain_back_s'] = chain
    return out


def _stop_note(res, manual=False):
    """(text, is_warning) describing the stop check / Fix A / Fix B of a bout."""
    fix = ''
    if res.get('chain_back_s'):
        fix += f"; start moved {res['chain_back_s']:.1f} s earlier (already stepping)"
    if res.get('walkback_cut'):
        return fix + '; walk back removed (ended at the turn)', False
    if res.get('walkback_steps'):
        return (fix + f"; walks back after the turn ({res['walkback_steps']} steps)"
                ' - move the end earlier', True)
    if res.get('stop_found') and not res.get('end_extended_s'):
        return fix, False
    if res.get('stop_found'):
        return (fix + f"; window extended +{res['end_extended_s']:.0f} s to find "
                f"the stop", False)
    if manual:
        return fix + '; NO clear stop in window - your end used as is', True
    searched = res.get('stop_searched_s') or 0
    return (fix + f'; NO clear stop' + (f' (searched +{searched:.0f} s past the '
                                  f'press; end kept at the press)' if searched
                                  else '') + ' - check the end', True)


def _bout_title(row, walked_m=None, came_back=False):
    """Bout block title: target vs the walked distance AS DRAWN (snugged gait,
    mean of both feet, start -> last footfall), plus the alignment's 'align
    max' (largest single-foot excursion over the padded button window - the
    number used ONLY to match snips to trials, before anything is measured;
    it includes pre/post-walk motion and drift, so it is usually larger than
    the drawn walk)."""
    txt = (f"rep {int(row['rep'])} bout {int(row['bout'])}: {row['walker']} "
           f"walks - target {row['distance_m']:.1f} m")
    if walked_m is not None and np.isfinite(walked_m):
        txt += f", walked {walked_m:.1f} m (as drawn)"
    if came_back:
        txt += ' [walked back after a turn - check the end]'
    if pd.notna(row.get('align_max_m')):
        txt += f", align max {row['align_max_m']:.1f} m"
    return txt + (' [inferred stop]' if row.get('inferred') else '')


def _render_inspect_block(axes5, feet, pairs, row, period,
                          prefix_seconds=5.0,
                          ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                imu.STEPSPEED_YMAX),
                          initial_separation=0.2, anchor_mode='firstonly',
                          i1_abs=None, snug_abs=None, i0_abs=None,
                          context_s=None, resnug=True):
    """(Re)draw ONE bout block of the inspection figure onto `axes5`.

    Shared by inspect_snipped_trial() and the draggable interactive version.
    Clears the axes first, so it can redraw in place. `i0_abs` / `i1_abs`
    override the search window (absolute samples; default = the row's
    start_s/stop_s). Snugging follows bout_sync_strides_steps(resnug=...); `snug_abs`
    pins the gait start only when resnug=False. Returns the
    bout_sync_strides_steps() result dict, or None when the pipeline failed
    (the failure is written on the axes).
    """
    for ax in axes5:
        for ch in list(getattr(ax, 'child_axes', [])):
            ch.remove()                   # stale secondary (sample-index) axes
        ax.clear()
    subject = 's1' if str(row['walker']).endswith('a') else 's2'
    title = _bout_title(row)
    # Anchor the title to the BLOCK top, not the equal-scale overhead's: for a
    # short walk that panel shrinks vertically (aspect='equal') and its title
    # sank mid-block, hidden under the wide overhead. x = narrow panel's left,
    # y = wide panel's top (full height) + 34 pt. Owned by the wide panel, so
    # a redraw's ax.clear() removes it.
    import matplotlib.transforms as mtransforms
    anchor = mtransforms.blended_transform_factory(axes5[0].transAxes,
                                                   axes5[1].transAxes)
    title_artist = axes5[1].text(0, 1, title, fontsize=10, fontweight='bold', ha='left',
                  va='bottom', clip_on=False,
                  transform=anchor + mtransforms.ScaledTranslation(
                      0, 34 / 72, axes5[1].figure.dpi_scale_trans))
    if (subject, 'left') not in pairs or (subject, 'right') not in pairs:
        axes5[2].text(0.5, 0.5, f'{subject}: foot IMU missing',
                      transform=axes5[2].transAxes, ha='center')
        return None
    i0 = int(i0_abs) if i0_abs is not None else int(row['start_s'] / period)
    i1 = int(i1_abs) if i1_abs is not None else int(row['stop_s'] / period)
    try:
        res = bout_sync_strides_steps(pairs, subject, i0, i1, period,
                           manual=row.get('manual', False) == True,
                           resnug=resnug, snug_abs=snug_abs,
                           end_overridden=i1_abs is not None,
                           initial_separation=initial_separation,
                           anchor_mode=anchor_mode,
                           expected_m=(float(row['distance_m'])
                                       if pd.notna(row.get('distance_m')) else None))
        expected = float(row['distance_m']) if pd.notna(row.get('distance_m')) else None
        imu.draw_bout_block(axes5, res, ymax=ymax, expected_m=expected)
        # the walked distance of THIS drawing (updates on every drag/redraw)
        travel = imu.overhead_travel(res)
        res['walked_m'], res['came_back'] = travel['mean'], travel['came_back']
        title_artist.set_text(_bout_title(row, walked_m=res['walked_m'],
                                          came_back=res['came_back']))
        if res['came_back']:
            title_artist.set_color('tab:red')

        ax_acc = axes5[2]
        t0_abs = res['t0_abs']            # absolute sample at t=0 (snug)
        j0 = res['slice'][0]              # first sample of the drawn slice
        # grey CONTEXT outside the analysed slice (drawn in colour by
        # draw_bout_block): `pre` s before t=0 and `post` s after the slice,
        # raw |A| AND foot speed - the speed from one mechanization of the
        # whole context window, so you can see where walking really starts
        # and ends before dragging the bounds out there.
        pre, post = context_s if context_s is not None else (prefix_seconds,
                                                              prefix_seconds)
        suffix_ab = 'a' if subject == 's1' else 'b'
        a_pre = max(0, t0_abs - int(pre / period))
        j1 = res['slice'][1]
        b_post = min(len(next(iter(feet.values()))), j1 + int(post / period))
        ax_vel = axes5[3]
        for side in ('left', 'right'):
            rec = feet[f'{side}_foot_{suffix_ab}']
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ctx = imu.compute_position(rec.Wb[a_pre:b_post],
                                               rec.Ab[a_pre:b_post], period)
            except Exception:                 # context is optional decoration
                ctx = None
            for aa, bb in ((a_pre, j0), (j1, b_post)):   # before AND after
                if bb <= aa:
                    continue
                tt = (np.arange(aa, bb) - t0_abs) * period
                ax_acc.plot(tt, np.linalg.norm(rec.Ab[aa:bb], axis=1), lw=0.5,
                            color='0.6', alpha=0.8, zorder=1)
                if ctx is not None:
                    ax_vel.plot(tt, ctx.Vm[aa - a_pre:bb - a_pre], lw=0.6,
                                color='0.6', alpha=0.8, zorder=1,
                                ls='-' if side == 'left' else '--')
        # the button presses (the snip bounds), as vertical trigger lines
        for t_ev, color, lab in (((i0 - t0_abs) * period, '#1a7f37', 'Start press'),
                                 ((i1 - t0_abs) * period, '#666666', 'Stop press')):
            for ax in axes5[2:]:
                ax.axvline(t_ev, color=color, lw=1.0, ls='--', alpha=0.8,
                           zorder=2)
            ax_acc.text(t_ev, ymax[0] * 0.97, lab, color=color, fontsize=7,
                        ha='center', va='top')
        reaction_s = (t0_abs - i0) * period    # Start press -> snug t=0
        duration_s = (res['end_abs'] - t0_abs) * period
        # the end is only un-snugged in legacy (resnug=False) manual/dragged use
        end_is_manual = (not resnug and
                         (i1_abs is not None or row.get('manual', False) == True))
        end_label = ('Manual end' if end_is_manual else
                     'Snug end' if res['steps']['end_snap_foot'] is not None else
                     'Window end (no strong swing)')
        trimmed_s = max(0., (i1 - 1 - res['end_abs']) * period)
        for ax in axes5[2:]:
            ax.axvline(duration_s, color='tab:purple', lw=2, ls='-',
                       label=end_label, zorder=4)
            if trimmed_s > 0:
                ax.axvspan(duration_s, (i1 - t0_abs) * period,
                           color='tab:purple', alpha=.07, zorder=0)
        ax_acc.annotate(end_label, xy=(duration_s, .83),
                        xycoords=('data', 'axes fraction'),
                        xytext=(-6, 0), textcoords='offset points',
                        ha='right', va='top', color='tab:purple', fontsize=8,
                        fontweight='bold', bbox=dict(fc='white', ec='none', alpha=.8))
        stop_note, stop_warn = _stop_note(res, manual=end_is_manual or
                                          row.get('manual', False) == True
                                          or i1_abs is not None)
        ax_acc.text(0.01, 0.04,
                    f'reaction ≈ {reaction_s:.2f} s, '
                    f'gait duration ≈ {duration_s:.1f} s; '
                    + (f'end trimmed {trimmed_s:.2f} s' if not end_is_manual
                       else 'manual end') + stop_note,
                    transform=ax_acc.transAxes, fontsize=8, va='bottom',
                    color='tab:red' if stop_warn else 'black',
                    bbox=dict(fc='white', ec='tab:red' if stop_warn else '0.8',
                              alpha=0.8, pad=2))
        ax_acc.set_xlim(left=min((a_pre - t0_abs) * period,
                                 (i0 - t0_abs) * period - 0.5),
                        right=(b_post - t0_abs) * period)
        return res
    except Exception as e:
        axes5[2].text(0.5, 0.5, f'stride pipeline failed: {e}',
                      transform=axes5[2].transAxes, ha='center',
                      fontsize=8, wrap=True)
        return None


def _inspect_rows(aligned, trial, rep=None):
    """The matched bouts of a trial (optionally one rep); ValueError if none."""
    if rep is not None and (isinstance(rep, bool) or rep not in (1, 2)):
        raise ValueError('rep must be 1 or 2')
    aligned = _upgrade_columns(aligned)
    rows = aligned[(aligned['trial'] == trial)].dropna(subset=['start_s',
                                                               'stop_s'])
    if rep is not None:
        rows = rows[rows['rep'] == rep]
    if not len(rows):
        raise ValueError(f'trial {trial}'
                         + (f', rep {rep}' if rep is not None else '')
                         + ': no matched bouts in the aligned table - '
                         'nothing to draw')
    return rows


def _build_inspect_figure(feet, aligned, trial, rep=None, session_tag='',
                          prefix_seconds=None,
                          ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                imu.STEPSPEED_YMAX),
                          initial_separation=0.2, anchor_mode='firstonly',
                          pad_s=5.0, plot_every=1,
                          interactive=False, resnug=True,
                          neighbor_s=NEIGHBOR_S):
    """THE inspection-figure layout, shared by the static and drag versions.

    Per repetition: a foot-speed overview (_draw_foot_speed_overview) across
    that rep's bouts, then one block per matched bout (_render_inspect_block).
    `interactive=True` only changes presentation - a dpi that fits the
    notebook column and a bottom strip for the save/close buttons - and the
    caller adds the clickable parts. Returns (fig, blocks) with
    blocks = [{'axes5', 'row', 'res', 'ov_ax', 'shade'}, ...] ('ov_ax' = its
    rep's overview axes, 'shade' = the bout's span + label artists there);
    raises ValueError when the
    trial (rep) has no matched bouts.
    """
    import matplotlib.pyplot as plt
    rows = _inspect_rows(aligned, trial, rep)
    pairs = feet_pairs_from_labels(feet)
    period = next(iter(feet.values())).period
    # One SECTION per repetition (a trial's two reps are typically an hour
    # apart, so a shared overview would be ~4000 s wide and useless): its
    # foot-speed overview, then its bout blocks. Vertical budget [in]:
    # title 0.5 | per section: overview ov_h, its x label + the first bout's
    # hanging title (34 pt over its axes) 1.75, 5.4 per bout block | between
    # sections / at the bottom: the last block's hanging legend (+ the button
    # strip when interactive). Manual placement: constrained_layout collapses
    # with >2 nested bout blocks.
    ov_h, head_h, block_h, gap_h = 1.6, 1.75, 5.4, 1.6
    ov_pad = 0.6          # above each overview: rep title + sample-index axis
    bottom_in = 1.25 if interactive else 0.8
    sections = [grp for _, grp in rows.groupby('rep', sort=True)]
    fig_h = (0.5 + sum(ov_pad + ov_h + head_h + block_h * len(g)
                       for g in sections)
             + gap_h * (len(sections) - 1) + bottom_in)
    if interactive:   # built hidden; the caller displays it explicitly
        with plt.ioff():
            fig = plt.figure(figsize=(12.5, fig_h), dpi=_interactive_dpi(12.5))
    else:             # a normal figure: shows inline like any other
        fig = plt.figure(figsize=(12.5, fig_h))
    blocks, top = [], 0.5                    # inches from the figure top
    for sec in sections:
        top += ov_pad
        ov_ax = fig.add_axes([0.07, 1 - (top + ov_h) / fig_h, 0.84,
                              ov_h / fig_h])
        _draw_foot_speed_overview(ov_ax, feet, sec, period,
                                  neighbor_s=neighbor_s, every=plot_every,
                                  aligned=aligned)
        ov_ax.set_title(f"rep {int(sec['rep'].iloc[0])}", loc='left',
                        fontsize=9, fontweight='bold')
        top += ov_h + head_h
        bottom = top + block_h * len(sec)
        outer = fig.add_gridspec(len(sec), 1, top=1 - top / fig_h,
                                 bottom=1 - bottom / fig_h, left=0.07,
                                 right=0.98, hspace=0.55)
        for k, (_, row) in enumerate(sec.iterrows()):
            axes5 = imu.make_bout_axes(fig, outer[k])
            context = [pad_s if prefix_seconds is None else prefix_seconds,
                       pad_s]
            res = _render_inspect_block(axes5, feet, pairs, row, period,
                                        context_s=context, resnug=resnug,
                                        ymax=ymax,
                                        initial_separation=initial_separation,
                                        anchor_mode=anchor_mode)
            _decimate_lines(axes5, plot_every)
            shade = _shade_overview_bout(ov_ax, row, float(row['start_s']),
                                         float(row['stop_s']))
            blocks.append({'axes5': list(axes5), 'row': row, 'res': res,
                           'ov_ax': ov_ax, 'shade': shade,
                           'context': context})
        top = bottom + gap_h
    title = f'{session_tag} trial {int(trial)}'.strip()
    if rep is not None:
        title += f' rep {int(rep)}'
    fig.suptitle(title, y=1 - 0.1 / fig_h, va='top')
    return fig, blocks


def inspect_snipped_trial(feet, aligned, trial, session_tag='',
                          prefix_seconds=None, save_path=None,
                          ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                imu.STEPSPEED_YMAX),
                          initial_separation=0.2, anchor_mode='firstonly',
                          pad_s=5.0, plot_every=1, rep=None, resnug=True,
                          neighbor_s=NEIGHBOR_S):
    """Draw the full inspection figure for ONE trial and return the Figure.

    On top, a foot-speed OVERVIEW: every foot on one session-time axis from
    `pad_s` before the first bout to `pad_s` after the last
    (person A black, B green; right foot dashed), each bout's window shaded
    - the quickest check that the snips sit on the actual walks. Below it,
    every matched bout of the trial is stacked (up to 4 blocks: 2 reps x 2
    walkers; `rep=1|2` shows one repetition). Each block shows two overhead foot maps (equal-scale + wide) on
    the left and raw |A| / foot speed / step speed on the right, plus:

      * `pad_s` of grey CONTEXT before and after the analysed walk - raw |A|
        and foot speed - what was happening around the button presses;
      * the Start/Stop button presses as vertical lines (green/purple);
      * the approximate REACTION time (Start press -> snug gait start, i.e.
        t=0) and the bout DURATION (t=0 -> Stop press), written on the plot.

    Parameters
    ----------
    feet : dict {label -> ImuRecording}
        The matched foot recordings of BOTH subjects, keyed by sensor label:
        'left_foot_a', 'right_foot_a' (subject a) and 'left_foot_b',
        'right_foot_b' (subject b) — exactly what load_available_feet(H5_FILE)
        returns. Each ImuRecording holds the gyro Wb, accelerometer Ab and
        the sample period, time-synced across sensors.
    aligned : pandas.DataFrame
        The per-bout results table from align_snips_to_trial_table(): one row
        per EXPECTED bout. The columns used here: trial/rep/bout (which bout
        this is), start_s/stop_s (the SNIP — the button-press bounds of the
        bout, in seconds of session time; NaN for a missed bout), walker (the
        foot label that moved farthest, so its suffix names the subject),
        distance_m (expected), align_max_m, inferred (True when the Stop was
        inferred from the feet rather than clicked).
    trial : int
        Trial number (the trial table's 1..48).
    prefix_seconds : float or None
        Override for the grey context BEFORE t=0 only (None = `pad_s`).
    save_path : str or None
        When given, the figure is also written to this file (format from the
        extension). None = just build it.
    pad_s : float
        Seconds of grey context each bout block shows on both sides.
    neighbor_s : float
        Seconds the top foot-speed overview spans before the first / after
        the last bout (default NEIGHBOR_S = 60, usually enough to bring in
        the previous and next trials). Neighbouring trials' bouts are shaded
        in their own colours and labelled 'T2 b1 (A)'.
    resnug : bool
        Snug manual windows like button presses (default; see bout_sync_strides_steps).
        False draws a manual window as the gait itself.
    plot_every : int
        Plot only every N-th sample (smaller/faster figures; plotting only -
        all computation stays full-rate). 1 = every sample.

    Returns
    -------
    matplotlib.figure.Figure — catch it to save/show it yourself or to hand
    it to other tooling, e.g. `fig = inspect_snipped_trial(feet, aligned, 5)`.
    For a version where you can DRAG the gait start and bout end, see
    interactive_inspect_trial().
    """
    fig, _ = _build_inspect_figure(
        feet, aligned, trial, rep=rep, session_tag=session_tag,
        prefix_seconds=prefix_seconds, ymax=ymax,
        initial_separation=initial_separation, anchor_mode=anchor_mode,
        pad_s=pad_s, plot_every=plot_every, resnug=resnug,
        neighbor_s=neighbor_s)
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def save_trial_figures(feet, aligned, session_tag, out_dir=None,
                       suffix='viz', trials=None, fmt='svg',
                       ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                             imu.STEPSPEED_YMAX), initial_separation=0.2,
                       anchor_mode='firstonly', pad_s=5.0,
                       plot_every=1, resnug=True, neighbor_s=NEIGHBOR_S):
    """Batch wrapper: inspect_snipped_trial() per trial REPETITION, to disk.

    One file per (trial, rep) - each rep's foot-speed overview plus its bout
    blocks (the two reps of a trial are usually far apart in time):
    <out_dir>/<session_tag>/brock_<session_tag>_trial<N>_rep<R>_<suffix>.<fmt>
    (folder created as necessary). Returns the list of paths. `out_dir`
    defaults to the 'figures' folder NEXT TO THE SESSION'S .h5 (taken from the
    recordings' file_path) — with the data in Dropbox that is the same
    '<imu data>/figures' the dataset-1 batch writes to, and it keeps generated
    figures out of the git repo. `trials=None` draws every trial-rep in
    `aligned`; otherwise a list of trial numbers (both reps) and/or
    (trial, rep) pairs. A rep with no matched bouts is skipped.

    `feet` and `aligned` are exactly as for inspect_snipped_trial(): the
    label-keyed dict of both subjects' foot recordings from
    load_available_feet(), and the per-bout table from
    align_snips_to_trial_table().
    """
    import matplotlib.pyplot as plt

    if out_dir is None:
        src = next(iter(feet.values())).file_path
        if not src:
            raise ValueError('out_dir not given and the recordings carry no '
                             'file_path to derive it from - pass out_dir=')
        out_dir = os.path.join(os.path.dirname(os.path.abspath(src)), 'figures')

    folder = os.path.join(out_dir, session_tag)
    os.makedirs(folder, exist_ok=True)
    want = None
    if trials is not None:
        want = set()
        for item in trials:
            if isinstance(item, (tuple, list)):
                want.add((int(item[0]), int(item[1])))
            else:
                want.update({(int(item), 1), (int(item), 2)})
    pairs = aligned[['trial', 'rep']].drop_duplicates().astype(int)
    paths = []
    for trial, rep in pairs.itertuples(index=False):
        if want is not None and (trial, rep) not in want:
            continue
        path = os.path.join(folder, f'brock_{session_tag}_trial{trial}_rep'
                                    f'{rep}_{suffix}.{fmt}')
        try:
            fig = inspect_snipped_trial(
                feet, aligned, trial, rep=rep, session_tag=session_tag,
                save_path=path, ymax=ymax,
                initial_separation=initial_separation, anchor_mode=anchor_mode,
                pad_s=pad_s, plot_every=plot_every, resnug=resnug,
                neighbor_s=neighbor_s)
        except ValueError:
            continue                      # no matched bouts for this rep
        plt.close(fig)
        paths.append(path)
        if len(paths) % 10 == 0:
            print(f'  ... {len(paths)} trial-rep figures written')
    return paths


def save_manual_windows(csv_path, records):
    """Upsert edits by trial/rep/person; preserve other windows, atomically.

    Both editors, ManualScoring.set_window() and .unmatch() use this shared
    file format: trial, rep, person (a/b), start_s, stop_s in session seconds.
    A record with BOTH start_s and stop_s empty/NaN means "this bout did not
    happen" (unmatch: the aligned row becomes a missed bout). Existing
    append-only histories are collapsed to their latest entries on the next
    save.
    """
    import tempfile
    columns = ['trial', 'rep', 'person', 'start_s', 'stop_s']
    updates = pd.DataFrame(records, columns=columns)
    if updates.empty:
        return
    t = updates[['start_s', 'stop_s']].to_numpy(float)
    no_bout = np.isnan(t).all(axis=1)
    finite = np.isfinite(t).all(axis=1)
    if not (updates['person'].isin(['a', 'b']).all()
            and updates['rep'].isin([1, 2]).all()
            and (no_bout | finite).all()
            and (updates.start_s[finite] >= 0).all()
            and (updates.stop_s[finite] > updates.start_s[finite]).all()):
        raise ValueError('Manual windows require person a/b, rep 1/2, and '
                         '0 <= start < stop (or BOTH empty = no bout)')
    old = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=columns)
    merged = (pd.concat([old, updates], ignore_index=True) if len(old) else updates
              ).drop_duplicates(['trial', 'rep', 'person'], keep='last'
              ).sort_values(['trial', 'rep', 'person'])
    folder = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(folder, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=folder, suffix='.csv', delete=False) as f:
            tmp = f.name
            merged.to_csv(f, index=False)
        os.replace(tmp, csv_path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _run_on_save(controller):
    """Call the editor's on_save hook (set by ManualScoring); return a note.

    Runs inside a widget callback, where a raised exception would vanish into
    the browser log - so failures are reported on the figure's status line.
    """
    hook = getattr(controller, 'on_save', None)
    if hook is None:
        return ''
    try:
        return hook(controller.trial) or ''
    except Exception as e:                      # noqa: BLE001 - shown to user
        return f' - BUT updating the corrected table failed: {e!r}'


def _save_controller_rows(controller, records):
    # Do not replay unchanged editor state over a newer save from another editor.
    previous = getattr(controller, '_saved_windows', {})
    pending = [r for r in records if previous.get((r['trial'], r['rep'], r['person']))
               != (r['start_s'], r['stop_s'])]
    if getattr(controller, '_discarded', False) or not pending:
        return False
    save_manual_windows(controller.out_csv, pending)
    controller._saved_windows = dict(previous)
    for r in pending:
        controller._saved_windows[(r['trial'], r['rep'], r['person'])] = (r['start_s'], r['stop_s'])
    return True


def _decimate_lines(axes, every):
    """Plot-only downsampling: keep every `every`-th point of long lines.

    Speeds up ipympl redraws. Only dense traces are thinned (> 200 points,
    no markers), so step dots, vertical press/drag lines and legends are
    untouched. The underlying data and every computation stay full-rate.
    """
    if not every or every <= 1:
        return
    for ax in axes:
        for a in [ax] + list(getattr(ax, 'child_axes', [])):
            for ln in a.lines:
                x, y = ln.get_xdata(), ln.get_ydata()
                if len(x) > 200 and ln.get_marker() in (None, '', 'None'):
                    ln.set_data(x[::every], y[::every])


OVERVIEW_COLOR = {'a': 'black', 'b': 'tab:green'}   # person 1 / person 2


def _neighbor_bouts(aligned, trial, rep, lo, hi):
    """Other trial-reps' bouts with a window overlapping [lo, hi] session s:
    [(label 'T2 b1 (A)', start_s, stop_s, color), ...]. Missed bouts (no
    window) are left out; works at the session's first and last trials."""
    if aligned is None:
        return []
    a = aligned[aligned['start_s'].notna() & aligned['stop_s'].notna()]
    a = a[~((a['trial'] == trial) & (a['rep'] == rep))]
    a = a[(a['stop_s'] >= lo) & (a['start_s'] <= hi)]
    out = []
    for r in a.sort_values('start_s').itertuples():
        w = str(r.walker) if pd.notna(r.walker) else ''
        who = f' ({w[-1].upper()})' if w else ''
        out.append((f'T{int(r.trial)} b{int(r.bout)}{who}', float(r.start_s),
                    float(r.stop_s),
                    NEIGHBOR_COLORS[int(r.trial) % len(NEIGHBOR_COLORS)]))
    return out


def _draw_neighbor_bouts(ax, neighbors, label=True, y=0.98):
    """Shade + label neighbouring bouts on one time axis (session seconds):
    a light band in the trial's colour, dashed lines at its start/stop (the
    button presses unless corrected) and a 'T2 b1 (A)' label at the top."""
    tr = ax.get_xaxis_transform()
    for lab, a, b, col in neighbors:
        ax.axvspan(a, b, color=col, alpha=0.15, lw=0)
        for t in (a, b):
            ax.axvline(t, color=col, lw=1.0, ls=':', alpha=0.9)
        if label:   # white backing: the label sits over the speed traces
            ax.text(a, y, f' {lab}', transform=tr, va='top', ha='left',
                    fontsize=7, fontweight='bold', color=col, clip_on=True,
                    zorder=5, bbox=dict(boxstyle='square,pad=0.15', fc='white',
                                        ec='none', alpha=0.85))


def _bout_window_s(row, st, period):
    """(start_s, stop_s) of a bout's window, with any drag adjustments."""
    start = (st['i0_abs'] * period if 'i0_abs' in st else
             st['snug_abs'] * period if 'snug_abs' in st else
             float(row['start_s']))
    stop = st['i1_abs'] * period if 'i1_abs' in st else float(row['stop_s'])
    return start, stop


def _shade_overview_bout(ax, row, start_s, stop_s):
    """Shade + label one bout's window on the overview; returns the artists."""
    person = str(row['walker'])[-1]
    color = OVERVIEW_COLOR.get(person, '0.5')
    arts = [ax.axvspan(start_s, stop_s, color=color, alpha=0.10, lw=0),
            ax.text(start_s, 0.98,
                    f" r{int(row['rep'])} b{int(row['bout'])} "
                    f"({person.upper()})",
                    transform=ax.get_xaxis_transform(), va='top',
                    fontsize=7, color=color)]
    return arts


def _draw_foot_speed_overview(ax, feet, rows, period, neighbor_s=NEIGHBOR_S,
                              every=1, aligned=None):
    """Every foot's speed across all shown bouts, on one session-time axis.

    Spans neighbor_s before the first bout start to neighbor_s after the last
    bout end (clipped to the recording), and - when `aligned` is given - marks
    the neighbouring trials' bouts in that span (_draw_neighbor_bouts), so the
    viewer can see which walk is which.
    Person a (s1) black, person b (s2) green; left foot solid, right dashed.
    Foot speed is heading-agnostic, so both walkers compare directly. Each
    foot is mechanized over the whole window (ZUPT keeps stance at ~0).
    Returns the window (lo_s, hi_s).
    """
    n_samples = min(len(rec) for rec in feet.values())
    lo = max(0.0, float(rows['start_s'].min()) - neighbor_s)
    hi = min(n_samples * period, float(rows['stop_s'].max()) + neighbor_s)
    j0, j1 = int(lo / period), int(hi / period)
    t = np.arange(j0, j1) * period
    step = max(1, int(every or 1))
    for label, rec in sorted(feet.items()):
        person = label.rsplit('_', 1)[-1]
        ls = '-' if label.startswith('left') else '--'
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                traj = imu.compute_position(rec.Wb[j0:j1], rec.Ab[j0:j1],
                                            period)
        except Exception as e:                  # noqa: BLE001 - shown on plot
            ax.text(0.01, 0.85, f'{label}: mechanization failed ({e})',
                    transform=ax.transAxes, fontsize=7)
            continue
        ax.plot(t[::step], traj.Vm[::step], lw=0.7, ls=ls,
                color=OVERVIEW_COLOR.get(person, '0.5'),
                label=label.replace('_foot', ''))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, imu.FOOTSPEED_YMAX)
    _draw_neighbor_bouts(ax, _neighbor_bouts(
        aligned, int(rows['trial'].iloc[0]), int(rows['rep'].iloc[0]), lo, hi),
        y=0.86)    # below this trial's own bout labels (y=0.98)
    ax.set_ylabel('foot speed [m/s]')
    ax.set_xlabel('session time [s]', labelpad=1)
    ax.grid(alpha=0.3)
    # second time base on top: absolute sample index (session time / period)
    import matplotlib.ticker as mticker
    secax = ax.secondary_xaxis('top', functions=(lambda x: x / period,
                                                 lambda n: n * period))
    secax.set_xlabel('absolute sample index', fontsize=8, labelpad=2)
    secax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    secax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:.0f}'))
    secax.tick_params(labelsize=8)
    # legend outside on the right: the top belongs to the sample axis, and
    # inside it would cover the bout labels
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.005, 0.5),
              frameon=False, borderaxespad=0)
    return lo, hi


class _DraggableTrial:
    """Drag-to-retime controller for one trial's inspection figure.

    Each bout block carries two grab-able vertical lines on its time axes:
    the SNUG gait start (green, at t=0) and the BOUT END (purple, at the Stop
    press). Mouse-down grabs whichever is closer to the cursor, dragging
    moves it live, and on release the whole bout is RE-RUN with the adjusted
    bounds and redrawn (the time axis re-zeroes to the new snug, so the green
    line lands back on t=0 after a snug drag — that is expected).
    Adjustments live in .state[(rep, bout)] = {'snug_abs': ..., 'i1_abs': ...}
    (absolute sample indices), printed to the status line as you go.
    """
    GRAB = ('snug', 'end')
    on_save = None    # ManualScoring hook: called with the trial after a save
    plot_every = 1    # plot-only downsampling (see _decimate_lines)
    resnug = True     # drags move the SEARCH window; gait bounds re-snug in it
    extend_s = 5.0    # double-click: widen that side of a block's view by this

    def __init__(self, fig, feet, pairs, period, blocks, prefix_seconds,
                 ymax, initial_separation, anchor_mode, trial, out_csv,
                 session_tag=''):
        self.fig, self.feet, self.pairs, self.period = fig, feet, pairs, period
        self.blocks = blocks          # list of dicts: axes5, row, res, lines
        self.prefix_seconds, self.ymax = prefix_seconds, ymax
        self.initial_separation, self.anchor_mode = (initial_separation,
                                                     anchor_mode)
        self.trial, self.out_csv = trial, out_csv
        self.session_tag = session_tag
        self.state = {}               # (rep, bout) -> {'snug_abs', 'i1_abs'}
        self.drag = None              # (block_idx, 'snug'|'end') while held
        self.status = fig.text(0.01, 0.002, 'drag the green (gait start) or '
                               'purple (bout end) line; release to recompute. '
                               'Double-click left/right to see more time.',
                               fontsize=9, color='tab:red')
        from matplotlib.widgets import Button
        h = 0.30 / fig.get_figheight()          # button strip: fixed ~0.3 inch
        y = 0.10 / fig.get_figheight()
        self._buttons = []
        for x, w, label, cb in ((0.60, 0.185, 'save adjustments', self.save),
                                (0.80, 0.185, 'continue without saving',
                                 self.discard)):
            bax = fig.add_axes([x, y, w, h])
            btn = Button(bax, label)
            btn.label.set_fontsize(8)
            btn.on_clicked(cb)
            self._buttons.append(btn)
        self.status.set_position((0.01, y))
        for blk in self.blocks:
            self._add_lines(blk)
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('button_release_event', self._on_release)

    # -- drawing ------------------------------------------------------------
    def _add_lines(self, blk):
        res, row = blk['res'], blk['row']
        if res is None:
            blk['lines'] = {}
            return
        t_end = res['end_abs'] - res['t0_abs']
        pos = {'snug': 0.0, 'end': t_end * self.period}
        color = {'snug': 'tab:green', 'end': 'tab:purple'}
        blk['lines'] = {name: [ax.axvline(pos[name], color=color[name], lw=2.0,
                                          alpha=0.65, zorder=6)
                               for ax in blk['axes5'][2:]]
                        for name in self.GRAB}

    def _redraw_block(self, bi):
        blk = self.blocks[bi]
        rep_bout = (int(blk['row']['rep']), int(blk['row']['bout']))
        st = self.state.get(rep_bout, {})
        res = _render_inspect_block(
            blk['axes5'], self.feet, self.pairs, blk['row'], self.period,
            context_s=blk['context'], ymax=self.ymax, resnug=self.resnug,
            initial_separation=self.initial_separation,
            anchor_mode=self.anchor_mode,
            i1_abs=st.get('i1_abs'), snug_abs=st.get('snug_abs'),
            i0_abs=st.get('i0_abs'))
        blk['res'] = res
        blk['i1_abs'] = st.get('i1_abs')
        self._add_lines(blk)
        _decimate_lines(blk['axes5'], self.plot_every)
        self._shade_overview(blk)
        self.fig.canvas.draw_idle()

    def _extend_view(self, bi, event):
        """Double-click: show `extend_s` more context on the clicked side.

        Left of the view's centre widens the pre-walk side, right of it the
        post-walk side. View only - the bout's bounds are unchanged (drag a
        line into the new stretch to move them; dragging the gait start
        before the analysed slice re-cuts the bout from there).
        """
        blk = self.blocks[bi]
        lo, hi = event.inaxes.get_xlim()
        side = 0 if event.xdata < (lo + hi) / 2 else 1
        blk['context'][side] += self.extend_s
        where = 'before' if side == 0 else 'after'
        self._say(f'showing {blk["context"][side]:g} s {where} - redrawing...')
        self._redraw_block(bi)
        self._say(f'now showing {blk["context"][0]:g} s before / '
                  f'{blk["context"][1]:g} s after (view only; drag a line '
                  f'to change the bout)')

    def _shade_overview(self, blk):
        """(Re)shade one bout's current window on its rep's overview."""
        if blk.get('ov_ax') is None:
            return
        row = blk['row']
        key = (int(row['rep']), int(row['bout']))
        for art in blk.get('shade', []):
            art.remove()
        start, stop = _bout_window_s(row, self.state.get(key, {}), self.period)
        blk['shade'] = _shade_overview_bout(blk['ov_ax'], row, start, stop)

    def _say(self, msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    # -- events -------------------------------------------------------------
    def _find_block(self, ax):
        for bi, blk in enumerate(self.blocks):
            if ax in blk['axes5'][2:]:
                return bi
        return None

    def _on_press(self, event):
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        if getattr(self, '_discarded', False):
            return                          # 'continue without saving' froze it
        if getattr(self.fig.canvas.toolbar, 'mode', ''):
            self._say('zoom/pan tool is active - turn it off to drag')
            return
        bi = self._find_block(event.inaxes)
        if bi is None:
            return
        if event.dblclick:
            self.drag = None
            self._extend_view(bi, event)
            return
        if not self.blocks[bi]['lines']:
            return
        lines = self.blocks[bi]['lines']
        name = min(self.GRAB,
                   key=lambda n: abs(lines[n][0].get_xdata()[0] - event.xdata))
        self.drag = (bi, name)
        self._drag_from = lines[name][0].get_xdata()[0]
        self._say(f'dragging the {"gait start" if name == "snug" else "bout end"}'
                  f' - release to recompute')

    def _on_motion(self, event):
        if self.drag is None or event.inaxes is None or event.xdata is None:
            return
        bi, name = self.drag
        for ln in self.blocks[bi]['lines'][name]:
            ln.set_xdata([event.xdata, event.xdata])
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.drag is None:
            return
        bi, name = self.drag
        self.drag = None
        blk = self.blocks[bi]
        x = blk['lines'][name][0].get_xdata()[0]
        if abs(x - getattr(self, '_drag_from', np.nan)) < 1e-9:
            self._say('drag the green (gait start) or purple (bout end) line; '
                      'double-click left/right of centre to see '
                      f'{self.extend_s:g} s more on that side')
            return                         # a plain click is not an adjustment
        res, row = blk['res'], blk['row']
        t0_abs = res['t0_abs']
        rep_bout = (int(row['rep']), int(row['bout']))
        st = self.state.setdefault(rep_bout, {})
        new_abs = int(round(t0_abs + x / self.period))
        if self.resnug:
            self._release_resnug(bi, name, st, new_abs, rep_bout)
            return
        if name == 'snug':
            j0, j1 = res['slice']
            end_abs = st.get('i1_abs') or int(row['stop_s'] / self.period)
            st['snug_abs'] = int(min(max(new_abs, 0), end_abs - 1))
            extend = ''
            if st['snug_abs'] < j0:
                # dragged BEFORE the mechanized slice: re-cut the bout from
                # there (the old behaviour silently clamped to the slice edge)
                st['i0_abs'] = st['snug_abs']
                extend = ', slice extended back'
            moved = (st['snug_abs'] - t0_abs) * self.period
            self._say(f'rep {rep_bout[0]} bout {rep_bout[1]}: gait start moved '
                      f'{moved:+.2f} s (abs sample {st["snug_abs"]}{extend}) - '
                      f'recomputing... (axes re-zero to the new t=0)')
        else:
            snug_abs = st.get('snug_abs', t0_abs)
            st['i1_abs'] = max(new_abs, snug_abs + int(1.0 / self.period))
            self._say(f'rep {rep_bout[0]} bout {rep_bout[1]}: bout end moved to '
                      f'abs sample {st["i1_abs"]} '
                      f'({(st["i1_abs"] - snug_abs) * self.period:.1f} s after '
                      f'gait start) - recomputing...')
        self._redraw_block(bi)
        res = self.blocks[bi]['res']
        n = len(res['steps']['time']) if res is not None else 0
        snapped = ''
        if (name == 'snug' and res is not None
                and res['t0_abs'] != st.get('snug_abs')):
            # dropped in standing time: the pipeline starts the gait where
            # walking actually begins - save what is SHOWN, not the drop point
            snapped = (f' Gait start snapped '
                       f'{(res["t0_abs"] - st["snug_abs"]) * self.period:+.2f} s'
                       f' to the walking onset.')
            st['snug_abs'] = int(res['t0_abs'])
            self._shade_overview(self.blocks[bi])
        self._say(f'rep {rep_bout[0]} bout {rep_bout[1]} recomputed: {n} steps.'
                  f'{snapped} Press \'save adjustments\' to save'
                  f' (original clicks stay untouched in the aligned table)')

    def _release_resnug(self, bi, name, st, new_abs, rep_bout):
        """Drop with re-snugging on: the line moves the SEARCH window edge.

        Like moving a button press: the gait start/end are then re-found by
        the same snugging as every automatic bout, inside the new window.
        """
        row = self.blocks[bi]['row']
        start = st.get('i0_abs', int(row['start_s'] / self.period))
        end = st.get('i1_abs', int(row['stop_s'] / self.period))
        one_s = int(1.0 / self.period)
        if name == 'snug':
            st['i0_abs'] = int(min(max(new_abs, 0), end - one_s))
            what = 'window start'
        else:
            st['i1_abs'] = int(max(new_abs, start + one_s))
            what = 'window end'
        st.pop('snug_abs', None)
        self._say(f'rep {rep_bout[0]} bout {rep_bout[1]}: {what} moved - '
                  f're-snugging the gait inside it...')
        self._redraw_block(bi)
        res = self.blocks[bi]['res']
        if res is None:
            self._say(f'rep {rep_bout[0]} bout {rep_bout[1]}: the pipeline '
                      f'failed for this window - move the line back')
            return
        n = len(res['steps']['time'])
        g0 = (res['t0_abs'] - st.get('i0_abs', start)) * self.period
        dur = (res['end_abs'] - res['t0_abs']) * self.period
        note, _ = _stop_note(res, manual=True)
        self._say(f'rep {rep_bout[0]} bout {rep_bout[1]}: {n} steps; gait '
                  f'snugged to start {g0:+.2f} s into your window, lasting '
                  f"{dur:.1f} s{note}. Press 'save adjustments' to save")

    def save(self, _event=None):
        """Save adjusted bouts, replacing existing trial/rep/person entries.

        Writes the same (trial, rep, person, start_s, stop_s) rows that
        manual_correct() saves, so the notebook's apply_manual_rescore cell
        folds them in identically (stamping manual=True and the person). The
        original button-press bounds are NEVER modified — they stay in the
        aligned table / the .h5 annotations; this only records the edit.
        """
        if getattr(self, '_discarded', False):
            self._say('edits were discarded (continue without saving) - '
                      'nothing saved; reopen the trial to rescore it')
            return
        if not self.state:
            self._say('nothing adjusted yet - drag a line first')
            return
        recs = []
        for blk in self.blocks:
            rep_bout = (int(blk['row']['rep']), int(blk['row']['bout']))
            st = self.state.get(rep_bout)
            if not st:
                continue
            row = blk['row']
            person = str(row['walker'])[-1]
            # resnug: save the WINDOW (like button presses; snugging is
            # re-derived from it). Legacy: the pinned gait start.
            start_s = (st['i0_abs'] * self.period
                       if self.resnug and 'i0_abs' in st else
                       st['snug_abs'] * self.period if 'snug_abs' in st
                       else float(row['start_s']))
            stop_s = (st['i1_abs'] * self.period if 'i1_abs' in st
                      else float(row['stop_s']))
            recs.append({'trial': self.trial, 'rep': rep_bout[0],
                         'person': person, 'start_s': round(start_s, 3),
                         'stop_s': round(stop_s, 3)})
        if not _save_controller_rows(self, recs):
            self._say('No new changes to save.')
            return
        if self.on_save is not None:
            self._say(f'saved {len(recs)} adjusted bout(s) to '
                      f'{os.path.basename(self.out_csv)}{_run_on_save(self)}')
            return
        fig_note = ''
        src_file = next(iter(self.feet.values())).file_path
        if src_file:   # adjusted figure beside the batch ones, _manual suffix
            folder = os.path.join(os.path.dirname(os.path.abspath(src_file)),
                                  'figures', self.session_tag)
            os.makedirs(folder, exist_ok=True)
            reps = sorted({int(b['row']['rep']) for b in self.blocks})
            tag = f'_rep{reps[0]}' if len(reps) == 1 else ''
            fpath = os.path.join(folder, f'brock_{self.session_tag}_trial'
                                         f'{self.trial}{tag}_viz_manual.svg')
            self.fig.savefig(fpath)
            fig_note = f' + {os.path.basename(fpath)}'
        self._say(f'saved {len(recs)} adjusted bout(s) to {self.out_csv}'
                  f'{fig_note} - run the apply_manual_rescore cell to fold in')

    def discard(self, _event=None):
        """'continue without saving': drop this figure's edits, write nothing.

        The figure stays on screen (closing an ipympl view from inside its
        own callback is unreliable across browsers); it is frozen instead -
        further drags are ignored and saving is refused - and the next rerun
        of a scoring cell (or H) closes it without saving.
        """
        self._discarded = True
        self.drag = None
        self._say('edits discarded - nothing will be saved from this figure. '
                  'Carry on: rerun a scoring cell for the next trial.')

    close = discard   # historical name


def interactive_inspect_trial(feet, aligned, trial, session_tag='',
                              out_csv='brock_manual_rescore.csv',
                              prefix_seconds=None,
                              ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                    imu.STEPSPEED_YMAX),
                              initial_separation=0.2,
                              anchor_mode='firstonly', rep=None,
                              pad_s=5.0, plot_every=1, resnug=True,
                              neighbor_s=NEIGHBOR_S):
    """inspect_snipped_trial(), but with DRAGGABLE gait start and bout end.

    A foot-speed OVERVIEW sits above the bout blocks: every foot on one
    session-time axis from `neighbor_s` (60 s) before the first shown bout to
    `neighbor_s` after the last (person A black, B green; right foot
    dashed), each bout's current window shaded, and the NEIGHBOURING trials'
    bouts shaded in their own colours and labelled 'T2 b1 (A)' - so you see
    which walk is which. Each bout block also shows `pad_s` of grey
    context (raw |A| and foot speed) on both sides of the analysed walk;
    DOUBLE-CLICK a block's time axes left or right of centre to see
    `extend_s` (5 s) more on that side - view only, then drag a line out
    there to move the bout. A click that doesn't move a line changes
    nothing. `plot_every=N` plots only every N-th sample (faster redraws;
    plotting only - all computation stays full-rate).

    Pass rep=1 or rep=2 to display only that repetition. Omitting rep keeps
    the historical all-repetitions behavior for existing callers.

    Every bout block gets two grab-able vertical lines on its time axes: the
    green line is the SNUG gait start (t=0) and the purple line the BOUT END
    (initially the Stop press). Press the mouse near either (the closer one
    is grabbed), drag, release — the bout is re-run with the adjusted bounds
    and redrawn. Dragging the green line re-zeroes the time axis to the new
    gait start, so it snaps back onto t=0 after the recompute: read the
    status line at the bottom for what changed, and find the adjusted
    absolute sample indices in the returned controller's
    .state[(rep, bout)] = {'snug_abs': ..., 'i1_abs': ...}.

    Saving: the ORIGINAL button-press bounds are never modified. Press the
    'save adjustments' button (bottom strip) to update the adjusted bouts in
    `out_csv` — the same (trial, rep, person, start_s, stop_s) file that
    manual_correct() writes, so the notebook's apply_manual_rescore cell
    folds them into the aligned table identically, stamping each row
    manual=True with the person. No need to catch a return value for the
    edits (the figure lives on after this function returns; the CSV is the
    hand-off). 'continue without saving' discards this figure's edits (nothing
    is written; the figure is frozen and closes on the next scoring-cell rerun).

    `feet` and `aligned` are as for inspect_snipped_trial() (the dict of both
    subjects' foot recordings from load_available_feet(), and the per-bout
    table from align_snips_to_trial_table()).

    Needs an interactive matplotlib backend — run it in Jupyter (browser) or
    from the terminal; ensure_interactive_backend() raises setup instructions
    if the Python backend cannot load. The figure retains its controller;
    the return value also lets you inspect pending edits.
    """
    import matplotlib.pyplot as plt
    _inspect_rows(aligned, trial, rep)   # nothing to draw -> raise BEFORE
    mode = ensure_interactive_backend()  # switching backend; figure after it
    fig, blocks = _build_inspect_figure(
        feet, aligned, trial, rep=rep, session_tag=session_tag,
        prefix_seconds=prefix_seconds, ymax=ymax,
        initial_separation=initial_separation, anchor_mode=anchor_mode,
        pad_s=pad_s, plot_every=plot_every,
        interactive=True, resnug=resnug, neighbor_s=neighbor_s)
    pairs = feet_pairs_from_labels(feet)
    period = next(iter(feet.values())).period
    ctrl = _DraggableTrial(fig, feet, pairs, period, blocks, prefix_seconds,
                           ymax, initial_separation, anchor_mode,
                           trial=int(trial), out_csv=out_csv,
                           session_tag=session_tag)
    ctrl.plot_every = plot_every
    ctrl.resnug = resnug
    # Matplotlib callbacks use weak references; the figure owns its editor.
    fig._rescore_controller = ctrl
    if mode == 'script':
        plt.show(block=True)
    else:
        _display_interactive_figure(fig)
    return ctrl


INTERACTIVE_MAX_WIDTH_PX = 860   # fits JupyterLab with the file browser open


def _interactive_dpi(width_in):
    """dpi at which a `width_in`-wide figure fits the notebook output column.

    A 12.5 in figure at 100 dpi is 1250 px - wider than JupyterLab's output
    area on most screens, hiding the right-hand time panels (and the drag
    lines) behind a horizontal scrollbar. A lower dpi shrinks everything
    proportionally. Pass it when CREATING the figure: changing the dpi of an
    ipympl figure afterwards sends a resize before the view exists, and the
    figure can come up blank.
    """
    import matplotlib
    return min(matplotlib.rcParams['figure.dpi'],
               INTERACTIVE_MAX_WIDTH_PX / width_in)


def display_interactive_figure(fig):
    """Show an ipympl figure so it renders completely on first appearance.

    ipympl sends one full PNG, then only DIFF frames (changed pixels). The
    browser view clears its canvas when it initializes/resizes - which can
    happen after the kernel has already sent a frame (the kernel is often
    still busy running the rest of the cell) - and diffs painted onto that
    cleared canvas leave everything unchanged blank: a half-drawn figure
    that fills in only where you click. So every frame is sent FULL until
    the first mouse press in the figure; after that the view is settled and
    diffs are safe (and keep dragging responsive).
    """
    from IPython.display import display
    canvas = fig.canvas
    if hasattr(canvas, '_force_full'):          # ipympl / webagg canvas
        def _full(_event):
            canvas._force_full = True           # read right after the draw
        cids = [canvas.mpl_connect('draw_event', _full)]

        def _settled(_event):
            for cid in cids:
                canvas.mpl_disconnect(cid)
        cids.append(canvas.mpl_connect('button_press_event', _settled))
    canvas.draw()
    display(canvas)
    canvas.draw_idle()


_display_interactive_figure = display_interactive_figure


INTERACTIVE_HELP = """\
HOW TO GET INTERACTIVE PLOTS WORKING (read this if manual_correct errors out)

1. Most common cause: the notebook is running on THE WRONG PYTHON (a kernel
   that doesn't have this project's packages). ipympl lives in the project's
   .venv, so the kernel must be the .venv one:
     - VS Code: kernel picker (top-right of the notebook) > Python
       Environments > pick '.venv' (.../stride_estimation_imu/.venv/bin/python),
       then Restart the kernel and re-run from the top.
     - Jupyter in the browser: start it with `uv run jupyter lab` from the
       repo folder (that pins the right python automatically).
   The error message above prints which python the kernel is on - if it does
   not end in stride_estimation_imu/.venv/bin/..., this is your problem.
2. If the kernel IS the .venv one but ipympl is missing: run `uv sync` in a
   terminal from the repo folder, then Restart the kernel.
3. If the right kernel + ipympl still won't go interactive in VS Code, run in
   the BROWSER instead (reliable path):
       cd <this repo folder>
       uv run jupyter lab
   then open this notebook from the left sidebar.
4. If clicks do nothing: the figure toolbar's zoom/pan tool may be active -
   click the zoom icon to turn it off, then click in the plot again.
"""


def ensure_interactive_backend():
    """Make matplotlib interactive, or raise with plain-language fix-it steps.

    Returns 'notebook' (Jupyter + ipympl widget backend) or 'script' (a GUI
    backend under plain `python`). Raises RuntimeError with INTERACTIVE_HELP
    when the Python backend cannot be enabled, before building a figure.
    This cannot verify the browser widget connection; use the notebook check.
    """
    import matplotlib
    try:
        ip = get_ipython()   # only defined inside IPython/Jupyter
    except NameError:
        ip = None
    backend = matplotlib.get_backend().lower()
    if ip is not None:
        if 'widget' in backend or 'ipympl' in backend:
            return 'notebook'
        try:
            ip.run_line_magic('matplotlib', 'widget')
            return 'notebook'
        except Exception as e:
            venv = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), '.venv')
            venv_py = os.path.join(venv, 'bin', 'python')
            on_venv = os.path.abspath(sys.prefix) == os.path.abspath(venv)
            diagnosis = (
                'ipympl is missing from this kernel'
                if not on_venv else 'ipympl failed to load')
            raise RuntimeError(
                f"Could not switch to the interactive 'widget' backend: "
                f"{diagnosis} (error: {e}).\n"
                f"This kernel's python:  {sys.executable}\n"
                f"The project's python:  {venv_py}"
                + ('' if on_venv else '   <-- switch the kernel to THIS one')
                + '\n\n' + INTERACTIVE_HELP) from e
    if backend in ('agg', 'pdf', 'svg', 'ps', 'template', 'module://matplotlib_inline.backend_inline'):
        for cand in ('MacOSX', 'QtAgg', 'TkAgg'):
            try:
                matplotlib.use(cand, force=True)
                return 'script'
            except Exception:
                continue
        raise RuntimeError(
            "No interactive GUI backend is available in this python.\n\n"
            + INTERACTIVE_HELP)
    return 'script'


class _RescoreFigure:
    """One (trial, rep) inspection figure with click-to-rescore state.

    Three stacked time axes (raw |A|, mechanized foot speed, horizontal
    excursion) for every foot, the current bout windows shaded per person,
    and buttons: 'rescore A' / 'rescore B' arm a two-click capture (first
    click = new start, second = new stop) for that person's bout; 'save'
    appends the corrections to the CSV. Nothing is written until 'save'.
    Two text fields hold the displayed window [from, to] in session seconds;
    pressing Enter in either re-slices, re-runs the mechanization, redraws.
    """
    SUBJECT_COLOR = {'a': 'tab:blue', 'b': 'tab:orange'}
    on_save = None    # ManualScoring hook: called with the trial after a save

    def __init__(self, fig, axes, trial, rep, rows, out_csv, session_tag,
                 feet, period, n_samples, window, plot_every=1, context=None,
                 resnug=True):
        self.fig, self.axes = fig, axes
        self.resnug = resnug      # how the walked-distance readout snugs
        # orientation aids (see _draw_context): target distance, the other
        # trials' bouts, button presses, estimated position of missed bouts
        self.context = context or {}
        self.trial, self.rep = trial, rep
        self.rows = rows                       # person -> (start_s, stop_s) or None
        self.out_csv, self.session_tag = out_csv, session_tag
        self.feet, self.period, self.n_samples = feet, period, n_samples
        self.plot_every = max(1, int(plot_every or 1))   # plotting only
        self.pending = None                    # (person, [first_click_or_None])
        self.new = {}                          # person -> [start_s, stop_s]
        self.spans = {}
        self.boxes = {}                        # 'from'/'to' -> TextBox
        self.status = fig.text(0.01, 0.005, '', fontsize=9, color='tab:red')
        self.set_window(*window)
        fig.canvas.mpl_connect('button_press_event', self._on_click)

    def __repr__(self):
        bouts = {p: [round(float(v), 1) for v in w]
                 for p, w in self.rows.items() if np.all(np.isfinite(w))}
        fixed = {p: [round(float(v), 1) for v in w]
                 for p, w in self.new.items()}
        return (f"<Rescore trial {self.trial} rep {self.rep}: window "
                f"[{self.window[0]:.1f}, {self.window[1]:.1f}] s, bout spans "
                f"{bouts or '(none scored)'}, rescored {fixed or 'nothing'}>")

    def set_window(self, lo_s, hi_s):
        """Slice [lo_s, hi_s], mechanize every foot, (re)draw the three axes."""
        j0 = max(0, int(lo_s / self.period))
        j1 = min(self.n_samples, int(hi_s / self.period))
        if j1 - j0 < 10:
            self._say(f'window [{lo_s:.1f}, {hi_s:.1f}] s is empty or too '
                      f'short - not redrawn')
            return
        self.window = (j0 * self.period, j1 * self.period)
        t = np.arange(j0, j1) * self.period
        k = self.plot_every                    # plot-only downsampling
        for ax in self.axes:
            ax.clear()
        self.spans = {}                        # cleared with the axes
        for label, rec in sorted(self.feet.items()):
            person = label.rsplit('_', 1)[-1]
            color = self.SUBJECT_COLOR.get(person, '0.5')
            ls = '-' if label.startswith('left') else '--'
            self.axes[0].plot(t[::k], np.linalg.norm(rec.Ab[j0:j1:k], axis=1),
                              lw=0.5, color=color, ls=ls, label=label)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    traj = imu.compute_position(rec.Wb[j0:j1], rec.Ab[j0:j1],
                                                self.period)
                self.axes[1].plot(t[::k], traj.Vm[::k], lw=0.8, color=color,
                                  ls=ls)
                self.axes[2].plot(
                    t[::k], np.linalg.norm(traj.P[::k, :2] - traj.P[0, :2],
                                           axis=1),
                    lw=0.8, color=color, ls=ls)
            except Exception as e:
                self.axes[1].text(0.01, 0.9,
                                  f'{label}: mechanization failed ({e})',
                                  transform=self.axes[1].transAxes, fontsize=7)
        self.axes[0].set_ylabel('raw |A| [m/s²]')
        self.axes[0].legend(fontsize=7, ncol=4, loc='upper right')
        self.axes[1].set_ylabel('foot speed [m/s]')
        self.axes[2].set_ylabel('horiz. excursion [m]')
        self.axes[2].set_xlabel('session time [s]')
        self.axes[0].set_xlim(self.window)
        self._draw_context()
        for name, val in zip(('from', 'to'), self.window):
            if name in self.boxes:
                box = self.boxes[name]
                box.eventson = False           # set_val fires on_submit otherwise
                box.set_val(f'{val:.1f}')
                box.eventson = True
        self._draw_spans()

    def _draw_context(self):
        """What else is in view, so you can be sure WHICH walk you score:
        dashed target distance on the excursion panel; every OTHER trial's
        bout shaded in its trial's colour and labelled (T2 b1 (A)), dotted
        lines at its start/stop; Start/Stop presses as
        thin green/purple lines; a missed bout's ESTIMATED position (from
        its neighbours in the trial sequence) as a red dashed line."""
        ctx, (lo, hi) = self.context, self.window
        ax0, ax_ex = self.axes[0], self.axes[2]
        top = ax0.get_xaxis_transform()
        if ctx.get('target_m') is not None:
            ax_ex.axhline(ctx['target_m'], color='0.25', ls='--', lw=1.0,
                          label=f"target {ctx['target_m']:g} m")
            ax_ex.legend(fontsize=7, loc='upper left')
        shown = [o for o in ctx.get('others', []) if o[2] >= lo and o[1] <= hi]
        for k, ax in enumerate(self.axes):
            _draw_neighbor_bouts(ax, shown, label=(k == 0))
        for t_s, lab in ctx.get('presses', []):
            if lo <= t_s <= hi:
                col = '#1a7f37' if lab == 'Start' else '#8250df'
                for ax in self.axes:
                    ax.axvline(t_s, color=col, lw=0.7, alpha=0.6)
        for lab, t_s in ctx.get('estimates', []):
            if lo <= t_s <= hi:
                for ax in self.axes:
                    ax.axvline(t_s, color='tab:red', lw=1.2, ls='--')
                ax0.text(t_s, 0.85, f' {lab}\n (estimated)', transform=top,
                         fontsize=7, color='tab:red', va='top')

    def on_window_submit(self, _text=None):
        """Enter pressed in a window text field: parse both, redraw."""
        try:
            lo_s = float(self.boxes['from'].text)
            hi_s = float(self.boxes['to'].text)
        except ValueError:
            self._say(f"could not read the window fields as numbers "
                      f"('{self.boxes['from'].text}', '{self.boxes['to'].text}')")
            return
        if hi_s <= lo_s:
            self._say(f'window from {lo_s:.1f} to {hi_s:.1f} s is backwards - '
                      f'not redrawn')
            return
        if (lo_s, hi_s) == getattr(self, 'window', None):
            return                             # unchanged (e.g. focus-out echo)
        self._say(f'recomputing window [{lo_s:.1f}, {hi_s:.1f}] s ...')
        self.set_window(lo_s, hi_s)
        self._say(f'window set to [{self.window[0]:.1f}, '
                  f'{self.window[1]:.1f}] s')

    def _draw_spans(self):
        for person, artists in self.spans.items():
            for art in artists:
                art.remove()
        self.spans = {}
        for person in ('a', 'b'):
            win = self.new.get(person) or self.rows.get(person)
            if win is None or not np.all(np.isfinite(win)):
                continue
            color = self.SUBJECT_COLOR[person]
            arts = []
            for ax in self.axes:
                arts.append(ax.axvspan(win[0], win[1], color=color,
                                       alpha=0.30 if person in self.new else 0.12))
            self.spans[person] = arts
        self.fig.canvas.draw_idle()

    def arm(self, person):
        def cb(_event):
            self.pending = (person, [])
            self._say(f"rescoring person {person.upper()}: click the plot at "
                      f"the bout START (1st click), then the STOP (2nd click)")
        return cb

    def _say(self, msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if self.pending is None or event.inaxes not in self.axes:
            return
        if self.fig.canvas.toolbar is not None and \
                getattr(self.fig.canvas.toolbar, 'mode', ''):
            self._say("zoom/pan tool is active - click its toolbar icon to "
                      "turn it off, then click again")
            return
        person, clicks = self.pending
        clicks.append(float(event.xdata))
        if len(clicks) == 1:
            self._say(f"person {person.upper()} start = {clicks[0]:.1f} s - "
                      f"now click the STOP")
        else:
            t0, t1 = sorted(clicks)
            self.new[person] = [t0, t1]
            self.pending = None
            self._say(f"person {person.upper()} rescored to "
                      f"[{t0:.1f}, {t1:.1f}] s - computing...")
            self._draw_spans()
            self._say(f"person {person.upper()} rescored to "
                      f"[{t0:.1f}, {t1:.1f}] s{self._walked_note(person, t0, t1)}"
                      f" - press 'save' to keep it (or rescore again)")

    def _walked_note(self, person, t0, t1):
        """'; walked ~ X m (target Y m)' for a just-clicked window: the SAME
        pipeline the saved table/figures will run on it (snugged inside the
        window), so the number you see while scoring is the number you get."""
        try:
            pairs = feet_pairs_from_labels(self.feet)
            subject = {'a': 's1', 'b': 's2'}[person]
            with contextlib.redirect_stdout(io.StringIO()):
                res = bout_sync_strides_steps(
                    pairs, subject, int(t0 / self.period), int(t1 / self.period),
                    self.period, manual=True, resnug=self.resnug)
            walked = imu.overhead_travel(res)['mean']
            target = self.context.get('target_m')
            note = f'; walked \u2248 {walked:.1f} m'
            if target is not None:
                note += f' (target {target:g} m)'
            if not res.get('stop_found'):
                note += ' - no clear stop in this window'
            return note
        except Exception as e:                       # noqa: BLE001 - shown
            return f'; (walked distance not computed: {e})'

    def save(self, _event=None):
        if not self.new:
            self._say("nothing rescored yet - use the rescore buttons first")
            return
        recs = [{'trial': self.trial, 'rep': self.rep, 'person': person,
                 'start_s': round(win[0], 3), 'stop_s': round(win[1], 3)}
                for person, win in sorted(self.new.items())]
        if not _save_controller_rows(self, recs):
            self._say('No new changes to save.')
            return
        self._say(f"saved {len(recs)} correction(s) to "
                  f"{os.path.basename(self.out_csv)}{_run_on_save(self)}")


def manual_correct(feet, aligned, trials, out_csv='brock_manual_rescore.csv',
                   pad_s=10.0, neighbor_s=NEIGHBOR_S, session_tag='', plot_every=1, resnug=True):
    """Interactive inspection + click-to-resnip for a list of trials.

    For each requested (trial, rep) this brings up an inspection figure -
    raw |A|, the mechanized foot speed, and horizontal position over time for
    every foot, with the currently-scored bout windows shaded (person A blue,
    person B orange; a missing/missed bout simply has no shading). Buttons:

      * 'rescore A' / 'rescore B' - then click the plot twice: first click is
        the new bout START, second the new STOP, for that person only (usually
        just one person needs fixing, so each is prompted separately);
      * 'save' - replace the rescored trial/rep/person rows in `out_csv`
        (columns trial, rep, person, start_s, stop_s). Nothing is written
        until you press save, so a stray click never corrupts the file.

    `plot_every=N` plots only every N-th sample (faster redraws; plotting
    only - the mechanization and saved windows stay full-rate).

    `trials` is a list of trial numbers (both reps shown) and/or (trial, rep)
    tuples. `feet` is the matched foot recordings of BOTH subjects, keyed by
    sensor label ('left_foot_a' ... 'right_foot_b') as returned by
    load_available_feet(); `aligned` is the per-bout results table from
    align_snips_to_trial_table() (start_s/stop_s = the snip bounds in
    session seconds). The inspection window covers
    both bouts of the rep plus max(`pad_s`, `neighbor_s`) on each side
    (neighbor_s default 60 s: usually enough to show the previous and next
    trials, whose bouts are shaded in their own colours and labelled
    'T2 b1 (A)'; type a narrower from/to window to zoom in, or pass
    neighbor_s=0 for the old tight window); missed bouts (no snip)
    get their window from the time-interpolated bout position, so there is
    always something to look at.

    Requires an interactive matplotlib backend - ensure_interactive_backend()
    raises a message with setup instructions (see INTERACTIVE_HELP) BEFORE any
    figure comes up if the Python backend cannot load. Browser widget
    connectivity must be checked separately. Returns the list of
    figure controllers. Each figure retains its controller so callbacks remain
    alive; keeping the returned list lets you inspect the pending edits.
    In Jupyter this returns BEFORE you click: it cannot return future edits.
    Save writes the edits to CSV; apply_manual_rescore() explicitly applies
    saved edits to a table copy. It never mutates the input aligned table.
    """
    mode = ensure_interactive_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    period = next(iter(feet.values())).period
    n_samples = min(len(rec) for rec in feet.values())
    time_s, _missed = imu.assign_bout_times(aligned)
    aligned = aligned.assign(_est_t=time_s)
    typical = np.nanmedian((aligned['stop_s'] - aligned['start_s']).to_numpy(float))
    presses = []
    try:                                  # button presses, for orientation
        ev = load_events(next(iter(feet.values())).file_path)
        presses = list(zip(ev['time_s'].tolist(), ev['label'].tolist()))
    except Exception:
        pass

    wanted = []
    for item in trials:
        if isinstance(item, (tuple, list)):
            wanted.append((int(item[0]), int(item[1])))
        else:
            wanted.extend([(int(item), 1), (int(item), 2)])

    controllers = []
    for trial, rep in wanted:
        group = aligned[(aligned['trial'] == trial) & (aligned['rep'] == rep)]
        if not len(group):
            print(f'trial {trial} rep {rep}: not in the aligned table - skipped')
            continue
        # person -> current window (NaN start = missed bout, no shading)
        rows = {}
        for _, row in group.iterrows():
            # missed bouts carry '' or NaN in walker - they get no shading
            walker = str(row['walker']) if pd.notna(row['walker']) else ''
            person = walker[-1] if walker else None
            win = [row['start_s'], row['stop_s']]
            if person in ('a', 'b'):
                rows[person] = win
        lo = np.nanmin(np.r_[group['start_s'].to_numpy(float),
                             group['_est_t'].to_numpy(float) - typical])
        hi = np.nanmax(np.r_[group['stop_s'].to_numpy(float),
                             group['_est_t'].to_numpy(float) + typical])

        # built under ioff() and displayed explicitly at the end: relying on
        # ipympl auto-show fails when the backend was switched mid-cell (the
        # figure simply never appears - same fix as interactive_inspect_trial)
        with plt.ioff():
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                     dpi=_interactive_dpi(12))
        fig.subplots_adjust(bottom=0.16, hspace=0.08)
        fig.suptitle(f'{session_tag} trial {trial} rep {rep} '
                     f'(target {group["distance_m"].iloc[0]:g} m) - inspect / rescore\n'
                     f'A blue, B orange; other trials coloured + labelled; '
                     f'missed bouts: red dashed = estimated position; '
                     f'thin green/purple = Start/Stop presses', fontsize=10)

        context = {
            'target_m': float(group['distance_m'].iloc[0]),
            'others': _neighbor_bouts(aligned, trial, rep, -np.inf, np.inf),
            'presses': presses,
            'estimates': [(f"T{trial} r{rep} b{int(b)}", float(t))
                          for b, t in zip(group.loc[group['start_s'].isna(), 'bout'],
                                          group.loc[group['start_s'].isna(), '_est_t'])],
        }
        ctrl = _RescoreFigure(fig, list(axes), trial, rep, rows, out_csv,
                              session_tag, feet, period, n_samples,
                              window=(lo - max(pad_s, neighbor_s),
                                      hi + max(pad_s, neighbor_s)),
                              plot_every=plot_every, context=context,
                              resnug=resnug)
        # buttons + window text fields along the bottom
        slots = [('rescore A', ctrl.arm('a')), ('rescore B', ctrl.arm('b')),
                 ('save', ctrl.save)]
        ctrl._buttons = []
        for k, (label, cb) in enumerate(slots):
            bax = fig.add_axes([0.06 + 0.14 * k, 0.03, 0.11, 0.055])
            btn = Button(bax, label)
            btn.on_clicked(cb)
            ctrl._buttons.append(btn)
        for k, name in enumerate(('from', 'to')):
            tax = fig.add_axes([0.60 + 0.17 * k, 0.03, 0.10, 0.055])
            box = TextBox(tax, f'{name} [s] ', textalignment='center',
                          initial=f'{ctrl.window[k]:.1f}')
            box.on_submit(ctrl.on_window_submit)
            ctrl.boxes[name] = box
        fig._rescore_controller = ctrl  # keep weak-reference callbacks alive
        controllers.append(ctrl)
        if mode == 'script':
            print(f'trial {trial} rep {rep}: close the window to move on '
                  f'(save first if you rescored)')
            plt.show(block=True)
    if mode == 'notebook' and controllers:
        for ctrl in controllers:
            _display_interactive_figure(ctrl.fig)
        print(f'{len(controllers)} figure(s) above. Rescore with the buttons, '
              f'then press save on each figure you changed; corrections land '
              f'in {out_csv}.')
    return controllers


def apply_manual_rescore(aligned, csv_path):
    """Fold saved manual corrections back into an aligned table copy.

    Window-only for backward compatibility: use refresh_manual_distances()
    afterwards to update measured distances, or plot_omnibus() to refresh/plot.

    Reads `csv_path` (trial, rep, person, start_s, stop_s - as written by
    manual_correct; the LAST correction wins when a bout was rescored twice)
    and overwrites start_s/stop_s/duration_s of the matching rows: same trial
    and rep, walker suffix == person. A correction for a MISSED bout (walker
    is NaN) fills the first unmatched row of that trial-rep instead, marking
    walker 'manual_<person>'. A record with EMPTY start/stop unmatches the
    bout (it becomes missed; see ManualScoring.unmatch). Adds a boolean
    'manual' column.
    """
    aligned = _upgrade_columns(aligned)
    aligned['manual'] = False
    if not os.path.exists(csv_path):
        return aligned
    fixes = pd.read_csv(csv_path).drop_duplicates(
        subset=['trial', 'rep', 'person'], keep='last')
    for _, fx in fixes.iterrows():
        sel = (aligned['trial'] == fx['trial']) & (aligned['rep'] == fx['rep'])
        walk = aligned.loc[sel, 'walker'].astype(str)
        hit = sel & walk.str.endswith(str(fx['person'])).reindex(
            aligned.index, fill_value=False)
        if not hit.any():   # missed bouts carry '' or NaN in walker
            blank = aligned['walker'].isna() | aligned['walker'].astype(str).eq('')
            hit = sel & blank
        idx = aligned.index[hit]
        if not len(idx):
            print(f"rescore trial {fx['trial']} rep {fx['rep']} person "
                  f"{fx['person']}: no matching aligned row - ignored")
            continue
        i = idx[0]
        if pd.isna(fx['start_s']) and pd.isna(fx['stop_s']):
            # "this bout did not happen": back to a MISSED bout. Everything
            # measured from the (wrong) snip goes; the person label stays so
            # the row can be addressed again.
            for col in ('snip', 'start_s', 'stop_s', 'duration_s',
                        'align_max_m', 'align_error_m', 'reaction_s',
                        'gait_start_s', 'gait_end_s', 'gait_duration_s',
                        'end_extended_s', 'walked_m', 'walked_error_m'):
                if col in aligned:
                    aligned.loc[i, col] = np.nan
            for col in ('inferred', 'jumped_gun'):
                if col in aligned:
                    aligned.loc[i, col] = False
            for col in ('stop_found', 'came_back'):
                if col in aligned:
                    aligned.loc[i, col] = pd.NA
            aligned.loc[i, 'manual'] = True
            if pd.isna(aligned.loc[i, 'walker']) or str(aligned.loc[i, 'walker']) == '':
                aligned.loc[i, 'walker'] = f"manual_{fx['person']}"
            continue
        aligned.loc[i, ['start_s', 'stop_s']] = fx['start_s'], fx['stop_s']
        aligned.loc[i, 'duration_s'] = fx['stop_s'] - fx['start_s']
        aligned.loc[i, 'manual'] = True
        if pd.isna(aligned.loc[i, 'walker']) or str(aligned.loc[i, 'walker']) == '':
            aligned.loc[i, 'walker'] = f"manual_{fx['person']}"
    return aligned


def refresh_manual_distances(feet, aligned, pad_seconds=1.0, cache=None):
    """Return a copy with manual-window distances remeasured from the IMUs.

    apply_manual_rescore intentionally retains its historical window-only
    behavior. Call this afterwards when saving/plotting corrected distances.
    Use the alignment's original metric (maximum horizontal excursion with
    one second of padding), restricted to the manually identified walker.
    Trial assignments and original snip IDs remain unchanged; this does not
    rerun sequence alignment or compute per-step/two-foot gait statistics.
    `cache` (optional dict) memoizes distances by (person, start, stop), so
    repeated refreshes only mechanize windows that changed.
    """
    refreshed = _upgrade_columns(aligned)
    manual = refreshed.get('manual', pd.Series(False, index=refreshed.index)).fillna(False)
    for idx in refreshed.index[manual.astype(bool)]:
        row = refreshed.loc[idx]
        start, stop = float(row.start_s), float(row.stop_s)
        if np.isnan(start) and np.isnan(stop):
            continue                      # unmatched by hand: nothing to measure
        if not (np.isfinite(start) and np.isfinite(stop) and 0 <= start < stop):
            raise ValueError(f'Invalid manual window at row {idx}: {start}, {stop}')
        person = str(row.walker).rsplit('_', 1)[-1]
        recordings = {k: v for k, v in feet.items() if k.endswith('_' + person)}
        if not recordings:
            raise ValueError(f'No foot recordings for manual walker {row.walker!r}')
        key = (person, start, stop, pad_seconds)
        if cache is not None and key in cache:
            distance = cache[key]
        else:
            distance = snip_distances(recordings, [[start, stop]],
                                      pad_seconds=pad_seconds).iloc[0].align_max_m
            if cache is not None:
                cache[key] = distance
        if not np.isfinite(distance):
            raise ValueError(f'Could not measure manual window at row {idx}')
        refreshed.loc[idx, 'align_max_m'] = distance
        refreshed.loc[idx, 'duration_s'] = stop - start
        refreshed.loc[idx, 'align_error_m'] = distance - row.distance_m
        # A manual gait boundary is no longer the original Start button cue.
        if 'reaction_s' in refreshed:
            refreshed.loc[idx, 'reaction_s'] = np.nan
        if 'jumped_gun' in refreshed:
            refreshed.loc[idx, 'jumped_gun'] = False
    return refreshed


def plot_omnibus(feet, events, result, aligned=None, report=None,
                  refresh_distances=True, **plot_kwargs):
    """Plot button presses and bout distances with optional scoring overrides.

    result is the unchanged dict returned by align_snips_to_trial_table.
    Omit aligned for the automatic baseline; supply a corrected table for
    overrides. Manual distances are refreshed by default. Returns
    (figure, axes, plotted_table); inputs are never modified and nothing is
    saved. Set refresh_distances=False only for an already-refreshed table.
    Uses draw_event_timeline, whose existing API remains unchanged.
    """
    table = _upgrade_columns(result['aligned'] if aligned is None else aligned)
    if refresh_distances:
        table = refresh_manual_distances(feet, table)
    # The legacy renderer counts matched snip IDs. A recovered manual bout
    # has a real window but no original snip: mark it only in the plot copy.
    display_table = table.copy(deep=True)
    recovered = (table.get('manual', pd.Series(False, index=table.index)).fillna(False)
                 & table.start_s.notna() & table.stop_s.notna() & table.snip.isna())
    display_table.loc[recovered, 'snip'] = -1
    fig, axes = imu.draw_event_timeline(display_table, events, report, **plot_kwargs)
    fig.suptitle('Omnibus — ' + ('automatic windows' if aligned is None
                                else 'with saved window adjustments'),
                 fontsize=10, y=0.995)
    return fig, axes, table


def delete_manual_windows(csv_path, pairs):
    """Drop every saved correction for the given (trial, rep) pairs.

    Returns the number of rows removed. Used to put a bout back to its
    automatic bounds before rescoring it from scratch.
    """
    if not os.path.exists(csv_path):
        return 0
    old = pd.read_csv(csv_path)
    drop = pd.Series([(int(t), int(r)) in set(pairs)
                      for t, r in zip(old['trial'], old['rep'])], index=old.index)
    if drop.any():
        old[~drop].to_csv(csv_path, index=False)
    return int(drop.sum())


def corrected_alignment(feet, auto_aligned, rescore_csv, cache=None,
                        resnug=True):
    """The automatic table with every saved manual correction applied.

    apply_manual_rescore() + refresh_manual_distances() + add_gait_timing():
    corrected windows, remeasured distances, a boolean 'manual' column, and
    gait_start_s/gait_end_s/gait_duration_s. By default (resnug=True) a
    manual window is snugged exactly like an automatic bout's button presses,
    so timing is comparable across both - this also re-snugs corrections
    saved before re-snugging existed. resnug=False keeps each manual window
    as the gait itself. Automatic rows keep their timing (computed here only
    if the table has none yet). `auto_aligned` is not modified.
    """
    out = refresh_manual_distances(
        feet, apply_manual_rescore(auto_aligned, rescore_csv), cache=cache)
    only = None if 'gait_start_s' not in auto_aligned else out['manual']
    return add_gait_timing(feet, out, only=only, resnug=resnug, cache=cache)


def manual_changes(auto_aligned, corrected):
    """One row per corrected bout: new window/distance next to the automatic one.

    walked_m / automatic_walked_m are the distance as the figures draw it
    (walked_change_m = the difference); align_max_m / automatic_align_max_m
    are the alignment feature (padded max excursion). automatic_* is NaN for
    a missed bout that was recovered by hand.
    """
    auto_aligned, corrected = _upgrade_columns(auto_aligned), _upgrade_columns(corrected)
    cols = ['trial', 'rep', 'bout', 'walker', 'start_s', 'stop_s', 'align_max_m']
    out = corrected.loc[corrected['manual'], cols].copy()
    out['automatic_align_max_m'] = auto_aligned.loc[out.index, 'align_max_m']
    out['align_change_m'] = out['align_max_m'] - out['automatic_align_max_m']
    if 'gait_duration_s' in corrected:      # snugged gait timing, both tables
        out['gait_start_s'] = corrected.loc[out.index, 'gait_start_s']
        out['gait_duration_s'] = corrected.loc[out.index, 'gait_duration_s']
        out['automatic_gait_s'] = (auto_aligned.loc[out.index, 'gait_duration_s']
                                   if 'gait_duration_s' in auto_aligned
                                   else np.nan)
    if 'walked_m' in corrected:            # distance as the figures draw it
        out['walked_m'] = corrected.loc[out.index, 'walked_m']
        out['automatic_walked_m'] = (auto_aligned.loc[out.index, 'walked_m']
                                     if 'walked_m' in auto_aligned else np.nan)
        out['walked_change_m'] = out['walked_m'] - out['automatic_walked_m']
    if 'stop_found' in corrected:
        out['stop_found'] = corrected.loc[out.index, 'stop_found']
    return out


class ManualScoring:
    """Section G's scoring session: open editors, and keep results saved.

    Every Save press in an editor (and every rerun of a scoring cell, which
    saves the open editors first) immediately:
      1. updates the shared corrections file `rescore_csv` (one row per
         trial/rep/person - a rescore REPLACES the old row);
      2. rewrites the complete corrected table `manual_aligned_csv`
         (the automatic table + all corrections, distances remeasured);
      3. re-exports that trial's figure as
         figures/<session_tag>/brock_<session_tag>_trial<N>_rep<R>_viz_manual.svg
         for each corrected rep (beside the automatic ..._viz.svg, which is
         never touched).
    So there is no separate 'save results' step to forget. `.corrected` is
    always the latest corrected table.

        scoring = ManualScoring(feet, res['aligned'], RESCORE_CSV,
                                MANUAL_ALIGNED_CSV, SESSION_TAG)
        scoring.drag([(4, 1)])      # draggable gait start / bout end
        scoring.click([(37, 2)])    # click a new start/stop (missed bouts)

    Opening a (trial, rep) that already has saved corrections NEVER prompts
    (an input() box used to block the cell - in VS Code it is easy to miss,
    so the cell looked like it ran forever). By default the editor opens
    showing the saved corrections; fix them again as often as you like -
    Save replaces only the person(s) you changed. `on_existing='clear'`
    (or scoring.clear(pairs)) puts a rep back to automatic first;
    scoring.clear_all() starts the whole session from a clean state.

    resnug=True (default): a manual window is treated like button presses -
    the gait start/end are snugged inside it exactly as for automatic bouts
    (bout_sync_strides_steps), in the table, the figures and the drag editor.
    """

    ON_EXISTING = ('rescore', 'skip', 'clear')

    def __init__(self, feet, auto_aligned, rescore_csv, manual_aligned_csv,
                 session_tag, previous=None, resnug=True):
        if previous is not None:        # re-created: save + close its editors
            previous.close_editors()
        self.feet, self.auto = feet, auto_aligned
        self.resnug = resnug
        self.period = next(iter(feet.values())).period
        self.rescore_csv, self.manual_aligned_csv = (rescore_csv,
                                                     manual_aligned_csv)
        self.session_tag = session_tag
        self.editors = []
        self._cache = {}
        self.corrected = None
        self.sync()

    def __repr__(self):
        return (f'<ManualScoring {self.session_tag}: '
                f'{int(self.corrected["manual"].sum())} corrected bouts in '
                f'{self.manual_trials()}, {len(self.editors)} open editor(s)>')

    # -- saving ---------------------------------------------------------------
    def manual_trials(self):
        return sorted(self.corrected.loc[self.corrected['manual'], 'trial']
                      .astype(int).unique().tolist())

    def saved_pairs(self):
        """(trial, rep) pairs with at least one saved correction."""
        if not os.path.exists(self.rescore_csv):
            return set()
        df = pd.read_csv(self.rescore_csv)
        return {(int(t), int(r)) for t, r in zip(df['trial'], df['rep'])}

    def figure_path(self, trial, rep):
        src = next(iter(self.feet.values())).file_path
        folder = os.path.join(os.path.dirname(os.path.abspath(src)), 'figures',
                              self.session_tag)
        return os.path.join(folder, f'brock_{self.session_tag}_trial'
                                    f'{int(trial)}_rep{int(rep)}_viz_manual.svg')

    def sync(self, trials=()):
        """Re-apply the corrections file; rewrite the table + `trials`' figures."""
        self.corrected = corrected_alignment(self.feet, self.auto,
                                             self.rescore_csv, cache=self._cache,
                                             resnug=self.resnug)
        self.corrected.to_csv(self.manual_aligned_csv, index=False)
        c = self.corrected
        manual = {(int(t), int(r)) for t, r in
                  zip(c.loc[c['manual'], 'trial'], c.loc[c['manual'], 'rep'])}
        if not next(iter(self.feet.values())).file_path:
            return c
        for trial in trials:
            for rep in (1, 2):              # only corrected reps get a figure
                path = self.figure_path(trial, rep)
                if (int(trial), rep) in manual:
                    save_trial_figures(self.feet, c, self.session_tag,
                                       resnug=self.resnug,
                                       suffix='viz_manual',
                                       trials=[(trial, rep)])
                elif os.path.exists(path):  # corrections cleared
                    os.remove(path)
        return c

    def _on_save(self, trial):
        self.sync(trials=[trial])
        return (f'; table + trial {trial} figure(s) updated '
                f'({int(self.corrected["manual"].sum())} corrected bouts)')

    def close_editors(self):
        """Save every open editor's finished edits, then close its figure."""
        import matplotlib.pyplot as plt
        for ed in self.editors:
            pend = getattr(ed, 'pending', None)      # (person, [clicks]) in G3
            if (pend and len(pend) > 1 and pend[1]   # armed AND a start clicked
                    and not getattr(ed, '_discarded', False)):
                raise RuntimeError(
                    f'trial {ed.trial}: a start was clicked without a stop - '
                    f'finish (or re-press rescore) before rerunning')
        for ed in self.editors:
            if not getattr(ed, '_discarded', False):
                ed.save()
            try:
                ed.fig.canvas.close()
            except Exception:
                pass
            plt.close(ed.fig)
        self.editors = []

    # -- opening --------------------------------------------------------------
    @staticmethod
    def _check_pairs(pairs):
        """[(trial, rep), ...] and/or bare trial numbers (= both reps)."""
        def is_int(v):
            return isinstance(v, (int, np.integer)) and not isinstance(v, bool)
        if not isinstance(pairs, (list, tuple)):
            pairs = None
        out = []
        for p in pairs or []:
            if is_int(p) and p >= 1:
                out += [(int(p), 1), (int(p), 2)]
            elif (isinstance(p, (list, tuple)) and len(p) == 2
                  and all(is_int(v) for v in p) and p[0] >= 1
                  and p[1] in (1, 2)):
                out.append((int(p[0]), int(p[1])))
            else:
                pairs = None
                break
        if pairs is None:
            raise ValueError('Use trial numbers and/or (trial, repetition) '
                             'pairs, e.g. [1, (4, 1), (5, 2)], or [] to just '
                             'save and close.')
        return list(dict.fromkeys(out))

    def _resolve_existing(self, pairs, on_existing):
        if on_existing == 'ask':           # old notebooks: never block on input()
            on_existing = 'rescore'
        if on_existing not in self.ON_EXISTING:
            raise ValueError(f'on_existing must be one of {self.ON_EXISTING}')
        done = [p for p in pairs if p in self.saved_pairs()]
        if not done:
            return pairs
        if on_existing == 'rescore':
            saved = pd.read_csv(self.rescore_csv)
            for t, r in done:
                who = sorted(saved.loc[(saved['trial'] == t) & (saved['rep'] == r),
                                       'person'].astype(str).str.upper().unique())
                print(f'trial {t} rep {r}: opening WITH your saved correction(s) '
                      f'for person {"/".join(who)}. Adjust again and Save to '
                      f'replace (only the person you change is replaced). To '
                      f'go back to automatic: scoring.clear([({t}, {r})])')
            return pairs
        if on_existing == 'skip':
            print('skipping ' + ', '.join(f'{t}:{r}' for t, r in done))
            return [p for p in pairs if p not in done]
        if on_existing == 'clear':
            n = delete_manual_windows(self.rescore_csv, done)
            self.sync(trials=sorted({t for t, _ in done}))
            print(f'cleared {n} saved correction(s); reopening from the '
                  f'automatic bounds')
        return pairs

    def _check_bout(self, trial, rep, person):
        trial, rep, person = int(trial), int(rep), str(person).lower()
        if person not in ('a', 'b') or rep not in (1, 2):
            raise ValueError("person must be 'a' or 'b', rep 1 or 2")
        a = self.auto
        if not ((a['trial'] == trial) & (a['rep'] == rep)).any():
            raise ValueError(f'trial {trial} rep {rep} is not in the table')
        return trial, rep, person

    def _report(self, trial, rep, person, verb):
        c = self.corrected
        sel = (c['trial'] == trial) & (c['rep'] == rep) & \
              c['walker'].astype(str).str.endswith(person)
        if not sel.any():
            print(f'{verb} trial {trial} rep {rep} person {person.upper()}')
            return None
        r = c[sel].iloc[0]
        if pd.isna(r['start_s']):
            print(f'{verb} trial {trial} rep {rep} person {person.upper()}: '
                  f'now a MISSED bout (no window)')
        else:
            print(f'{verb} trial {trial} rep {rep} person {person.upper()}: '
                  f"window {r['start_s']:.2f}-{r['stop_s']:.2f} s -> walked "
                  f"{r.get('walked_m', float('nan')):.2f} m (target "
                  f"{r['distance_m']:g} m), gait {r.get('gait_duration_s', float('nan')):.1f} s"
                  + ('' if r.get('stop_found', True) is not False else
                     ', NO clear stop in that window'))
        return r

    def set_window(self, trial, rep, person, start, stop, units='s'):
        """Type a bout's window - the full override, no clicking.

        `start`/`stop` in session seconds (units='s') or as sample indices
        (units='samples', the "absolute sample index" axis on the figures).
        Saved like a Save press: the gait is snugged inside the window by the
        same rule as every automatic bout, the table and the rep's figure are
        rewritten. Works for matched AND missed bouts, and for a bout the
        alignment attached to the wrong walk (see also unmatch()).
        """
        trial, rep, person = self._check_bout(trial, rep, person)
        if units == 'samples':
            start, stop = start * self.period, stop * self.period
        elif units != 's':
            raise ValueError("units must be 's' or 'samples'")
        start, stop = float(start), float(stop)
        if not (np.isfinite(start) and np.isfinite(stop) and 0 <= start < stop):
            raise ValueError(f'need 0 <= start < stop, got {start}, {stop}')
        self.close_editors()
        save_manual_windows(self.rescore_csv, [dict(
            trial=trial, rep=rep, person=person,
            start_s=round(start, 3), stop_s=round(stop, 3))])
        self.sync(trials=[trial])
        return self._report(trial, rep, person, 'set')

    def unmatch(self, trial, rep, person):
        """Declare that this bout did NOT happen (or that the alignment
        attached the wrong walk to it): it becomes a missed bout - window,
        distances and gait timing cleared, `manual` = True. Undo with
        clear([(trial, rep)]) or by set_window()."""
        trial, rep, person = self._check_bout(trial, rep, person)
        self.close_editors()
        save_manual_windows(self.rescore_csv, [dict(
            trial=trial, rep=rep, person=person, start_s=np.nan, stop_s=np.nan)])
        self.sync(trials=[trial])
        return self._report(trial, rep, person, 'unmatched')

    def clear(self, pairs):
        """Delete the saved corrections of these (trial, rep) pairs (bare
        trial numbers = both reps): back to the automatic windows, table and
        figures updated. Open editors are saved and closed first."""
        pairs = self._check_pairs(pairs)
        self.close_editors()
        n = delete_manual_windows(self.rescore_csv, pairs)
        self.sync(trials=sorted({t for t, _ in pairs}))
        print(f'cleared {n} saved correction(s) for '
              + ', '.join(f'trial {t} rep {r}' for t, r in pairs))
        return n

    def clear_all(self):
        """Start from a CLEAN state: no manual edits at all. The corrections
        file is not deleted but renamed to <name>.bak-<timestamp> (so a
        mistake can be undone by renaming it back); every _viz_manual.svg is
        removed. The automatic table (tied only to the raw data and the
        button presses) is never touched by manual scoring."""
        import datetime
        self.close_editors()
        trials = sorted({t for t, _ in self.saved_pairs()})
        backup = None
        if os.path.exists(self.rescore_csv):
            backup = (f'{self.rescore_csv}.bak-'
                      f'{datetime.datetime.now():%Y%m%d-%H%M%S}')
            os.replace(self.rescore_csv, backup)
        self.sync(trials=trials)
        removed = 0                       # stale figures (also when the file was
        if next(iter(self.feet.values())).file_path:   # moved aside already)
            import glob
            folder = os.path.dirname(self.figure_path(1, 1))
            for f in glob.glob(os.path.join(
                    folder, f'brock_{self.session_tag}_trial*_rep*_viz_manual.svg')):
                os.remove(f)
                removed += 1
        print('clean state: no manual corrections'
              + (f' (previous corrections kept as {backup})' if backup else '')
              + (f'; removed {removed} _viz_manual figure(s)' if removed else ''))
        return backup

    def _open(self, pairs, open_one, on_existing):
        pairs = self._check_pairs(pairs)
        self.close_editors()            # saves them -> table/figures updated
        if not pairs:
            print(f'Saved and closed. {self!r}')
            return self.editors
        for trial, rep in self._resolve_existing(pairs, on_existing):
            try:
                for ed in open_one(trial, rep):
                    ed.on_save = self._on_save
                    self.editors.append(ed)
            except ValueError as e:
                print(f'Skipped trial {trial}, rep {rep}: {e}')
        print(f'Each Save updates {os.path.basename(self.manual_aligned_csv)}'
              f' and the rep\'s _viz_manual.svg. Rerunning either scoring '
              f'cell saves these figures\' finished edits first.')
        return self.editors

    def drag(self, pairs, on_existing='rescore', **kwargs):
        """Open the draggable inspector for each (trial, rep)."""
        return self._open(pairs, lambda t, r: [interactive_inspect_trial(
            self.feet, self.corrected, trial=t, rep=r,
            session_tag=self.session_tag, out_csv=self.rescore_csv,
            resnug=self.resnug, **kwargs)],
            on_existing)

    def click(self, pairs, on_existing='rescore', **kwargs):
        """Open the click-to-rescore figure for each (trial, rep)."""
        return self._open(pairs, lambda t, r: manual_correct(
            self.feet, self.corrected, [(t, r)],
            session_tag=self.session_tag, out_csv=self.rescore_csv,
            resnug=self.resnug, **kwargs),
            on_existing)
