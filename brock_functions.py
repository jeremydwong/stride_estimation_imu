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


def automatically_score_movements_from_events(events, bounding_window_s=30.0,
                                              min_bout_s=1.5):
    """Pair Start/Stop button presses into scored movement bouts.

    Walks the annotation stream in time order holding at most one pending
    Start. A Stop within `bounding_window_s` of the pending Start closes a
    bout; a later Start before any Stop replaces the pending one (a restart /
    double-press); a Stop with no viable Start is an orphan. Bouts shorter
    than `min_bout_s` are kept but flagged (likely accidental presses).

    Returns (snips, report):
      snips  - (n, 2) float array of [start_s, stop_s] per accepted bout
      report - dict of anomaly times: 'restarts' (superseded Starts),
               'expired_starts' (no Stop within the window),
               'orphan_stops', and 'short' (row indices into snips)
    """
    order = np.argsort(events['time_s'])
    t, lab = events['time_s'][order], events['label'][order]
    snips, restarts, expired, orphans = [], [], [], []
    pending = None
    for ti, li in zip(t, lab):
        if li == 'Start':
            if pending is not None:
                restarts.append(pending)
            pending = ti
        elif li == 'Stop':
            if pending is None:
                orphans.append(ti)
            elif ti - pending > bounding_window_s:
                expired.append(pending)
                orphans.append(ti)
                pending = None
            else:
                snips.append((pending, ti))
                pending = None
    if pending is not None:
        expired.append(pending)
    snips = np.array(snips) if snips else np.empty((0, 2))
    durations = snips[:, 1] - snips[:, 0] if len(snips) else np.empty(0)
    report = {'restarts': np.array(restarts), 'expired_starts': np.array(expired),
              'orphan_stops': np.array(orphans),
              'short': np.where(durations < min_bout_s)[0],
              'n_start': int(np.sum(lab == 'Start')),
              'n_stop': int(np.sum(lab == 'Stop')), 'n_snips': len(snips)}
    return snips, report


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
    """Load whatever foot IMUs a session file contains, keyed by label.

    Unlike load_feet() this tolerates missing sensors (e.g. the 20260622
    session has no left_foot_a) — it loads every sensor whose label contains
    any of `patterns`. Pass patterns=('foot', 'box') to also treat the box
    IMU like a foot (its distance is ZUPT-anchored only while it rests, so
    expect drift over a carry).
    """
    return {label: imu.load_imu_recording(file_path, sensor_id=sid)
            for sid, label in imu.list_sensors(file_path).items()
            if any(p in label for p in patterns)}


def snip_distances(data, snips, pad_seconds=1.0):
    """Per-snip walked distance from the foot IMUs.

    For each [start_s, stop_s] snip, runs single-foot inertial mechanization
    (compute_position) on every recording in `data` (label -> ImuRecording)
    over the padded snip window and takes the maximum horizontal excursion
    from the starting position. The bout distance is the max over feet (the
    walker's feet move the trial distance; the stander's stay ~0) and the
    moving foot's label identifies the walker.

    Returns a DataFrame with start_s/stop_s/duration_s, distance_m, walker
    (label of the foot that moved farthest), and every per-foot distance.
    """
    rec0 = next(iter(data.values()))
    period = rec0.period
    rows = []
    for start_s, stop_s in np.asarray(snips):
        row = {'start_s': start_s, 'stop_s': stop_s,
               'duration_s': stop_s - start_s}
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
        row['distance_m'] = dists[best]
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


def align_snips_to_expected(measured, expected, gap_penalty=0.55):
    """Needleman-Wunsch alignment of measured snip distances to the expected
    bout-distance sequence.

    Match cost = |measured - expected| / max(measured, expected) (0 = perfect);
    skipping either an expected bout (missed / not captured by button presses)
    or a measured snip (extra press pair) costs `gap_penalty`. Returns
    (matches, missed_idx, extra_idx): matches is a list of (snip_i, expected_j)
    pairs; missed_idx indexes expected rows with no snip; extra_idx indexes
    snips with no expected bout.
    """
    obs = np.asarray(measured, float)
    exp = np.asarray(expected, float)
    n, m = len(obs), len(exp)

    def cost(o, e):
        if np.isnan(o):
            return gap_penalty * 0.9   # unmeasurable snip: near-neutral match
        return abs(o - e) / max(o, e, 0.5)

    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * gap_penalty
    D[0, :] = np.arange(m + 1) * gap_penalty
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i-1, j-1] + cost(obs[i-1], exp[j-1]),
                          D[i-1, j] + gap_penalty,
                          D[i, j-1] + gap_penalty)
    matches, missed, extra = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(D[i, j], D[i-1, j-1] + cost(obs[i-1], exp[j-1])):
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


def compare_to_trials(data, snips, processed_trialtable, pad_seconds=1.0,
                      reps=2, walkers_per_rep=2, order='rep_major',
                      gap_penalty=0.55, measured=None):
    """Compare event-scored snips against the condition table's distances.

    Measures each snip's walked distance from the foot IMUs (`snip_distances`),
    expands the table into the expected in-order bout sequence
    (`expected_bouts`), and aligns the two so missed / skipped trials fall out
    as expected bouts with no matching snip.

    `order` sets the expected sequence layout (see expected_bouts). Pass a
    precomputed `measured` DataFrame (from snip_distances) to skip the slow
    per-snip mechanization when re-aligning.

    Returns a dict:
      'measured' - per-snip DataFrame (times, per-foot + best distance, walker)
      'expected' - expected bout DataFrame (trial/rep/bout/distance_m/status)
      'aligned'  - expected joined with its matched snip (NaNs where missed),
                   plus distance_error_m
      'missed'   - the aligned rows with no matching snip
      'extra'    - measured rows matching no expected bout
    """
    if measured is None:
        measured = snip_distances(data, snips, pad_seconds=pad_seconds)
    expected = expected_bouts(processed_trialtable, reps=reps,
                              walkers_per_rep=walkers_per_rep, order=order)
    matches, missed_idx, extra_idx = align_snips_to_expected(
        measured['distance_m'], expected['distance_m'], gap_penalty=gap_penalty)

    aligned = expected.copy()
    for col in ('snip', 'start_s', 'stop_s', 'duration_s', 'measured_m'):
        aligned[col] = np.nan
    aligned['walker'] = ''
    for si, ej in matches:
        aligned.loc[ej, ['snip', 'start_s', 'stop_s', 'duration_s']] = (
            si, *measured.loc[si, ['start_s', 'stop_s', 'duration_s']])
        aligned.loc[ej, 'measured_m'] = measured.loc[si, 'distance_m']
        aligned.loc[ej, 'walker'] = measured.loc[si, 'walker']
    aligned['distance_error_m'] = aligned['measured_m'] - aligned['distance_m']
    return {'measured': measured, 'expected': expected, 'aligned': aligned,
            'missed': aligned.iloc[missed_idx],
            'extra': measured.iloc[extra_idx]}
