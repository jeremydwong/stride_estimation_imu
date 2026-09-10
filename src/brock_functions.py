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
                  gravity_seconds=1.0, force_snug_abs=None, snug_end_enabled=True):
    """Walk-onset snip + stride pipeline for one bout; pull out the series the
    figure needs (|A| and |V| per foot, footfalls, steps, snug start).

    anchor_mode='firstonly' anchors the common frame at the first foot contacts
    so the assumed `initial_separation` lateral offset sits at the gait start
    (visible at the beginning of the overhead); 'auto' minimizes drift instead.

    gravity_seconds: keep up to this many seconds of the verified BOTH-feet-quiet
    stationary block just before walk onset inside the slice (default 1.0; 0 =
    snip exactly at onset, the pre-2026-09 behaviour). The mechanization's own
    footfall/stationarity criterion then pins that lead-in as the opening
    stance: the first stride's ZUPT averages gravity over real standing
    (halves first-step vertical drift), both feet register their opening
    stance touchdown (so the step train's gait-init logic sees the classic
    two-stance pattern), and snug_start's valley lands at the stance end —
    t=0 becomes "where the pinned stance ends". Was shelved in June because
    the standing footfall fought the OLD unconditional gait-init drop (fixed
    2026-09-02). The separate An[0] init bug fix (in compute_position) is
    always on.

    snug_end_enabled: trim to the settling valley after the last committed
    swing (snug_start in reverse), then rerun mechanization/strides on that
    window so the final ZUPT and step metrics use the trimmed end. An explicit
    inspector end override disables automatic end trimming.
    """
    # Walk onset + opening-stance lead-in, from the subject's RAW stillness
    # (|gyro| < STILL_RAD_S rad/s on either foot = moving). detect_quiet_time
    # is NOT used here: it demands 1.5 s runs and prunes samples (it is a
    # bias-window hunter), so it misses short genuine stances and can even
    # place "onset" after the first swing already happened.
    #
    # onset: first moving sample at/after the click — but if a foot is
    # already mid-swing AT the click (jumped the gun / late press), walk back
    # to that motion run's start. lead: consecutive BOTH-feet-still samples
    # immediately before onset, up to `grav`, allowed to cross the click
    # (the true side-by-side stance often sits just before the Start press).
    # The kept stretch is pinned as stance by the mechanization's own
    # footfall criterion; snug_start's valley then lands at its end.
    STILL_RAD_S = 0.35
    grav = int(round(gravity_seconds / period))
    a0 = max(0, i0 - grav)                       # search floor (pre-click cap)
    b0 = min(i1, i0 + int(round(5.0 / period)))  # onset must be near the click
    moving = np.zeros(b0 - a0, bool)
    for side in ('left', 'right'):
        Wm = np.linalg.norm(feet[(subject, side)].Wb[a0:b0], axis=1) / period
        moving |= Wm >= STILL_RAD_S
    click = i0 - a0
    if moving[click]:                            # mid-swing at the click
        run = np.flatnonzero(~moving[:click][::-1])
        onset = i0 - (int(run[0]) if len(run) else click)
    else:
        after = np.flatnonzero(moving[click:])
        onset = i0 + (int(after[0]) if len(after) else b0 - i0)
    still_before = ~moving[:onset - a0]
    run = np.flatnonzero(~still_before[::-1])
    lead = min(grav, int(run[0]) if len(run) else len(still_before))
    snip = max(0, onset - lead)
    result = process_bout(feet, subject, snip, i1, period, pad_seconds=0.0)
    end_foot = None
    if snug_end_enabled:
        end_snap, end_foot = imu.snug_end(result['left_info'], result['right_info'])
        if end_snap is not None:
            # snug_end is inclusive; processing slices are exclusive at i1.
            trimmed_stop = result['slice'][0] + end_snap + 1
            if snip + 2 <= trimmed_stop < result['slice'][1]:
                result = process_bout(feet, subject, snip, trimmed_stop, period,
                                      pad_seconds=0.0)
    # force_snug_abs: absolute sample to pin the snug gait start to (interactive
    # override of snug_start's detection); converted to slice-relative here
    force_snap = (None if force_snug_abs is None
                  else int(force_snug_abs) - result['slice'][0])
    steps = imu.steps_from_strides(result['left_strides'], result['right_strides'],
                                   result['left_info'], result['right_info'], period,
                                   initial_separation=initial_separation,
                                   anchor_mode=anchor_mode, force_snap=force_snap)
    j0, j1 = result['slice']
    steps['end_snap'] = j1 - j0 - 1
    steps['end_snap_foot'] = end_foot
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
        td = imu.touchdown_map(info.stationary_periods, period)
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
            't_ref': t_ref, 't0_abs': t0_abs, 'end_abs': j1 - 1,
            'slice': (j0, j1), 't': t,
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


def snip_distances(data, snips, pad_seconds=1.0, inferred=None):
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
                   plus distance_error_m
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
        measured['distance_m'], expected['distance_m'], gap_penalty=gap_penalty,
        times=measured['start_s'], same_trial=same_trial, spacing=spacing,
        time_weight=time_weight)

    aligned = expected.copy()
    for col in ('snip', 'start_s', 'stop_s', 'duration_s', 'measured_m'):
        aligned[col] = np.nan
    aligned['walker'] = ''
    aligned['inferred'] = False
    for si, ej in matches:
        aligned.loc[ej, ['snip', 'start_s', 'stop_s', 'duration_s']] = (
            si, *measured.loc[si, ['start_s', 'stop_s', 'duration_s']])
        aligned.loc[ej, 'measured_m'] = measured.loc[si, 'distance_m']
        aligned.loc[ej, 'walker'] = measured.loc[si, 'walker']
        if 'inferred' in measured.columns:
            aligned.loc[ej, 'inferred'] = bool(measured.loc[si, 'inferred'])
    aligned['distance_error_m'] = aligned['measured_m'] - aligned['distance_m']
    aligned['reaction_s'] = estimate_reaction_s(feet, aligned)
    aligned['jumped_gun'] = flag_jumped_gun(feet, aligned,
                                            reaction_s=aligned['reaction_s'])
    return {'measured': measured, 'expected': expected, 'aligned': aligned,
            'missed': aligned.iloc[missed_idx],
            'extra': measured.iloc[extra_idx]}


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


def _render_inspect_block(axes5, feet, pairs, row, period,
                          prefix_seconds=5.0,
                          ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                imu.STEPSPEED_YMAX),
                          initial_separation=0.2, anchor_mode='firstonly',
                          i1_abs=None, snug_abs=None, i0_abs=None):
    """(Re)draw ONE bout block of the inspection figure onto `axes5`.

    Shared by inspect_snipped_trial() and the draggable interactive version.
    Clears the axes first, so it can redraw in place. `i1_abs` overrides the
    bout end (absolute sample; default = the Stop press from `row`), and
    `snug_abs` pins the snug gait start instead of detecting it. Returns the
    bout_sync_strides_steps() result dict, or None when the pipeline failed
    (the failure is written on the axes).
    """
    for ax in axes5:
        for ch in list(getattr(ax, 'child_axes', [])):
            ch.remove()                   # stale secondary (sample-index) axes
        ax.clear()
    subject = 's1' if str(row['walker']).endswith('a') else 's2'
    title = (f"rep {int(row['rep'])} bout {int(row['bout'])}: "
             f"{row['walker']} walks - expected {row['distance_m']:.1f} m, "
             f"measured {row['measured_m']:.1f} m"
             + (' [inferred stop]' if row.get('inferred') else ''))
    axes5[0].set_title(title, loc='left', fontsize=10, fontweight='bold', pad=34)
    if (subject, 'left') not in pairs or (subject, 'right') not in pairs:
        axes5[2].text(0.5, 0.5, f'{subject}: foot IMU missing',
                      transform=axes5[2].transAxes, ha='center')
        return None
    i0 = int(i0_abs) if i0_abs is not None else int(row['start_s'] / period)
    i1 = int(i1_abs) if i1_abs is not None else int(row['stop_s'] / period)
    try:
        # the stride-variability polyfit warns on very short bouts; harmless
        import warnings
        rank_warning = (   # np.RankWarning in numpy 1.x, moved in 2.x
            getattr(getattr(np, 'exceptions', None), 'RankWarning', None)
            or getattr(np, 'RankWarning', RuntimeWarning))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', rank_warning)
            res = bout_sync_strides_steps(
                pairs, subject, i0, i1, period,
                initial_separation=initial_separation,
                anchor_mode=anchor_mode, force_snug_abs=snug_abs,
                snug_end_enabled=i1_abs is None and row.get('manual', False) != True)
        imu.draw_bout_block(axes5, res, ymax=ymax)

        ax_acc = axes5[2]
        t0_abs = res['t0_abs']            # absolute sample at t=0 (snug)
        j0 = res['slice'][0]              # first sample of the drawn slice
        # grey pre-walk |A| context: from t0-prefix up to the slice start
        # (the slice itself is already drawn in colour by draw_accel)
        suffix_ab = 'a' if subject == 's1' else 'b'
        a_pre = max(0, t0_abs - int(prefix_seconds / period))
        j1 = res['slice'][1]
        b_post = min(len(next(iter(feet.values()))),
                     j1 + int(prefix_seconds / period))
        for side in ('left', 'right'):
            rec = feet[f'{side}_foot_{suffix_ab}']
            for aa, bb in ((a_pre, j0), (j1, b_post)):   # before AND after
                seg = rec.Ab[aa:bb]
                tt = (np.arange(aa, bb) - t0_abs) * period
                ax_acc.plot(tt, np.linalg.norm(seg, axis=1), lw=0.5,
                            color='0.6', alpha=0.8, zorder=1)
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
        end_is_manual = i1_abs is not None or row.get('manual', False) == True
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
        ax_acc.text(0.01, 0.04,
                    f'reaction ≈ {reaction_s:.2f} s, '
                    f'gait duration ≈ {duration_s:.1f} s; '
                    + (f'end trimmed {trimmed_s:.2f} s' if not end_is_manual
                       else 'manual end'),
                    transform=ax_acc.transAxes, fontsize=8, va='bottom',
                    bbox=dict(fc='white', ec='0.8', alpha=0.8, pad=2))
        ax_acc.set_xlim(left=min(-prefix_seconds,
                                 (i0 - t0_abs) * period - 0.5),
                        right=(b_post - t0_abs) * period)
        return res
    except Exception as e:
        axes5[2].text(0.5, 0.5, f'stride pipeline failed: {e}',
                      transform=axes5[2].transAxes, ha='center',
                      fontsize=8, wrap=True)
        return None


def _inspect_layout(fig, n_rows):
    """Gridspec for n stacked bout blocks (constrained_layout collapses >2)."""
    return fig.add_gridspec(n_rows, 1, top=0.90, bottom=0.12,
                            left=0.07, right=0.98, hspace=0.55)


def inspect_snipped_trial(feet, aligned, trial, session_tag='',
                          prefix_seconds=5.0, save_path=None,
                          ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                imu.STEPSPEED_YMAX),
                          initial_separation=0.2, anchor_mode='firstonly'):
    """Draw the full inspection figure for ONE trial and return the Figure.

    Every matched bout of the trial is stacked (up to 4 blocks: 2 reps x 2
    walkers). Each block shows two overhead foot maps (equal-scale + wide) on
    the left and raw |A| / foot speed / step speed on the right, plus:

      * `prefix_seconds` of the raw accelerometer BEFORE the walk, in grey —
        what was happening just before/around the button press;
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
        distance_m (expected), measured_m, inferred (True when the Stop was
        inferred from the feet rather than clicked).
    trial : int
        Trial number (the trial table's 1..48).
    prefix_seconds : float
        How much grey pre-walk accelerometer context to draw before t=0.
    save_path : str or None
        When given, the figure is also written to this file (format from the
        extension). None = just build it.

    Returns
    -------
    matplotlib.figure.Figure — catch it to save/show it yourself or to hand
    it to other tooling, e.g. `fig = inspect_snipped_trial(feet, aligned, 5)`.
    For a version where you can DRAG the gait start and bout end, see
    interactive_inspect_trial().
    """
    import matplotlib.pyplot as plt

    pairs = feet_pairs_from_labels(feet)
    period = next(iter(feet.values())).period
    rows = aligned[(aligned['trial'] == trial)].dropna(subset=['start_s',
                                                               'stop_s'])
    if not len(rows):
        raise ValueError(f'trial {trial}: no matched bouts in the aligned '
                         f'table - nothing to draw')
    fig = plt.figure(figsize=(12, 5.0 * len(rows)),
                     constrained_layout=False)
    outer = _inspect_layout(fig, len(rows))
    for k, (_, row) in enumerate(rows.iterrows()):
        axes5 = imu.make_bout_axes(fig, outer[k])
        _render_inspect_block(axes5, feet, pairs, row, period,
                              prefix_seconds=prefix_seconds, ymax=ymax,
                              initial_separation=initial_separation,
                              anchor_mode=anchor_mode)
    fig.suptitle(f'{session_tag} trial {int(trial)}' if session_tag
                 else f'trial {int(trial)}')
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def save_trial_figures(feet, aligned, session_tag, out_dir=None,
                       suffix='viz', trials=None, fmt='svg',
                       ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                             imu.STEPSPEED_YMAX), initial_separation=0.2,
                       anchor_mode='firstonly'):
    """Batch wrapper: inspect_snipped_trial() for every trial, saved to disk.

    Writes <out_dir>/<session_tag>/brock_<session_tag>_trial<N>_<suffix>.<fmt>
    (folder created as necessary) and returns the list of paths. `out_dir`
    defaults to the 'figures' folder NEXT TO THE SESSION'S .h5 (taken from the
    recordings' file_path) — with the data in Dropbox that is the same
    '<imu data>/figures' the dataset-1 batch writes to, and it keeps generated
    figures out of the git repo. `trials=None` draws every trial in `aligned`
    (a trial with no matched bouts is skipped).

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
    want = set(trials) if trials is not None else None
    paths = []
    for trial in aligned['trial'].unique():
        if want is not None and trial not in want:
            continue
        path = os.path.join(
            folder, f'brock_{session_tag}_trial{int(trial)}_{suffix}.{fmt}')
        try:
            fig = inspect_snipped_trial(
                feet, aligned, int(trial), session_tag=session_tag,
                save_path=path, ymax=ymax,
                initial_separation=initial_separation, anchor_mode=anchor_mode)
        except ValueError:
            continue                      # no matched bouts for this trial
        plt.close(fig)
        paths.append(path)
        if len(paths) % 10 == 0:
            print(f'  ... {len(paths)} trial figures written')
    return paths


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
                               'purple (bout end) line; release to recompute',
                               fontsize=9, color='tab:red')
        from matplotlib.widgets import Button
        h = 0.30 / fig.get_figheight()          # button strip: fixed ~0.3 inch
        y = 0.10 / fig.get_figheight()
        self._buttons = []
        for x, w, label, cb in ((0.60, 0.185, 'save adjustments', self.save),
                                (0.80, 0.185, 'close without saving',
                                 self.close)):
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
            prefix_seconds=self.prefix_seconds, ymax=self.ymax,
            initial_separation=self.initial_separation,
            anchor_mode=self.anchor_mode,
            i1_abs=st.get('i1_abs'), snug_abs=st.get('snug_abs'),
            i0_abs=st.get('i0_abs'))
        blk['res'] = res
        blk['i1_abs'] = st.get('i1_abs')
        self._add_lines(blk)
        self.fig.canvas.draw_idle()

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
        if getattr(self.fig.canvas.toolbar, 'mode', ''):
            self._say('zoom/pan tool is active - turn it off to drag')
            return
        bi = self._find_block(event.inaxes)
        if bi is None or not self.blocks[bi]['lines']:
            return
        lines = self.blocks[bi]['lines']
        name = min(self.GRAB,
                   key=lambda n: abs(lines[n][0].get_xdata()[0] - event.xdata))
        self.drag = (bi, name)
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
        res, row = blk['res'], blk['row']
        t0_abs = res['t0_abs']
        rep_bout = (int(row['rep']), int(row['bout']))
        st = self.state.setdefault(rep_bout, {})
        new_abs = int(round(t0_abs + x / self.period))
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
        self._say(f'rep {rep_bout[0]} bout {rep_bout[1]} recomputed: {n} steps.'
                  f" Press 'save adjustments' to append to {self.out_csv}"
                  f' (original clicks stay untouched in the aligned table)')

    def save(self, _event=None):
        """Append every adjusted bout to the manual-rescore CSV.

        Writes the same (trial, rep, person, start_s, stop_s) rows that
        manual_correct() saves, so the notebook's apply_manual_rescore cell
        folds them in identically (stamping manual=True and the person). The
        original button-press bounds are NEVER modified — they stay in the
        aligned table / the .h5 annotations; this only records the edit.
        """
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
            start_s = (st['snug_abs'] * self.period if 'snug_abs' in st
                       else float(row['start_s']))
            stop_s = (st['i1_abs'] * self.period if 'i1_abs' in st
                      else float(row['stop_s']))
            recs.append({'trial': self.trial, 'rep': rep_bout[0],
                         'person': person, 'start_s': round(start_s, 3),
                         'stop_s': round(stop_s, 3)})
        df = pd.DataFrame(recs)
        header = not os.path.exists(self.out_csv)
        df.to_csv(self.out_csv, mode='a', header=header, index=False)
        fig_note = ''
        src_file = next(iter(self.feet.values())).file_path
        if src_file:   # adjusted figure beside the batch ones, _manual suffix
            folder = os.path.join(os.path.dirname(os.path.abspath(src_file)),
                                  'figures', self.session_tag)
            os.makedirs(folder, exist_ok=True)
            fpath = os.path.join(folder, f'brock_{self.session_tag}_trial'
                                         f'{self.trial}_viz_manual.svg')
            self.fig.savefig(fpath)
            fig_note = f' + {os.path.basename(fpath)}'
        self._say(f'saved {len(recs)} adjusted bout(s) to {self.out_csv}'
                  f'{fig_note} - run the apply_manual_rescore cell to fold in')

    def close(self, _event=None):
        """Discard: close the figure without writing anything."""
        import matplotlib.pyplot as plt
        try:
            self.fig.canvas.close()   # ipympl: destroys the widget view;
        except Exception:             # plt.close alone can leave it visible
            pass                      # when called from inside a callback
        plt.close(self.fig)


def interactive_inspect_trial(feet, aligned, trial, session_tag='',
                              out_csv='brock_manual_rescore.csv',
                              prefix_seconds=5.0,
                              ymax=(imu.ACCEL_YMAX, imu.FOOTSPEED_YMAX,
                                    imu.STEPSPEED_YMAX),
                              initial_separation=0.2,
                              anchor_mode='firstonly'):
    """inspect_snipped_trial(), but with DRAGGABLE gait start and bout end.

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
    'save adjustments' button (bottom strip) to append the adjusted bouts to
    `out_csv` — the same (trial, rep, person, start_s, stop_s) file that
    manual_correct() writes, so the notebook's apply_manual_rescore cell
    folds them into the aligned table identically, stamping each row
    manual=True with the person. No need to catch a return value for the
    edits (the figure lives on after this function returns; the CSV is the
    hand-off). 'close without saving' discards everything and closes the
    figure.

    `feet` and `aligned` are as for inspect_snipped_trial() (the dict of both
    subjects' foot recordings from load_available_feet(), and the per-bout
    table from align_snips_to_trial_table()).

    Needs an interactive matplotlib backend — run it in Jupyter (browser) or
    from the terminal; ensure_interactive_backend() raises setup instructions
    if the environment can't do it. KEEP THE RETURN VALUE in a variable
    (`ctrl = interactive_inspect_trial(...)`) or the callbacks are garbage-
    collected and dragging stops working.
    """
    mode = ensure_interactive_backend()
    import matplotlib.pyplot as plt

    pairs = feet_pairs_from_labels(feet)
    period = next(iter(feet.values())).period
    rows = aligned[(aligned['trial'] == trial)].dropna(subset=['start_s',
                                                               'stop_s'])
    if not len(rows):
        raise ValueError(f'trial {trial}: no matched bouts in the aligned '
                         f'table - nothing to draw')
    # a bit taller than the static figure, with a reserved bottom strip for
    # the save/close buttons (below the last block's hanging legend). Built
    # under ioff() and displayed EXPLICITLY below: relying on ipympl's
    # auto-show made a second call in one session silently not render.
    fig_h = 5.4 * len(rows) + 0.7
    with plt.ioff():
        fig = plt.figure(figsize=(12.5, fig_h))
    outer = fig.add_gridspec(len(rows), 1, top=1 - 0.85 / fig_h,
                             bottom=1.25 / fig_h, left=0.07, right=0.98,
                             hspace=0.55)
    blocks = []
    for k, (_, row) in enumerate(rows.iterrows()):
        axes5 = imu.make_bout_axes(fig, outer[k])
        res = _render_inspect_block(axes5, feet, pairs, row, period,
                                    prefix_seconds=prefix_seconds, ymax=ymax,
                                    initial_separation=initial_separation,
                                    anchor_mode=anchor_mode)
        blocks.append({'axes5': list(axes5), 'row': row, 'res': res})
    fig.suptitle(f'{session_tag} trial {int(trial)}' if session_tag
                 else f'trial {int(trial)}')
    ctrl = _DraggableTrial(fig, feet, pairs, period, blocks, prefix_seconds,
                           ymax, initial_separation, anchor_mode,
                           trial=int(trial), out_csv=out_csv,
                           session_tag=session_tag)
    if mode == 'script':
        plt.show(block=True)
    else:
        from IPython.display import display
        display(fig.canvas)
    return ctrl


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
    when interactivity cannot be achieved, BEFORE any figure is built, so the
    caller never crashes halfway through a rescoring session.
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
            venv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '.venv')
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

    def __init__(self, fig, axes, trial, rep, rows, out_csv, session_tag,
                 feet, period, n_samples, window):
        self.fig, self.axes = fig, axes
        self.trial, self.rep = trial, rep
        self.rows = rows                       # person -> (start_s, stop_s) or None
        self.out_csv, self.session_tag = out_csv, session_tag
        self.feet, self.period, self.n_samples = feet, period, n_samples
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
        for ax in self.axes:
            ax.clear()
        self.spans = {}                        # cleared with the axes
        for label, rec in sorted(self.feet.items()):
            person = label.rsplit('_', 1)[-1]
            color = self.SUBJECT_COLOR.get(person, '0.5')
            ls = '-' if label.startswith('left') else '--'
            self.axes[0].plot(t, np.linalg.norm(rec.Ab[j0:j1], axis=1),
                              lw=0.5, color=color, ls=ls, label=label)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    traj = imu.compute_position(rec.Wb[j0:j1], rec.Ab[j0:j1],
                                                self.period)
                self.axes[1].plot(t, traj.Vm, lw=0.8, color=color, ls=ls)
                self.axes[2].plot(
                    t, np.linalg.norm(traj.P[:, :2] - traj.P[0, :2], axis=1),
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
        for name, val in zip(('from', 'to'), self.window):
            if name in self.boxes:
                box = self.boxes[name]
                box.eventson = False           # set_val fires on_submit otherwise
                box.set_val(f'{val:.1f}')
                box.eventson = True
        self._draw_spans()

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
                      f"[{t0:.1f}, {t1:.1f}] s - press 'save' to keep it "
                      f"(or rescore again)")
            self._draw_spans()

    def save(self, _event=None):
        if not self.new:
            self._say("nothing rescored yet - use the rescore buttons first")
            return
        recs = [{'trial': self.trial, 'rep': self.rep, 'person': person,
                 'start_s': round(win[0], 3), 'stop_s': round(win[1], 3)}
                for person, win in sorted(self.new.items())]
        df = pd.DataFrame(recs)
        header = not os.path.exists(self.out_csv)
        df.to_csv(self.out_csv, mode='a', header=header, index=False)
        self._say(f"saved {len(recs)} correction(s) to {self.out_csv}")


def manual_correct(feet, aligned, trials, out_csv='brock_manual_rescore.csv',
                   pad_s=10.0, session_tag=''):
    """Interactive inspection + click-to-resnip for a list of trials.

    For each requested (trial, rep) this brings up an inspection figure -
    raw |A|, the mechanized foot speed, and horizontal position over time for
    every foot, with the currently-scored bout windows shaded (person A blue,
    person B orange; a missing/missed bout simply has no shading). Buttons:

      * 'rescore A' / 'rescore B' - then click the plot twice: first click is
        the new bout START, second the new STOP, for that person only (usually
        just one person needs fixing, so each is prompted separately);
      * 'save' - append the rescored [start, stop] rows to `out_csv`
        (columns trial, rep, person, start_s, stop_s). Nothing is written
        until you press save, so a stray click never corrupts the file.

    `trials` is a list of trial numbers (both reps shown) and/or (trial, rep)
    tuples. `feet` is the matched foot recordings of BOTH subjects, keyed by
    sensor label ('left_foot_a' ... 'right_foot_b') as returned by
    load_available_feet(); `aligned` is the per-bout results table from
    align_snips_to_trial_table() (start_s/stop_s = the snip bounds in
    session seconds). The inspection window covers
    both bouts of the rep plus `pad_s` on each side; missed bouts (no snip)
    get their window from the time-interpolated bout position, so there is
    always something to look at.

    Requires an interactive matplotlib backend - ensure_interactive_backend()
    raises a message with setup instructions (see INTERACTIVE_HELP) BEFORE any
    figure comes up if the environment can't do it. Returns the list of
    figure controllers (keep the return value in a variable in notebooks, or
    the button callbacks are garbage-collected and clicks stop working).
    """
    mode = ensure_interactive_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    period = next(iter(feet.values())).period
    n_samples = min(len(rec) for rec in feet.values())
    time_s, _missed = imu.assign_bout_times(aligned)
    aligned = aligned.assign(_est_t=time_s)
    typical = np.nanmedian((aligned['stop_s'] - aligned['start_s']).to_numpy(float))

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
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.subplots_adjust(bottom=0.16, hspace=0.08)
        fig.suptitle(f'{session_tag} trial {trial} rep {rep} - inspect / rescore'
                     f'   (A blue, B orange; missed bouts have no shading)')

        ctrl = _RescoreFigure(fig, list(axes), trial, rep, rows, out_csv,
                              session_tag, feet, period, n_samples,
                              window=(lo - pad_s, hi + pad_s))
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
        controllers.append(ctrl)
        if mode == 'script':
            print(f'trial {trial} rep {rep}: close the window to move on '
                  f'(save first if you rescored)')
            plt.show(block=True)
    if mode == 'notebook' and controllers:
        from IPython.display import display
        for ctrl in controllers:
            display(ctrl.fig.canvas)
        print(f'{len(controllers)} figure(s) above. Rescore with the buttons, '
              f'then press save on each figure you changed; corrections land '
              f'in {out_csv}.')
    return controllers


def apply_manual_rescore(aligned, csv_path):
    """Fold saved manual corrections back into an aligned table copy.

    Reads `csv_path` (trial, rep, person, start_s, stop_s - as written by
    manual_correct; the LAST correction wins when a bout was rescored twice)
    and overwrites start_s/stop_s/duration_s of the matching rows: same trial
    and rep, walker suffix == person. A correction for a MISSED bout (walker
    is NaN) fills the first unmatched row of that trial-rep instead, marking
    walker 'manual_<person>'. Adds a boolean 'manual' column.
    """
    aligned = aligned.copy()
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
        aligned.loc[i, ['start_s', 'stop_s']] = fx['start_s'], fx['stop_s']
        aligned.loc[i, 'duration_s'] = fx['stop_s'] - fx['start_s']
        aligned.loc[i, 'manual'] = True
        if pd.isna(aligned.loc[i, 'walker']) or str(aligned.loc[i, 'walker']) == '':
            aligned.loc[i, 'walker'] = f"manual_{fx['person']}"
    return aligned
