# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stride estimation system using IMU (Inertial Measurement Unit) data. The codebase consists of both MATLAB and Python implementations for processing IMU sensor data to estimate walking strides and gait parameters.

## Commands

### Conda Environment (Required for Claude)
Claude Code does not have access to shell profile configurations. Before running any Python commands, activate conda:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate stride_estimation_imu
```

### Python Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main Python demo
python demo4.py
```

### MATLAB
The original MATLAB implementation is in the `matlab/` directory. Key demo files:
- `demo.m`, `demo1.m`, `demo2.m`, etc. - Various processing demonstrations
- `demo4.m` - Main single IMU processing example

## Architecture

### Python Implementation (`src/stride/`)
- **`apdm.py`** - APDM sensor data loading and preprocessing
  - `getdata_apdm()` - Load .h5 files with orientation correction
  - `detect_quite_time()` - Auto-detect static periods for bias correction
  - `getdata()` - Section data, apply bias correction, normalize accelerometer
  
- **`inertial.py`** - Core inertial navigation and stride processing
  - Quaternion operations: `eul2qua()`, `qua2eul()`, `qua2rot()`
  - Footfall detection: `foot_fall()`, `acc_tilt()`
  - Main processing pipeline: `compute_pos()` - performs inertial mechanization
  - Stride segmentation: `stride_segmentation()`, `get_strides()`
  - Kalman filtering for tilt correction: `kf_tilt()`
  - Zero velocity updates: `zupts()`

- **`plotting.py`** - Visualization functions
  - `plt_ltrl_frwd_strides()` - Lateral vs forward stride plots
  - `plt_frwd_elev_strides()` - Forward vs elevation stride plots  
  - `plt_stride_var()` - Stride variability ellipse analysis

### Data Processing Pipeline
1. **Load IMU data** from APDM .h5 files with orientation correction
2. **Detect static periods** for bias correction and gravity normalization
3. **Inertial mechanization** - integrate angular velocity to get orientation, transform accelerations to navigation frame
4. **Footfall detection** - identify stationary periods during stance phase
5. **Zero velocity updates** - correct velocity drift during stance phases
6. **Stride segmentation** - extract individual strides from walking data
7. **Visualization** - plot stride patterns and variability analysis

### Key Constants
- `GRAVITY = 9.80297286843` - Gravity constant used throughout
- Footfall detection thresholds: W_FF (angular velocity), A_FF (acceleration), T_FF (time)
- Orientation mapping for different APDM sensor placements

### Data Structure
The main data structure from `compute_pos()` contains:
- `FF` - Footfall detection boolean array
- `FF_walking` - Footfall detection for walking periods only
- `An`, `Anz` - Accelerations in navigation frame (with/without ZUPT correction)
- `V`, `Vm` - Velocity vectors and magnitudes
- `P` - Position trajectory
- `quaternion` - Orientation quaternions
- `euler` - Euler angles (roll, pitch, yaw)

### MATLAB Legacy
The `matlab/` directory contains the original MATLAB implementation with equivalent functionality. The Python version is a port of key MATLAB functions with similar naming conventions.

## Log

### 2026-06-11

**Done and verified:**

1. **The step→stride rename** — the stride pipeline used "step" naming throughout
   despite segmenting strides (footfall → next footfall, same foot). Now:
   `get_strides`, `cut_stride_section`, `filter_strides`, the `stride_samples`
   dict key, plus the misleading locals in `demo_two_feet_head.py`
   (`compute_stride_metrics`, `stride_speeds`, `n_strides`, etc.).
   Compile-checked; `demo_one_foot.py` runs clean.

2. **`step_segmentation()` exists and works** — new in `inertial.py`, exported
   from the package. Step = opposite-foot footfall to footfall; gait-initiation
   steps included; missed contacts and standing footfalls handled. On synthetic
   data it recovers time/length/width exactly. On real Brock bouts, **step time
   is solid everywhere** (~0.50 s, splitting cleanly by side).

**The honest caveat — spatial step measures:** length-split and width depend on
anchoring the two feet's frames at a side-by-side stance, and the Brock bouts
are hostile to that: several un-ZUPTed seconds of box-handling before the walk,
turns after it. Three anchor strategies were tried; the one that stuck scores
each candidate stance by the **longest un-ZUPTed gap connecting it to the
walking steps**, reported as `anchor_quality`. It works as a filter: bouts with
quality ≲ 1 s give plausible numbers (e.g. trial 76 bout 2: quality 0.06 s,
width 0.24±0.03 m, lengths 0.77/0.44 by side — possibly real box asymmetry, s2
was carrying); bouts with quality > 2 s give garbage splits (negative lengths)
and are correctly flagged.

### 2026-06-11 (later)

- `step_segmentation` gained `anchor_mode=` ('auto' | 'firstonly'). 'firstonly'
  forces the anchor to the first contact of each foot (the protocol's initial
  side-by-side stance), for trials where pre-walk shuffling drags the auto
  anchor onto a drifted stance.
- Renamed `anchor_quality` → `anchor_drift_seconds` (it's a worst-case
  un-ZUPTed time gap in seconds, bigger = worse — it was never a "quality").

**Not done yet:**
- Haven't run `step_segmentation` over all 192 bouts / added step columns to
  `brock_trial_table.csv`.
- Footfall recall is the next real lever: 1–2 missed contacts per bout (the
  `skips`/`slow` counts) — restoring the `foot_fall_opposite_velocity` merge in
  `compute_position_two_imus` (commented call sites, function already ported)
  would likely fix those.

### 2026-06-26

**Consolidated to one step function (user-directed: "only one of these should
exist").** `step_segmentation()` was DELETED; `steps_from_strides()` was moved
from `demo_brock_velocity.py` into the library (`inertial.py`, exported from
`stride_imu`), along with `touchdown_map()` and `snug_start()`. All consumers
re-sourced to `imu.steps_from_strides` (velocity demo + batch, acc-sanity,
steps-trial, debug).

`steps_from_strides` now also produces the **spatial** measures that only
`step_segmentation` used to: it places both feet in ONE common frame —
**walking axis +Y, lateral X, gravity Z** — returning per-step `length`/`width`
and `left_xyz`/`right_xyz` (columns [lat, fwd, up]). This folded in the old
anchor logic (`initial_separation`, `min_stride_displacement`, `anchor_mode`,
`anchor_drift_seconds`); trust the length L/R split + mean width only when
`anchor_drift_seconds` ≲ 2 s. Speed/time still come from the validated strides
(zero nans) and the first step is snugged via `snug_start`.

Plotting helpers (`draw_accel/velocity/step_speed/bout_column`, `side_color`,
`VMAX`) moved to `stride_imu.plotting` (Agg-free) so the batch and the new
notebook share them.

**`brock_analysis.ipynb`** — streamlined pipeline: `load_metadata()` (condition
CSV + scored bounds), then per bout A) load, B) `compute_position_two_imus`,
C) `stride_segmentation`, D) `steps_from_strides`, E) the 3×2 speed panel.
Builds a per-bout table → `brock_analysis_table.csv` (190 bouts; same 2
`LinAlgError` bouts skipped — trial 76 b1, 96 b2). Verified it reproduces the
batch's `brock_trial<N>_velocity.svg` panel.

Per-trial figure layout (user-directed): the two bouts are stacked, and each
bout is an **overhead foot-position map on the left** (common frame, walk climbs
**+Y**, lateral X not-to-scale) plus **raw accel / foot speed / step speed on the
right half**. Built with a gridspec via `make_bout_axes` + `draw_bout_block`
(in `stride_imu.plotting`, replacing the old `draw_bout_column`); figsize 12×9 so
it fits on screen. The overhead needs `ff_idx` in the per-bout `data` dict
(`bout_velocity` / the notebook's `process_walking_bout` store it).

First (snug) step length/width (user-directed): drift before the snug is ignored,
so `length[0]` is the leading foot's forward travel from the snug start (its
position there subtracted) and `width[0]` is the assumed `initial_separation`
(0.3 m). Earlier they used the original opposite-foot contact pairing.

Time axis = snug gait start (user-directed): the rightward plots (|A|, foot
speed, step speed) are referenced to the SNUG-UP sample, so t=0 is the best
estimate of gait start and the manually-clipped pre-walk sits at negative t. The
per-bout `data` dict carries `t_ref` (slice-relative t=0) and `t0_abs` (absolute
sample at t=0, for the |A| secondary axis); the reference is `steps['start_snap']`
with the detected onset as fallback. (We still snip from the detected onset, so
the negative region is the onset→snug gap, not the full scored pre-walk.)

Y-axis ceilings are configurable at the top of `demo_brock_velocity_batch.py`
(ACCEL_MAX 50, FOOTSPEED_MAX 3, STEPSPEED_MAX 1.6) and passed via
`draw_bout_block(..., ymax=YMAX)`; library defaults match in `stride_imu.plotting`
(ACCEL_YMAX/FOOTSPEED_YMAX/STEPSPEED_YMAX), so all trials compare directly. The
old single `VMAX` (1.75) is gone.

Overhead foot map — real metres + correct sides (user-directed). The common
frame's lateral convention was flipped so **+X = the walker's right** (was
left-of-travel): `_common_frame` negates lateral on output and the width formula
became right-minus-left (value unchanged, normal +, crossover −). The overhead
now plots REAL lateral metres (no display negation), left foot on the left.
Overhead axes are FIXED for cross-trial comparison (user-directed, replacing the
earlier `lateral_amp`/`set_aspect` magnification): **X always [-1, 1] m**, **Y from
the start (~0) to 1.10x the furthest forward point**. Forward is re-origined to the
**snug start** (Y=0 at gait start). To make the assumed foot
offset visible AT the start, the figure pipeline now anchors `firstonly` by
default and uses `INITIAL_SEPARATION = 0.2 m` — both exposed as config at the top
of the batch and in the notebook's Configuration cell (and `bout_velocity` /
`process_walking_bout` params). 'auto' (drift-min) often anchors mid-walk, so the
feet had drifted together by the start (~0.07 m); 'firstonly' puts ~0.2 m at the
start. This changes the table's width/length vs the previous auto/0.3 run.

**Still open:** the spatial anchor remains the weak link (`step_segmentation`'s
old caveats now live in `steps_from_strides`); footfall recall (the `skips`
counts) is unchanged.

### 2026-06-29

**`An[0]` init bug fixed (always on).** `compute_position`'s mechanization loop
starts at `i=1`, so `An[0]` was left `[0,0,0]`. The first-stride ZUPT averages
gravity over a *tiny* window (8-12 samples, ~0.1 s — `first_FF` lands almost
immediately after onset), so that one zero pulled the gravity magnitude ~10%
low (`|g|≈9.803·(n-1)/n`, e.g. 8.7 vs 9.803) and injected a spurious `-g` at
`Anz[0]`; degenerate bouts with `first_FF=0` were subtracting ~0 gravity
entirely. Fix: compute `An[0,:]` from `A[0]` rotated by the init quaternion.
Restores `|g|` err to ~0.1 everywhere. NOTE: it barely moves the trajectory
(that window is near-stationary), so it is a correctness fix, not the cure for
first-step drift.

**Gravity-stance lead-in: SHELVED.** Added `gravity_seconds` to `bout_velocity`
(+ `leading_static_block`, batch `INITIAL_GRAVITY_SECONDS`) to keep ~1 s of clean
pre-onset stance in the slice so the first stride's ZUPT anchors on real stance
— halves first-step vertical drift, but the lead-in's standing footfall perturbs
the anchor-fragile spatial split (and a `FF_walking`-trim attempt fought
`steps_from_strides`' gait-init logic, giving `length[0]=1.8 m`). Left **off by
default** (`gravity_seconds=0`) pending a re-slice-free reimplementation
(inject pre-onset stance gravity into only the first stride's offset).

**Overhead per-foot rotation fixed (viz-only).** `_common_frame` rotated each
foot by its `swing[0]→swing[-1]` heading, which for the RIGHT foot diverged from
its true travel (up to +7° off +Y) → right track bowed while left stood straight,
feet ~1 m apart from a drifted anchor. `draw_overhead` now re-rotates EACH foot
independently so its own **snug-start→last-contact** direction points +Y, then
places the feet `±initial_separation/2` apart at the snug start (left −, right +;
Y=0 at the snug start). Spatial table measures (from `_common_frame`) are
untouched — purely the final visualization. `initial_separation` is now carried
in the per-bout `data` dict (`bout_velocity` + the notebook's
`process_walking_bout`); `draw_overhead` defaults to 0.2 if absent. A genuinely
curved foot path (e.g. trial 8 bout 2 right) still bows — rotation aligns the
endpoints, it can't straighten a real curve.

**Deleted `notebooks/stride_estimation_imu/`** — a 329 MB untracked accidental
full clone of the repo (own `src/`, `matlab/`, nested `notebooks/`), frozen Jun
11. Nothing imported it (Colab notebooks clone from GitHub; the local notebook
uses the main `src/`). It was the stale copy whose `compute_position` still
returned a dict (the live one returns `FootTrajectory` via `FootTrajectory(**result)` at
`inertial.py:484`).

**Regenerated** `brock_analysis_table.csv` (notebook, 190 bouts) and all 96
`brock_trial<N>_velocity.svg` + 2 subject grids (batch). Same 2 `LinAlgError`
skips (trial 76 b1, 96 b2).

### 2026-08-31

**New sessions s05_s06 (20260703) and s07_s08 (20260708) — event auto-scoring.**

- **`parse_subinfo_trialtables.py`** (new): extracts the "Experiment Trials"
  block from each session sheet of the experimenter's `SubInfo2*.xlsx` and
  writes `trialtable_<DATE>.csv` in the format `load_condition_table()` already
  reads (no more hand-made trial tables). Wrote `trialtable_20260703.csv` and
  `trialtable_20260708.csv` into the Dropbox imu-data folder. Note the xlsx
  also has a per-sheet "Trial Conditions" mini-table and subject info block;
  only the trial block is parsed.
- **`demo_brock_s05_s06_auto.ipynb` / `demo_brock_s07_s08_auto.ipynb`**: clones
  of `demo_brock_dataset2_auto.ipynb` with paths/titles per session; the
  s03_s04-specific prose (3-IMU note, press counts) genericized, and section F
  marked as documenting examples from the session where the method was built.
- **s07_s08 RAN CLEAN**: 4 foot IMUs present, 192 Starts/189 Stops → 190 snips
  (4 inferred stops). **188/192 expected bouts matched, median |distance error|
  0.51 m** → `brock_s07_s08_auto_aligned.csv`. Missed: trial 1 rep 1 (both
  bouts — no presses under them at all; the session's first trial was never
  clicked), trial 37 rep 2 bout 2 ("likely real miss"), trial 45 rep 2 bout 2
  (rep note: "start button accidentally clicked"). Pacing 15 s within-rep /
  36 s between trials.
- **s05_s06 BLOCKED — truncated h5.** `imuData_s05_s06_20260703.h5` is 76.7 MB
  on disk but its HDF5 superblock says 202.8 MB (`OSError: truncated file` on
  open). Unchanged for hours with Dropbox running and no cached copy — the
  *upload* to Dropbox was likely incomplete at the source. The notebook is
  ready; re-sync the file and run it.
- **Tooling:** `jupyter` was resolving to a global install whose `python3`
  kernel had no pandas — `ipykernel`, `nbconvert`, `nbformat`, `openpyxl` are
  now project (dev) deps; execute notebooks with
  `uv run python -m nbconvert --to notebook --execute --inplace <nb>`.

### 2026-08-31 (later)

**Stop inference rewritten (user: "four-feet-simultaneous is a bad rule").**
The old `infer_stop_from_quiet` required ALL feet static ≥1.5 s at once — a
moment that structurally never occurs at the true bout end (the walker hands
off and turns straight back), so inferred stops slid past the un-clicked
return walk: 28–41 s "bouts" for 10 s walks (only distance survived, being a
max-excursion; the snip windows were garbage). New rule: per SUBJECT (own two
feet, grouped by label suffix), still gaps < `quiet_seconds` are absorbed into
the motion run (closes within-gait double-stance, ~0.3 s), so the first
sustained (≥ `min_bout_s`) walk run by either subject ends exactly at a
≥`quiet_seconds` walker-pair stance — the hand-off. `quiet_seconds` default
0.75 s: below the shortest observed hand-off stance (0.9 s, s07_s08 t=4952),
above double-stance. s07_s08 result: all 5 inferred bouts at clicked-typical
durations, plus a 6th bout recovered that the restart logic used to drop
(t=8077, 5.4 m) → **189/192 matched** (trial 45 rep 2 regained; remaining
misses: trial 1 rep 1 never clicked at all, trial 37 rep 2 likely real).
NOTE: whole-window "most active pair" was tried and is fragile for long
search windows (the partner's later walks dominate) — first-sustained-run
avoids choosing a walker at all.

**New in `brock_functions.py`:**
- `save_trial_figures(data, aligned, session_tag)` — per-trial SVGs, every
  matched bout stacked (up to 4 blocks: overhead + |A|/foot speed/step speed
  via `imu.draw_bout_block`), `figures/<tag>/brock_<tag>_trial<N>_viz.svg`.
  constrained_layout collapses at >2 nested bout blocks → manual gridspec
  spacing there. RankWarning from short-bout polyfits suppressed
  (`np.RankWarning` moved to `np.exceptions` in numpy 2 — getattr fallback).
- `feet_pairs_from_labels()` — label-keyed feet → ('s1'|'s2','left'|'right')
  keys the stride pipeline wants (a→s1, b→s2).
- `manual_correct(data, aligned, trials, out_csv=…)` — interactive
  inspection/rescore: per (trial, rep) figure of raw |A| + mechanized foot
  speed + horizontal excursion for all feet, current bout windows shaded;
  'rescore A'/'rescore B' buttons arm a two-click (start, stop) capture per
  person, 'save' appends to CSV. `ensure_interactive_backend()` raises
  student-readable setup instructions (INTERACTIVE_HELP: use
  `uv run jupyter lab` in the browser, not VS Code; `uv sync` + kernel
  restart for ipympl) BEFORE drawing. Keep the returned controllers in a
  variable or the buttons get GC'd. `apply_manual_rescore(aligned, csv)`
  folds corrections back (person→walker-suffix match; missed bouts fill the
  unmatched row, walker='manual_<p>').
- `ipympl` + `jupyterlab` added as dev deps.

**`make_brock_session_notebook.py`** (new) — generates a session notebook from
the s03_s04 master: patches paths/prose/QUIET_SECONDS and appends sections G
(per-trial figures, `FIG_TRIALS`) and H (manual rescore, `MANUAL_INSPECT=[]`
+ the interactive-setup instructions). `uv run python
make_brock_session_notebook.py s09_s10 <date>` for the next session.

**Known dubious case, s07_s08 trial 45** (both reps carry accidental-click
notes): the trial-45 figure shows two `[inferred stop]` blocks with FLAT |A|
and zero steps yet "measured ~5.4 m" — junk snips whose distance is pure
integration drift that happens to land near the expected 5 m, so the aligner
preferred them over the GENUINE b-walk at t=3720 s (4.31 m, now sitting in
'extra'). All four trial-45 bouts claim an 'a' walker, which is impossible.
First candidate for `MANUAL_INSPECT = [45]`. Deeper fix if it recurs: make
snip_distances require footfalls/steps (drift snips have none) instead of raw
max excursion.

**s05_s06 still blocked** — h5 still truncated at 76.7/202.8 MB.

### 2026-09-02

Iteration from user testing the s07_s08 notebook:

- **Renamed `compare_to_trials` → `align_snips_to_trial_table`** (user-directed:
  names should say what happens/returns) across brock_functions, all three
  auto notebooks (source-only patches; outputs kept), the generator, and
  plotting.py docs. It wraps the NW core `align_snips_to_expected`.
- **`rescore_brock.py`** (new CLI): the manual_correct figures as native
  windows from the terminal — `uv run python rescore_brock.py s07_s08 1 37:2`
  — the zero-Jupyter rescoring path (VS Code interactive plots are
  unreliable). Reads `brock_<tag>_auto_aligned.csv`, writes the same
  rescore CSV the notebook fold-in cell reads.
- **manual_correct window text fields**: each figure has 'from [s]'/'to [s]'
  TextBoxes; Enter re-slices + re-mechanizes + redraws (plotting moved into
  `_RescoreFigure.set_window`; `box.eventson=False` around set_val or
  on_submit recurses). Bad input → status message, no crash.
- **manual_correct bug**: missed bouts carry `walker=''` (not NaN) —
  `pd.notna('')` is True so `''[-1]` crashed; blank-or-NaN handled in both
  manual_correct and apply_manual_rescore.
- **Event-timeline top axis** (user: labels had "no rhyme or reason"): was
  every-8th-matched-bout; now one tick per trial-rep pair anchored at the
  pair's first bout snip START (fallback: interpolated time), ha='left'.
- **Figure location** (user: not in the git repo): `save_trial_figures`
  originally wrote to a repo-relative `figures/` — the 48 s07_s08 SVGs were
  MOVED to `…/Dropbox/Treadmill Brock 2025/imu data/figures/s07_s08/` and the
  default `out_dir` now derives from the recordings' `file_path` (the
  'figures' folder next to the session .h5 — the same `imu data/figures` the
  dataset-1 batch's FIG_DIR points at). `figures/` is gitignored as a
  backstop. Earlier log lines saying `figures/<tag>/` mean the Dropbox one.
- **All figures are SVG** (user-directed): saved figures already were
  (`save_trial_figures` fmt='svg', the batch's per-trial SVGs); the auto
  notebooks now also render INLINE figures as vector via
  `%config InlineBackend.figure_formats = ['svg']` in the setup cell (all
  three notebooks; the generator inherits it from the master). Keep new
  figure code SVG-first; only drop to PNG if a notebook grows prohibitively
  large (not the case so far).
- **Event-timeline legend** moved to dedicated figure space (fig.legend under
  the suptitle, tight_layout rect) — it used to collide with row 1's top-axis
  labels, worse after the per-pair tick change.
- **First-step-erased bug in `steps_from_strides` FIXED** (user spotted it on
  s07_s08 trial 5 rep 2, both bouts: "we snugged too close, missed the first
  left step"). The snug itself was fine (snug_start's 1.25/0.2 m/s thresholds
  put t=0 in the valley before the first committed swing); the gait-init rule
  was the culprit: it UNCONDITIONALLY dropped "the first-swinging foot's
  opening stance contact", but when the snip starts at walk onset the swinging
  foot has no stance plateau in-slice — the heuristic (which infers the first
  swinger from each foot's SECOND touchdown) then deleted the STANDING foot's
  genuine stance, so the first landing had no predecessor, no step ended
  there, and the snugged step 1 silently spanned two real steps. Now the drop
  fires only when BOTH feet registered the shared pre-walk stance: two
  contacts at/before snug-g + 0.15 s (`OPENING_MARGIN_S` — the standing
  foot's plateau often restarts a few samples after g on weight shift, while
  a real first landing is never before ~0.3 s after g). Full s07_s08 sweep:
  189 bouts, first-step median 0.52 s, zero sub-0.1 s micro-steps, zero
  failures; ~30 bouts gained their genuine first step. NOTE: this is in the
  shared library, so dataset-1 (`brock_analysis.ipynb`) step tables/figures
  change slightly on next regeneration — notebooks NOT re-executed
  (user-directed: don't overwrite the ipynb, backend only).
- **Stance lead-in ON (user-directed redesign of the bout start).** Instead of
  snip-at-onset + backwards snug, `bout_sync_strides_steps` keeps the opening
  stance in-slice and lets the mechanization pin it (extended ZUPT):
  `gravity_seconds` default 0 → 1.0 (un-shelved; its June blockers died with
  the gait-init fix). Onset + lead-in come from the plain raw-stillness mask
  (|gyro| < 0.35 rad/s, either-foot-moving) — NOT `detect_quiet_time`, which
  is a bias-window hunter (thresholds d|W|/dt, demands 1.5 s runs, prunes to
  ±2σ), found no lead in ~150/189 bouts, and could even place onset after the
  first swing. Two crucial details: (a) the backwards stance walk may CROSS
  the click (floor i0 − grav) — the true side-by-side stance often sits just
  before the Start press, and a foot mid-swing AT the click (jumped gun)
  walks back to its motion-run start; (b) onset must land within 5 s of the
  click. snug_start's valley then falls at the pinned-stance end, so t=0 =
  "where stance ends" with no snug logic change. s07_s08 sweep: 135/189 bouts
  get ≥0.3 s stance (18 get none — genuinely unreachable, they use the fixed
  no-stance fallback), negative step lengths 296→183, trial 5's alternating
  length artifact gone ([0.71 0.64 0.76 0.66] m). All 48 figures regenerated
  (Dropbox figures/s07_s08). OPEN: `anchor_drift_seconds` reads HIGHER with
  the lead-in (median 1.7→2.2 s; >2 s flag now on 122/189 bouts) — probably
  the metric counting the pinned stance as an un-ZUPTed gap; re-derive it
  before trusting the flag. Also 3 bouts show a ~0.05 s first "step" from a
  genuine pre-walk foot replant (false start) — consider a min-step-time
  filter if they pollute step stats.
- **ensure_interactive_backend** now prints kernel python vs project .venv
  python with a "switch to THIS one" arrow; notebook H instructions lead with
  the terminal script, then VS Code kernel-picker (Select Another Kernel →
  .venv → Restart ↻), then `uv run jupyter lab`. Also documented: after any
  brock_functions.py edit the kernel must be RESTARTED (module caching).

### 2026-09-03

- **Tutorial section I appended to the session notebooks** ("Student
  playground — the three data types"): feet (ImuRecording: Wb/Ab/period,
  sliceable), the tables (conditions + aligned, plain pandas), and the
  manual_correct controllers. Four runnable cells: raw-signal plot of a bout,
  DIY compute_position_two_imus re-run, aligned⋈conditions merge with
  bout-speed-by-condition boxplots, controller inspection. Cells live in
  `make_brock_session_notebook.TUTORIAL_CELLS` (module-level, reused by
  in-place patches) and were smoke-run against real s07_s08 data.
  `_RescoreFigure` gained a `__repr__` (trial/rep, window, bout spans,
  rescores) so printing controllers is informative.
- Section G cell now sets `FIGURES_DIR = dirname(H5_FILE)/figures` explicitly
  and passes `out_dir=` (same place the library default derives — visible to
  students now). Config cell documents the `os.environ.get` hook (reads an
  env var if set, never sets one — for pointing the notebook at other files
  without editing).

### 2026-09-03 (later) — two step-train fixes from user figure review

- **"First step faster than second" SOLVED — it was the SECOND step.** User
  twice suspected snug zeroing; numerics disproved that (P[g]≈0, step-1
  displacement == step length). The real bug: step 2's speed came from its
  foot's first STRIDE, clocked from that foot's PRE-WALK stance touchdown —
  standing time in the denominator (trial 4 r2b1: 0.37 m/s vs step 1's honest
  1.0). Fix in steps_from_strides: any step whose inbound contact predates the
  snug g is recomputed from the trajectory with the clock starting at g ("no
  step's clock starts before g"). Sweep: s1/s2 medians now 0.95/0.95 vs
  cruise 1.29 (the natural ramp); old dip class (s2 < 0.5x cruise) 28→4.
  8 bouts keep s1 > 1.5x cruise = the genuine no-stance drift class.
- **Terminal landing vanished from the step train FIXED** (trial 1 r2b1: no
  blue step at ~4.5 s). The final landing's stance runs into the terminal
  standing, broken only by ~0.1 s weight-shift blips; foot_fall puts ONE
  footfall per quiet segment at arg-min gyro, which fell in a LATER plateau,
  so touchdown_map snapped it past the landing. Fix: touchdown_map(period=,
  bridge_s=0.15) bridges sub-0.15 s non-stationary blips (real swings are
  >=0.3 s) so chained plateaus form one stance starting at the true landing;
  the phantom late shuffle-contacts merge away too. period=None keeps the
  historic behaviour; both call sites pass period.
- All 48 s07_s08 figures regenerated. NOTE: the June "footfall recall /
  foot_fall_opposite_velocity" lever is still untouched — mid-walk missed
  contacts (69 same-foot skips across the session) remain the open item.

### 2026-09-04

- **`inspect_snipped_trial(feet, aligned, trial, ...)` -> Figure** (new,
  user-directed): the per-trial inspection figure as a first-class playground
  function. Adds over the old batch-only path: `prefix_seconds` of GREY raw
  |A| before the walk (what happened around the button press), the Start/Stop
  presses as green/purple dashed vlines on all right-hand rows, and a
  "reaction ≈ X s, duration ≈ Y s" box per bout (reaction = Start press ->
  snug t=0; NEGATIVE = jumped the gun, e.g. trial 4 r1b2 at -0.48 s).
  Returns the Figure handle. `save_trial_figures` is now a thin batch wrapper
  over it (same files, same defaults). Tutorial gained cell 5 demoing it.
- **Naming/docs pass (user: "why is it named data with no type")**: the
  label-keyed recordings dict is now called `feet` everywhere
  (inspect_snipped_trial, save_trial_figures, manual_correct,
  _RescoreFigure); load_available_feet's docstring spells out the
  {label -> ImuRecording} shape and that it IS the `feet` argument;
  inspect_snipped_trial's docstring defines `aligned` column-by-column
  (start_s/stop_s = the snip bounds in session seconds). Playground-facing
  functions carry undergrad-level docstrings.
- All 48 s07_s08 figures regenerated with the new annotations.

### 2026-09-04 (later)

- **`interactive_inspect_trial(feet, aligned, trial)` (new)**: the inspection
  figure with DRAGGABLE bounds. Per bout block, two grab-able vlines on the
  time axes: green = snug gait start (t=0), purple = bout end (initially the
  Stop press). Mouse-down grabs the closer line, drag moves it live, release
  re-runs the bout with the adjusted bounds and redraws in place (time axis
  re-zeroes to the new snug — the green line snaps back to 0 by design; the
  status line narrates). Adjustments accumulate in ctrl.state[(rep, bout)] =
  {'snug_abs', 'i1_abs'} (absolute samples). Plumbing: steps_from_strides
  gained `force_snap` (slice-rel snug override), bout_sync_strides_steps
  `force_snug_abs`. inspect_snipped_trial's per-bout drawing was extracted to
  `_render_inspect_block` (clears axes incl. stale secondary sample-index
  child axes, so it can redraw in place) — shared by static + interactive.
  Headless-simulated press/drag/release verified (end-drag 16→13 steps,
  forced snug lands exactly). Tutorial cell 6 added (both notebooks +
  generator). Keep the returned controller in a variable or callbacks die.

### 2026-09-04 (later still)

- **interactive_inspect_trial: status + save (user feedback).** (a) The
  "recomputing..." status was never replaced after the (synchronous) redraw
  finished — cosmetic, now ends with "recomputed: N steps. Press save...".
  (b) New 'save adjustments' button (bottom right): appends adjusted bouts to
  `out_csv` as the SAME (trial, rep, person, start_s, stop_s) rows
  manual_correct writes, so the existing apply_manual_rescore cell folds them
  in (manual=True + person attribution) — no table passing / return-catching
  needed (the figure outlives the function return; the CSV is the hand-off).
  Original button-press bounds are never modified. start_s saved = the
  dragged snug time; stop_s = the dragged end; un-dragged sides keep the
  row's original value. Note: a release only takes effect if the line MOVED
  (release reads the line position, not the cursor). Tutorial cell 6 now
  passes out_csv=f'brock_{SESSION_TAG}_manual_rescore.csv'.

### 2026-09-04 (evening)

- **Manual-tool hierarchy clarified (user)**: the draggable inspector is now
  the documented primary tool for bouts with bad bounds; manual_correct's
  remaining niche is MISSED bouts (no snip -> nothing to drag; its window is
  placed from the interpolated bout time). Tutorial cell 4 rewritten around
  that division; section H markdown leads with it (notebooks + generator).
- **Reaction time + "press not a go cue" icon (plot E)**:
  `estimate_reaction_s(feet, aligned)` (new) computes per-bout reaction
  (Start press -> snug gait start) via a MINI-mechanization: only the
  walker's two feet over ~8 s around the click, window anchored on a
  verified both-feet-still stretch, then snug_start's valley — but the first
  high-speed crossing must open a sustained WALKING RUN (>=3 crossings, <=2.5
  s gaps) whose end reaches the click, else the first run after it (raw-gyro
  proxies could not separate energetic box handling from walking: 0.35 rad/s
  flagged 105/189, swing-level 2.5 rad/s still flagged fumbles). Validated
  against pipeline annotations (t4: +0.06/-0.48/+0.31 vs +0.05/-0.48/+0.31);
  t4 r2b2 reads -2.8 vs pipeline +0.75 — genuinely ambiguous pre-click
  stepping that chains into the walk (the pipeline's slice-at-click hides
  it). ~6 s for a session. align_snips_to_trial_table now adds reaction_s +
  jumped_gun columns (and its feet param was renamed from `data`);
  draw_event_timeline draws an orange open triangle at the Start press of
  every negative-reaction bout, legend "press not a go cue (already
  walking)". FINDING: s07_s08 median reaction is -0.27 s and ~60% of bouts
  are negative — the experimenter generally clicked in RESPONSE to seeing
  gait start, not as a go cue. E cells in all three notebooks compute the
  column if absent (estimate_reaction_s import inline).

- **Interactive figure buttons (user)**: taller layout (5.4 in/block + 0.7,
  always-manual gridspec with a reserved ~1.25 in bottom strip) so the
  buttons sit BELOW the last block's hanging legend instead of on it; added
  'close without saving' next to 'save adjustments' (plt.close, discards).

### 2026-09-04 (repo restructure + notebook cache rewiring, user-directed)

**New repo layout (root was "a shitshow"; only config/docs + folders remain):**
- `src/` — the `stride_imu` package AND `brock_functions.py` (its internal
  sys.path insert now adds its own dir, so `import stride_imu` works).
- `notebooks/` — ALL .ipynb: brock_analysis, demo_brock_dataset2_auto (the
  master), demo_brock_s05_s06_auto, demo_brock_s07_s08_auto, + the two colab
  ones. Their config cells now do `sys.path.insert(0, abspath('../src'))`.
- `examples/` — every demo_*.py + matlab_compare.py + debug_missing_stride.py
  (sys.path inserts point at ../src; demo_matlab_check_two_feet's REPO_DIR is
  the parent and its .mat ref moved).
- `scripts/` — rescore_brock.py, parse_subinfo_trialtables.py,
  make_brock_session_notebook.py (MASTER resolved via __file__ to
  ../notebooks/, output written there too).
- `data/` — + matlab_demo4.mat, matlab_demo6.mat.
- Generated CSVs/SVG left the repo entirely (see cached data below);
  .ipynb_checkpoints/ gitignored. README run commands updated. NOTE: old
  demo_two_feet.py-style cwd-relative 'matlab/...' data paths still assume
  running from repo root (legacy, untouched).

**"cached data" dir (beside figures/, in the imu data folder):** all generated
tables live there now. Notebook config defines CACHE_DIR / ALIGNED_CSV /
RESCORE_CSV; cell 15, section H, the fold-in cell, tutorial cell 6, the
generator, and scripts/rescore_brock.py all use them. Existing
brock_*_auto_aligned.csv, brock_analysis_table.csv, brock_trial_table.csv,
brock_trial1_velocity.svg were MOVED to
`.../imu data/cached data/`. brock_analysis.ipynb writes its table there too.

**Interactive/figure fixes (user feedback):**
- 'close without saving' now really closes under ipympl: `fig.canvas.close()`
  (plt.close alone from inside a widget callback can leave the view).
- Second interactive_inspect_trial call not rendering: figures are now built
  under `plt.ioff()` and displayed EXPLICITLY via display(fig.canvas) —
  deterministic single display per call, no reliance on ipympl auto-show.
- Reaction/duration annotation moved to the |A| axes' bottom-left.
- Grey raw-|A| context now drawn AFTER the bout end too (same
  `prefix_seconds` amount), xlim extended right.
- 'save adjustments' additionally saves the adjusted trial figure as
  `figures/<tag>/brock_<tag>_trial<N>_viz_manual.svg` (same prefix, _manual
  suffix) beside the batch figures.
- Fixed my earlier regex slip in examples/demo_brock_velocity_batch.py
  (`FOOTSPEED_MAX = 4.5.0` -> 4.5).

Everything compiles; imports verified from notebooks/ cwd; generator builds
into notebooks/ with correct paths. Git status shows the moves as renames
(uncommitted — commit when ready). Restart the Jupyter server from the repo
root and open notebooks from notebooks/.

### 2026-09-08

- **Committed the restructure** (9bd85ee) — moves recorded as renames.
- **demo_brock_s07_s08_auto on Colab**: new bootstrap cell after the title
  (no-op locally): clones the repo, %cd notebooks, pip installs ipympl +
  enables the custom widget manager (so the interactive cells work), mounts
  Google Drive. Config cell gained a DATA_DIR switch — Drive path
  ('/content/drive/MyDrive/Treadmill Brock 2025/imu data') on Colab, Dropbox
  locally — with H5_FILE/CONDITION_CSV joined from it (the ~200 MB .h5 cannot
  live in the repo; users copy the 'imu data' folder into their Drive).
  README got the Colab badge. Generator emits all of it for future sessions
  (its CACHE-block anchor updated for the DATA_DIR-form config; ordering of
  the config replaces matters).
- **Tutorial cell (0): introspection** (user: working memory is ~7 items —
  teach students to ask objects what they contain): `list(d)`/`.items()` for
  dicts (feet), `vars(obj)` field->shape loop for objects (ImuRecording),
  `.dtypes`/`.head()`/`.describe()` for DataFrames (aligned, conditions),
  plus dir()/help() pointers. Playground markdown frames it. In both session
  notebooks + generator.
- Notebook cell validation now replaces !/% magic lines with `pass` (plain
  stripping broke `if IN_COLAB:` bodies).

### 2026-09-08 (later) — Colab without Google permissions (user: drive.mount
scope "too powerful", and it was also failing with 'credential propagation
was unsuccessful' on partial consent)

- drive.mount is GONE. Data in: the lab link-shares just the session .h5 +
  trialtable .csv ('Anyone with the link', Viewer) and pastes the two links
  into H5_URL / TRIALTABLE_URL in the bootstrap cell; students' notebooks
  gdown them into /content/brock_data (DATA_DIR on Colab). Zero consent
  screens. Results out: a final cell zips ONLY 'cached data' + 'figures'
  (not the .h5) from DATA_DIR and hands the zip to the browser via
  files.download() — also permission-free. Everything mirrored in the
  generator (COLAB_CELL/RESULTS_CELL constants taken verbatim from the
  patched notebook). Trade-off to note: link-shared files are readable by
  anyone holding the link.
- REMAINING for the lab owner: create the two share links and paste them
  into the bootstrap cell (placeholders say PASTE_DRIVE_LINK_...).

- **xlsx -> trialtable in the notebook (user)**: new optional cell after the
  config cell — `BUILD_TRIALTABLE_FROM_XLSX = True` uploads the SubInfo2
  .xlsx on Colab (files.upload; reads the Dropbox copy locally), prints
  `list_xlsx_sheets()` so the student picks from the actual sheet list
  (default: the sheet named like SESSION_TAG), and
  `trialtable_from_xlsx(xlsx, sheet, DATA_DIR)` writes trialtable_<date>.csv
  and points CONDITION_CSV at it. Parsing moved into brock_functions;
  scripts/parse_subinfo_trialtables.py is now a thin CLI over it. In both
  notebooks + generator.

### 2026-09-08 (evening) — interactive widgets were broken by a version clash

The "tried to get a widget" / VS Code dying / Lab+Colab live-scoring failures
traced to ONE real bug, captured in the notebook's saved traceback:
`ImportError: cannot import name 'backend2gui' from IPython.core.pylabtools`.
matplotlib was pinned 3.8.4, which still imports `backend2gui`; IPython
removed it, and the recently-added ipykernel dev-dep pulled IPython 9.x. Any
backend switch (the `%matplotlib widget` in ensure_interactive_backend, the
plt.ioff() exit) exploded. Fix: matplotlib pin 3.8.4 -> 3.9.4 (first version
compatible with modern IPython). Regression-tested at the kernel level:
nbconvert-executed a %matplotlib widget + ioff + Button + display(canvas)
notebook — clean widget-view output. After pulling: restart the Jupyter
SERVER (new env), hard-refresh the browser. Colab uses its own modern
matplotlib/IPython so this particular bug never applied there — if Colab
still misbehaves it is a separate issue (need its error text).

- **Snug drag past the slice start now EXTENDS the slice (user: dragged to
  ~651500, snapped to ~652500).** The release used to clamp the gait-start to
  the mechanized slice's first sample (slice begins <=1 s before detected
  onset — nothing integrated earlier to snap to). Now dropping the green line
  before the slice sets an `i0_abs` override: the bout is re-cut from the
  dropped sample and re-mechanized (_render_inspect_block gained i0_abs;
  drag state carries it; status says "slice extended back"). Verified: drop
  at 651500 -> new slice 651460, t0_abs exactly 651500, steps recomputed.
