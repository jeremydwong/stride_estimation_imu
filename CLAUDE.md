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

### 2026-09-25 — manual scoring saves as you go (user: students will forget a separate save step)

- **`ManualScoring`** (new, brock_functions): section G's session object.
  `scoring.drag(pairs)` / `scoring.click(pairs)` open the editors; every Save
  press (and every rerun, which saves+closes the open editors first) rewrites
  `RESCORE_CSV` (upsert per trial/rep/person), `MANUAL_ALIGNED_CSV` (full
  corrected table, distances remeasured, distance cache so only changed
  windows re-mechanize) and `figures/<tag>/brock_<tag>_trial<N>_viz_manual.svg`
  (regenerated via save_trial_figures; replaces the drag editor's old
  snapshot of its interactive figure when a session hook is set). Editors get
  an `on_save` hook; errors inside it go to the figure status line.
- Opening a (trial, rep) that already has saved corrections asks
  s(kip, default)/r(escore)/c(lear back to auto) via input(); `on_existing=`
  bypasses it; `[]` never prompts (nbconvert-safe). `delete_manual_windows`,
  `corrected_alignment`, `manual_changes` added.
- Notebook s07_s08: G = G1 (widget check + creates `scoring`,
  `previous=` adopts/saves a prior session) + two one-line cells; H is one
  read-only compare cell (closes open editors first, rebuilds corrected from
  the CSV, change table + both omnibus plots). Old H1/H2 and the singular
  `figure/<tag>_Manual/` folder are gone.
- Tests: tests/test_manual_scoring.py (mocked, fast). Smoke-tested against the
  real s07_s08 data with the real drag/click editors (Agg, patched display).
- NOT updated: scripts/make_brock_session_notebook.py is still far behind the
  s07_s08 notebook's G/H (it predates Codex's G/H rework too).
- **Interactive figure rendering (same day, verified in a real JupyterLab via
  Chrome):** the drag/click figures were 1250 px wide inside a ~890 px output
  column, so the right-hand time panels + drag lines hid behind a horizontal
  scrollbar. They are now CREATED at a dpi that fits 860 px
  (`_interactive_dpi`; changing dpi after creation sent a pre-view resize and
  blanked the first figure — reproduced). `display_interactive_figure` (public;
  G1 uses it) forces full ipympl frames until the first click, a defence
  against diff frames painted on a freshly-cleared canvas — theory-backed but
  the user's G1 blank-check was NOT reproduced here. Figure title moved above
  the first bout title (they overlapped). Observed ipympl quirk, not fixed: in
  one test only the first figure per kernel came up HiDPI; later ones were
  half-resolution.

### 2026-09-26 — drag-editor overview strip + plot-only downsampling

- `interactive_inspect_trial(..., overview_pad_s=5.0, plot_every=1)`: a
  foot-speed OVERVIEW axis above the bout blocks — all four feet mechanized
  over [first bout start - pad, last bout end + pad] on session time; person
  A black, B green, right foot dashed (`OVERVIEW_COLOR`); each bout's CURRENT
  window shaded + labelled, re-shaded after every drag (`_bout_window_s`,
  `_DraggableTrial._shade_overview`). Foot speed chosen as the
  heading-agnostic signal.
- `plot_every=N` (default 1 = no downsampling) on both editors: drag editor
  thins dense lines after each render (`_decimate_lines`: >200 pts, no
  markers — step dots/vlines untouched); manual_correct slices its traces.
  Plotting only; mechanization and saved windows stay full-rate. ~40% faster
  draw at N=2. ManualScoring passes both through (**kwargs); notebook G2/G3
  expose OVERVIEW_PAD_S / PLOT_EVERY.
- **One figure builder for static + interactive (user-directed):**
  `_build_inspect_figure` now draws the inspection figure for BOTH
  `inspect_snipped_trial`/`save_trial_figures` and `interactive_inspect_trial`
  (which only adds drag lines/buttons, a fitted dpi and a bottom strip).
  Layout = one SECTION PER REP: that rep's foot-speed overview, then its bout
  blocks. Per-rep, not per-trial, because a trial's two reps are ~an hour
  apart (trial 4: 1016 s vs 5100 s) — a shared overview was ~4000 s wide and
  took ~50 s to mechanize per trial; per-rep the batch is ~1.7 s/trial again.
  `overview_pad_s` / `plot_every` / `rep` on all three functions.
  `_inspect_rows` validates before the interactive backend switch (the figure
  must be created AFTER `%matplotlib widget`). Static figures are NOT built
  under ioff, so the playground's inline display still works. The 48 Dropbox
  s07_s08 figures were not regenerated.
- **save_trial_figures is one file per (trial, rep)** (user-directed):
  `brock_<tag>_trial<N>_rep<R>_<suffix>.svg`; `trials=` takes trial numbers
  (both reps) and/or (trial, rep) pairs. ManualScoring exports/removes
  `..._trial<N>_rep<R>_viz_manual.svg` only for CORRECTED reps
  (`figure_path(trial, rep)`). Old per-trial `..._trial<N>_viz.svg` files in
  Dropbox figures/s07_s08 are stale leftovers under the old naming — not
  deleted, not regenerated.
- Overview strip gained a second time base (user): absolute sample index on
  a top secondary axis (session time / period), like the bout |A| panels;
  its legend moved outside to the right (overview width 0.84) and each
  section got 0.6 in of headroom (`ov_pad`) for the new ticks.
- Drag editor's 'close without saving' renamed **'continue without saving'**
  (user: it never actually closed in their browser). It now `discard()`s:
  marks the editor discarded, freezes drags, refuses save with a status
  message, and leaves the figure for the next scoring-cell rerun / H to close
  (closing an ipympl view from its own callback is unreliable). `close` kept
  as an alias.
- `ManualScoring` pair lists accept bare trial numbers = both reps (user hit
  a ValueError with `scoring.click([1])`): `[1, (37, 2)]`. drag() skips a rep
  with no matched bouts (message); click() opens it (missed-bout scoring).
- **One pad + scrubbing context (user)**: `overview_pad_s` renamed `pad_s`
  on inspect_snipped_trial / save_trial_figures / interactive_inspect_trial
  (notebook G2: `PAD_S`). It now sets BOTH the overview span and each bout
  block's grey context, which shows raw |A| AND foot speed (one
  mechanization of the whole context window) outside the analysed slice.
  `prefix_seconds` is an optional pre-side override (default None = pad_s).
  `_render_inspect_block(context_s=(pre, post))`.
- Drag editor: DOUBLE-CLICK a block's time axes left/right of centre =
  `extend_s` (5 s) more context on that side (view only, blk['context']).
  A click that doesn't move a line is now a no-op (it used to re-run the bout
  and mark it adjusted, so Save would write it). Dropping the gait start in
  standing time snaps t=0 to the walking onset; the saved start now follows
  the DISPLAYED t0 (it used to save the raw drop point) and the status says
  "snapped +X s".
- Bout-block title fix (user: unreadable on very short bouts): it was the
  equal-aspect narrow overhead's set_title, and that panel shrinks vertically
  for a short walk, sinking the title mid-block under the wide overhead. Now
  a text owned by the wide overhead (axes5[1]), blended transform x = narrow
  panel left, y = wide panel top + 34 pt — fixed at the block top; cleared
  with the axes on interactive redraws. Checked on trial 16 rep 1 (1.6 m).

### 2026-09-26 (later) — walker assignment from the protocol order

- Diagnosed from s07_s08 trial 13 rep 1 (user: bout 2's per-bout plots showed
  no foot speed although the overview showed green walking): NOT a projection
  problem — the bout was attributed to the wrong PERSON. snip_distances names
  the walker by max horizontal excursion over single feet; on 2.5 m trials a
  standing foot's integration drift (3.34 m) beat the real walker (1.4-1.8 m),
  so both bouts of the trial-rep were pinned on 'a' and the blocks plotted a
  standing person (slice collapsed to 1 s, 0 steps). 20 trial-reps had both
  bouts on one person. Naive "more swing time" was no better (both people
  move during hand-offs).
- **`assign_walkers_by_protocol(feet, result, first_walker='a')`** (new,
  user-directed rule): rep 1 = `first_walker` walks bout 1, other person
  bout 2; rep 2 flipped. Relabels matched bouts, takes distance from that
  person's farther foot (per-foot `dist_*` already in result['measured'] — no
  re-mechanization), recomputes reaction_s/jumped_gun; adds `walker_auto` +
  `walker_changed`. Snip->trial matching NOT redone. s07_s08: 42/189 changed,
  same-person trial-reps 20 -> 0, |distance error| 95th pct 2.07 -> 1.08 m
  (independent check — the rule never sees expected distance).
- Also unblocks manual fixes: apply_manual_rescore matches corrections by
  walker suffix, so a person-B correction on an all-'a' trial-rep was
  silently ignored.
- Notebook: `FIRST_WALKER = 'a'` in config (after the xlsx parse); C cell
  applies it right after align_snips_to_trial_table. The cached
  ALIGNED_CSV / figures are stale until the notebook is rerun.
  tests/test_walker_protocol.py.

### 2026-09-27 — load previous manual scoring OR score now (section G)

- New **G0** cell: `LOAD_PREVIOUS_MANUAL_SCORING`. True = load RESCORE_CSV via
  `corrected_alignment` into `aligned` (+ writes MANUAL_ALIGNED_CSV, prints the
  corrected trials; FileNotFoundError with instructions if the CSV is
  missing; saves+closes any open editors if the mode is switched mid-session)
  and G1-G3 print "skipped" instead of opening widgets — Run All safe.
  False = score now, G1-G3 as before. `MANUAL_ALIGNED_CSV` moved from G1 to
  G0. G1's body is indented under the guard (IPython transforms the indented
  `%matplotlib widget` — verified). H closes all figures after saving the
  editors (G1's widget check was leaking into H's output).
- Verified end-to-end with nbconvert on real s07_s08 data in BOTH modes
  (notebook's actual G0-G3 + H cells, scratch RESCORE_CSV).
- Intro markdown (cell 0) gained **"End results — what to look at, and
  where"**: figures (per trial-rep `_viz.svg` from F, `_viz_manual.svg` from
  G, `omnibus_auto` / `omnibus_plus_manual` from H) and CSVs (auto_aligned =
  E, manual_rescore = the corrections, manual_aligned = final table); C/G/H
  bullets updated to the protocol-walker fix, G0 switch, and compare step.
- H now SAVES `figures/<tag>/brock_<tag>_omnibus_auto.svg` always and
  `..._omnibus_plus_manual.svg` when corrections exist. Verified via nbconvert
  (load mode, scratch H5_FILE so nothing landed in Dropbox).
- G0 load mode now reports what was loaded (user: so students can check it
  worked): N saved corrections -> M bouts applied, one line per trial-rep
  (bout, person, new window), a WARNING if any saved row matched no bout, and
  a before/after table (auto vs manual window + distance, expected). Verified
  via nbconvert with a normal, a recovered-missed, and an unmatchable row.

### 2026-09-28 — manual windows are snugged like button presses (consistent timing)

- **`snugged_bout(pairs, subject, i0, i1, period, manual, resnug, ...)`** —
  THE snugging decision, factored in front of bout_sync_strides_steps and
  used by the figures (`_render_inspect_block`), the drag editor and the
  table. resnug=True (default): [i0, i1] is only the SEARCH window, exactly
  like Start/Stop presses — snug_start/snug_end find the gait inside it, for
  automatic AND manual bouts. resnug=False = legacy: a manual window IS the
  gait (start pinned, end untrimmed).
- **`add_gait_timing(feet, aligned, only=, resnug=, cache=)`** — new table
  columns gait_start_s / gait_end_s / gait_duration_s (session s) from
  snugged_bout; the figures' t=0 and end use the same numbers. s07_s08: 189
  bouts in 13.5 s, 0 failures, median gait 6.2 s. Notebook C adds them right
  after the protocol walker fix.
- `corrected_alignment(..., resnug=True)` re-snugs manual rows (and computes
  all rows if the automatic table has no gait columns yet), so LOADED old
  corrections are re-snugged by default. `manual_changes` reports gait start
  / duration + automatic_gait_s.
- Drag editor with resnug: green/purple drags move the WINDOW edges
  (state i0_abs/i1_abs); the gait re-snugs inside and the lines land on the
  snugged bounds; Save writes the window (not the snugged t0). Legacy path
  kept for resnug=False. End label reads "Snug end" unless legacy.
- `ManualScoring(resnug=)` threads it to the table, figure exports and the
  drag editor. Notebook G0: `RESNUG_MANUAL = True`, passed in G0-load, G1
  and H; G0 table shows auto vs manual gait duration.
- Verified on real data: a manual window equal to the press window now gives
  IDENTICAL gait timing to the automatic bout (10.98 s; legacy gave 10.49 s);
  an old-style saved correction re-snugs (1762.80 -> 1763.14 s); a drag's
  saved window reproduces the displayed gait exactly; G0-load + H end to end
  via nbconvert. Cached CSVs/figures are stale until the notebook is rerun.

### 2026-09-29 — snug_end fix: end = last landing COMPLETED inside the window

- User: after dragging the end, shouldn't the snug end move? It did, but
  only ~0.2 s before every drop regardless of gait phase. Cause: the
  mechanization stopped AT the window edge, so a swing in progress there was
  truncated into a fake settling valley; snug_end took it as the landing.
  Checked against the uncut trajectory: 3/6 mid-walk drops put the "end"
  where a foot was really swinging at 1.4-2.6 m/s.
- Fix (shared engine, so automatic bouts too): bout_sync_strides_steps
  mechanizes `END_LOOKAHEAD_S` (1 s) past the window just to find the end,
  and `imu.snug_end(..., limit=)` takes the latest swing whose settling
  valley is at/before the window's last sample (limit=None = historic rule,
  kept for other callers/tests). Then the bout is re-run on the trimmed
  window as before.
- Now: drag ends land 0.02-0.19 s before the last real touchdown (the <=0.2
  m/s valley precedes the stance marker; same rule as auto). Automatic bouts:
  gait start unchanged on all 189; gait end unchanged on 163, 26 moved - most
  <=0.05 s (recompute noise); the real moves (-0.2..-2.1 s) are all bouts
  whose old end sat within 0.01-0.09 s of the Stop press (walker still
  moving/shuffling at the press), e.g. t1 r2 b2: last step landed ~2 s before
  the press, then standing + a turn-start shuffle at the press — the old end
  counted 2 s of standing. Manual==auto identity and table==figure still
  hold. add_gait_timing now ~22 s/session. Tests: SnugEndLimitTests.

### 2026-09-29 (later) — stop check after the gait end; extend automatic windows only

- User concern: the snugged end assumes the window contains a true stop.
  `stop_after(feet, subject, end_abs, period)`: both feet raw-still (|gyro| <
  STILL_RAD_S = 0.35 rad/s, as walk onset) for STOP_HOLD_S, starting within
  STOP_SEARCH_S (1.5 s) of the end. Raw gyro only (reads past any window).
  STOP_HOLD_S calibrated to 0.25 s: mid-walk, both feet are essentially never
  still together even 0.15 s (132/151 windows; the 19 exceptions are real
  pauses >=0.5 s), while hand-off pauses are often 0.25-0.5 s — 0.5 s missed
  them (139 vs 161 stops found).
- snugged_bout: AUTOMATIC bouts whose end is still the Stop press and show no
  stop are re-snugged with the window +1 s at a time up to STOP_EXTEND_MAX_S
  (3 s); if none is found the ORIGINAL press-window result is kept (never the
  most-extended one) and flagged. Manual/dragged ends are NEVER extended
  (user: the scientist's judgement stands when subjects didn't stay still) —
  only flagged. snug_end only ever moves the end BACKWARD from the window end.
- Result dict + add_gait_timing: `stop_found`, `end_extended_s` (and
  `stop_searched_s` in the dict). s07_s08: 163 stop at the press, 18 found
  after extending (11 at +1 s), 8 no stop (kept at the press). 13/18 extended
  had the walker mid-swing within 0.5 s of the press (press came early); a
  few long ones (e.g. t36 r2 b2 +2.8 s, 8 swings) may have run into
  post-hand-off turn steps — speed can't tell turning from walking; they're
  listed for review.
- Figures: the |A| note says "window extended +N s to find the stop" or, in
  red, "NO clear stop ... - check the end" (manual: "your end used as is").
  Drag status line too. Notebook: config `STOP_EXTEND_MAX_S` (set on the
  module; re-applied after G1's importlib.reload), C prints counts + the list
  of bouts to check, G0/H tables show stop_found. Tests: StopAfterTests.

### 2026-10-01 — student feedback on section G (s07_s08 notebook)

- **G2/G3 "runs forever" = the input() prompt** in ManualScoring for trials with
  saved corrections (VS Code shows it as an easy-to-miss box). Prompt REMOVED:
  default `on_existing='rescore'` reopens showing the saved windows (fix again
  any number of times; Save replaces only the person changed — "fixed A, forgot
  B" works). 'ask' is accepted as an alias. New `scoring.clear(pairs)` and
  `scoring.clear_all()` (renames RESCORE_CSV to .bak-<date>, removes this
  session's _viz_manual.svg). Notebook G0 `START_FRESH` switch; G markdown
  explains the clean state (automatic table never touched by manual scoring).
- **Missed trial 1 was placed ON trial 2 in G3**: assign_bout_times used
  np.interp, which clamps outside the matched range. Edge missed bouts are now
  extrapolated by the median per-bout spacing (s07_s08 trial 1 r1: ~823/856 s
  vs trial 2 at 889 s). G3 figure now shows the target distance, other
  trials' bouts grey + labelled, Start/Stop presses, and the missed bout's
  ESTIMATED position (red dashed).
- **Distance shown vs drawn**: titles printed measured_m (alignment feature =
  single-foot max excursion over the padded button window) — never updated by
  a drag and not what the overhead draws. New `imu.overhead_travel(res)` =
  walked distance AS DRAWN (snugged gait, mean of both feet, gait start ->
  FARTHEST footfall); titles show "target / walked (as drawn) / snip max" and
  update on every drag; overheads get a dashed target line; add_gait_timing
  adds `walked_m` + `came_back`; manual_changes reports both.
- **Overhead flipped / collapsed on out-and-back windows** (likely the
  students' "funky" trials): each foot was aimed at its LAST footfall; when the
  window contains a turn + walk back, that is near the start. Now aimed at the
  FARTHEST footfall (identical for straight walks). `came_back` flags these
  (title red): 17/189 bouts in s07_s08, e.g. trial 8 r1 b2.
- The two overhead panels per bout are labelled "true scale" / "same walk,
  lateral stretched" (students read ±1 m vs ±2 m as approach vs return).
- s07_s08 after the fixes: measured_m − walked_m median 0.14 m, >1 m on 8/189.
  Cached CSVs/figures in Dropbox are stale until the notebook is rerun (C, E, F).
- Students' trial 7 r1 (8.4 m) / trial 12 r1 numbers do NOT match s07_s08
  (13.1 m / clean) — they are on another session; not reproduced.

### 2026-10-01 — FEEDBACK (first real student feedback on the s07_s08 notebook)

Comments verbatim-ish, each with its status. Most were addressed in commit
`369fb67` (made in another session); reviewed here: 40/40 tests pass, all
notebook cells parse, START_FRESH is consumed in G0 (file -> .bak-<date>) and
G1 (clear_all removes the _viz_manual figures), drag/click default to
on_existing='rescore'.

1. "Running G2/G3 just runs ad infinitum - seems to happen if that trial was
   already manually adjusted." — DONE (369fb67). It was the blocking
   `input()` skip/rescore/clear prompt (invisible in VS Code). Removed: an
   already-corrected trial-rep now simply reopens WITH its saved corrections.
2. "Would be nice to manually fix multiple times / we rescored A but not B
   and can't pull it up again." — DONE. Reopen as often as you like; Save
   replaces only the person you changed. `scoring.clear(pairs)` puts a rep
   back to automatic.
3. "Best way to run from a clean state without manual edits?" — DONE.
   `START_FRESH = True` in G0 (scoring mode) or `scoring.clear_all()`: the
   corrections file is renamed `.bak-<timestamp>` (rename back to undo), the
   `_viz_manual.svg` files are removed. The automatic table (E) is never
   touched by manual scoring.
4. "Target distance plotted in G2/G3 for a sanity check?" — DONE. Dashed
   target line on both overheads (G2 / saved figures) and on the excursion
   panel (G3); target in every title.
5. "If trial 1 is not found, how can we be confident G3 pulls up the actual
   trial 1?" — DONE. Missed bouts before the first match were placed ON the
   first match (np.interp clamps) so G3 showed trial 2; edge bouts are now
   extrapolated by the median bout spacing. The G3 figure shows the other
   trials' bouts (grey, labelled "T2 r1 b1 A"), the Start/Stop presses, and
   the missed bout's estimated position (red dashed).
6. ".svg: approach trajectory x-axis -1..1, return -2..2" — DONE (labels).
   Both panels are the SAME walk (true scale / lateral stretched), not
   approach vs return; they are now titled so.
7. "In G2, made a fix, figures update but measured distance did not." —
   DONE. The title now shows `walked X m (as drawn)` from `imu.overhead_travel`
   (snugged gait, mean of both feet, start -> farthest footfall), recomputed
   on every drag; `walked_m` / `came_back` columns in the tables. `measured_m`
   ("snip max" in the title) is the alignment feature — single-foot max
   excursion over the PADDED button window — and is meant to differ.
8. "Trial 7, rep 1, person A, measured 8.4 but looks closer to 7 m; measured
   often doesn't match the trajectory." — RESOLVED, and it was TRIAL 8: the
   only bout in the session with measured ~8.4 is trial 8 rep 1 bout 1
   (person A, target 8.0): snip max 8.36 m, walked 7.09 m as drawn. The
   snip max is inflated by foot drift after the walk (e.g. t7 r1 b1: 13.1 m
   snip max, peak 4 s after the gait end, walked 11.5). Item 7's title fix
   is the answer. NOTE the student's trial numbering was off by one here.
9. "Trial 12, rep 1, person A+B, weird drift or step, funky situation." —
   NOT REPRODUCED. Re-rendered trial 12 rep 1 with the Sep-27 code they had
   (git worktree at 6a41b51) and with current code: both bouts clean (15/16
   steps, walked 10.7/10.8 m vs 12 target, stop found, no walk-back). The
   overview strip does show A's un-clicked return walk and a slow ~0.5 m/s
   drift hump in A's standing foot between the bouts - possibly what they
   meant; it does not affect either scored bout. Given item 8's off-by-one,
   ask whether they meant trial 13 rep 1 (the misassigned-walker case, since
   fixed by the protocol rule, and the one with a saved manual correction),
   or trial 8 rep 1 bout 2 (the old flipped/collapsed overhead of a bout
   containing the walk back — exactly what 369fb67's farthest-footfall aim
   + red "walked back after a turn" flag fixed).

### 2026-10-02 — FEEDBACK round 2 + the three changes from the depth review

Second batch of student questions (same notebook), with answers; then the
code changes (user-directed: "do all three").

**Feedback 2**
1. "Corrected Walk 1 Person A in G3, saved; Walk B wasn't shown afterwards;
   G2 then said the trial was already corrected though B wasn't." — Same
   root cause as round 1 item 1 (the blocking skip/rescore/clear prompt,
   default skip). Verified the exact flow on real data: G3 fix A -> save ->
   rerun G3 same pair reopens WITH A's window and B still scorable (message
   names the saved person) -> fix B -> save -> CSV holds both rows; G2 on the
   same pair reopens too; no prompt anywhere. Found+fixed a related bug while
   testing: pressing 'rescore B' and then rerunning a cell WITHOUT clicking
   raised "a start was clicked without a stop" (an armed button with no
   click counted as pending). close_editors now blocks only when a start was
   actually clicked.
2. "When G2/G3 keeps running, what is it doing? wait or stuck?" — It was
   waiting on input(); nothing to wait for. No prompt exists now. Answer
   written into the section-G markdown ("Answers to the first round").
3. "How is total measured distance calculated; why no update after a manual
   change?" — It was `measured_m` = alignment feature from the button
   window; now `walked_m` in titles/tables, recomputed on every drag/save.
   Explained in C and G markdown.
4. "In G3, show measured/target while correcting?" — DONE: after the 2nd
   click the status line shows "walked ≈ X m (target Y m)" for the clicked
   window, computed by the same pipeline (bout_sync_strides_steps, manual,
   resnug) that fills the table, + "no clear stop in this window" when so.
   `_RescoreFigure._walked_note`; manual_correct/_RescoreFigure gained
   `resnug=`; ManualScoring.click passes it.

**The three changes**
- (1) `snugged_bout` MERGED into `bout_sync_strides_steps(..., manual=,
  resnug=, snug_abs=, end_overridden=)`: the old body is a nested
  `attempt(i1)`; the flag translation + stop-extension loop live in the same
  function. Depth to compute_position: add_gait_timing 4 (was 5), figures 6
  (was 7); README table updated. `force_snug_abs`/`snug_end_enabled` are no
  longer public parameters (examples pass only initial_separation/anchor_mode).
- (2) G0 load mode no longer raises FileNotFoundError when there is no
  corrections file: prints a note and continues with the automatic table
  (Run-All safe for a new student).
- (3) RENAME (user: "measured" reads as the result; "snip_max" also
  confusing): `measured_m` -> **`align_max_m`**, `distance_error_m` ->
  `align_error_m`; `snip_distances` output column `distance_m` ->
  `align_max_m` (it was colliding with the TARGET's name); manual_changes:
  `automatic_m`/`change_m` -> `automatic_align_max_m`/`align_change_m`, +
  `walked_change_m`; add_gait_timing adds `walked_error_m`. Titles say
  "align max". `_upgrade_columns()` (OLD_COLUMN_NAMES) maps old names on
  every table-consuming entry point, so cached CSVs from before today still
  load (verified with the Dropbox auto_aligned.csv). Omnibus legend: "align
  max (single-foot excursion)". Notebooks s07_s08 / s05_s06 / dataset2 and
  the generator patched (source only; saved outputs still show old headers
  until rerun); s07_s08 bout_speed_mps is now walked_m / gait_duration_s.
  Framing (user's, confirmed correct): one crude distance is needed BEFORE
  analysis to align snips to trials; the real distance/time measurement
  comes AFTER identification and snugging - two different quantities.
  README "bout table" section rewritten around that; G0 table shows
  auto/manual WALKED, not align max.
- Verified: 40/40 tests; nbconvert G0(load)+H end to end on a rebuilt
  alignment (new columns); the G3/G2 flow above.

### 2026-10-02 (later) — full override: type a window, or unmatch a bout

- User: "can scoring.click() take two timepoints or frame numbers? what if
  the alignment screwed up?" The corrections CSV was already the override,
  but its only writer was a mouse, and nothing could say "this bout did not
  happen" — so a snip the alignment pinned on the WRONG trial could not be
  removed. Added:
  - `ManualScoring.set_window(trial, rep, person, start, stop, units='s'|
    'samples')` — writes the window through save_manual_windows, syncs
    (snugged by the same rule, table + rep figure rewritten), prints the
    resulting walked/gait. Seconds vs samples verified identical on real data.
  - `ManualScoring.unmatch(trial, rep, person)` — a corrections record with
    BOTH times empty = "no bout": apply_manual_rescore turns the row back
    into a missed bout (snip/window/align/gait/walked NaN, flags cleared,
    walker kept so it stays addressable, manual=True); refresh_manual_
    distances skips it; add_gait_timing clears gait columns of in-scope rows
    without a window (they used to keep the automatic values). Undo with
    clear() or set_window(). Verified on trial 45 rep 2 bout 2 (the known
    junk snip).
  - Notebook cell **G4** (TYPED_WINDOWS / UNITS / UNMATCH, guarded like
    G1-G3) + G markdown "What if the alignment itself got it wrong?" (how to
    spot it: align_error_m, figure with no steps, res['extra']; fix per bout
    with unmatch + set_window, or retune TIME_WEIGHT and rerun C). README
    corrections-file paragraph + `manual` row updated.
- User: drag vs click had different list CONSTANTS (INTERACTIVE_INSPECT vs
  MANUAL_INSPECT) on top of different function names. Both cells now use
  `TRIALS`, `PAD_S`, `PLOT_EVERY` (manual_correct already took pad_s); all
  prose references updated.
- 41/41 tests (new: set_window s/samples, unmatch + undo, no-bout record
  validation).
- **Deleted `scripts/make_brock_session_notebook.py`** (user: no value;
  new sessions are a copy of the s07_s08 notebook with the EDIT HERE config
  lines changed). README says so. Nothing imported it; CLAUDE.md history
  mentions remain as history.
- **Deleted `scripts/make_brock_session_notebook.py`** (user: no value;
  new sessions are a copy of the s07_s08 notebook with the EDIT HERE config
  lines changed). README says so. Nothing imported it; CLAUDE.md history
  mentions remain as history.

### 2026-10-03 — Hallee's full review: Fix A (walk back) + Fix B (first step), A/B review tool

- **Hallee's review** (hallee_1..3.jpg, all 96 trial-reps) transcribed:
  `reports/s07_s08_hallee_notes.csv` (+ `no_change` flag) and per-bout
  categories `reports/s07_s08_hallee_labels.csv` (70 flagged bouts: WALKBACK
  22, DIST 12, LAST 11, STOP? 10, FIRST 7, NODATA 4, EARLY 3, UNDER 1).
  Report: `reports/s07_s08_scoring_failures.html` (also published).
- **Fix A — walk-back cut** (`WALKBACK_CUT`, `walkback_info`): automatic bout
  with >= 2 footfalls stepping back along each foot's own direction (per-foot
  projection, ref = first footfall -> first > 0.5 m away), or gait > 1.4x
  (target/1.25 m/s + 1 s; needs `expected_m`, now passed by add_gait_timing
  and the figures), ends at the turn (later of the feet's farthest footfalls
  + 0.4 s) and is NEVER stop-extended (the 09-29 extension had extended 6
  walk-backs past the return). Manual: flag only (`walkback_steps`).
  Isolated vs old code: 22/22 flagged walk-backs cut, 3/72 clean touched
  (23.2 b1, 31.2 b1, 44.2 b1: turn-in-place steps after the farthest footfall;
  gait -1.0..-1.6 s, walked unchanged), 6 other (16.2 b1 5.7->2.8 s, 32.2 b2
  5.7->2.5 s, ...).
- **Fix B — chain-back start** (`CHAIN_BACK`, `chain_back_start`): NOT a
  threshold change — a blanket 3 s onset search pulled 12/72 clean bouts onto
  pre-walk activity. Automatic only: if committed swings (>= 1.25 m/s) run
  back-to-back (gaps <= 0.6 s) into the detected gait start, restart the
  search at the first (<= 3 s back). 5/7 first-step bouts moved (4 gain a
  step; 45.1 b2 +2 steps +1.6 m), 2/72 clean touched (26.1 b2, 44.2 b2: +1
  step each), still missed 12.1 b2, 28.2 b2 (slow first step).
- Result flags `walkback_cut`, `walkback_steps`, `chain_back_s`; table
  columns `walkback_cut`, `chain_back_s`; figure note says what happened.
  `attempt(i1, i0=i0)` now takes the start too. tests/test_bout_fixes.py.
- **Visuals**: contacts before the snug gait start drawn hollow grey, t<0
  shaded with "before gait start: not analysed" (`shade_pre_gait`,
  `PRE_GAIT_COLOR`); overhead target line in the omnibus 'expected' grey.
- **`scripts/ab_review.py`** (new): reusable A/B review — runs every matched
  bout with module switches BEFORE vs AFTER, groups changed bouts by the
  labels CSV (fixed / still_missed / touched_clean / touched_other), writes
  one PDF per group with BEFORE/AFTER pages interleaved per trial-rep + an
  index page, and changes.csv, to `<imu data>/figures/review/<date>_<name>/`.
- Figures: previous SVGs copied to `figures/s07_s08_old/`; `figures/s07_s08/`
  regenerated with both fixes. Removed stray test folders (flowtest, ovr, t)
  that my earlier ManualScoring tests had written into Dropbox figures/.
- OPEN (explained to user, not changed): step SPEED. Step k's speed is the
  leading foot's STRIDE speed (stride length/time over its last two steps),
  except step 1 (lead-foot displacement g->landing / time) and any step whose
  stride started before g (displacement since g / time since g — includes the
  time that foot stood still). So the first two steps are distorted (14.1 b1:
  1.08, 0.98 vs the obvious ramp) and all steps lag/average over two. Proposed:
  midpoint progression = lead-foot displacement within [prev touchdown, this
  touchdown] / 2 / step time (14.1 b1: 0.54, 1.11, 1.35, 1.28, 1.30).

### 2026-10-03 (later) — one place for start/end decisions; step speed, not stride speed

- User: the step-speed panel was showing STRIDE speed ("lying to the
  viewer") with hacks at the front; consolidate all start/end wiggling into
  one place before compute_position_two_imus / steps_from_strides.
- **`find_gait_window()`** (renamed `find_straight_gait_window()` the same day:
  the window is one straight walk, reversals rejected) (brock_functions) now makes EVERY start/end
  decision, in order: onset + standing lead-in (`_onset_and_snip`, raw gyro),
  Fix B chain-back, snug start, snug end (1 s look-ahead), Fix A walk-back
  cut, stop check + extension. It looks at look-ahead runs of the ENGINE only
  (`_mechanize` = compute_position_two_imus, no strides/steps). Returns
  absolute snip/onset/start/end + start_from ('snug'|'pinned'|'onset') +
  flags. **`bout_sync_strides_steps()`** = find_gait_window, then ONE
  process_bout on [snip, end] and steps_from_strides with the start handed
  over (force_snap; None when start_from == 'onset', i.e. no snug exists).
  Result also carries 'window'. Same signature/return keys as before.
  VERIFIED a pure refactor before the speed change: 189/189 bouts identical
  (start, end, slice, every step index/speed/length, walked, flags) vs a
  snapshot. Two details needed for that: the walk-back check runs on the
  gait window [snip, end] (footfall detection near a window edge depends on
  where it ends), and 'onset' starts are passed as "no snug".
- **Step speed** (steps_from_strides `frwd_speed`): a step = one foot's
  touchdown -> the other foot's; speed = (|dP_left| + |dP_right|) / 2 / step
  time = how far the MIDPOINT BETWEEN THE FEET moved (each foot's own
  displacement between the two touchdowns; no strides, no cross-foot frame).
  No step starts before the gait start g; steps ending at/before g dropped
  (7 bouts lose a pre-walk "step"). The stride-speed lookup and both
  first-step special cases are gone (strides args now unused, kept for
  callers). s07_s08: median speeds step1/2/3/steady OLD 0.88/0.92/1.28/1.34,
  NEW 0.58/1.11/1.30/1.33; step1 > step2 in 93/188 bouts OLD vs 6/188 NEW.
  The "planted" foot moves 0.1-0.26 m within a step (heel-off before the
  other foot's touchdown registers at its flat-foot plateau) - counted, so
  this is the true midpoint; each foot's motion is partitioned, never
  double-counted. Tests: StepSpeedTests.
- **ab_review.py** compares CODE VERSIONS too: --before-src/--after-src or
  --before-rev/--after-rev (git worktree), plus switches; each side runs in
  its own subprocess (two copies of brock_functions never mix); --compare
  window|speed; --trials always rendered ('requested'). Figures don't pickle
  (lambda axes), so workers write one PDF per trial-rep and the main process
  merges in BEFORE/AFTER order (pypdf via `uv run --with pypdf`, else
  pdfunite) - pyproject untouched.
- **One path, no side routes (user: "everything through the same path")**:
  deleted `_mechanize` - find_straight_gait_window calls
  compute_position_two_imus directly. bout_sync_strides_steps no longer uses
  process_bout: it slices the bout to the decided window and calls
  compute_position_two_imus ONCE, then **`steps_from_footfalls(left_info,
  right_info, period, ...)`** (new name - it never used strides after the
  step-speed change; `steps_from_strides` kept as a 1-line deprecated wrapper
  for the dataset-1 examples). No strides computed in the bout path; result
  drops left_strides/right_strides/n_strides, adds n_steps (demo_brock_velocity
  print + brock_analysis.ipynb table updated). process_bout kept, marked
  LEGACY, for the 4 dataset-1 example scripts only. Engine print-noise
  silenced once per bout. Verified bit-identical (189/189) vs before.

### 2026-10-05 — manual worklist, 12.1 b2 diagnosis, threshold 0.9 trial, neighbour trials in the overview

- **`reports/s07_s08_manual_worklist.html/.csv`** (published): Hallee's 70
  flagged bouts rerun on main afbf137 - 35 fixed, 31 still manual (grouped:
  no walk recorded 4, first step 4, over target 3, needs video 9, last step
  11), 3 real early starts, 1 team decision. 10.1 b1 already rescored.
- **12.1 b2 missed first step, diagnosed (not yet fixed):** walker started
  ~1.9 s before the Start press (search floor is press - 1 s); Fix B should
  catch the 0.53 m initiation step but `chain_back_start` measures speed via
  `_foot_speeds` (single-foot compute_position, a separate route) over ~4 s
  of standing that ENDS at the gait start: ZUPT pins velocity only at
  footfalls, so standing reads 0.6-0.8 m/s and the step 1.05 m/s (< 1.25).
  Same engine 1 s past the start reads 1.61 m/s. Proposed: chain-back via
  compute_position_two_imus with the END_LOOKAHEAD_S look-ahead, delete
  _foot_speeds. Tested on 189: 5 change (12.1 b2 fixed +1 step; 18.1 b2 +1
  step; 44.2 b2 loses its Fix B chain = likely a correction). Awaiting OK.
  The worklist's "first steps are slower than 1.25 m/s" wording is wrong for
  12.1 b2 (measurement, not the step).
- **Swing threshold now one knob:** snug_start / snug_end /
  steps_from_footfalls take high/low=None -> read
  `inertial.DEFAULT_START_FOOTSPEED_HIGH/LOW` at call time (defaults were
  bound at def time, so changing the constant did nothing). Default still
  1.25. ab_review switches accept `module.NAME` for stride_imu settings
  (`--after inertial.DEFAULT_START_FOOTSPEED_HIGH=0.9`).
- **0.9 m/s trial (NOT adopted in code):** 48/189 bouts change. 23 gain the
  closing feet-together step at the end (median +0.38 s, +0.06 m walked;
  5/11 of Hallee's LAST, but also 9 bouts she called clean - systematic, a
  definition change); 14 starts move earlier (~0.6-1.0 s; 12.1 b2, 18.1 b2,
  28.2 b1, 23.1 b2 gain; 4.2 b2 jumps -2.35 s onto pre-walk shuffles);
  6 ends run on 0.7-2.9 s into post-hand-off shuffling (25.1 b2, 46.1 b2,
  15.2 b1, 25.2 b1, 38.2 b2, 40.2 b2: gait time up, walked ~unchanged); 5
  ends move earlier (the walk-back detector now sees a turn step: 44.1 b2,
  17.2 b1 ...). Idea if adopted: require end swings within CHAIN_GAP_S of
  the previous one (mirror of chain-back) to stop the run-ons.
- **Figures:** previous s07_s08 set stashed to
  `figures/s07_s08_stash_2026-10-05_threshold1.25/`; `figures/s07_s08/` (95
  trial-rep + 2 omnibus + 2 _viz_manual) REGENERATED AT 0.9 m/s with the
  neighbour overview (regenerated via a scratch script mirroring notebook
  A-C/F/H; the code default is still 1.25, so the notebook will not reproduce
  them until the threshold is adopted).
- **Neighbouring trials in the top plot (user):** `NEIGHBOR_S = 60.0`
  (s07_s08: last Stop -> next trial's first Start ~36 s, 90% within 54 s).
  The drag/static overview strip spans +/- neighbor_s around the rep's bouts
  and `_draw_neighbor_bouts` shades other trial-reps' bouts in a per-trial
  colour (NEIGHBOR_COLORS) with dotted lines at their bounds and a 'T2 b1 (A)'
  label (`_neighbor_bouts` clips to the recording; first/last trials just
  show one side). pad_s now only sets each bout block's grey context.
  manual_correct (click) gets the same colouring and its window widens to
  max(pad_s, neighbor_s) (neighbor_s=0 = old tight window). `neighbor_s` on
  inspect_snipped_trial / save_trial_figures / interactive_inspect_trial /
  manual_correct (and via ManualScoring.drag/click **kwargs); notebook G2/G3
  expose NEIGHBOR_S.

### 2026-10-08 — swing threshold 0.9 m/s + end chain + chain-back fix (ADOPTED)

User-directed ("do both"). All three in the shared engine/pipeline:
- **`DEFAULT_START_FOOTSPEED_HIGH` 1.25 -> 0.9 m/s** (a committed swing).
  Slow first steps and the closing feet-together step now count. Affects
  dataset-1 too on next regeneration.
- **End chain (`END_CHAIN`, automatic bouts):** `snug_end(start=, max_gap=)`
  groups swings after the gait start into chains (gaps <= CHAIN_GAP_S 0.6 s)
  and only counts swings up to the end of the LONGEST chain (the main walk).
  Stops 0.9's run-ons into post-hand-off shuffling (25.1 b2, 46.1 b2, 15.2
  b1, 25.2 b1, 38.2 b2, 40.2 b2 were +0.7..2.9 s). First version used the
  chain FROM the start - broke 28.2 b1 (lone initiation step, 0.66 s pause:
  gait collapsed to 1 step) -> longest chain.
- **Chain-back fix (Fix B):** `chain_back_start` uses
  compute_position_two_imus run END_LOOKAHEAD_S past t0 (`_foot_speeds`
  deleted - it was single-foot compute_position ending AT t0, which read
  standing as 0.6-0.8 m/s and squashed 12.1 b2's first step to 1.05 m/s).
  New `floor=` (the onset search floor): when t0 sits within 0.15 s of it,
  the floor cut a step in progress, so a swing straddling t0 joins the chain
  (34.1 b1, 20.2 b1 lost 0.3 m without it); elsewhere straddling swings are
  excluded (allowing them always pulled 16.1 b1, 30.1 b1, 35.2 b1, 43.2 b1
  2-3 s back onto box handling, 30.1 b1 -1.4 m).
- **Net vs c7207bd (ab_review `2026-10-08_thr0p9_endchain_chainfix`):** 48/189
  bouts change: 23 gain the closing step (median +0.38 s, ~+0.06 m), 17
  starts earlier (12.1 b2, 18.1 b2, 23.1 b2, 28.2 b1 gain a first step),
  8 ends earlier (post-walk swings after a 0.8-1.2 s pause dropped: 4.2 b1,
  17.2 b1, 20.1 b2, 30.1 b2, 34.2 b1 ...), 0 run-ons. Hallee's 22 walk-backs:
  16 unchanged, 6 gained only the closing step (walk-back cuts fire 11x now -
  the end chain stops most bouts before the turn). Known bad: 4.2 b2 start
  -2.35 s onto pre-walk shuffling (-0.49 m; Hallee: needs video). 13.2 b2
  unchanged (not a threshold problem).
- Tests: EndChainTests (3), ChainBackTests rewritten to fake the engine +
  floor cases. 41/41.
- **Figures regenerated** (all 95 trial-rep + 2 omnibus + 2 _viz_manual,
  Dropbox figures/s07_s08) with this code; 1.25 set stashed in
  `figures/s07_s08_stash_2026-10-05_threshold1.25/`. Cached auto_aligned CSV
  NOT rewritten (notebook C regenerates it). The manual worklist report
  (reports/s07_s08_manual_worklist.*) predates this change - stale.
