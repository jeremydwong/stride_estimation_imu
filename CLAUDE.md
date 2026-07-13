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