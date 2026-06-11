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

**Not done yet:**
- Haven't run `step_segmentation` over all 192 bouts / added step columns to
  `brock_trial_table.csv`.
- Footfall recall is the next real lever: 1–2 missed contacts per bout (the
  `skips`/`slow` counts) — restoring the `foot_fall_opposite_velocity` merge in
  `compute_position_two_imus` (commented call sites, function already ported)
  would likely fix those.