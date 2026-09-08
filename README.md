# Stride Estimation IMU

A Python library for estimating walking strides and gait parameters from IMU (Inertial Measurement Unit) sensor data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremydwong/stride_estimation_imu/blob/main/notebooks/demo_colab_one_foot.ipynb) Single Foot Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremydwong/stride_estimation_imu/blob/main/notebooks/demo_colab_two_feet_head_exphand.ipynb) Two Feet + Head Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremydwong/stride_estimation_imu/blob/main/notebooks/demo_brock_s07_s08_auto.ipynb) Brock two-subject session (s07/s08): auto bout scoring, per-trial inspection, interactive rescoring — bring the session data via Google Drive

## Features

- Load and process APDM sensor data from `.h5` files
- Inertial mechanization: integrate angular velocity and acceleration to compute position
- Automatic footfall detection during stance phases
- Zero velocity updates (ZUPT) for drift correction
- Stride segmentation and gait parameter extraction
- Walking bout detection using quiet period analysis
- Multi-IMU synchronization (feet, head, hand)
- Visualization tools for stride patterns and variability

## Installation

```bash
# Clone the repository
git clone https://github.com/jeremydwong/stride_estimation_imu.git
cd stride_estimation_imu

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import stride_imu as imu
from stride_imu.apdm import load_imu_recording

# Load IMU data from APDM .h5 file
recording = load_imu_recording('path/to/sensor.h5')

# Compute position trajectory
walk_info = imu.compute_position(recording.Wb, recording.Ab, recording.period)

# Segment strides
strides = imu.stride_segmentation(walk_info, recording.period)

# Plot results
imu.plt_ltrl_frwd_strides(strides)
imu.plt_stride_var(strides)
```

## Demos

### Track walking investigation

[`notebooks/debug_track_foot_difference.ipynb`](notebooks/debug_track_foot_difference.ipynb)
compares the included October 29 left/right track bout, diagnoses endpoint-based
plot alignment, and sweeps tilt/footfall settings and gyro-bias calibration.
Reproduce the extraction, metrics, and figures headlessly:

```bash
python scripts/debug_track_foot_difference.py
python scripts/debug_track_extended.py
python -m unittest discover -s tests -v
```

The strongest candidate, `stride_imu.experimental.compute_position_stride_gravity`,
uses the whole contact-to-contact stride to estimate tilt. It reduces both
horizontal disagreement and vertical drift, without enforcing track geometry or
loop closure. It is offline and explicitly opt-in; existing single- and two-foot
processing remains unchanged. The notebook includes a separate real bout,
synthetic tracks with known truth, calibration sensitivity, and rejected controls.

### Single Foot Analysis (`examples/demo_one_foot.py`)

Basic stride estimation using a single foot-mounted IMU:

```bash
uv run python examples/demo_one_foot.py
```

This demo:
1. Loads IMU data from a single foot sensor
2. Performs inertial mechanization to compute position
3. Visualizes the 3D trajectory and stride patterns

### Two Feet + Head Analysis (`examples/demo_two_feet_head.py`)

Comprehensive gait analysis using multiple synchronized IMUs:

```bash
uv run python examples/demo_two_feet_head.py
```

This demo:
1. Loads and synchronizes 4 IMUs (left foot, right foot, head, hand)
2. Detects walking bouts by finding quiet periods that bound active walking
3. Interactive bout selection with accelerometer visualization
4. Computes step metrics (speed, length, duration) for each foot
5. Analyzes head motion during walking
6. Generates comparison plots across walking bouts
7. Caches results for faster re-analysis

## API Reference

### Data Loading (`stride_imu.apdm`)

| Function | Description |
|----------|-------------|
| `load_imu_recording(filepath)` | Load APDM .h5 file as an `ImuRecording` object |
| `find_overlapping_recordings(recordings)` | Synchronize multiple IMU recordings |
| `sync_apdm(files)` | Synchronize multiple APDM files |

### Inertial Processing (`stride_imu.inertial`)

| Function | Description |
|----------|-------------|
| `compute_position(Wb, Ab, period)` | Main inertial mechanization pipeline |
| `compute_position_two_imus(...)` | Process two foot IMUs together |
| `stride_segmentation(walk_info, period)` | Extract individual strides |
| `detect_walking_section(walk_info, period)` | Find walking vs stationary periods |
| `detect_walking_bouts(Wb, Ab, period)` | Detect walking bouts using quiet periods |
| `detect_quiet_periods(Wb, Ab, period)` | Find stationary periods |
| `find_bouts_near_time(bouts, time, target)` | Find bouts near a target time |

### Visualization (`stride_imu.plotting`)

| Function | Description |
|----------|-------------|
| `plt_ltrl_frwd_strides(strides)` | Plot lateral vs forward stride trajectories |
| `plt_frwd_elev_strides(strides)` | Plot forward vs elevation stride trajectories |
| `plt_stride_var(strides)` | Plot stride variability ellipse (2D Gaussian) |
| `plt_walk_info_position(walk_info)` | 3D plot of position trajectory |

### Data Structures

**`ImuRecording`**: Container for synchronized IMU data
- `Wb`: Angular velocity (rad/sample)
- `Ab`: Acceleration (m/s^2)
- `time_datetime`: Timestamps as datetime objects
- `period`: Sampling period (seconds)
- `tz_offset_hours`: UTC offset from sensor config (e.g., -6.0 for MDT)

**`WalkingBout`**: Detected walking segment
- `start_idx`, `end_idx`: Sample indices
- `duration_seconds`: Bout duration
- `quiet_before_idx`, `quiet_after_idx`: Bounding quiet periods

## Google Colab

To run in Google Colab:

1. Click the "Open in Colab" badge above (requires GitHub mirror), or
2. Upload the notebook manually:
   - Download `notebooks/demo_colab.ipynb` from this repository
   - Go to [Google Colab](https://colab.research.google.com/)
   - File > Upload notebook
   - Upload your IMU `.h5` files to the Colab runtime

**Note**: The Colab badge requires the repository to be mirrored to GitHub. See instructions below.

### Setting up GitHub Mirror for Colab

The repository is already mirrored at https://github.com/jeremydwong/stride_estimation_imu

To sync future changes:
```bash
git push github main
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

[jeremydwong.github.io](https://jeremydwong.github.io)
