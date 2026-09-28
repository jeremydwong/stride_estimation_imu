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

## Student guide: keep your work and update the library

You do not need to be a programmer to use these notebooks. A **repository** (or
“repo”) is a project folder whose changes Git can record. **GitHub** hosts a copy
online. A **notebook** (`.ipynb`) mixes instructions, runnable code, and figures.
An **import** lets your notebook use the library's functions without copying them.

Choose one route; you do not need to do both:

| Route | Best for | Where your work lives |
| --- | --- | --- |
| **A — separate work folder (recommended first)** | Running and adapting the supplied notebooks | Beside the library folder, outside it |
| **B — your own repository** | A separate research project with its own history and recorded software versions | In your own project; this library is installed as a dependency |

Both routes use **uv**, a tool that installs Python and the required packages,
and **JupyterLab**, the application that opens notebooks in your browser. The
notebooks run on your computer even though the interface is in a browser.

### Before you start (either route)

1. Install [Git](https://git-scm.com/downloads) and
   [uv using its official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
   Pick the instructions for your operating system. These routes use Python 3.12;
   uv can download it for you. This library currently requires Python below 3.13.
2. Open **Terminal** on a Mac, or **PowerShell** on Windows. This is the window
   where you type commands. Close and reopen it after installing the tools.
3. Check that both tools respond:

   ```sh
   git --version
   uv --version
   ```

4. Use Finder/File Explorer to create a folder called `walking-study` somewhere
   you can find again. Open a terminal in that folder (Windows: right-click the
   folder and choose **Open in Terminal**; Mac: right-click it and look under
   **Services → New Terminal at Folder**). Alternatively, type `cd `, drag the
   folder into the terminal, and press Enter on a Mac. `cd` means “go to this
   folder”; `cd ..` means “go up one folder.”

Run each command below one line at a time. Do not type the surrounding backticks.
An initial installation may take several minutes. If a command fails, stop there
and check the error before running the next command.

### A. Keep a personal work folder beside the library

**1. Download the library and install its notebook tools.** Starting in
`walking-study`, run:

```sh
git clone https://github.com/jeremydwong/stride_estimation_imu.git
cd stride_estimation_imu
uv sync --python 3.12
uv run python -m ipykernel install --user --name stride-study --display-name "Stride study"
cd ..
mkdir my-work
```

The `ipykernel` command gives this Python installation a name you can select in
Jupyter. A **kernel** is the Python process that actually runs your notebook.
Your folders now look like this:

```text
walking-study/
    stride_estimation_imu/    Library downloaded from GitHub
    my-work/                 Your notebooks, data, and results
```

**2. Copy a notebook into `my-work`.** Use Finder/File Explorer to copy (not move)
a notebook from `stride_estimation_imu/notebooks`. For example, copy
`demo_colab_one_foot.ipynb` for the single-foot demo, or
`demo_brock_s07_s08_auto.ipynb` for that Brock session. Rename your copy, for
example `my_first_analysis.ipynb`. Other sessions need their corresponding data
and settings; changing the notebook's filename does not change the session.

Keep personal work outside `stride_estimation_imu`. A new folder inside the
clone is less reliable: future library files could use the same name, and some
Git cleanup commands can remove untracked files.

**3. Open your work folder.** From `walking-study`, run:

```sh
cd my-work
uv run --project ../stride_estimation_imu jupyter lab .
```

Open your copied notebook and select **Kernel → Change Kernel → Stride study**.
Follow [Preparing your copied notebook](#preparing-your-copied-notebook) below.

**4. Return another day.** Open a terminal in `my-work` and run the same
`uv run --project ../stride_estimation_imu jupyter lab .` command. You do not need
to clone or install the kernel again. Save your notebook before closing Jupyter;
Ctrl+C in the terminal stops its server (follow any shutdown prompt).

**5. Get library improvements when you are ready.** Save your work and stop
Jupyter. Open a terminal in `stride_estimation_imu`, then run:

```sh
git pull --ff-only
uv sync --python 3.12
```

Restart Jupyter using step 3. These commands update the library, while your
`my-work` files stay in their separate folder. Your copied notebook will **not**
automatically gain new notebook cells: compare it with the latest original and
copy across changes you want. Updated analysis functions can change results, so
keep previous outputs before rerunning. To record the library version used for
an analysis, run `git rev-parse HEAD` in the library folder and paste the result
into a text cell in your notebook.

If Git says local changes would be overwritten, stop and ask for help. Do not
use `git reset --hard` or `git clean` to get past the message. Back up `my-work`
separately: Git in the library folder does not back it up.

### B. Create your own repository and import this library

This route keeps your notebook project independent. You do not need to copy the
library's source code into your project, fork it, or set up a Git submodule.

**1. Create your project.** Starting in `walking-study`, run:

```sh
mkdir my-stride-project
cd my-stride-project
uv init --python 3.12 --vcs git
```

**2. Set the supported Python range.** Open the new `pyproject.toml` file in a
plain-text editor (for example VS Code or Notepad, not Word). Find the line
starting with `requires-python` and replace that whole line with:

```toml
requires-python = ">=3.12,<3.13"
```

Save the file. This tells uv to choose versions compatible with this library.

**3. Install the library and notebook tools.** In `my-stride-project`, run:

```sh
uv add "stride-imu @ git+https://github.com/jeremydwong/stride_estimation_imu.git@main"
uv add --dev jupyterlab ipykernel ipympl
uv run python -m ipykernel install --user --name my-stride-project --display-name "My stride project"
uv run jupyter lab .
```

The installation name is `stride-imu`; the Python import name is `stride_imu`.
uv records the library's exact Git revision in `uv.lock`, so your project can
keep using that version until you deliberately update it.
([How uv locks Git dependencies](https://docs.astral.sh/uv/concepts/projects/sync/).)

**4. Add your notebook and data.** Create folders called `notebooks`, `data`, and
`results` inside your project. Download a demo from this repository's
[`notebooks` folder](notebooks) using GitHub's **Download raw file** button, and
save it in your `notebooks` folder. Alternatively, copy a demo from a local clone.
Installing the library does not copy demo notebooks or recording files into your
project. Open your notebook in Jupyter and choose the **My stride project**
kernel. Follow the preparation steps below.

**5. Record your own work with Git.** Before adding files, open `.gitignore` in
your project and append these lines. This tells Git which files to leave out:

```gitignore
data/
results/
.ipynb_checkpoints/
.env
*.h5
```

Keep raw recordings and generated results in those folders. Notebook outputs
can also contain participant information; clear sensitive outputs before
sharing. Back up excluded data and scoring files using your lab's approved
storage. A Git commit does not back up ignored files.

Save your notebook, then run:

```sh
git status
git add .gitignore pyproject.toml uv.lock .python-version notebooks
git commit -m "Start my walking analysis"
```

`git status` shows what changed; `git add` selects files for the next saved
snapshot; `git commit` records that snapshot locally. If Git asks for your name
and email, follow its instructions and repeat the commit. For later snapshots,
repeat these commands with a message describing the changes.

To put the project on GitHub, create a new **private**, empty repository in your
own account (do not add a README, licence, or `.gitignore` there). Follow GitHub's
“push an existing repository” commands, using **your** repository URL. A local
commit is not uploaded until you push. Check your lab's sharing rules first.

**6. Reopen or update deliberately.** On a normal day, open a terminal in
`my-stride-project` and run `uv run jupyter lab .`. To adopt a newer library
version, save your work, stop Jupyter, and run:

```sh
uv lock --upgrade-package stride-imu
uv sync
```

Restart Jupyter and rerun your analysis to check the results, then commit the
updated `uv.lock` along with any related notebook changes. `git pull` in your
own project retrieves your project's changes; it does not update the library
dependency. Someone cloning your project can run `uv sync --python 3.12` to
install its recorded dependencies, then register their kernel as in step 3.
They will still need authorized access to the data.

### Preparing your copied notebook

1. **Check the imports.** In a new code cell, run this with Shift+Enter:

   ```python
   import stride_imu as imu
   import brock_functions  # Helpers used by the Brock notebooks
   print(imu.__file__)
   print(brock_functions.__file__)
   ```

   Both should print file locations without an error. In your copied demo,
   remove any `sys.path.insert(...)` line that points to `../src` or a developer's
   computer. Installation now makes these imports work. Keep the regular imports.
   Colab-only installation cells are not needed locally; cells guarded by an
   `IN_COLAB` check normally skip themselves.
2. **Point the notebook at your data.** Obtain the correct recording and trial
   table from your supervisor. Find the notebook's configuration cell (look for
   names such as `DATA_DIR`, `H5_FILE`, or `FILE`) and replace the example paths.
   Copying a notebook does not copy its data, and paths such as `../data/...` or
   `/Users/jeremy/...` may not work on your computer. An absolute path is easiest
   at first; use your actual path and filename, for example:

   ```python
   DATA_DIR = r"C:\Users\YourName\walking-study\my-stride-project\data"  # Windows
   # Mac alternative: DATA_DIR = "/Users/YourName/walking-study/my-stride-project/data"
   ```

   Keep the `r` before a Windows path. Update the recording and trial-table
   filenames too. Use the `my-work` location instead if following route A.
3. **Check where results will be saved.** Read the configuration/output cells
   before running. The Brock s07/s08 notebook writes its scoring CSVs into a
   `cached data` folder beside the input recording. Other notebooks may use
   different locations. Keep your inputs and outputs in your personal work area.
4. **Run from the top down.** Use Shift+Enter for each cell, reading the text
   between cells. After changing settings or updating the library, use
   **Kernel → Restart Kernel and Run All Cells** so old in-memory results do not
   get mixed with new ones. Save the notebook with Ctrl+S (Windows) or Cmd+S (Mac).

### Saving Brock scoring changes

Saving the notebook alone does **not** save newly dragged scoring boundaries.
In the s07/s08 notebook, after adjusting a window:

1. Click **Save adjustments** in the scoring controls. Repeat for each edited
   trial. This writes the edits to `RESCORE_CSV`.
2. After your edits, run this cell to apply the saved windows and write the
   updated aligned table:

   ```python
   from brock_functions import apply_manual_rescore, refresh_manual_distances

   aligned = apply_manual_rescore(aligned, RESCORE_CSV)
   aligned = refresh_manual_distances(feet, aligned)
   aligned.to_csv(ALIGNED_CSV, index=False)
   print("Saved adjustments:", RESCORE_CSV)
   print("Saved aligned windows:", ALIGNED_CSV)
   ```

3. The notebook's save-and-compare cell in section H remeasures corrected bout
   distances and redraws the **omnibus** figure before and after adjustments.
   Rerun other downstream analysis and figure cells to reflect the revised
   windows; refreshing bout distances does not recalculate per-step measurements. Save
   your notebook and back up both CSV files. In Colab, also download the files
   before the temporary session ends.

### Getting help without needing to know the terminology

If running a notebook cell does nothing and the kernel indicator never becomes
busy, click inside the code cell (not its figure) and try Shift+Enter or the Run
button. Test a new cell with `print("Kernel is responding", flush=True)`. If
that also does nothing, save the notebook and **refresh the browser page**.
This can reconnect to the existing kernel and normally preserves loaded data;
**Restart Kernel** clears variables and requires rerunning the analysis.
Save scoring edits with the figure's Save button first if it still responds;
saving the notebook alone does not save pending scoring edits.

If you see `ModuleNotFoundError`, first check the selected kernel. If you see
`FileNotFoundError`, check the configured data path and filename. If `git` or
`uv` is “not recognized” or “not found,” reopen your terminal and check the
installation. The **terminal commands** above go in Terminal/PowerShell;
**Python code** such as `import stride_imu` goes in a notebook code cell.

You can give your own ChatGPT this prompt, along with this README and the exact
error text (remove private data and credentials):

> I am a student, not a programmer. I am following route [A or B] in the
> stride_estimation_imu README on [Mac or Windows]. I reached step [number].
> My project folder is [path]. I ran [exact command or cell] and got [error].
> Explain the next step in plain language, say whether to use the terminal or
> notebook, and help me preserve my existing work. Do not change the analysis
> methods just to make an error disappear.

## Installation (short version)

For readers already comfortable with Python projects:

```sh
git clone https://github.com/jeremydwong/stride_estimation_imu.git
cd stride_estimation_imu
uv sync --python 3.12
uv run jupyter lab
```

For an existing Python 3.10–3.12 environment, `python -m pip install -e .`
installs the library from a local checkout; install notebook tools separately.

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

For a browser-only start, click one of the **Open in Colab** badges above and
save a personal copy to your Google Drive before editing. Follow that notebook's
setup and data instructions. Alternatively, download
[`demo_colab_one_foot.ipynb`](notebooks/demo_colab_one_foot.ipynb) and use
**File → Upload notebook** in [Google Colab](https://colab.research.google.com/).

Colab's runtime storage is temporary. Saving a notebook to Drive does not by
itself preserve CSVs, uploaded recordings, or exported figures; download your
results or explicitly save them to your authorized Drive storage before ending
the session. The two local workflows above keep files on your own computer.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

[jeremydwong.github.io](https://jeremydwong.github.io)
