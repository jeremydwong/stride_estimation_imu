"""Generate a per-session Brock auto-scoring notebook from the master.

Clones demo_brock_dataset2_auto.ipynb (the s03_s04 notebook where the method
was developed), patches session-specific paths/prose, updates the stop-
inference description to the current per-person rule, and appends:

  G. per-trial visualization figures (figures/<tag>/brock_<tag>_trial<N>_viz.svg)
  H. manual inspection / click-to-rescore (manual_correct)

Usage:
    uv run python make_brock_session_notebook.py s05_s06 20260703
    uv run python make_brock_session_notebook.py s07_s08 20260708
"""
import copy
import json
import os
import sys

MASTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'notebooks', 'demo_brock_dataset2_auto.ipynb')



# Colab bootstrap cell inserted after the title of every session notebook.
COLAB_CELL = """\
# --- Google Colab setup (safe to run anywhere: it is a NO-OP locally) -------
# On Colab this clones the code, installs the interactive-plot backend, and
# mounts your Google Drive for the session data (the .h5 is ~200 MB - too big
# for the git repo, so put a copy of the 'imu data' folder in your Drive and
# check DATA_DIR in the next cell points at it).
import os, sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    if not os.path.isdir('/content/stride_estimation_imu'):
        !git clone -q https://github.com/jeremydwong/stride_estimation_imu.git /content/stride_estimation_imu
    %cd /content/stride_estimation_imu/notebooks
    !pip install -q ipympl
    from google.colab import output, drive
    output.enable_custom_widget_manager()   # lets the interactive cells work
    drive.mount('/content/drive')
"""


# ---------------------------------------------------------------------------
# Tutorial playground appended at the end of every session notebook.
# Kept as module-level data so patch scripts can reuse the exact same cells.
# Each entry: ('markdown'|'code', source). '{tag}' is filled per session.
# ---------------------------------------------------------------------------
TUTORIAL_CELLS = [
    ('markdown', """\
## I. Student playground — the three data types

Everything in this notebook is functions applied to **three kinds of data**.
If you are comfortable with these, you can re-analyse anything:

1. **Raw IMU recordings** — `feet` is a plain dict `{{label -> ImuRecording}}`
   (`'left_foot_a'`, `'right_foot_b'`, ...). An `ImuRecording` holds the gyro
   `Wb` (rad/sample), the accelerometer `Ab` (m/s²) and the sample `period`
   (s); `rec[a:b]` slices it like a list. Every trajectory, footfall and speed
   in this notebook is *derived* from these two signals.
2. **Tables** — `conditions` (one row per trial: distance, package, pose,
   experimenter notes) and `aligned` (one row per expected bout: when it
   happened, measured distance, who walked). Both are ordinary pandas
   DataFrames: filter, merge, group, plot.
3. **Rescore controllers** — the return value of `manual_correct()`: one
   object per inspected trial-rep that knows its window, the current bout
   spans and any rescores you clicked. `print()` one to see its state.

The cells below poke at each in turn — copy, edit and re-run them freely.

None of this needs memorizing — human working memory holds ~7 items, and these
objects hold far more. Cell (0) below shows the three introspection patterns
(`list(d)` / `.items()` for dicts, `vars(obj)` for objects, `.dtypes`/`.head()`
for tables) that let you ask *any* variable what it contains.
"""),
    ('code', """\\
# --- 0) don't memorize - ask the object what it contains --------------------
# Human working memory holds ~7 items; this notebook's objects hold far more.
# So never try to remember what is inside something - ASK it. Three patterns
# cover every named variable in this notebook:

# (a) a DICT, like `feet`: list its keys, or loop over key -> value pairs
print('feet is a', type(feet).__name__, 'with keys:', list(feet))
for name, rec in feet.items():
    print(f'   {{name}}: {{len(rec)}} samples @ {{1/rec.period:.0f}} Hz')

# (b) an OBJECT, like one ImuRecording: vars(obj) is a dict of its fields.
# Print names + types + shapes instead of the values (arrays are huge):
rec = feet['left_foot_a']
print('\\none ImuRecording contains:')
for field, value in vars(rec).items():
    desc = getattr(value, 'shape', None) or type(value).__name__
    print(f'   rec.{{field:22s}} {{desc}}')
# (for anything else: dir(obj) lists methods too, help(obj) prints the docs)

# (c) a DataFrame, like `aligned` or `conditions`: .columns/.dtypes name the
# columns, .head() peeks at rows, .describe() summarizes numeric columns
print('\\naligned columns:'); print(aligned.dtypes.to_string())
aligned.head(3)
"""),
    ('code', """\
# --- 1) raw IMU data: pick a bout, look at the actual signals ---------------
rec = feet['left_foot_a']
print(f"left_foot_a: {{len(rec)}} samples @ {{1/rec.period:.0f}} Hz "
      f"({{len(rec)*rec.period/60:.1f}} min of recording)")

bout = aligned.dropna(subset=['start_s']).iloc[0]      # first scored bout
j0, j1 = int(bout.start_s / rec.period), int(bout.stop_s / rec.period)
piece = rec[j0:j1]                                     # slicing = new recording
t = np.arange(len(piece)) * rec.period

fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
axes[0].plot(t, np.linalg.norm(piece.Ab, axis=1), lw=0.6)
axes[0].set_ylabel('|A| [m/s²]')
axes[1].plot(t, np.linalg.norm(piece.Wb, axis=1) / rec.period, lw=0.6)
axes[1].set(ylabel='|gyro| [rad/s]', xlabel='time in bout [s]')
fig.suptitle(f"trial {{bout.trial:.0f}} rep {{bout.rep:.0f}} bout {{bout.bout:.0f}}: "
             f"raw left_foot_a signals")
plt.show()
"""),
    ('code', """\
# --- 2) re-run the mechanization yourself -----------------------------------
# compute_position_two_imus integrates gyro + accel into foot trajectories,
# applying zero-velocity updates at every detected stance. It returns one
# FootTrajectory per foot: .P (position, m), .Vm (speed, m/s), .FF_walking
# (footfall mask), .euler ... — this is the engine under every figure above.
suffix = str(bout.walker)[-1]          # 'a' or 'b': who walked this bout
L = feet[f'left_foot_{{suffix}}'][j0:j1]
R = feet[f'right_foot_{{suffix}}'][j0:j1]
left_info, right_info = imu.compute_position_two_imus(L.Wb, L.Ab, R.Wb, R.Ab,
                                                      L.period)

fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
axes[0].plot(t, left_info.Vm, lw=0.7, label='left')
axes[0].plot(t, right_info.Vm, lw=0.7, label='right')
axes[0].set_ylabel('foot speed [m/s]'); axes[0].legend()
axes[1].plot(t, np.linalg.norm(left_info.P[:, :2] - left_info.P[0, :2], axis=1),
             lw=0.8, label='left')
axes[1].plot(t, np.linalg.norm(right_info.P[:, :2] - right_info.P[0, :2], axis=1),
             lw=0.8, label='right')
axes[1].set(ylabel='distance from start [m]', xlabel='time in bout [s]')
plt.show()
print(f"net horizontal travel: "
      f"left {{np.linalg.norm(left_info.P[-1,:2]-left_info.P[0,:2]):.2f}} m, "
      f"right {{np.linalg.norm(right_info.P[-1,:2]-right_info.P[0,:2]):.2f}} m "
      f"(trial table says {{bout.distance_m}} m)")
"""),
    ('code', """\
# --- 3) tables: merge results with conditions, plot anything vs anything ----
# `aligned` already carries distance_m/package per bout; the merge adds the
# hand-off pose and the experimenter's per-rep notes from `conditions`.
table = (aligned.dropna(subset=['start_s'])
         .merge(conditions[['trial', 'pose', 'rep1_status', 'rep2_status']],
                on='trial'))
# crude bout-average speed: walked distance over the whole scored window
# (includes box handling — the per-STEP speeds in the figures are the real
# gait measure; this is just an easy table exercise)
table['bout_speed_mps'] = table['measured_m'] / table['duration_s']

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
table.boxplot(column='bout_speed_mps', by='distance_m', ax=axes[0])
table.boxplot(column='bout_speed_mps', by='pose', ax=axes[1])
for ax in axes:
    ax.set_ylabel('bout speed [m/s]'); ax.set_title('')
fig.suptitle('bout-average speed by condition')
plt.tight_layout(); plt.show()

table.groupby('distance_m')['bout_speed_mps'].agg(['mean', 'std', 'count']).round(2)
"""),
    ('code', '''\
# --- 5) the per-trial inspection figure, on demand --------------------------
# inspect_snipped_trial draws ONE trial's full figure (all matched bouts) and
# RETURNS the matplotlib Figure: grey pre-walk accel context, the Start/Stop
# button presses as vertical lines, and the reaction time / duration written
# on each bout. Change the trial number and re-run. Full details:
# help(inspect_snipped_trial)
from brock_functions import inspect_snipped_trial

fig = inspect_snipped_trial(feet, aligned, trial=5, session_tag=SESSION_TAG,
                            prefix_seconds=5.0)
'''),
    ('code', '''\
# --- 6) interactive: DRAG the gait start / bout end -------------------------
# Same figure as (5) but live: the GREEN line is the snug gait start, the
# PURPLE line the bout end. Mouse-down grabs whichever is closer, drag it,
# release -> that bout re-runs with the adjusted bounds and redraws (the
# time axis re-zeroes to the new gait start). The ORIGINAL clicks are never
# modified: press 'save adjustments' (bottom right) to append your edits to
# the manual-rescore CSV - the fold-in cell below then stamps them
# manual=True with the person. KEEP the return value or dragging dies.
# Needs the interactive backend - same setup notes as section H.
from brock_functions import interactive_inspect_trial

drag_ctrl = interactive_inspect_trial(
    feet, aligned, trial=5, session_tag=SESSION_TAG,
    out_csv=RESCORE_CSV)
'''),
    ('code', """\
# --- 4) rescore controllers (from section H's manual_correct) ---------------
# With the draggable inspector above, you mostly do NOT need manual_correct
# for bouts that exist but have bad bounds - just drag them. manual_correct's
# remaining job is MISSED bouts: when no snip exists there is nothing to drag,
# and its inspection window (placed from the interpolated bout time) lets you
# click a start/stop where the alignment found none. Its controllers know
# their state - print one to see it:
if 'controllers' in dir() and controllers:
    for c in controllers:
        print(c)          # trial/rep, shown window, bout spans, your rescores
else:
    print('no controllers - only needed for MISSED bouts; run the '
          'MANUAL_INSPECT cell in section H with a trial that has a red X')
"""),
]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'outputs': [],
            'execution_count': None, 'source': text.splitlines(keepends=True)}


def build(tag, date):
    subjects = tag.replace('_', '/')
    nb = json.load(open(MASTER))
    out = copy.deepcopy(nb)
    cells = out['cells']

    # --- title cell ---
    src = ''.join(cells[0]['source'])
    src = src.replace(
        '# Brock dataset 2 — automatic bout scoring from button-press events',
        f'# Brock {tag} ({date}) — automatic bout scoring from button-press events')
    src = src.replace(
        'The 20260622 session (s03/s04) has no hand-scored `ExpTrialBounds.mat`.',
        f'The {date} session ({subjects}) has no hand-scored `ExpTrialBounds.mat`.')
    src = src.replace(
        '\nNOTE: this session has only 3 foot IMUs (`left_foot_a` is absent), '
        'so subject-a bouts\nare measured from the right foot alone.\n',
        '\nThe cell in section A prints which foot IMUs the file actually '
        'contains — if one is\nabsent, that subject\'s bouts are measured from '
        'the remaining foot alone.\n')
    cells[0]['source'] = src.splitlines(keepends=True)

    # --- Colab bootstrap cell right after the title ---
    cells.insert(1, code(COLAB_CELL))

    # --- config cell: paths, session tag, new inference parameters ---
    src = ''.join(cells[2]['source'])
    src = src.replace(
        "H5_FILE = os.environ.get('STRIDE_DATA_FILE',\n"
        "    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/imuData_s03_s04_20260622.h5')",
        "# where the session data lives: your Drive on Colab, Dropbox locally\n"
        "DATA_DIR = ('/content/drive/MyDrive/Treadmill Brock 2025/imu data'\n"
        "            if 'google.colab' in sys.modules else\n"
        "            '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data')\n"
        "H5_FILE = os.environ.get('STRIDE_DATA_FILE',\n"
        f"    os.path.join(DATA_DIR, 'imuData_{tag}_{date}.h5'))")
    src = src.replace(
        "CONDITION_CSV = os.environ.get('STRIDE_TRIALTABLE_FILE',\n"
        "    '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/trialtable_20260622.csv')",
        "CONDITION_CSV = os.environ.get('STRIDE_TRIALTABLE_FILE',\n"
        f"    os.path.join(DATA_DIR, 'trialtable_{date}.csv'))")
    src = src.replace(
        'from brock_functions import (load_events, automatically_score_movements_from_events,\n'
        '                             load_condition_table, load_available_feet,\n'
        '                             align_snips_to_trial_table, explain_missed, bout_spacing)',
        'from brock_functions import (load_events, automatically_score_movements_from_events,\n'
        '                             load_condition_table, load_available_feet,\n'
        '                             align_snips_to_trial_table, explain_missed, bout_spacing,\n'
        '                             save_trial_figures, manual_correct,\n'
        '                             apply_manual_rescore)')
    src = src.replace(
        "CONDITION_CSV = os.environ.get('STRIDE_TRIALTABLE_FILE',\n"
        f"    os.path.join(DATA_DIR, 'trialtable_{date}.csv'))",
        "CONDITION_CSV = os.environ.get('STRIDE_TRIALTABLE_FILE',\n"
        f"    os.path.join(DATA_DIR, 'trialtable_{date}.csv'))\n"
        "\n"
        "# all generated tables live beside the imu data, next to figures/\n"
        "CACHE_DIR = os.path.join(os.path.dirname(H5_FILE), 'cached data')\n"
        "os.makedirs(CACHE_DIR, exist_ok=True)\n"
        "ALIGNED_CSV = os.path.join(CACHE_DIR, f'brock_{SESSION_TAG}_auto_aligned.csv')\n"
        "RESCORE_CSV = os.path.join(CACHE_DIR, f'brock_{SESSION_TAG}_manual_rescore.csv')")
    src = src.replace(
        "H5_FILE = os.environ.get('STRIDE_DATA_FILE',",
        f"SESSION_TAG = '{tag}'\n"
        "# os.environ.get(NAME, default) = use the environment variable when one\n"
        "# is set, else the default path. It never SETS the variable - it is a\n"
        "# hook so a batch run can point this notebook at other files unedited.\n"
        "H5_FILE = os.environ.get('STRIDE_DATA_FILE',")
    src = src.replace(
        '# Recover bouts the experimenter STARTED but forgot to end: close them where\n'
        '# every foot settles into stance. Set INFER_STOPS = False for clicks-only.\n'
        'INFER_STOPS = True\n'
        'QUIET_SECONDS = 1.5        # stance long enough to call the bout over\n',
        '# Recover bouts the experimenter STARTED but forgot to end: close them where\n'
        '# the walker plants both feet (the hand-off stance). INFER_STOPS = False for\n'
        '# clicks-only.\n'
        'INFER_STOPS = True\n'
        'QUIET_SECONDS = 0.75       # walker-pair stance that ends the bout: must be\n'
        '                           # shorter than a hand-off stance (~0.9 s observed)\n'
        '                           # and longer than within-gait double-stance (~0.3 s)\n')
    cells[2]['source'] = src.splitlines(keepends=True)

    # --- section A markdown: describe the per-person stop inference ---
    src = ''.join(cells[3]['source'])
    src = src.replace(
        'With\n`INFER_STOPS = True`, `infer_stop_from_quiet()` closes it at the '
        'first moment **every\nfoot is simultaneously static for `QUIET_SECONDS`** '
        '(via `detect_quiet_time`), searching\nonly up to the next button press so '
        'an inferred bout can never swallow the following\none.',
        'With\n`INFER_STOPS = True`, `infer_stop_from_quiet()` closes it at the end '
        'of the **first\nsustained walk**: per person (their own two feet only), still '
        'gaps shorter than\n`QUIET_SECONDS` are absorbed into the motion run, so the '
        'walk ends exactly where the\nwalker plants both feet for `QUIET_SECONDS` — '
        'the hand-off stance. (Requiring all\nFOUR feet still at once was the first '
        'attempt, and is wrong for this protocol: that\nmoment structurally never '
        'happens before the walker\'s un-clicked return walk, so\ninferred bouts ran '
        '30-40 s for 10 s walks.) The search runs only up to the next button\npress '
        'so an inferred bout can never swallow the following one.')
    cells[3]['source'] = src.splitlines(keepends=True)

    # --- section E markdown: press counts are session-specific -> genericize ---
    src = ''.join(cells[12]['source'])
    src = src.replace(
        'blue = measured, red X = missed. Only two annotation types exist in this file\n'
        '(200 `Start`, 183 `Stop`, all from the `event` sensor) — there are no other button kinds.',
        'blue = measured, red X = missed. Only two annotation types exist in these files\n'
        '(`Start`/`Stop`, all from the `event` sensor) — there are no other button kinds.')
    cells[12]['source'] = src.splitlines(keepends=True)

    # --- section F markdown: examples come from the method-development session ---
    src = ''.join(cells[14]['source'])
    src = src.replace(
        '## F. Why is each bout missing? (documenting the logic)\n',
        '## F. Why is each bout missing? (documenting the logic)\n\n'
        '*(The worked examples below — trials 37-40, trial 2 rep 2, etc. — are from the\n'
        's03/s04 20260622 session where this method was developed; the logic applies\n'
        'unchanged here.)*\n')
    cells[14]['source'] = src.splitlines(keepends=True)

    # --- output csv name ---
    src = ''.join(cells[16]['source'])
    src = src.replace(
        "aligned.to_csv('brock_dataset2_auto_aligned.csv', index=False)\n"
        "print('wrote brock_dataset2_auto_aligned.csv,', len(aligned), 'rows')",
        'aligned.to_csv(ALIGNED_CSV, index=False)\n'
        "print(f'wrote {ALIGNED_CSV},', len(aligned), 'rows')")
    cells[16]['source'] = src.splitlines(keepends=True)

    # --- G. per-trial figures ---
    cells.append(md(
        '## G. Per-trial visualization figures\n'
        '\n'
        'One SVG per trial with every matched bout stacked (up to 2 reps × 2 walkers):\n'
        'overhead foot map on the left, raw |A| / foot speed / step speed on the right —\n'
        'the same panel as the dataset-1 velocity batch. Written to the\n'
        f'`figures/{tag}/` folder **next to the session .h5** (in Dropbox, the same\n'
        '`imu data/figures` the dataset-1 batch uses; created as needed) as\n'
        f'`brock_{tag}_trial<N>_viz.svg` — kept out of the git repo.\n'
        'This runs the two-IMU stride pipeline per bout, so expect several minutes for\n'
        'all 48 trials; set `FIG_TRIALS` to a short list (e.g. `[1, 2]`) while testing.\n'))
    cells.append(code(
        'FIG_TRIALS = None    # None = all trials; or a list like [1, 2, 37]\n'
        '# figures live in a "figures" folder NEXT TO THE IMU DATA (not in the repo)\n'
        "FIGURES_DIR = os.path.join(os.path.dirname(H5_FILE), 'figures')\n"
        '\n'
        'fig_paths = save_trial_figures(feet, aligned, session_tag=SESSION_TAG,\n'
        '                               trials=FIG_TRIALS, out_dir=FIGURES_DIR)\n'
        "print(f'{len(fig_paths)} trial figures -> {os.path.dirname(fig_paths[0])}')\n"))

    # --- H. manual inspection / rescore ---
    cells.append(md(
        '## H. Manual inspection & click-to-rescore\n'
        '\n'
        '**Prefer the draggable inspector** (tutorial cell 6 / '
        '`interactive_inspect_trial`) for bouts\nthat exist but have bad '
        'bounds — drag the gait start / bout end and press *save\n'
        'adjustments*. `manual_correct` below is mainly for **missed** '
        'bouts: with no snip\nthere is nothing to drag, and its window '
        '(placed from the interpolated bout time)\nlets you click a '
        'start/stop where the alignment found none.\n'
        '\n'
        'When a bout above looks wrong (a `missed` verdict you disagree with, a nonsense\n'
        'duration, an `[inferred stop]` figure showing extra walking), list its trial here\n'
        'and rescore it by clicking. For each trial-rep an inspection figure shows raw |A|,\n'
        'the mechanized foot speed, and horizontal excursion for every foot, with the\n'
        'current bout windows shaded (person A blue, person B orange — a missed bout has\n'
        'no shading). On the figure:\n'
        '\n'
        "1. press **rescore A** (or **rescore B**) — usually only one person's bout needs\n"
        '   fixing, so each is rescored separately;\n'
        '2. **click the plot twice**: first click = the new bout START, second = STOP;\n'
        '3. press **save** — the correction is appended to\n'
        f'   `brock_{tag}_manual_rescore.csv` (nothing is written until you press save).\n'
        '\n'
        'The **from [s]** / **to [s]** text fields set the displayed time window —\n'
        'type a number and press Enter to re-slice and redraw (useful when the\n'
        'walking you care about sits outside the default window).\n'
        '\n'
        'Afterwards `apply_manual_rescore(aligned, csv)` folds the saved corrections back\n'
        'into the table (last save wins if you rescored twice).\n'
        '\n'
        '### If the interactive figure does not appear / errors out\n'
        '\n'
        '**Easiest reliable path — skip Jupyter entirely.** The same click-to-rescore\n'
        'figures open as native windows from the terminal (from the repo folder):\n'
        '\n'
        '```\n'
        f'uv run python rescore_brock.py {tag} 1 37:2\n'
        '```\n'
        '\n'
        '(each argument is a trial — `1` shows both reps, `37:2` just rep 2; close each\n'
        'window to move to the next; press **save** before closing if you rescored).\n'
        'Corrections land in the same CSV, so the fold-in cell below works unchanged.\n'
        'Run the notebook once first — the script reads the\n'
        f'`brock_{tag}_auto_aligned.csv` it writes.\n'
        '\n'
        "If you'd rather stay in the notebook, `manual_correct` checks the environment\n"
        'BEFORE drawing anything and its error message prints which python the kernel is\n'
        'on and what to fix:\n'
        '\n'
        '* **`ModuleNotFoundError: ipympl` / "Could not switch to the interactive\n'
        '  \'widget\' backend"** → the notebook is running on the WRONG PYTHON. The\n'
        "  kernel must be this project's `.venv`: in VS Code, kernel picker (top-right)\n"
        '  → *Select Another Kernel* → *Python Environments* → **`.venv`**, then\n'
        '  *Restart* the kernel (↻) and re-run from the top. If the kernel already IS\n'
        '  `.venv`, run `uv sync` in a terminal, then restart the kernel.\n'
        "* **VS Code still won't go interactive** (it often won't) → use the terminal\n"
        '  script above, or the browser: `uv run jupyter lab` from the repo folder.\n'
        '* **Clicks do nothing** → the zoom/pan tool in the figure toolbar is switched\n'
        '  on; click its icon to turn it off, then click in the plot again.\n'
        '\n'
        '**After editing anything in `brock_functions.py`** (or pulling changes):\n'
        'restart the kernel — Python caches imported modules, so a re-run without a\n'
        'restart keeps executing the old code.\n'))
    cells.append(code(
        'MANUAL_INSPECT = []   # trials to inspect, e.g. [1, (37, 2)]\n'
        '                      # a bare number shows both reps; (trial, rep) just one\n'
        '\n'
        'if MANUAL_INSPECT:\n'
        '    controllers = manual_correct(   # keep the return value! (buttons die otherwise)\n'
        '        feet, aligned, MANUAL_INSPECT,\n'
        "        out_csv=f'brock_{SESSION_TAG}_manual_rescore.csv',\n"
        '        session_tag=SESSION_TAG)\n'
        'else:\n'
        "    print('MANUAL_INSPECT is empty - nothing to inspect. Add trial numbers'\n"
        "          ' (or (trial, rep) tuples) to the list above and re-run this cell.')\n"))
    cells.append(code(
        '# after rescoring + saving above, fold the corrections into the table:\n'
        "csv = RESCORE_CSV\n"
        'if os.path.exists(csv):\n'
        '    aligned_fixed = apply_manual_rescore(aligned, csv)\n'
        "    aligned_fixed.to_csv(ALIGNED_CSV, index=False)\n"
        "    print(f\"{int(aligned_fixed['manual'].sum())} manually-rescored bout(s) \"\n"
        "          f'folded in and saved to {ALIGNED_CSV}')\n"
        'else:\n'
        "    print('no manual corrections saved yet')\n"))

    for kind, text in TUTORIAL_CELLS:
        cells.append(md(text.format(tag=tag)) if kind == 'markdown'
                     else code(text.format(tag=tag)))

    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    fname = os.path.join(os.path.dirname(MASTER),
                         f'demo_brock_{tag}_auto.ipynb')
    json.dump(out, open(fname, 'w'), indent=1)
    print('wrote', fname)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2])
