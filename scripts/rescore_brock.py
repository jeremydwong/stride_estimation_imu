"""Rescore Brock bouts by clicking, from the TERMINAL (no Jupyter needed).

Opens the manual_correct inspection figures in native windows — this avoids
every VS Code / Jupyter interactive-plot problem. For each trial-rep:

  1. press 'rescore A' (or 'rescore B'), then click the plot twice:
     first click = new bout START, second = STOP;
  2. press 'save' — corrections append to brock_<tag>_manual_rescore.csv;
  3. CLOSE the window to move on to the next figure.

Afterwards run the notebook's last cell (apply_manual_rescore) to fold the
corrections into brock_<tag>_auto_aligned.csv.

Usage (from the repo folder):
    uv run python rescore_brock.py s07_s08 1          # trial 1, both reps
    uv run python rescore_brock.py s07_s08 1 37:2     # + trial 37 rep 2 only
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from brock_functions import load_available_feet, manual_correct

H5_DIR = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data'


def parse_trial(arg):
    """'12' -> 12 (both reps); '12:2' or '12.2' -> (12, 2)."""
    for sep in (':', '.'):
        if sep in arg:
            t, r = arg.split(sep)
            return (int(t), int(r))
    return int(arg)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    tag, trials = sys.argv[1], [parse_trial(a) for a in sys.argv[2:]]

    hits = glob.glob(os.path.join(H5_DIR, f'imuData_{tag}_*.h5'))
    if len(hits) != 1:
        sys.exit(f"expected exactly one {H5_DIR}/imuData_{tag}_*.h5, found: {hits}")
    cache = os.path.join(H5_DIR, 'cached data')
    aligned_csv = os.path.join(cache, f'brock_{tag}_auto_aligned.csv')
    if not os.path.exists(aligned_csv):
        sys.exit(f'{aligned_csv} not found - run the demo_brock_{tag}_auto '
                 f'notebook first (it writes the aligned table).')

    print(f'loading {os.path.basename(hits[0])} ...')
    feet = load_available_feet(hits[0])
    aligned = pd.read_csv(aligned_csv)
    out_csv = os.path.join(cache, f'brock_{tag}_manual_rescore.csv')
    manual_correct(feet, aligned, trials, out_csv=out_csv, session_tag=tag)
    print(f"done - any saved corrections are in {out_csv};"
          f" re-run the notebook's apply_manual_rescore cell to fold them in.")


if __name__ == '__main__':
    main()
