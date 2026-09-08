"""Extract per-session trial tables from the SubInfo2 experimenter workbook.

Each session sheet (s01_s02, s03_s04, ...) of SubInfo2*.xlsx carries an
"Experiment Trials" block; this writes one trialtable_<DATE>.csv per sheet in
the format brock_functions.load_condition_table() reads. The actual parsing
lives in brock_functions.trialtable_from_xlsx (also exposed in the session
notebooks, where the xlsx can be uploaded and the sheet picked from a list).

Usage:
    uv run python scripts/parse_subinfo_trialtables.py [XLSX] [OUT_DIR] \\
        [--sheets s05_s06,s07_s08]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from brock_functions import list_xlsx_sheets, trialtable_from_xlsx

DEFAULT_XLSX = ('/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data/'
                'SubInfo2(2).xlsx')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('xlsx', nargs='?', default=DEFAULT_XLSX)
    ap.add_argument('out_dir', nargs='?', default=None,
                    help='default: directory of the xlsx')
    ap.add_argument('--sheets', default=None,
                    help='comma-separated sheet names (default: every '
                         'session-named sheet with a filled trial block)')
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.xlsx))
    wanted = args.sheets.split(',') if args.sheets else None

    for name in list_xlsx_sheets(args.xlsx):
        if wanted is not None and name not in wanted:
            continue
        if not re.match(r's\d+_s\d+', name):
            continue
        try:
            out = trialtable_from_xlsx(args.xlsx, name, out_dir)
        except ValueError as e:
            print(f'skip {name}: {e}')
            continue
        print(f'{name} -> {out}')


if __name__ == '__main__':
    main()
