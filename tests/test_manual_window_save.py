import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from brock_functions import save_manual_windows, _save_controller_rows


class ManualWindowSaveTests(unittest.TestCase):
    def test_replace_preserve_and_do_not_replay_stale_editor(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manual.csv'
            def row(trial=1, person='a', start=10):
                return dict(trial=trial,rep=1,person=person,start_s=start,stop_s=20)
            save_manual_windows(path,[row(),row(person='b'),row(trial=2)])
            old=SimpleNamespace(out_csv=path)
            new=SimpleNamespace(out_csv=path)
            self.assertTrue(_save_controller_rows(old,[row(start=11)]))
            self.assertTrue(_save_controller_rows(new,[row(start=12)]))
            self.assertFalse(_save_controller_rows(old,[row(start=11)]))
            table=pd.read_csv(path)
            self.assertEqual(len(table),3)
            self.assertEqual(table.query('trial == 1 and person == "a"').iloc[0].start_s,12)
            self.assertEqual(table.query('person == "b"').iloc[0].start_s,10)
            old._discarded=True
            self.assertFalse(_save_controller_rows(old,[row(start=13)]))

    def test_bad_window_does_not_replace_file(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manual.csv'
            good=dict(trial=1,rep=1,person='a',start_s=1,stop_s=3)
            save_manual_windows(path,[good])
            before=path.read_bytes()
            with self.assertRaises(ValueError):
                save_manual_windows(path,[dict(good,start_s=4)])
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
