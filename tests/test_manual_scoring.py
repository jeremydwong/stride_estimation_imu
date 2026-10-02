"""ManualScoring: every save rewrites the corrected table; reruns never prompt."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import brock_functions as bf


class FakeEditor:
    """Stands in for an interactive editor: save() goes through the real path."""
    def __init__(self, scoring, trial, rep, start):
        self.trial, self.out_csv, self.pending = trial, scoring.rescore_csv, None
        self.fig = SimpleNamespace(canvas=SimpleNamespace(close=lambda: None))
        self.rec = dict(trial=trial, rep=rep, person='a', start_s=start, stop_s=start + 5)
        self.note = None

    def save(self):
        if bf._save_controller_rows(self, [self.rec]):
            self.note = bf._run_on_save(self)


class ManualScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.auto = pd.DataFrame(dict(trial=[4, 4], rep=[1, 2], bout=[1, 1],
            walker=['left_foot_a'] * 2, start_s=[10., 30.], stop_s=[16., 36.],
            duration_s=[6., 6.], distance_m=[5., 5.], align_max_m=[5., 5.],
            align_error_m=[0., 0.], snip=[0., 1.]))
        feet = {'left_foot_a': SimpleNamespace(file_path=str(d / 'x.h5'), period=0.01)}
        self.figs = []
        patches = [
            patch.object(bf, 'snip_distances', side_effect=lambda rec, w, **k:
                         pd.DataFrame({'align_max_m': [w[0][1] - w[0][0]]})),
            patch.object(bf, 'save_trial_figures', side_effect=lambda *a, **k:
                         self.figs.append(k['trials'])),
            patch('matplotlib.pyplot.close')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.s = bf.ManualScoring(feet, self.auto, str(d / 'r.csv'),
                                  str(d / 'm.csv'), 'tag')
        self.opened = []
        def opener(t, r):
            self.opened.append((t, r))
            return [FakeEditor(self.s, t, r, start=11. + len(self.opened))]
        self.open = lambda pairs, on_existing='rescore': self.s._open(
            pairs, opener, on_existing)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_rewrites_table_and_figure(self):
        ed, = self.open([(4, 1)])
        ed.save()
        table = pd.read_csv(self.s.manual_aligned_csv)
        self.assertEqual(table['manual'].tolist(), [True, False])
        self.assertEqual(table.loc[0, 'align_max_m'], 5.)
        self.assertEqual(self.figs, [[(4, 1)]])   # only the corrected rep
        self.assertIn('table + trial 4 figure(s) updated', ed.note)
        pd.testing.assert_frame_equal(self.s.auto, self.auto)

    def test_rerun_saves_open_editors_and_reopens_saved_pairs_without_prompt(self):
        self.open([(4, 1)])                       # edited but Save not pressed
        with patch('builtins.input', side_effect=AssertionError('prompted')):
            self.open([(4, 1), (4, 2)])           # rerun saves, then REOPENS (4,1)
            self.assertEqual(self.opened, [(4, 1), (4, 1), (4, 2)])
            self.assertEqual(self.s.manual_trials(), [4])
            self.open([(4, 1)], on_existing='ask')   # legacy value: no prompt
            self.open([(4, 1)], on_existing='skip')  # explicit skip still works
        self.assertEqual(self.opened[-1], (4, 1))
        self.assertEqual(self.opened.count((4, 1)), 3)

    def test_clear_one_rep_and_clean_state(self):
        ed, = self.open([(4, 1)])
        ed.save()
        ed2, = self.open([(4, 2)])
        ed2.save()
        self.assertEqual(self.s.saved_pairs(), {(4, 1), (4, 2)})
        self.open([(4, 1)], on_existing='clear')  # (4,2) saved on the way in
        self.assertEqual(self.s.saved_pairs(), {(4, 2)})
        # the open (4,1) editor is saved on the way in, so both reps clear
        self.assertEqual(self.s.clear([4]), 2)
        self.assertEqual(self.s.saved_pairs(), set())
        ed3, = self.open([(4, 2)])
        ed3.save()
        backup = self.s.clear_all()
        self.assertFalse(os.path.exists(self.s.rescore_csv))
        self.assertTrue(os.path.exists(backup))
        self.assertFalse(self.s.corrected['manual'].any())

    def test_bad_pairs_rejected_before_anything_is_saved(self):
        self.open([(4, 1)])
        for bad in ([(4, 3)], [0], [True], 4):
            with self.assertRaises(ValueError):
                self.open(bad)
        self.assertEqual(len(self.s.editors), 1)

    def test_bare_trial_number_means_both_reps(self):
        self.assertEqual(bf.ManualScoring._check_pairs([7, (4, 1), 7]),
                         [(7, 1), (7, 2), (4, 1)])


if __name__ == '__main__':
    unittest.main()


    def test_set_window_seconds_and_samples_and_unmatch(self):
        r = self.s.set_window(4, 1, 'a', 12.0, 18.0)
        self.assertEqual(self.s.saved_pairs(), {(4, 1)})
        self.assertEqual((r['start_s'], r['stop_s'], bool(r['manual'])), (12.0, 18.0, True))
        self.s.set_window(4, 2, 'a', 3100, 3700, units='samples')   # period 0.01
        row = self.s.corrected[(self.s.corrected.trial == 4) & (self.s.corrected.rep == 2)].iloc[0]
        self.assertEqual((row['start_s'], row['stop_s']), (31.0, 37.0))
        with self.assertRaises(ValueError):
            self.s.set_window(4, 1, 'c', 1, 2)
        with self.assertRaises(ValueError):
            self.s.set_window(99, 1, 'a', 1, 2)
        with self.assertRaises(ValueError):
            self.s.set_window(4, 1, 'a', 5, 5)
        r = self.s.unmatch(4, 1, 'a')
        self.assertTrue(pd.isna(r['start_s']) and pd.isna(r['align_max_m']) and bool(r['manual']))
        self.assertTrue(pd.isna(r['walked_m']) and pd.isna(r['gait_duration_s']))
        saved = pd.read_csv(self.s.rescore_csv)
        self.assertTrue(saved.loc[saved.rep == 1, 'start_s'].isna().all())
        self.s.clear([(4, 1)])                                       # undo
        self.assertEqual(self.s.corrected.loc[0, 'start_s'], 10.0)   # automatic again


class ManualWindowNoBoutTests(unittest.TestCase):
    def test_file_accepts_both_empty_but_not_half(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'r.csv'
            bf.save_manual_windows(path, [dict(trial=1, rep=1, person='a',
                                               start_s=float('nan'), stop_s=float('nan'))])
            self.assertTrue(pd.read_csv(path)['start_s'].isna().all())
            with self.assertRaises(ValueError):
                bf.save_manual_windows(path, [dict(trial=1, rep=1, person='a',
                                                   start_s=float('nan'), stop_s=5.)])
