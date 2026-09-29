"""ManualScoring: every save rewrites the corrected table; reruns ask first."""
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
            duration_s=[6., 6.], distance_m=[5., 5.], measured_m=[5., 5.],
            distance_error_m=[0., 0.], snip=[0., 1.]))
        feet = {'left_foot_a': SimpleNamespace(file_path=str(d / 'x.h5'), period=0.01)}
        self.figs = []
        patches = [
            patch.object(bf, 'snip_distances', side_effect=lambda rec, w, **k:
                         pd.DataFrame({'distance_m': [w[0][1] - w[0][0]]})),
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
        self.open = lambda pairs, on_existing='ask': self.s._open(
            pairs, opener, on_existing)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_rewrites_table_and_figure(self):
        ed, = self.open([(4, 1)])
        ed.save()
        table = pd.read_csv(self.s.manual_aligned_csv)
        self.assertEqual(table['manual'].tolist(), [True, False])
        self.assertEqual(table.loc[0, 'measured_m'], 5.)
        self.assertEqual(self.figs, [[(4, 1)]])   # only the corrected rep
        self.assertIn('table + trial 4 figure(s) updated', ed.note)
        pd.testing.assert_frame_equal(self.s.auto, self.auto)

    def test_rerun_saves_open_editors_then_asks_about_saved_pairs(self):
        self.open([(4, 1)])                       # edited but Save not pressed
        with patch('builtins.input', return_value='') as ask:
            self.open([(4, 1), (4, 2)])           # rerun saves, then asks: skip
        ask.assert_called_once()
        self.assertEqual(self.opened, [(4, 1), (4, 2)])
        self.assertEqual(self.s.manual_trials(), [4])
        with patch('builtins.input', return_value='c'):
            self.open([(4, 1)])                   # (4,2) saved on the way in
        self.assertEqual(self.s.saved_pairs(), {(4, 2)})
        self.assertEqual(self.opened[-1], (4, 1))

    def test_no_prompt_for_empty_list_or_explicit_choice(self):
        self.open([(4, 1)])
        with patch('builtins.input', side_effect=AssertionError):
            self.open([])
            self.open([(4, 1)], on_existing='rescore')
        self.assertEqual(self.opened, [(4, 1), (4, 1)])

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
