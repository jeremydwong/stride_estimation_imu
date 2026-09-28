"""Corrected windows must refresh distances without changing the baseline."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brock_functions as bf


class OmnibusTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame(dict(trial=[1, 1], rep=[1, 1], bout=[1, 2],
            distance_m=[6., 6.], measured_m=[5., np.nan],
            distance_error_m=[-1., np.nan], start_s=[10., np.nan],
            stop_s=[16., np.nan], duration_s=[6., np.nan],
            walker=['left_foot_a', ''], snip=[0., np.nan], inferred=[False, False]))

    def test_save_refresh_and_recovered_bout_plot(self):
        original = self.base.copy(deep=True)
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d)/'edits.csv'
            pd.DataFrame(dict(trial=[1, 1], rep=[1, 1], person=['a', 'b'],
                              start_s=[11., 20.], stop_s=[15., 25.])).to_csv(csv,index=False)
            corrected = bf.apply_manual_rescore(self.base, csv)
        feet = {'left_foot_a': object(), 'right_foot_a': object(), 'left_foot_b': object()}
        calls = []
        def measure(data, windows, **kwargs):
            calls.append((set(data), windows))
            return pd.DataFrame({'distance_m':[4. if windows[0][0] == 11 else 6.]})
        events = {'time_s':np.array([10.,16.,20.,25.]),
                  'label':np.array(['Start','Stop','Start','Stop'])}
        with patch.object(bf, 'snip_distances', side_effect=measure):
            fig, axes, table = bf.plot_omnibus(feet, events, {'aligned':self.base},
                                              aligned=corrected, rows=1)
        self.assertEqual(calls[0][0], {'left_foot_a','right_foot_a'})
        self.assertEqual(calls[1][0], {'left_foot_b'})
        np.testing.assert_allclose(table.measured_m,[4.,6.])
        np.testing.assert_allclose(table.distance_error_m,[-2.,0.])
        self.assertTrue(pd.isna(table.snip.iloc[1]))  # preserve provenance
        measured = next(line for line in axes[0].lines if line.get_label()=='measured')
        np.testing.assert_allclose(measured.get_ydata(),[4.,6.])
        pd.testing.assert_frame_equal(self.base,original)
        self.assertEqual(corrected.measured_m.iloc[0],5.)
        plt.close(fig)
        with patch.object(bf,'snip_distances') as recompute:
            fig, _, table = bf.plot_omnibus(feet,events,{'aligned':self.base},rows=1)
            recompute.assert_not_called()
        pd.testing.assert_frame_equal(table,self.base)
        plt.close(fig)

    def test_invalid_manual_window_fails_instead_of_saving_old_distance(self):
        table=self.base.copy()
        table['manual']=[True,False]
        table.loc[0,'stop_s']=9.
        with self.assertRaises(ValueError):
            bf.refresh_manual_distances({'left_foot_a':object()},table)


if __name__ == '__main__':
    unittest.main()
