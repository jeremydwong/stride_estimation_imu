"""assign_walkers_by_protocol: rep 1 = first walker first, rep 2 flipped."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import brock_functions as bf


class WalkerProtocolTests(unittest.TestCase):
    def setUp(self):
        # snip 1: a standing foot DRIFTS farther (3.3 m) than b's real walk
        self.measured = pd.DataFrame({
            'dist_left_foot_a': [2.3, 3.3, 0.2, 2.4],
            'dist_right_foot_a': [2.1, 0.9, 0.1, 2.2],
            'dist_left_foot_b': [0.1, 1.4, 2.6, 0.3],
            'dist_right_foot_b': [0.2, 1.8, 2.4, 0.1]})
        self.aligned = pd.DataFrame(dict(
            trial=[13, 13, 13, 13, 14], rep=[1, 1, 2, 2, 1], bout=[1, 2, 1, 2, 1],
            distance_m=[2.5] * 5, snip=[0., 1., 2., 3., np.nan],
            align_max_m=[2.3, 3.3, 2.6, 2.4, np.nan],
            walker=['left_foot_a', 'left_foot_a', 'left_foot_b', 'left_foot_a', '']))
        self.result = {'measured': self.measured, 'aligned': self.aligned,
                       'missed': self.aligned.iloc[[4]]}
        for name in ('estimate_reaction_s', 'flag_jumped_gun'):
            p = patch.object(bf, name, return_value=np.zeros(5))
            p.start()
            self.addCleanup(p.stop)

    def test_protocol_order_and_distance_from_that_persons_feet(self):
        out = bf.assign_walkers_by_protocol({}, self.result)['aligned']
        self.assertEqual(out['walker'].tolist()[:4],
                         ['left_foot_a', 'right_foot_b', 'left_foot_b', 'left_foot_a'])
        np.testing.assert_allclose(out['align_max_m'][:4], [2.3, 1.8, 2.6, 2.4])
        self.assertEqual(out['walker_changed'].tolist(),
                         [False, True, False, False, False])
        self.assertEqual(out['walker_auto'][1], 'left_foot_a')
        self.assertEqual(out['walker'][4], '')            # missed: untouched
        self.assertEqual(self.aligned['walker'][1], 'left_foot_a')  # input kept

    def test_first_walker_b_flips_everything(self):
        out = bf.assign_walkers_by_protocol({}, self.result, first_walker='b')
        persons = out['aligned']['walker'].str[-1].tolist()[:4]
        self.assertEqual(persons, ['b', 'a', 'a', 'b'])
        with self.assertRaises(ValueError):
            bf.assign_walkers_by_protocol({}, self.result, first_walker='c')


if __name__ == '__main__':
    unittest.main()
