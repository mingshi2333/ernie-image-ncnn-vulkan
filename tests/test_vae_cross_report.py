"""Crossed decoders must distinguish decoder error from upstream latent drift."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from diagnose_vae_cross import cross_terms, compare_arrays


class CrossedVaeTests(unittest.TestCase):
    def test_decoder_and_input_error_are_separate(self):
        z = np.zeros((3, 2, 2), dtype=np.float64)
        terms = cross_terms(z, z + .01, z + .1, z + .11)
        self.assertAlmostEqual(terms['decoder_on_official_max'], .01)
        self.assertAlmostEqual(terms['input_through_official_max'], .1)
        self.assertAlmostEqual(terms['interaction_max'], 0.0)
        self.assertAlmostEqual(terms['connected_max'], .11)

    def test_interaction_cannot_be_attributed_as_independent_error(self):
        z = np.zeros((3, 2, 2))
        terms = cross_terms(z, z + 1, z + 2, z + 6)
        self.assertEqual(terms['interaction_max'], 3)
        self.assertEqual(terms['decoder_on_native_max'], 4)
        self.assertEqual(terms['input_through_native_max'], 5)

    def test_shapes_and_finiteness_are_required_for_all_four_inputs(self):
        z = np.zeros((3, 2, 2))
        for index in range(4):
            for bad in (np.zeros((12,)), np.full_like(z, np.nan),
                        np.full_like(z, np.inf), np.empty((0,))):
                values = [z.copy() for _ in range(4)]
                values[index] = bad
                with self.subTest(index=index, shape=bad.shape):
                    with self.assertRaises(ValueError):
                        cross_terms(*values)

    def test_localized_failure_is_not_hidden_by_mean_error(self):
        reference = np.ones((100, 100))
        candidate = reference.copy()
        candidate[0, 0] += .02
        metrics = compare_arrays(reference, candidate,
                                 {'nrmse': .003, 'global_rtol': .01, 'atol': .0002})
        self.assertLess(metrics['nrmse'], .003)
        self.assertFalse(metrics['passed'])
        self.assertEqual(metrics['max_index'], [0, 0])


if __name__ == '__main__':
    unittest.main()
