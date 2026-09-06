"""Execution order and evidence completeness are independent of error magnitude."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_trajectory import first_failed_boundary, ordered_boundaries, build_rows


class TrajectoryTests(unittest.TestCase):
    def test_earlier_failure_is_not_replaced_by_larger_late_error(self):
        rows = [{'boundary': 'step-1/block-4', 'passed': False, 'max_abs_error': .01},
                {'boundary': 'step-7/final', 'passed': False, 'max_abs_error': 100}]
        self.assertEqual(first_failed_boundary(rows), 'step-1/block-4')
        self.assertIsNone(first_failed_boundary([{'boundary': 'final', 'passed': True}]))

    def test_ambiguous_status_cannot_hide_late_missing_verdict(self):
        for bad in (None, 'false', 0, 1):
            with self.assertRaises(ValueError):
                first_failed_boundary([{'boundary': 'early', 'passed': False},
                                       {'boundary': 'late', 'passed': bad}])

    def test_predictions_precede_each_euler_step(self):
        order = [name for name, _, _ in ordered_boundaries(12)]
        self.assertLess(order.index('prediction-2'), order.index('step-2'))
        self.assertLess(order.index('step-2'), order.index('prediction-10'))
        self.assertLess(order.index('final'), order.index('unpacked'))
        self.assertLess(order.index('unpacked'), order.index('decoded'))

    def test_omitted_duplicated_or_unexpected_boundaries_reject(self):
        comparison = {'tensor': 'text', 'passed': True}
        for rows in ([], [comparison], [comparison, comparison], [{'tensor': 'unknown'}]):
            with self.assertRaises(ValueError):
                build_rows({'comparisons': rows}, {'steps': 8})


if __name__ == '__main__':
    unittest.main()
