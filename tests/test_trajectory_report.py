"""Execution order and evidence completeness are independent of error magnitude."""
from pathlib import Path
import sys
import tempfile
import copy
from types import SimpleNamespace
import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_trajectory import first_failed_boundary, ordered_boundaries, build_rows, conditioning_identity, verify_candidate_snapshot


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


class SavedCandidateTests(unittest.TestCase):
    def prepare(self,p):
        from validate_pipeline import snapshot_diagnostic_embeddings
        source=p/'candidate.f32';np.arange(3072,dtype='<f4').tofile(source)
        fixture={'ids':[1],'inputs':{'text':{'shape':[1,1,3072]}}}
        meta=snapshot_diagnostic_embeddings(source,p,fixture)
        (p/'trace').mkdir();(p/'trace/text.f32').write_bytes(source.read_bytes())
        result={'conditioning_source':'saved_candidate_diagnostic','native_acceptance_eligible':False,
                'diagnostic_embeddings':meta,'comparisons':[{'tensor':'text','sha256':meta['sha256']}],
                'command':['runner','--embeddings',str(p/meta['snapshot_file'])]}
        return fixture,result

    def test_snapshot_and_actual_runner_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);fixture,result=self.prepare(p)
            verify_candidate_snapshot(p,result,fixture)
            for mutate in (lambda r:r.update(native_acceptance_eligible=True),
                           lambda r:r['diagnostic_embeddings'].update(dtype='<f8'),
                           lambda r:r['diagnostic_embeddings'].update(shape=[1,2,3072]),
                           lambda r:r['diagnostic_embeddings'].update(token_ids_sha256='0'*64),
                           lambda r:r['comparisons'][0].update(sha256='0'*64),
                           lambda r:r['command'].__setitem__(-1,str(p/'candidate.f32'))):
                bad=copy.deepcopy(result);mutate(bad)
                with self.assertRaises(ValueError):verify_candidate_snapshot(p,bad,fixture)
            (p/'trace/text.f32').write_bytes(b'bad')
            with self.assertRaises(ValueError):verify_candidate_snapshot(p,result,fixture)

    def test_raw_fp32_storage_size_finite_and_exclusivity(self):
        from validate_pipeline import snapshot_diagnostic_embeddings,check_conditioning_options
        fixture={'ids':[1],'inputs':{'text':{'shape':[1,1,3072]}}}
        for data in (np.zeros(3072,dtype='<f8'),np.full(3072,np.nan,dtype='<f4')):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);source=p/'input.f32';data.tofile(source)
                with self.assertRaises(ValueError):snapshot_diagnostic_embeddings(source,p,fixture)
        base=dict(diagnostic_embeddings=Path('input.f32'),reference_embeddings=False,pe_model=None,pe_reference=None,reference_only=False)
        check_conditioning_options(SimpleNamespace(**base))
        for flag in ('reference_embeddings','pe_model','pe_reference','reference_only'):
            with self.assertRaises(ValueError):check_conditioning_options(SimpleNamespace(**{**base,flag:True}))

if __name__ == '__main__':
    unittest.main()
