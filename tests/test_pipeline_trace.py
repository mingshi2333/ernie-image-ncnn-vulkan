"""Trace storage changes must not hide model or mask differences."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from validate_pipeline import compare_trace_tensor, sha256


class PipelineTraceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference, self.native = self.root/'reference', self.root/'native'
        self.reference.mkdir(); self.native.mkdir()
        self.gate = {'nrmse': .0002, 'atol': .0002, 'global_rtol': .0002}
        self.row = np.array([0, 0, -1e30], dtype='<f4')
        self.dense = np.tile(self.row, (3, 1))

    def compare(self, expected, actual, name='constant-2'):
        expected = np.asarray(expected, dtype='<f4')
        path = self.reference/(name+'.f32')
        expected.tofile(path)
        np.asarray(actual, dtype='<f4').tofile(self.native/path.name)
        item = {'file': path.name, 'shape': list(expected.shape), 'sha256': sha256(path)}
        return compare_trace_tensor(name, item, self.reference, self.native, self.gate)

    def test_old_dense_and_new_row_match_the_same_official_fixture(self):
        dense = self.compare(self.dense, self.dense)
        row = self.compare(self.dense, self.row)
        self.assertTrue(dense['passed']); self.assertTrue(row['passed'])
        self.assertEqual(row['nrmse'], 0)
        self.assertEqual(row['logical_shape'], [3, 3])
        self.assertEqual(row['native_storage_shape'], [1, 3])
        self.assertNotEqual(dense['sha256'], row['sha256'])

    def test_every_reference_row_is_checked(self):
        expected = self.dense.copy(); expected[2, 0] = -1e30
        self.assertFalse(self.compare(expected, self.row)['passed'])

    def test_large_mask_sentinel_cannot_hide_an_incorrect_visible_value(self):
        actual = self.row.copy(); actual[0] = 1
        self.assertFalse(self.compare(self.dense, actual)['passed'])
        actual = self.dense.copy(); actual[1, 0] = 1
        self.assertFalse(self.compare(self.dense, actual)['passed'])

    def test_wrong_size_or_nonfinite_mask_is_rejected(self):
        for actual in (self.row[:2], np.array([0, 0, np.nan]), np.array([0, 0, np.inf])):
            with self.assertRaises(RuntimeError):
                self.compare(self.dense, actual)

    def test_other_tensors_and_rectangular_masks_cannot_broadcast(self):
        for name in ('constant-0', 'prediction-0', 'decoded'):
            with self.assertRaisesRegex(RuntimeError, 'shape'):
                self.compare(self.dense, self.row, name)
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            self.compare(self.dense[:2], self.row)

    def test_checksum_and_partial_float_corruption_are_rejected(self):
        self.compare(self.dense, self.row)
        path = self.reference/'constant-2.f32'
        item = {'file': path.name, 'shape': [3, 3], 'sha256': sha256(path)}
        native = self.native/path.name
        native.write_bytes(native.read_bytes() + b'x')
        with self.assertRaises(RuntimeError):
            compare_trace_tensor('constant-2', item, self.reference, self.native, self.gate)
        item['sha256'] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            compare_trace_tensor('constant-2', item, self.reference, self.native, self.gate)


if __name__ == '__main__':
    unittest.main()
