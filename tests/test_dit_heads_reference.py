import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import export_dit_heads as heads
import torch


class HeadReference(unittest.TestCase):
    def test_reference_only_keeps_full_denominator_without_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'fixture'
            x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            with patch.object(heads.torch.jit, 'trace', side_effect=AssertionError('must not trace')):
                heads.export_component(torch.nn.Identity(), (x,), (x,), output,
                                       {'component': 'input'}, reference_only=True)
            meta = json.loads((output / 'fixture.json').read_text())
            self.assertEqual(set(p.name for p in output.iterdir()), {'fixture.json', 'in0.f32', 'out0.f32'})
            self.assertEqual(meta['expected']['out0']['shape'], [1, 3, 4])
            self.assertEqual((output/'out0.f32').stat().st_size, 48)
            self.assertEqual(meta['wrapper_vs_official'][0]['max_abs_error'], 0)
            self.assertEqual(meta['gates']['fp32'], {'atol': .0002, 'rtol': .0002, 'nrmse': .00002})

    def test_wrapper_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = torch.ones(1, 2)
            with self.assertRaisesRegex(ValueError, 'wrapper differs'):
                heads.export_component(torch.nn.Identity(), (x,), (x+1,), Path(tmp)/'bad',
                                       {'component': 'output'}, reference_only=True)

    def test_explicit_root_threads_and_fixed_shape_route_before_loading(self):
        class StopBeforeModel(Exception): pass
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp)/'weights'
            argv = ['export_dit_heads.py', '--output', str(Path(tmp)/'new'), '--height', '48', '--width', '86',
                    '--text-tokens', '64', '--only', 'input', '--reference-only', '--official-root', str(weights), '--threads', '2']
            with patch.object(sys, 'argv', argv), patch.object(heads, 'load_heads', side_effect=StopBeforeModel) as load:
                with self.assertRaises(StopBeforeModel): heads.main()
            load.assert_called_once_with(False, weights)
            self.assertEqual(torch.get_num_threads(), 2)


if __name__ == '__main__': unittest.main()
