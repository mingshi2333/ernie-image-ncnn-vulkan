"""The single-step command must use authenticated runtime geometry and inputs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import diagnose_pipeline_step as diagnostic
from test_pipeline_reference import fixture


class PipelineStepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.reference = self.root/'model', self.root/'reference'
        self.model.mkdir(); self.reference.mkdir()
        self.saved = fixture()
        for entry in [*self.saved['inputs'].values(), self.saved['outputs'][0]['prediction']]:
            path = self.reference/entry['file']
            np.zeros(entry['shape'], dtype='<f4').tofile(path)
            entry['sha256'] = diagnostic.sha256(path)
        self.write_reference()
        (self.model/'manifest.json').write_text(json.dumps({'config': self.saved['config']}))
        (self.root/'runner').write_bytes(b'synthetic runner; execution is mocked')
        (self.root/'tools').mkdir()
        (self.root/'tools/snapshot.py').write_text('# synthetic source snapshot\n')

    def write_reference(self):
        (self.reference/'fixture.json').write_text(json.dumps(self.saved))

    def invoke(self, binding):
        argv = ['diagnose_pipeline_step.py', '--model', str(self.model),
                '--reference', str(self.reference), '--step', '0',
                '--output', str(self.root/'output'), '--runner', str(self.root/'runner'),
                '--threads', '2', '--host-weights']
        with patch.object(sys, 'argv', argv), patch.object(diagnostic, 'ROOT', self.root), \
                patch.object(diagnostic, 'validation_package', return_value=(self.saved['config'], binding)) as select, \
                patch.object(diagnostic, 'verify_package', return_value=({'config': self.saved['config']}, {})) as fixed, \
                patch.object(diagnostic, 'reviewed_shared_reference') as authenticate, \
                patch.object(diagnostic, 'run', return_value={'passed': True}) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostic.main(), 0)
        select.assert_called_once_with(self.model, 32, 32, reference=self.reference)
        return fixed, authenticate, execute.call_args

    def test_shared_cli_passes_package_and_real_token_count_without_raw_models(self):
        binding = {'schema_version': 3, 'source_manifest_sha256': 'b'*64}
        fixed, authenticate, call = self.invoke(binding)
        fixed.assert_not_called()
        authenticate.assert_called_once_with(self.reference/'fixture.json', binding, self.model/'manifest.json')
        self.assertEqual(call.args[0], [])
        extra = call.kwargs['extra_args']
        self.assertNotIn('--input-head', extra)
        self.assertEqual(extra[extra.index('--package')+1], str(self.model))
        self.assertEqual(extra[extra.index('--valid-text-tokens')+1], '2')
        self.assertEqual(extra[extra.index('--width')+1], '2')
        self.assertEqual(extra[extra.index('--threads')+1], '2')
        self.assertIn('--host-weights', extra)
        saved = json.loads((self.root/'output/fixture/fixture.json').read_text())
        self.assertFalse(saved['native_acceptance_eligible'])
        self.assertEqual(saved['package_binding'], binding)
        self.assertEqual(saved['inputs']['in0']['sha256'], self.saved['inputs']['initial']['sha256'])

    def test_fixed_cli_retains_36_blocks_and_both_heads(self):
        fixed, authenticate, call = self.invoke(None)
        fixed.assert_called_once_with(self.model)
        authenticate.assert_not_called()
        self.assertEqual(call.args[0], [self.model/f'dit/block-{i:02d}' for i in range(36)])
        extra = call.kwargs['extra_args']
        self.assertNotIn('--package', extra)
        self.assertEqual(extra[extra.index('--input-head')+1], str(self.model/'dit/input'))
        self.assertEqual(extra[extra.index('--output-head')+1], str(self.model/'dit/output'))

    def test_unknown_shared_oracle_is_rejected_before_output(self):
        with patch.object(diagnostic, 'validation_package', return_value=(self.saved['config'], {})), \
                self.assertRaisesRegex(ValueError, 'no matching reviewed source binding'):
            diagnostic.reference_package(self.model, self.reference, 0)
        self.assertFalse((self.root/'output').exists())

    def test_invalid_step_or_incomplete_reference_precedes_package_read(self):
        with patch.object(diagnostic, 'validation_package') as select:
            for step in (-1, 8, True):
                with self.assertRaises(ValueError):
                    diagnostic.reference_package(self.model, self.reference, step)
            self.saved['outputs'].pop()
            self.write_reference()
            with self.assertRaises(ValueError):
                diagnostic.reference_package(self.model, self.reference, 0)
            select.assert_not_called()

    def test_selected_text_source_or_geometry_cannot_differ_from_oracle(self):
        for key in ('packed_width', 'text_bucket'):
            different = dict(self.saved['config'], **{key: self.saved['config'][key]+1})
            with patch.object(diagnostic, 'validation_package', return_value=(different, {})), \
                    self.assertRaisesRegex(ValueError, 'selected package configuration'):
                diagnostic.reference_package(self.model, self.reference, 0)

    def test_corrupt_input_is_rejected_before_runner(self):
        (self.reference/self.saved['inputs']['initial']['file']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum or shape differs'):
            self.invoke(None)
        self.assertFalse((self.root/'output/native').exists())


if __name__ == '__main__':
    unittest.main()
