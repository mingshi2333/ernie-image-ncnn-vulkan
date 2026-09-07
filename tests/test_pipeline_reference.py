"""A complete, source-bound reference is required before native acceptance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import pipeline_reference as reference


def fixture():
    cfg = dict(packed_width=2, packed_height=2, text_bucket=32,
               dit_text_tokens=32, text_layers=25, dit_layers=36)
    def entry(name, shape):
        return dict(file=name + '.f32', shape=shape, dtype='float32_le', sha256='a' * 64)
    latent = [1, 128, 2, 2]
    return dict(complete=True, config=cfg, prompt='test', steps=8, ids=[1, 2],
                inputs={name: entry(name, shape) for name, shape in {
                    'initial': latent, 'text': [1, 2, 3072], 'padded-text': [1, 32, 3072],
                    'constant-0': [1, 36, 128], 'constant-1': [1, 36, 128],
                    'constant-2': [36, 36]}.items()},
                outputs=[{name: entry(name + '-' + str(i), latent) for name in ('prediction', 'step')}
                         for i in range(8)],
                final={name: entry(name, shape) for name, shape in {
                    'final': latent, 'unpacked': [1, 32, 4, 4], 'decoded': [1, 3, 32, 32]}.items()})


class ReferenceContractTest(unittest.TestCase):
    def check(self, data):
        return reference.full_reference_contract(data, fixture()['config'], 'test', 8)

    def test_complete_reference_has_exact_25_boundaries(self):
        self.assertEqual(self.check(fixture()), 25)

    def test_empty_or_missing_step_is_rejected(self):
        for count in (0, 7, 9):
            data = fixture()
            data['outputs'] = (data['outputs'] * 2)[:count]
            with self.assertRaisesRegex(ValueError, 'every denoising step'):
                self.check(data)

    def test_suffix_bool_count_and_changed_config_are_rejected(self):
        for key, value in (('start_step', 1), ('start_step', False), ('steps', True), ('complete', 1)):
            data = fixture(); data[key] = value
            with self.assertRaises(ValueError): self.check(data)
        data = fixture(); data['config']['packed_width'] = 3
        with self.assertRaises(ValueError): self.check(data)

    def test_missing_conditioning_prediction_or_final_is_rejected(self):
        for group, key in (('inputs', 'constant-2'), ('final', 'decoded'), ('output', 'prediction')):
            data = fixture()
            del (data['outputs'][4] if group == 'output' else data[group])[key]
            with self.assertRaises(ValueError): self.check(data)

    def test_step_identity_shape_dtype_and_token_type_are_checked(self):
        for key, value in (('file', 'prediction-0.f32'), ('file', '../prediction-6.f32'),
                           ('shape', [1, 128, 1, 4]), ('dtype', 'float16_le'), ('sha256', 'x' * 64)):
            data = fixture(); data['outputs'][6]['prediction'][key] = value
            with self.assertRaises(ValueError): self.check(data)
        data = fixture(); data['ids'] = [True, 2]
        with self.assertRaises(ValueError): self.check(data)

    def test_runtime_target_cannot_replace_source_config_or_omit_target_binding(self):
        import pipeline_package
        source_config = dict(packed_width=64, packed_height=64, text_bucket=64,
                             dit_text_tokens=64, text_layers=25, dit_layers=36)
        target_config = dict(source_config, packed_width=86, packed_height=48)
        files = {'dit/block-35/block.ncnn.bin': 'b' * 64}
        source = dict(schema_version=2, portable=True, config=source_config, files=files)
        source_bytes = json.dumps(source).encode()
        digest = hashlib.sha256(source_bytes).hexdigest()
        instance = dict(source_manifest_sha256=digest, config=source_config, runtime_bindings=files)
        contract = {'source_manifests': {digest: source_config},
                    'reviewed_runtime_targets': {digest: [target_config]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'objects').mkdir()
            (root / 'objects' / digest).write_bytes(source_bytes)
            fixture_path = root / 'fixture.json'
            fixture_path.write_text(json.dumps({'config': target_config, 'ids': [1, 2]}))
            fixture_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
            manifest = {'schema_version': 3, 'instances': [instance]}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            trusted = {fixture_digest: {'source_manifest_sha256': digest}}
            with patch.object(pipeline_package, 'shared_contract', return_value=contract), \
                    patch.dict(reference.REVIEWED_SHARED_REFERENCES, trusted, clear=True):
                _, target = pipeline_package.select_shared_instance([instance], 1376, 768)
                binding = {'source_manifest_sha256': digest, 'runtime_bindings': files,
                           'shared_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                           'runtime_target': target}
                reference.reviewed_shared_reference(fixture_path, binding, manifest_path)
                for altered in ({k: v for k, v in binding.items() if k != 'runtime_target'},
                                {**binding, 'runtime_target': {**target, 'source_config': target_config}}):
                    with self.assertRaises(ValueError):
                        reference.reviewed_shared_reference(fixture_path, altered, manifest_path)
                manifest['instances'][0]['config'] = target_config
                manifest_path.write_text(json.dumps(manifest))
                binding['shared_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    reference.reviewed_shared_reference(fixture_path, binding, manifest_path)

    def test_whole_fixture_digest_and_source_binding_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.json'; path.write_text(json.dumps(fixture()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            trusted = {digest: {'source_manifest_sha256': 'b' * 64, 'scope': 'synthetic unit reference'}}
            with patch.dict(reference.REVIEWED_SHARED_REFERENCES, trusted, clear=True):
                reference.reviewed_shared_reference(path, {'source_manifest_sha256': 'b' * 64})
                with self.assertRaises(ValueError):
                    reference.reviewed_shared_reference(path, {'source_manifest_sha256': 'c' * 64})
                data = fixture(); data['reference_environment'] = {'dtype': 'float16'}
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    reference.reviewed_shared_reference(path, {'source_manifest_sha256': 'b' * 64})

    def test_resigned_shared_metadata_cannot_replace_pinned_source_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'fixture.json'; path.write_text(json.dumps(fixture()))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source_bytes = json.dumps({'schema_version': 2, 'portable': True,
                                       'config': fixture()['config'], 'files': {'text/a': 'a' * 64}}).encode()
            source_digest = hashlib.sha256(source_bytes).hexdigest()
            trusted = {digest: {'source_manifest_sha256': source_digest}}
            forged = {'invented-file': '0' * 64}
            manifest = {'schema_version': 3, 'instances': [{'source_manifest_sha256': source_digest,
                        'config': fixture()['config'], 'runtime_bindings': forged}]}
            mp = root / 'manifest.json'; mp.write_text(json.dumps(manifest))
            binding = {'source_manifest_sha256': source_digest, 'runtime_bindings': forged,
                       'shared_manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest()}
            import pipeline_package
            with patch.dict(reference.REVIEWED_SHARED_REFERENCES, trusted, clear=True), \
                    patch.object(pipeline_package, 'shared_contract', return_value={
                        'source_manifests': {source_digest: fixture()['config']}}):
                with self.assertRaisesRegex(ValueError, 'source manifest object'):
                    reference.reviewed_shared_reference(path, binding, mp)
                (root / 'objects').mkdir(); (root / 'objects' / source_digest).write_bytes(source_bytes)
                with self.assertRaisesRegex(ValueError, 'full source inventory'):
                    reference.reviewed_shared_reference(path, binding, mp)


if __name__ == '__main__':
    unittest.main()
