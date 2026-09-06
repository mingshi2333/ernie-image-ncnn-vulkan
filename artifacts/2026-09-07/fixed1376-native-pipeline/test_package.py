"""Model-free corruption and relocation tests for Python and native verification."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from package_model import package_model, runtime_files, sha256, verify_package


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'model'
        self.root.mkdir()
        lock = json.loads((ROOT/'sources.lock.json').read_text())
        config = dict(packed_width=4, packed_height=4, text_bucket=32,
                      dit_text_tokens=64, text_layers=25, dit_layers=36)
        for name in runtime_files():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture: '+name.encode())
        (self.root/'model.cfg').write_text(''.join(f'{k} {v}\n' for k, v in config.items()))
        self.manifest = {'schema_version': 2, 'portable': True, 'config': config,
                         'official_model_revision': lock['official_model']['revision'],
                         'ncnn_revision': lock['ncnn']['revision'],
                         'files': {n: sha256(self.root/n) for n in runtime_files()},
                         'file_sizes': {n: (self.root/n).stat().st_size for n in runtime_files()}}
        self.write_manifest()
        self.runner = Path(os.environ.get('ERNIE_TEST_RUNNER', ROOT/'build/ernie-image'))

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))

    def check(self, valid, error=None):
        if valid:
            verify_package(self.root)
        else:
            with self.assertRaises((ValueError, OSError)):
                verify_package(self.root)
        if self.runner.exists():
            run = subprocess.run([str(self.runner.resolve()), '--model', str(self.root), '--verify-model'],
                                 capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode == 0, valid, run.stdout+run.stderr)
            if error:
                self.assertIn(error, run.stderr)

    def test_complete_inventory_and_native_check_without_model_loading(self):
        self.check(True)

    def test_same_size_bit_corruption(self):
        path = self.root/'dit/block-35/block.ncnn.bin'
        content = bytearray(path.read_bytes()); content[-1] ^= 1
        path.write_bytes(content)
        self.check(False, 'File checksum differs')

    def test_missing_last_text_layer(self):
        (self.root/'text/block-24/text.ncnn.bin').unlink()
        self.check(False)

    def test_omitted_file_cannot_evade_verification(self):
        del self.manifest['files']['vae/head.ncnn.bin']
        self.write_manifest(); self.check(False, 'Runtime file inventory differs')

    def test_truncated_weight(self):
        (self.root/'dit/block-00/block.ncnn.bin').write_bytes(b'x')
        self.check(False, 'File size differs')

    def test_reject_version_mismatch(self):
        self.manifest['ncnn_revision'] = 'unreviewed'
        self.write_manifest(); self.check(False, 'ncnn revision differs')

    def test_portable_flag_must_be_boolean(self):
        self.manifest['portable'] = 'true'
        self.write_manifest(); self.check(False, 'Missing portable package flag')

    def test_manifest_config_cannot_use_boolean_for_integer(self):
        self.manifest['config']['packed_width'] = True
        self.write_manifest(); self.check(False, 'Configuration and manifest disagree')

    def test_configuration_cannot_disagree_with_manifest(self):
        self.manifest['config']['packed_height'] = 64
        self.write_manifest(); self.check(False, 'Configuration and manifest disagree')

    def test_path_escape_cannot_enter_inventory(self):
        self.manifest['files']['../escaped'] = '0'*64
        self.write_manifest(); self.check(False)

    def test_portable_package_rejects_external_file_link(self):
        path = self.root/'vae/bn-mean.f32'
        outside = Path(self.temporary.name)/'external'
        path.rename(outside); path.symlink_to(outside)
        self.check(False, 'Portable package contains a symlink')

    def test_portable_package_rejects_directory_link(self):
        outside = Path(self.temporary.name)/'external'
        (self.root/'vae').rename(outside); (self.root/'vae').symlink_to(outside, target_is_directory=True)
        self.check(False, 'Portable package contains a symlink')

    def test_external_alias_to_complete_portable_root_is_allowed(self):
        alias = Path(self.temporary.name)/'latest'
        alias.symlink_to(self.root, target_is_directory=True)
        self.root = alias
        self.check(True)

    def test_copy_survives_source_removal_and_relocation(self):
        destination = Path(self.temporary.name)/'packed'
        package_model(self.root, destination)
        shutil.rmtree(self.root)
        self.root = Path(self.temporary.name)/'relocated with spaces'
        destination.rename(self.root)
        self.check(True)

    def test_development_links_are_explicit(self):
        destination = Path(self.temporary.name)/'linked'
        packed = package_model(self.root, destination, link=True)
        self.assertFalse(packed['portable'])
        self.root = destination
        self.check(True)

    def test_fixed_shape_rejects_unreviewed_source_before_writing(self):
        destination = Path(self.temporary.name)/'fixed'
        with self.assertRaisesRegex(ValueError, 'reviewed 1024x1024/s64'):
            package_model(self.root, destination, link=True, fixed1376=True)
        self.assertFalse(destination.exists())
        self.check(True)

    def test_fixed_shape_module_api_also_checks_source_identity(self):
        destination = Path(self.temporary.name)/'module-fixed'
        code = ('from tools.package_model import package_model; import sys; '
                'package_model(sys.argv[1], sys.argv[2], fixed1376=True)')
        run = subprocess.run([sys.executable, '-c', code, str(self.root), str(destination)],
                             cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('ValueError: Fixed 1376x768 requires the reviewed', run.stderr)
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
