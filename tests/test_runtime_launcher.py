"""First-run download and native invocation behavior, without real model weights."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runtime_launcher', SOURCE / 'tools/runtime/run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RuntimeLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ernie launcher ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'bin').mkdir()
        self.binary = self.root / 'bin' / ('ernie-image.exe' if os.name == 'nt' else 'ernie-image')
        self.binary.write_bytes(b'fixture')

    def invoke(self, args):
        with patch.object(launcher, 'prepare') as prepare, patch.object(launcher.subprocess, 'call', return_value=0) as call:
            result = launcher.main(args, root=self.root)
        return result, prepare, call

    def test_help_and_diagnose_never_download(self):
        for args in ([], ['--help'], ['--diagnose'], ['--help-all']):
            with self.subTest(args=args):
                result, prepare, call = self.invoke(args)
                self.assertEqual(result, 0)
                prepare.assert_not_called()
                self.assertEqual(call.call_args.args[0][0], str(self.binary))

    def test_default_main_model_and_small_image(self):
        result, prepare, call = self.invoke(['--prompt', '红苹果'])
        self.assertEqual(result, 0)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(prepare.call_args.args[0], self.root / 'models/turbo')
        self.assertEqual(call.call_args.args[0][1:], ['--prompt', '红苹果', '--model',
                         str(self.root / 'models/turbo'), '--width', '512', '--height', '512', '--output', 'output.png'])

    def test_explicit_model_dimensions_and_precision_preserved(self):
        args = ['--model', 'my model/turbo', '--prompt-file', '中文 prompt.txt',
                '--width', '1376', '--height', '768', '--precision', 'fp32', '--output', '湖.png']
        _, prepare, call = self.invoke(args)
        prepare.assert_not_called()
        self.assertEqual(call.call_args.args[0][1:], args)

    def test_pe_is_optional_and_explicit_pe_is_not_replaced(self):
        _, prepare, call = self.invoke(['--with-pe', '--prompt', 'village'])
        self.assertEqual(prepare.call_count, 2)
        self.assertTrue(prepare.call_args.kwargs['pe'])
        self.assertIn(str(self.root / 'models/pe'), call.call_args.args[0])
        _, prepare, _ = self.invoke(['--with-pe', '--pe-model', 'existing/pe', '--prompt', 'village'])
        self.assertEqual(prepare.call_count, 1)

    def test_download_only_does_not_generate(self):
        _, prepare, call = self.invoke(['--download-only', '--with-pe'])
        self.assertEqual(prepare.call_count, 2)
        call.assert_not_called()

    def test_model_storage_override(self):
        _, prepare, _ = self.invoke(['--models-dir', 'local models', '--prompt', 'village'])
        self.assertEqual(prepare.call_args.args[0], Path('local models').absolute() / 'turbo')

    def test_native_error_exit_code_survives(self):
        with patch.object(launcher, 'prepare'), patch.object(launcher.subprocess, 'call', return_value=7):
            self.assertEqual(launcher.main(['--prompt', 'apple'], root=self.root), 7)

    def test_no_prompt_does_not_start_large_download(self):
        with patch.object(launcher, 'prepare') as prepare, self.assertRaises(SystemExit):
            launcher.main(['--width', '512'], root=self.root)
        prepare.assert_not_called()

    def test_macos_driver_override_is_respected(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.environment(self.root, 'darwin')['VK_DRIVER_FILES'],
                             str(self.root / 'vulkan/MoltenVK_icd.json'))
        for key in ('VK_DRIVER_FILES', 'VK_ICD_FILENAMES'):
            with patch.dict(os.environ, {key: 'custom.json'}, clear=True):
                env = launcher.environment(self.root, 'darwin')
                self.assertEqual(env, {key: 'custom.json'})

    def test_existing_complete_model_skips_network(self):
        with patch.dict('sys.modules', {'download_model': __import__('tools.download_model', fromlist=['download']),
                                       'release_manifest': __import__('tools.release_manifest', fromlist=['load_manifest'])}):
            from tools import download_model, release_manifest
            model = self.root / 'model'
            model.mkdir()
            (model / 'weight').write_bytes(b'1234')
            manifest = {'files': [{'path': 'weight', 'size': 4}]}
            with patch.object(release_manifest, 'load_manifest', return_value=manifest), patch.object(download_model, 'download') as download:
                launcher.prepare(model, Path('unused'), self.binary, {})
                download.assert_not_called()
                (model / 'weight').unlink()
                with patch.object(subprocess, 'run') as native:
                    launcher.prepare(model, Path('unused'), self.binary, {})
                download.assert_called_once()
                self.assertIn('--verify-model', native.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
