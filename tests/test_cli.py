"""Reject ambiguous prompts and invalid requests before loading large weights."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(os.environ.get('ERNIE_TEST_RUNNER', ROOT/'build/ernie-image'))


@unittest.skipUnless(RUNNER.is_file(), 'Build the native generator first')
class CliTests(unittest.TestCase):
    def request(self, *args):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(RUNNER), '--model', str(Path(folder)/'absent'),
                                     '--output', str(Path(folder)/'new.png'), *args],
                                    text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_prompt_sources_are_mutually_exclusive(self):
        self.assertIn('one prompt source', self.request('--prompt', 'cat', '--prompt-file', 'absent.txt'))

    def test_duplicate_prompt_cannot_silently_replace_first(self):
        self.assertIn('one prompt source', self.request('--prompt', 'cat', '--prompt', 'dog'))

    def test_missing_prompt_file_is_reported_before_model_loading(self):
        self.assertIn('Cannot open prompt file', self.request('--prompt-file', 'absent.txt'))

    def test_invalid_utf8_prompt_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'prompt.txt'
            path.write_bytes(b'cat\xffdog')
            self.assertIn('UTF-8', self.request('--prompt-file', str(path)))

    def test_prompt_file_is_bounded_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'prompt.txt'
            with path.open('wb') as file:
                file.truncate(1024*1024 + 1)
            self.assertIn('1 MiB', self.request('--prompt-file', str(path)))

    def test_valid_bf16_request_reaches_model_validation(self):
        self.assertIn('Cannot open model.cfg', self.request('--prompt', 'cat', '--precision', 'bf16'))

    def test_dimensions_require_both_axes(self):
        self.assertIn('width and height together', self.request('--prompt', 'cat', '--width', '512'))

    def test_dimensions_must_be_multiples_of_sixteen(self):
        self.assertIn('multiples of 16', self.request('--prompt', 'cat', '--width', '513', '--height', '384'))

    def test_rectangular_request_reaches_model_validation(self):
        self.assertIn('Cannot open model.cfg', self.request('--prompt', 'cat', '--width', '512', '--height', '384'))

    def test_pe_options_require_an_explicit_model(self):
        self.assertIn('PE options require --pe-model', self.request('--prompt', 'cat', '--pe-greedy'))

    def test_pe_rejects_infinite_temperature(self):
        self.assertIn('finite number', self.request('--prompt', 'cat', '--pe-model', 'pe', '--pe-temperature', 'inf'))

    def test_pe_rejects_invalid_top_p(self):
        self.assertIn('top-p in (0,1]', self.request('--prompt', 'cat', '--pe-model', 'pe', '--pe-top-p', '1.1'))

    def test_pe_cannot_change_prompt_of_precomputed_embeddings(self):
        self.assertIn('cannot be combined', self.request('--prompt', 'cat', '--pe-model', 'pe', '--embeddings', 'text.f32'))

    def test_repeated_resolution_cannot_silently_override(self):
        self.assertIn('Duplicate option', self.request('--prompt', 'cat', '--width', '512', '--width', '768'))

    def test_threads_are_strictly_bounded(self):
        self.assertIn('Threads must be', self.request('--prompt', 'cat', '--threads', '0'))
        self.assertIn('Threads must be', self.request('--prompt', 'cat', '--threads', '257'))

    def test_gpu_requires_a_vulkan_stage(self):
        self.assertIn('requires a Vulkan', self.request('--prompt', 'cat', '--device', 'cpu',
                                                       '--precision', 'fp32', '--gpu', '0'))

    def test_unsupported_text_device_is_explicit(self):
        self.assertIn('currently supported', self.request('--prompt', 'cat', '--text-device', 'vulkan'))

    def test_img2img_is_recognized_but_fail_closed(self):
        self.assertIn('unsupported until the F2', self.request('--prompt', 'cat', '--input', 'missing.jpg',
                                                               '--strength', '.5', '--resize', 'fit',
                                                               '--background', '#102030'))

    def test_img2img_only_options_require_input(self):
        self.assertIn('require --input', self.request('--prompt', 'cat', '--strength', '.5'))

    def test_invalid_img2img_metadata_is_rejected(self):
        self.assertIn('Strength must be', self.request('--prompt', 'cat', '--input', 'a.png', '--strength', '1.1'))
        self.assertIn('Resize must be', self.request('--prompt', 'cat', '--input', 'a.png', '--resize', 'squash'))
        self.assertIn('Background must be', self.request('--prompt', 'cat', '--input', 'a.png', '--background', '#xx0000'))

    def test_output_format_is_checked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(RUNNER), '--model', str(Path(folder)/'absent'), '--prompt', 'cat',
                                     '--output', str(Path(folder)/'new.gif')], text=True,
                                    capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output extension', result.stderr)

    def test_diagnose_lists_runtime_without_loading_a_model(self):
        result = subprocess.run([str(RUNNER), '--diagnose'], text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r'vulkan_compiled=(true|false)')
        self.assertRegex(result.stdout, r'gpu_count=\d+')
        self.assertRegex(result.stdout, r'default_gpu_index=-?\d+')

    def test_diagnose_reads_only_bounded_model_config_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'model'
            model.mkdir()
            (model/'model.cfg').write_text(
                'packed_width 4\npacked_height 4\ntext_bucket 32\n'
                'dit_text_tokens 272\ntext_layers 25\ndit_layers 36\n')
            result = subprocess.run([str(RUNNER), '--diagnose', '--model', str(model)],
                                    text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('model_config_schema=1', result.stdout)
        self.assertIn('packed_width=4', result.stdout)
        self.assertIn('dit_layers=36', result.stdout)

    def test_diagnose_rejects_generation_options(self):
        result = subprocess.run([str(RUNNER), '--diagnose', '--prompt', 'cat'],
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('accepts only', result.stderr)


if __name__ == '__main__':
    unittest.main()
