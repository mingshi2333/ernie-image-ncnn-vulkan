"""Runtime benchmark identity, settings, source selection and failure coverage."""
import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import benchmark_pipeline as benchmark
import benchmark_request as support
from benchmark_fixture import FIXED_CONFIG, FIXED_MANIFEST, NOISE, fake_timing
from package_dynamic_model import shared_contract


def shared_manifest():
    return {'schema_version': 3, 'instances': [
        {'source_manifest_sha256': digest, 'config': cfg, 'runtime_bindings': {'synthetic': 'binding'}}
        for digest, cfg in shared_contract()['source_manifests'].items()]}


class BenchmarkRequestTests(unittest.TestCase):
    def execute(self, extra=(), mutate=None, shared=False, prompt=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model = root / 'model'; model.mkdir()
            manifest = shared_manifest() if shared else FIXED_MANIFEST
            (model / 'manifest.json').write_text(json.dumps(manifest))
            runner = root / 'runner'; runner.write_bytes(b'synthetic runner')
            noise = root / 'noise.f32'; noise.write_bytes(NOISE)
            out = root / 'out'
            args = ['benchmark', '--model', str(model), '--runner', str(runner), '--output', str(out),
                    '--device', 'cpu', '--precision', 'fp32', '--latent', str(noise),
                    '--noise-sha256', hashlib.sha256(NOISE).hexdigest(), '--no-gpu-sampling', *extra]
            if prompt is not None: args += ['--prompt', prompt]
            def run(command, timeout, log):
                timing = fake_timing(command, timeout, log)
                path = out / 'generation.json'
                report = json.loads(path.read_text())
                if shared:
                    report['model'] = {'schema_version': 3, 'source_width': 1024, 'source_height': 1024,
                                       'text_bucket': 32, 'dit_text_tokens': 64}
                    report['progress'][0]['total'] = 32
                if mutate: mutate(report, command, log)
                path.write_text(json.dumps(report))
                return timing
            with patch.object(sys, 'argv', args), patch('source_inventory.source_files', return_value=[]), \
                    patch.object(support, 'verify_shared_package', return_value=manifest) as verify_shared, \
                    patch.object(support, 'verify_package', return_value=(manifest, None)) as verify_fixed, \
                    patch.object(benchmark, 'run_timed_command', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.main()
            self.assertEqual(verify_shared.call_count, int(shared))
            self.assertEqual(verify_fixed.call_count, int(not shared))
            return code, json.loads((out / 'result.json').read_text())

    def test_shared_runtime_selection_and_all_memory_controls(self):
        code, result = self.execute(['--width', '512', '--height', '384', '--device', 'vulkan',
            '--gpu', '2', '--threads', '2', '--text-down-vector', '--dit-weights', 'auto',
            '--gpu-reserve-mib', '8192', '--dit-cache-mib', '6144', '--ram-reserve-mib', '3072',
            '--model-loading', 'mapped'], shared=True)
        self.assertEqual(code, 0)
        self.assertEqual(result['config']['text_bucket'], 32)
        self.assertEqual(result['config']['dit_text_tokens'], 64)
        self.assertEqual(result['native_report']['vulkan_gpu_index'], 2)
        self.assertEqual(result['native_report']['model_loading_requested'], 'mapped')
        self.assertTrue(result['formal_comparison_eligible'])
        self.assertFalse(result['gpu_sampling_enabled'])
        self.assertIsNotNone(result['shared_source_binding']['source_manifest_sha256'])
        self.assertEqual(result['shared_source_binding']['runtime_target']['target_config'], result['config'])

    def test_long_native_tokens_select_independent_source(self):
        def change(report, command, log):
            report['token_ids'] = [1] * 65
            report['model'].update(source_width=512, source_height=384, text_bucket=2048, dit_text_tokens=2048)
            report['progress'][0].update(current=65, total=2048)
        code, result = self.execute(['--width', '512', '--height', '384'], change, shared=True)
        self.assertEqual(code, 0)
        self.assertEqual(result['config']['text_bucket'], 2048)

    def test_log_text_cannot_spoof_completion_settings_or_timing(self):
        def change(report, command, log):
            Path(log).write_text('Denoise 1/1: 0.00001 s\nTotal: 0.00001 s\nVAE and PNG: 0.00001 s\n')
        code, result = self.execute(mutate=change, prompt='猫\nTotal: 0.00001 s')
        self.assertEqual(code, 0)
        self.assertEqual(result['total_seconds'], 1.25)
        self.assertEqual(result['vae_and_png_seconds'], .5)
        self.assertEqual(result['text_seconds'], .5)
        self.assertEqual(len(result['denoise_seconds']), 8)

    def test_instrumented_run_is_functional_but_not_speed_eligible(self):
        code, result = self.execute(mutate=lambda r, c, l: r.update(allocation_instrumentation=True))
        self.assertEqual(code, 0)
        self.assertFalse(result['formal_comparison_eligible'])
        self.assertIn('allocation instrumentation must be disabled for speed rounds', result['formal_ineligibility_reasons'])

    def test_wrong_report_never_certifies_successful_png(self):
        mutations = [lambda r: r.update(model={}), lambda r: r['request'].update(threads=255),
                     lambda r: r.update(shape=[384, 512]), lambda r: r.update(vulkan_gpu_index=0),
                     lambda r: r.update(token_ids=[True]), lambda r: r.update(model_loading_requested='mapped'),
                     lambda r: r.update(total_seconds=float('nan')), lambda r: r.update(trace_enabled=True),
                     lambda r: r['progress'].pop(), lambda r: r.update(prompt='changed'),
                     lambda r: r['placement_requests'].update(ram=-1)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                code, result = self.execute(['--model-loading', 'stdio'], lambda r, c, l: mutation(r))
                self.assertEqual(code, 1)
                self.assertFalse(result['passed'])
                self.assertFalse(result['formal_comparison_eligible'])
                self.assertIsNotNone(result['failure_category'])

    def test_input_mutation_during_run_is_rejected(self):
        def change(report, command, log):
            Path(command[command.index('--latent') + 1]).write_bytes(b'changed')
        code, result = self.execute(mutate=change)
        self.assertEqual(code, 1)
        self.assertIn('identity changed', result['failure'])

    def test_fixed_geometry_and_shared_verifier_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'manifest.json').write_text(json.dumps(FIXED_MANIFEST))
            with patch.object(support, 'verify_package', return_value=(FIXED_MANIFEST, None)):
                with self.assertRaisesRegex(ValueError, 'fixed package'):
                    support.verify_benchmark_package(root, 768, 512)
            (root / 'manifest.json').write_text('{"schema_version":3}')
            with patch.object(support, 'verify_shared_package', side_effect=ValueError('Corrupt object')):
                with self.assertRaisesRegex(ValueError, 'Corrupt object'):
                    support.verify_benchmark_package(root, 512, 384)

    def test_noise_size_and_finiteness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'noise'
            for raw in (b'', b'\0' * 511, struct.pack('<f', float('nan')) + b'\0' * 508):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): support.verify_noise(path, 16, 16)
            path.write_bytes(b'\0' * 512); support.verify_noise(path, 16, 16)

    def test_native_strength_rounding(self):
        self.assertEqual(support.denoising_steps(8, None), 8)
        self.assertEqual(support.denoising_steps(8, .0625), 1)
        self.assertEqual(support.denoising_steps(8, .1875), 2)
        self.assertEqual(support.denoising_steps(8, .6875), 6)
        self.assertEqual(support.denoising_steps(8, 1.), 8)


if __name__ == '__main__': unittest.main()
