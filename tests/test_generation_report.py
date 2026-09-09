"""Parse the actual C++ writer output, including Unicode and JSON control bytes."""
import json
import os
import subprocess
import unittest
from pathlib import Path

RUNNER = Path(os.environ.get('ERNIE_REPORT_RUNNER', Path(__file__).resolve().parents[1] / 'build/ernie-generation-report-contract'))

@unittest.skipUnless(RUNNER.is_file(), 'Build the native generation report contract first')
class GenerationReportTest(unittest.TestCase):
    def test_native_json_roundtrip(self):
        run = subprocess.run([str(RUNNER)], capture_output=True, check=True, timeout=10)
        report = json.loads(run.stdout)
        self.assertEqual(report['prompt'], '中文 "quote"\\\r\n\t\0')
        self.assertEqual(report['token_ids'], [1, 23, 131071])
        self.assertEqual(report['model']['text_bucket'], 32)
        self.assertEqual(report['model']['dit_text_tokens'], 64)
        self.assertEqual(report['weight_cache']['peak_charged_bytes'], 5687678976)
        self.assertEqual(report['placement_requests']['ram'], 262)
        self.assertEqual(report['request']['dit_prefetch_mib'], 1024)
        self.assertEqual(report['request']['gpu_memory'], 'auto')
        self.assertEqual(report['request']['gpu_spill_mib'], 2048)
        self.assertEqual(report['request']['oom_retries'], 3)
        self.assertEqual(report['weight_prefetch'], {
            'started': 3, 'used': 2, 'skipped': 1, 'peak_charged_bytes': 536870912, 'overlap_seconds': .25})
        self.assertEqual(report['gpu_memory'], {
            'device_allocations': 32, 'host_allocations': 16, 'host_peak_bytes': 536870912,
            'fallbacks': 8, 'allocation_failures': 2,
            'host_device_local_allocations': 4, 'host_non_device_local_allocations': 12})
        self.assertEqual(report['memory_recovery'], {'retries': 2, 'attention_query_rows': 32})
        self.assertEqual(report['model_loading_requested'], 'mapped')
        self.assertFalse(report['allocation_instrumentation'])
        self.assertFalse(report['trace_enabled'])
        self.assertIsNone(report['request']['strength'])
        self.assertEqual(report['total_seconds'], 1.375)
        self.assertEqual(report['vae_and_image_encode_seconds'], .875)
