import hashlib
from pathlib import Path
import signal
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import compare_ports
from port_adapters import PortAdapter


class LaunchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / 'corpus'
        self.inputs.mkdir()
        self.noise = struct.pack('<2048f', *range(2048))
        prompt = b'preserved\r\n'
        (self.inputs / 'noise.f32').write_bytes(self.noise)
        (self.inputs / 'prompt.txt').write_bytes(prompt)
        self.case = dict(id='development-example', shape=[64, 64], shape_order='WH',
                         steps=8, cfg=1, noise_shape=[128, 4, 4], noise_path='noise.f32',
                         noise_layout='CHW', noise_dtype='<f4',
                         noise_sha256=hashlib.sha256(self.noise).hexdigest(),
                         prompt_path='prompt.txt', prompt_sha256=hashlib.sha256(prompt).hexdigest(),
                         dtype_by_stage={stage: 'fp32' for stage in ('text', 'dit', 'vae', 'scheduler')},
                         model_identity={'source': 'test only'}, pe={'enabled': False})
        self.binary = self.root / 'runner'
        self.adapter = PortAdapter('candidate', self.binary, self.root / 'model', self.inputs)

    def run_program(self, body, timeout=10):
        self.binary.write_text('#!' + sys.executable + '\n' + body + '\n')
        self.binary.chmod(0o755)
        with patch.object(compare_ports, 'verify_package'):
            return compare_ports.run_side(self.adapter, self.case, self.root / 'run', False, timeout)

    def assert_clocks(self, record):
        self.assertGreater(record['finished_monotonic_ns'], record['started_monotonic_ns'])
        self.assertEqual(record['wall_seconds'],
                         (record['finished_monotonic_ns'] - record['started_monotonic_ns']) / 1e9)

    def test_positive_high_exit_code_is_not_signal(self):
        record = self.run_program('raise SystemExit(139)')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['exit_code'], 139)
        self.assertIsNone(record['termination_signal'])
        self.assertEqual(record['quality_status'], 'incomplete')
        self.assert_clocks(record)

    def test_child_signal_requires_wrapper_evidence(self):
        record = self.run_program('import os, signal\nos.kill(os.getpid(), signal.SIGTERM)')
        self.assertEqual(record['status'], 'crashed')
        self.assertEqual(record['termination_signal'], signal.SIGTERM)
        self.assertEqual(record['termination_evidence'], 'GNU time signal report')
        self.assert_clocks(record)

    def test_timeout_reaps_process_group_and_records_exit(self):
        record = self.run_program('import time\ntime.sleep(30)', timeout=.05)
        self.assertEqual(record['status'], 'timeout')
        self.assertEqual(record['exit_code'], -signal.SIGKILL)
        self.assert_clocks(record)

    def test_launch_error_still_has_complete_clocks(self):
        with patch.object(compare_ports.subprocess, 'Popen', side_effect=OSError('deliberate test failure')):
            record = self.run_program('raise SystemExit(0)')
        self.assertEqual(record['status'], 'incomplete')
        self.assertIn('deliberate test failure', record['reason'])
        self.assert_clocks(record)

    def test_executed_noise_is_snapshot_even_when_source_changes(self):
        program = """import sys
from pathlib import Path
args = sys.argv
noise = Path(args[args.index('--latent') + 1])
output = Path(args[args.index('--output') + 1])
Path(SOURCE).write_bytes(b'changed after verification')
(output.parent / 'observed.f32').write_bytes(noise.read_bytes())
output.write_bytes(b'placeholder only; no quality assertion')
""".replace('SOURCE', repr(str(self.inputs / 'noise.f32')))
        record = self.run_program(program)
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(record['quality_status'], 'incomplete')
        self.assertEqual((self.root / 'run/observed.f32').read_bytes(), self.noise)
        self.assertEqual(record['files']['inputs/noise.f32'], self.case['noise_sha256'])
        self.assertNotEqual((self.inputs / 'noise.f32').read_bytes(), self.noise)
        self.assert_clocks(record)


if __name__ == '__main__':
    unittest.main()
