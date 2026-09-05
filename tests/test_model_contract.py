"""Guard against corrupted artifacts, incompatible fixtures, and accidental lossy packing."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_block import sha256
from validate_dit_block import verify
from pack_block_weights import pack


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        (self.directory / 'in0.f32').write_bytes(struct.pack('<f', 0.5))
        fixture = {'tokens': 1, 'weights_sha256': 'test', 'inputs': {},
                   'expected': {'file': 'in0.f32', 'shape': [1], 'sha256': sha256(self.directory / 'in0.f32')}}
        (self.directory / 'fixture.json').write_text(json.dumps(fixture))
        self.manifest = {'tokens': 1, 'weights_sha256': 'test',
                         'ncnn_revision': json.loads((ROOT / 'sources.lock.json').read_text())['ncnn']['revision'],
                         'files': {name: sha256(self.directory / name) for name in ['fixture.json', 'in0.f32']}}
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        (self.directory / 'model.json').write_text(json.dumps(self.manifest))

    def test_valid_artifacts(self):
        self.assertEqual(verify(self.directory, self.directory)[1]['tokens'], 1)

    def test_corrupted_bytes_fail(self):
        (self.directory / 'in0.f32').write_bytes(struct.pack('<f', 0.25))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify(self.directory, self.directory)

    def test_wrong_static_shape_fails(self):
        self.manifest['tokens'] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Fixture shape'):
            verify(self.directory, self.directory)

    def test_wrong_runtime_revision_fails(self):
        self.manifest['ncnn_revision'] = 'unreviewed'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Runtime revision'):
            verify(self.directory, self.directory)


class PackingTests(unittest.TestCase):
    def make_stream(self, directory, value=0.5, trailing=b''):
        # Small sections exercise format conversion. Actual ERNIE geometry is
        # validated separately by the official-weight C++ fixtures.
        lines = ['7767517', '11 12']
        data = bytearray()
        for index in range(4):
            lines.append(f'RMSNorm norm{index} 1 1 x y 0=4 2=1')
            data += struct.pack('<4f', 0.1, 0.2, 0.3, 0.4)  # these must remain FP32
        for index in range(7):
            lines.append(f'Gemm linear{index} 1 1 x y 4=0 5=1 6=1 8=3 9=2 10=-1')
            data += b'\0' * 4 + struct.pack('<6f', *([value] * 6))
        data += trailing
        (directory / 'block.ncnn.param').write_text('\n'.join(lines) + '\n')
        (directory / 'block.ncnn.bin').write_bytes(data)
        (directory / 'model.json').write_text(json.dumps({'files': {name: sha256(directory / name)
                                         for name in ['block.ncnn.param', 'block.ncnn.bin']}}))
        return hashlib.sha256(data).hexdigest()

    def test_lossless_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = self.make_stream(directory)
            result = pack(directory, directory / 'packed')
            self.assertTrue(result['lossless'])
            self.assertEqual(result['reconstructed_fp32_sha256'], original)
            self.assertLess(result['stored_bytes'], result['original_bytes'])

    def test_reject_non_bf16_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_stream(directory, value=0.3)
            with self.assertRaisesRegex(ValueError, 'not exactly representable'):
                pack(directory, directory / 'packed')

    def test_reject_unconsumed_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_stream(directory, trailing=b'extra')
            with self.assertRaisesRegex(ValueError, 'complete reviewed block'):
                pack(directory, directory / 'packed')


if __name__ == '__main__':
    unittest.main()
