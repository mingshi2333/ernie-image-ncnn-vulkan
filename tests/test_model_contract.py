"""Guard against corrupted artifacts, incompatible fixtures, and accidental lossy packing."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_block import sha256
from validate_dit_block import verify
from pack_block_weights import pack
from build_dit_weights import graph_hash, GRAPH_SHA256
from fetch_component import bounded_ranges, fetch_component


class SingleFileFetchTests(unittest.TestCase):
    def test_unsharded_exact_component_and_reuse(self):
        payload = struct.pack('<2f', 1.25, -0.5)
        header = {'selected.weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]},
                  'other.weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [8, 12]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'selected.safetensors'
            with patch('fetch_component.read_header', return_value=(header, 100, 'header-hash')), \
                 patch('fetch_component.bounded_ranges', return_value=iter([payload])) as download, \
                 redirect_stdout(io.StringIO()):
                manifest = fetch_component('selected.', output, subfolder='text_encoder', single_file='model.safetensors')
            self.assertEqual(manifest['payload_bytes'], 8)
            self.assertIn('single_file_url', manifest)
            self.assertNotIn('index_url', manifest)
            self.assertEqual(manifest['tensors'][0]['source_offsets'], [100, 108])
            self.assertEqual(manifest['tensors'][0]['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(output.read_bytes()[-8:], payload)
            download.assert_called_once()
            with patch('fetch_component.read_header', return_value=(header, 100, 'header-hash')), \
                 patch('fetch_component.bounded_ranges') as download, redirect_stdout(io.StringIO()):
                reused = fetch_component('selected.', output, subfolder='text_encoder', single_file='model.safetensors')
            self.assertEqual(reused['sha256'], manifest['sha256'])
            download.assert_not_called()
            # The same prefix/header in a different component is a different source.
            with patch('fetch_component.read_header', return_value=(header, 100, 'header-hash')), \
                 patch('fetch_component.bounded_ranges') as download:
                with self.assertRaises(FileExistsError):
                    fetch_component('selected.', output, subfolder='wrong_component', single_file='model.safetensors')
            download.assert_not_called()

    def test_unsharded_path_rejected_before_network(self):
        with patch('fetch_component.read_header') as read_header:
            with self.assertRaises(ValueError):
                fetch_component('x.', Path('unused'), single_file='../model.safetensors')
            read_header.assert_not_called()


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

    def test_reviewed_old_revision_does_not_require_reexport(self):
        self.manifest['ncnn_revision'] = '6a1bf000f363714839a36793addc8c879d3d899e'
        self.write_manifest()
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


class DirectGraphTests(unittest.TestCase):
    def test_three_reviewed_buckets_share_the_exact_topology(self):
        for tokens in [24, 288, 4160]:
            graph = (ROOT / f'artifacts/2026-09-05/dit-block/models/s{tokens}/block.ncnn.param').read_text()
            self.assertEqual(graph_hash(graph, tokens), GRAPH_SHA256)

    def test_shape_mismatch_rejected(self):
        graph = (ROOT / 'artifacts/2026-09-05/dit-block/models/s24/block.ncnn.param').read_text()
        with self.assertRaisesRegex(ValueError, 'Static token dimension'):
            graph_hash(graph, 288)

    def test_math_changes_are_not_treated_as_weight_compatible(self):
        graph = (ROOT / 'artifacts/2026-09-05/dit-block/models/s24/block.ncnn.param').read_text()
        self.assertNotEqual(graph_hash(graph.replace('1=1.000000e-6', '1=1.000000e-5'), 24), GRAPH_SHA256)
        self.assertNotEqual(graph_hash(graph.replace('3=1 4=0', '3=0 4=0'), 24), GRAPH_SHA256)


class DownloadWindowTests(unittest.TestCase):
    def test_partial_response_is_retried_before_yielding_any_bytes(self):
        responses = [io.BytesIO(b'bad'), io.BytesIO(b'abcd'), io.BytesIO(b'ef')]
        with patch('fetch_component.request_range', side_effect=responses) as request, \
                patch('fetch_component.time.sleep'), redirect_stdout(io.StringIO()):
            chunks = list(bounded_ranges('fixture', 10, 15, window=4))
        self.assertEqual(chunks, [b'abcd', b'ef'])
        self.assertEqual([call.args[1:] for call in request.call_args_list], [(10, 13), (10, 13), (14, 15)])

    def test_persistent_truncation_fails_after_bounded_retries(self):
        with patch('fetch_component.request_range', side_effect=lambda *args: io.BytesIO(b'')) as request, \
                patch('fetch_component.time.sleep'), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Truncated range window'):
                list(bounded_ranges('fixture', 0, 3, window=4))
        self.assertEqual(request.call_count, 3)

    def test_oversized_response_is_not_accepted(self):
        with patch('fetch_component.request_range', side_effect=lambda *args: io.BytesIO(b'extra')), \
                patch('fetch_component.time.sleep'), redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                list(bounded_ranges('fixture', 0, 3, window=4))


if __name__ == '__main__':
    unittest.main()
