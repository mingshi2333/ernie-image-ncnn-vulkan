"""Independently exported buckets must differ only in reviewed token dimensions."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from rebucket_text import graph_hash, GRAPH_SHA256


class TextBucketTests(unittest.TestCase):
    def test_independent_exports_have_identical_equations(self):
        paths = {32: ROOT/'artifacts/2026-09-05/pipeline/models/text-s32-v1/block-00/text.ncnn.param',
                 64: ROOT/'tests/fixtures/text-s64.ncnn.param'}
        for tokens, path in paths.items():
            self.assertEqual(graph_hash(path.read_text(), tokens), GRAPH_SHA256)

    def test_wrong_manifest_bucket_rejected(self):
        with self.assertRaisesRegex(ValueError, 'static text dimension'):
            graph_hash((ROOT/'tests/fixtures/text-s64.ncnn.param').read_text(), 32)

    def test_normalization_equation_is_not_normalized_away(self):
        graph = (ROOT/'tests/fixtures/text-s64.ncnn.param').read_text()
        self.assertNotEqual(graph_hash(graph.replace('1=1.000000e-5', '1=1.000000e-6'), 64), GRAPH_SHA256)

    def test_gqa_head_count_is_not_a_token_dimension(self):
        graph = (ROOT/'tests/fixtures/text-s64.ncnn.param').read_text()
        self.assertNotEqual(graph_hash(graph.replace('0=128 1=8 2=64', '0=128 1=16 2=64'), 64), GRAPH_SHA256)


if __name__ == '__main__':
    unittest.main()
