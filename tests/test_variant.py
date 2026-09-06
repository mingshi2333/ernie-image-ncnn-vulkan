"""Shape requests must respect both spatial and complete attention capacity."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_variant import dimensions


class VariantTests(unittest.TestCase):
    def test_rectangular_axes_and_full_text_capacity(self):
        self.assertEqual(dimensions(512, 384, 2048), (32, 24, 2816))

    def test_full_resolution_and_maximum_prompt_fit(self):
        self.assertEqual(dimensions(1024, 1024, 2048), (64, 64, 6144))

    def test_invalid_sizes_do_not_round_or_truncate(self):
        for request in [(513, 384, 128), (0, 512, 128), (512, 384, 2049),
                        (512, 384, 0), (1040, 384, 128)]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                dimensions(*request)


if __name__ == '__main__':
    unittest.main()
