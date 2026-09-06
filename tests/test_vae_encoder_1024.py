import json
import unittest
from pathlib import Path

from tools.reference_vae_encoder_1024 import development_rgb, resource_plan
from tools.specialize_vae_encoder_1024 import reviewed_dimensions, specialized_lines


class Encoder1024PreparationTests(unittest.TestCase):
    def test_trusted_registry_is_fixed_to_reviewed_source_and_files(self):
        contract = json.loads(Path('tokenizer/schema3_contract.json').read_text())
        source = '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1'
        entry = contract['reviewed_encoders'][source]
        self.assertEqual((entry['width'], entry['height']), (1024, 1024))
        self.assertEqual(entry['files']['vae/encoder.ncnn.param'], {
            'sha256': 'd3207b56f558d65b9901ff73640b51ae2a0934143b43eaeab6cad45e275b9ceb',
            'size': 8667,
        })
        self.assertEqual(entry['evidence']['official_fixture_sha256'],
                         '88f2e8b7ad63a47fd993b069282dcb8bca9042cf56a470efa84237b711d9f7d9')

    def test_development_input_and_resource_plan(self):
        rgb = development_rgb()
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertEqual(rgb.dtype.str, '|u1')
        plan = resource_plan()
        self.assertEqual(plan['mean'], [1, 32, 128, 128])
        self.assertEqual(plan['packed'], [1, 128, 64, 64])
        self.assertEqual(plan['baseline_512x384_attention_positions'], 3072)
        self.assertAlmostEqual(plan['dense_attention_element_ratio_vs_512x384'], 256 / 9)
        self.assertEqual(plan['one_dense_fp32_attention_bytes'], 1024**3)
        self.assertEqual(plan['swap_max_bytes'], 0)

    def test_only_two_shape_records_change(self):
        source = ['7767517 8425958', 'Reshape reshape_77 1 1 89 90 0=16 1=512',
                  'Reshape reshape_78 1 1 94 95 0=4 1=4 2=512']
        result, changes = specialized_lines(source)
        self.assertEqual(len(changes), 2)
        self.assertEqual(result[1].split()[6:], ['0=16384', '1=512'])
        self.assertEqual(result[2].split()[6:], ['0=128', '1=128', '2=512'])

    def test_other_shapes_and_graphs_fail_closed(self):
        reviewed_dimensions(1024, 1024)
        for shape in ((1024, 768), (512, 512), (2048, 1024)):
            with self.assertRaises(ValueError):
                reviewed_dimensions(*shape)
        with self.assertRaises(ValueError):
            specialized_lines(['Reshape reshape_77 1 1 89 90 0=17 1=512'])


if __name__ == '__main__':
    unittest.main()
