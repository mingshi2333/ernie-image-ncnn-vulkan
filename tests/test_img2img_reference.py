import unittest

import numpy as np

from tools.reference_img2img import make_start, strength_plan, turbo_sigmas
from tools.reference_img2img_positive import reviewed_profile, validate_inputs, validate_start, validate_suffix
from tools.validate_img2img_positive_result import metrics


class Img2ImgReferenceTests(unittest.TestCase):
    def setUp(self):
        self.encoded = np.full((1, 128, 2, 3), np.float32(2), dtype=np.float32)
        self.noise = np.full_like(self.encoded, np.float32(-1))

    def test_eight_step_strength_contract(self):
        expected = [(0, 8), (2, 6), (4, 4), (6, 2), (8, 0)]
        for strength, pair in zip([0, .25, .5, .75, 1], expected):
            plan = strength_plan(8, strength)
            self.assertEqual((plan['denoise_steps'], plan['start_step']), pair)
        self.assertEqual(strength_plan(8, .3125)['denoise_steps'], 3)
        self.assertEqual(strength_plan(8, np.finfo(np.float32).tiny)['denoise_steps'], 1)

    def test_pinned_fp32_sigmas(self):
        expected = np.array([1, .9655172228813171, .9230769276618958, .8695651888847351,
                             .800000011920929, .7058823704719543, .5714285969734192,
                             .3636363744735718, 0], dtype=np.float32)
        np.testing.assert_array_equal(turbo_sigmas(8).view('u4'), expected.view('u4'))

    def test_endpoints_and_fp32_interpolation(self):
        zero, plan = make_start(self.encoded, np.empty((0,), np.float32), 8, 0)
        self.assertEqual(plan['start_step'], 8)
        np.testing.assert_array_equal(zero, self.encoded)
        self.assertFalse(np.shares_memory(zero, self.encoded))
        one, plan = make_start(self.encoded, self.noise, 8, 1)
        self.assertEqual(plan['start_step'], 0)
        np.testing.assert_array_equal(one, self.noise)
        for strength in [.25, .5, .75]:
            actual, plan = make_start(self.encoded, self.noise, 8, strength)
            sigma = np.float32(plan['sigma'])
            expected = sigma * self.noise + (np.float32(1) - sigma) * self.encoded
            np.testing.assert_array_equal(actual.view('u4'), expected.view('u4'))

    def test_invalid_math_inputs_fail_closed(self):
        for strength in [-1, 1.01, float('nan'), float('inf')]:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                make_start(self.encoded, self.noise, 8, strength)
        with self.assertRaises(ValueError):
            make_start(self.encoded.astype(np.float64), self.noise, 8, .5)
        bad = self.noise.copy(); bad.flat[0] = np.nan
        with self.assertRaises(ValueError):
            make_start(self.encoded, bad, 8, .5)
        with self.assertRaises(ValueError):
            strength_plan(True, .5)
        with self.assertRaises(ValueError):
            strength_plan(8, True)

    def test_positive_input_contract_rejects_self_attested_encoder(self):
        import hashlib, json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture.json'; fixture.write_text('{}')
            values = np.zeros((1, 128, 24, 32), dtype='<f4')
            values.tofile(root / 'saved-noise.f32'); values.tofile(root / 'start-4.f32')
            prompt = root / 'prompt.txt'; prompt.write_text('apple\n')
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            contract = {'request': {'width': 512, 'height': 384, 'steps': 8, 'strength': .5,
                                    'start_step': 4, 'denoise_steps': 4, 'pe': {'enabled': False},
                                    'text_precision': 'fp32', 'text_reduction': 'vector'},
                        'encoder_fixture': {'file': fixture.name, 'sha256': sha(fixture)},
                        'noise': {'file': 'saved-noise.f32', 'shape': list(values.shape), 'dtype': '<f4', 'sha256': sha(root/'saved-noise.f32')},
                        'start': {'file': 'start-4.f32', 'shape': list(values.shape), 'dtype': '<f4', 'sha256': sha(root/'start-4.f32')},
                        'prompt': {'file': prompt.name, 'text': 'apple', 'sha256': sha(prompt)}}
            (root / 'input-contract.json').write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, 'not reviewed'):
                validate_inputs(root)

    def test_positive_suffix_rejects_incomplete_denominator(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = {'complete': True, 'prompt': 'apple', 'steps': 8, 'start_step': 4, 'ids': [1],
                       'config': {'packed_width': 32, 'packed_height': 24, 'text_bucket': 2048,
                                  'dit_text_tokens': 2048, 'text_layers': 25, 'dit_layers': 36},
                       'inputs': {}, 'outputs': [], 'final': {}}
            (root / 'fixture.json').write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, 'four complete'):
                validate_suffix(root, fixture, 'apple', '0' * 64)

    def test_positive_start_must_be_recomputed_from_official_encoder(self):
        shape = (1, 128, 24, 32)
        encoded = np.full(shape, np.float32(2), dtype=np.float32)
        noise = np.full(shape, np.float32(-1), dtype=np.float32)
        start = np.float32(.8) * noise + (np.float32(1) - np.float32(.8)) * encoded
        validate_start(encoded, noise, start)
        start.flat[0] = 0
        with self.assertRaisesRegex(ValueError, 'reviewed FP32 mixture'):
            validate_start(encoded, noise, start)

    def test_positive_profiles_are_fixed_and_shape_specific(self):
        profile = reviewed_profile(1024, 1024)
        self.assertEqual(profile['latent_shape'], (1, 128, 64, 64))
        self.assertEqual(profile['text_bucket'], 64)
        self.assertEqual(profile['source_manifest'],
                         '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1')
        with self.assertRaisesRegex(ValueError, 'No reviewed'):
            reviewed_profile(768, 768)
        values = np.zeros(profile['latent_shape'], dtype=np.float32)
        validate_start(values, values, values, profile['latent_shape'])
        with self.assertRaisesRegex(ValueError, 'Invalid positive-strength'):
            validate_start(values[:, :, :-1], values[:, :, :-1], values[:, :, :-1], profile['latent_shape'])

    def test_positive_contract_requires_canonical_input_names(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input-contract.json').write_text(json.dumps({
                'request': {'width': 512, 'height': 384, 'steps': 8, 'strength': .5,
                            'start_step': 4, 'denoise_steps': 4, 'pe': {'enabled': False},
                            'text_precision': 'fp32', 'text_reduction': 'vector'},
                'noise': {'file': 'alternate-noise.f32'}, 'start': {'file': 'alternate-start.f32'},
                'prompt': {'file': 'prompt.txt'}}))
            with self.assertRaisesRegex(ValueError, 'not canonical'):
                validate_inputs(root)

    def test_result_metrics_reject_bad_denominator(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.zeros(2, dtype='<f4').tofile(root / 'candidate.f32')
            np.zeros(3, dtype='<f4').tofile(root / 'reference.f32')
            with self.assertRaisesRegex(ValueError, 'size or finite'):
                metrics(root / 'candidate.f32', root / 'reference.f32')
            np.array([0, np.nan], dtype='<f4').tofile(root / 'reference.f32')
            with self.assertRaisesRegex(ValueError, 'size or finite'):
                metrics(root / 'candidate.f32', root / 'reference.f32')


if __name__ == '__main__':
    unittest.main()
