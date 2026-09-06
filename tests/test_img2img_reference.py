import unittest

import numpy as np

from tools.reference_img2img import make_start, strength_plan, turbo_sigmas


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


if __name__ == '__main__':
    unittest.main()
