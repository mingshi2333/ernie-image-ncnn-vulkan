import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.export_vae_encoder import dimensions,rgb_fixture,normalize_rgb
from tools.validate_img2img_encoder import bounded_run,compare_arrays,FP32_GATES

class Img2ImgEncoderTests(unittest.TestCase):
    def test_export_dimensions_are_deliberately_small(self):
        for w,h in [(32,32),(64,32),(16,64)]:dimensions(w,h)
        for w,h in [(0,32),(15,32),(32,17),(128,32),(True,32)]:
            with self.subTest(shape=(w,h)),self.assertRaises(ValueError):dimensions(w,h)
    def test_rgb_identity_and_normalization_fp32_order(self):
        rgb=rgb_fixture(32,16);self.assertEqual(rgb.shape,(16,32,3));self.assertEqual(rgb.dtype,np.uint8)
        self.assertTrue(np.array_equal(rgb,rgb_fixture(32,16)))
        x=normalize_rgb(np.array([[[0,127,255]]],np.uint8))
        self.assertEqual(x.shape,(1,3,1,1));self.assertEqual(x.dtype,np.float32)
        self.assertEqual(x[0,0,0,0],np.float32(-1));self.assertEqual(x[0,2,0,0],np.float32(1))
        expected=np.float32(np.float32(127)-np.float32(127.5))*np.float32(1/127.5)
        self.assertEqual(x[0,1,0,0].view('u4'),expected.view('u4'))
    def test_normalization_rejects_non_rgb(self):
        for a in [np.zeros((1,1,4),np.uint8),np.zeros((1,1,3),np.float32)]:
            with self.assertRaises(ValueError):normalize_rgb(a)
    def test_boundary_quality_gate_failure_preserved(self):
        ref=np.ones(32,np.float32);self.assertTrue(compare_arrays(ref,ref,FP32_GATES)['passed'])
        bad=ref.copy();bad[0]+=1;self.assertFalse(compare_arrays(bad,ref,FP32_GATES)['passed'])
        for bad in [np.array([np.nan]),np.zeros(3)]:
            with self.assertRaises(ValueError):compare_arrays(bad,ref,FP32_GATES)
    def test_low_host_memory_blocks_before_launch_and_records_failure(self):
        with tempfile.TemporaryDirectory() as t,patch('tools.validate_img2img_encoder.available',return_value=1),patch('tools.validate_img2img_encoder.subprocess.Popen') as launch:
            out=Path(t)/'attempt';r=bounded_run(['never-launch'],out);self.assertFalse(r['passed']);launch.assert_not_called()
            self.assertIn('below3GiB',r['failure']);self.assertTrue((out/'result.json').exists())
    def test_failed_launch_recorded_no_unsupervised_fallback(self):
        with tempfile.TemporaryDirectory() as t,patch('tools.validate_img2img_encoder.available',return_value=10*1024**3),patch('tools.validate_img2img_encoder.subprocess.Popen',side_effect=FileNotFoundError('systemd-run missing')):
            r=bounded_run(['never-launch'],Path(t)/'attempt');self.assertFalse(r['passed']);self.assertIn('systemd-run missing',r['failure'])
if __name__=='__main__':unittest.main()
