import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.export_vae_encoder import dimensions,rgb_fixture,normalize_rgb
from tools.validate_img2img_encoder import bounded_run,compare_arrays,FP32_GATES,verify
from tools.package_model import ROOT,sha256

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
class EncoderCandidateIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/'outputs/img2img-encoder-32x32-v1/model'
        if not cls.source.exists():raise unittest.SkipTest('Independent real encoder candidate is not installed')
    def candidate(self, directory):
        target=Path(directory)
        for path in self.source.iterdir():
            if path.suffix=='.json':(target/path.name).write_bytes(path.read_bytes())
            elif path.is_file():(target/path.name).symlink_to(path.resolve())
        return target
    def reseal(self,target):
        path=target/'model.json';m=json.loads(path.read_text())
        m['files']={name:sha256(target/name) for name in m['files']}
        path.write_text(json.dumps(m))
    def test_real_candidates_keep_original_identity(self):
        for name in ['32x32','64x32']:
            self.assertEqual(verify(ROOT/f'outputs/img2img-encoder-{name}-v1/model')['component'],'vae-encoder')
    def test_resealed_manifest_cannot_change_exporter_or_weights(self):
        for field,value in [('source_sha256','0'*64),('schema_version',True),('weights',{})]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                t=self.candidate(d);p=t/'model.json';m=json.loads(p.read_text());m[field]=value;p.write_text(json.dumps(m));self.reseal(t)
                with self.assertRaises(ValueError):verify(t)
    def test_resealed_fixture_cannot_change_reviewed_contract(self):
        edits=[('posterior','sample'),('packing','unreviewed'),('official_source_sha256','0'*64),
               ('distribution_source_sha256','0'*64),('diffusers_revision','0'*40),('vae_config_sha256','0'*64),
               ('source_manifests',{}),('encoder_bn',{'eps':1e-5,'affine':False}),('wrapper_bitwise_equal',[1,1,1]),
               ('boundaries',['packed','mean','normalized']),('text_tokens',False),('inputs',{}),
               ('expected',{}),('rgb',{'file':'../input.rgb'})]
        for field,value in edits:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                t=self.candidate(d);p=t/'fixture.json';f=json.loads(p.read_text());f[field]=value;p.write_text(json.dumps(f));self.reseal(t)
                with self.assertRaises(ValueError):verify(t)
    def test_resealed_tensor_cannot_change_dtype_layout_or_name(self):
        for field,value in [('dtype','BF16'),('layout','NHWC'),('file','out1.f32'),('shape',[1,32,2,8])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                t=self.candidate(d);p=t/'fixture.json';f=json.loads(p.read_text());f['expected']['out0'][field]=value;p.write_text(json.dumps(f));self.reseal(t)
                with self.assertRaises(ValueError):verify(t)
    def test_resealed_graph_and_empty_binary_rejected(self):
        for name in ['head.ncnn.param','head.ncnn.bin']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                t=self.candidate(d);(t/name).unlink();(t/name).write_bytes(b'');self.reseal(t)
                with self.assertRaisesRegex(ValueError,'Unreviewed encoder graph'):verify(t)
    def test_resealed_trace_conversion_cannot_change_tools(self):
        for file,field,value in [('trace.json','exporter_sha256','0'*64),('conversion.json','pnnx_sha256','0'*64),('conversion.json','return_code',False),('conversion.json','unsupported_diagnostics',['unsupported'])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                t=self.candidate(d);p=t/file;f=json.loads(p.read_text());f[field]=value;p.write_text(json.dumps(f));self.reseal(t)
                with self.assertRaises(ValueError):verify(t)

if __name__=='__main__':unittest.main()
