import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from acceptance_manifest import freeze_inputs, verify_inputs, canonical, digest

class AcceptanceManifestTests(unittest.TestCase):
    def case(self):
        return dict(id='changed', split='development', prompt='a red apple', prompt_source='unit-fixture', shape=[64,64], seed=42, steps=8, pe={'enabled':False}, model_identity='unit-fixture', dtype_by_stage=dict(text='fp32',dit='fp32',vae='fp32',scheduler='fp32'))
    def test_freeze_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            m=freeze_inputs({'cases':[self.case()]},Path(d)); verify_inputs(m,Path(d))
            self.assertEqual(m['cases'][0]['noise_shape'],[128,4,4])
    def test_modified_prompt_is_not_same_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=freeze_inputs({'cases':[self.case()]},root)
            (root/'changed/prompt.txt').write_bytes(b'a red apple ')
            with self.assertRaises(ValueError):verify_inputs(m,root)
    def test_modified_last_noise_byte_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=freeze_inputs({'cases':[self.case()]},root);p=root/'changed/initial.f32'
            b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
            with self.assertRaises(ValueError):verify_inputs(m,root)
    def test_duplicate_formal_id_fails(self):
        c=self.case();c['split']='formal'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):freeze_inputs({'cases':[c,c]},Path(d))
    def test_verify_duplicate_id_fails(self):
        with tempfile.TemporaryDirectory() as d:
            m=freeze_inputs({'cases':[self.case()]},Path(d));m['cases']*=2
            unsigned=dict(m);unsigned.pop('manifest_sha256');m['manifest_sha256']=digest(canonical(unsigned))
            with self.assertRaises(ValueError):verify_inputs(m,Path(d))
    def test_refreeze_refused(self):
        with tempfile.TemporaryDirectory() as d:
            freeze_inputs({'cases':[self.case()]},Path(d))
            with self.assertRaises(ValueError):freeze_inputs({'cases':[self.case()]},Path(d))
    def test_missing_metadata_fails(self):
        for key in ['steps','pe','dtype_by_stage','model_identity','prompt_source','shape','split','seed']:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                c=self.case();del c[key]
                with self.assertRaises(ValueError):freeze_inputs({'cases':[c]},Path(d))
    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.case();c['id']='../escape'
            with self.assertRaises(ValueError):freeze_inputs({'cases':[c]},Path(d))
    def test_changed_metadata_fails(self):
        with tempfile.TemporaryDirectory() as d:
            m=freeze_inputs({'cases':[self.case()]},Path(d));m['cases'][0]['steps']=7
            with self.assertRaises(ValueError):verify_inputs(m,Path(d))
    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as d:
            m=freeze_inputs({'cases':[self.case()]},Path(d));(Path(d)/'changed/prompt.txt').unlink()
            with self.assertRaises(ValueError):verify_inputs(m,Path(d))

    def test_saved_noise_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            import numpy as np
            root=Path(d);noise=root/'noise';raw=np.arange(128*4*4,dtype='<f4').tobytes();noise.write_bytes(raw)
            c=self.case();del c['seed'];c['noise_file']=str(noise)
            m=freeze_inputs({'cases':[c]},root/'frozen');verify_inputs(m,root/'frozen')
            self.assertEqual((root/'frozen/changed/initial.f32').read_bytes(),raw)
    def test_corrupt_image_fails_verification(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);im=root/'input.png';Image.new('RGB',(32,24),'red').save(im)
            c=self.case();c.update(mode='img2img',input_image_file=str(im),strength=.5,resize_policy={'mode':'stretch'},image_source='self-created')
            m=freeze_inputs({'cases':[c]},root/'frozen');verify_inputs(m,root/'frozen')
            (root/'frozen/changed/decoded.rgb').write_bytes(b'bad')
            with self.assertRaises(ValueError):verify_inputs(m,root/'frozen')
    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=freeze_inputs({'cases':[self.case()]},root/'frozen')
            p=root/'frozen/changed/prompt.txt';raw=p.read_bytes();outside=root/'outside';outside.write_bytes(raw);p.unlink();p.symlink_to(outside)
            with self.assertRaises(ValueError):verify_inputs(m,root/'frozen')
    def test_frozen_summary_has_required_coverage(self):
        summary=json.loads((Path(__file__).parent/'fixtures/port-corpus.json').read_text())
        self.assertEqual(summary['counts'],dict(formal=72,development=8,performance=6,img2img=15))
        self.assertEqual(summary['boundary_count'],21)
        self.assertEqual(summary['formal_prompt_count'],24)
        self.assertFalse(summary['formal_results_revealed'])
        long=next(p for p in summary['prompts'] if p['name']=='long-interior')
        self.assertTrue(1025<=long['token_count']<=2048)
        cases=[c for c in summary['case_identities'] if c['split']=='formal']
        shapes=[[512,512],[1024,1024],[1376,768]]
        for i in range(24):
            for j,seed in enumerate([42,1234,2026]):
                case=next(c for c in cases if c['id']==f'formal-{i:02d}-{seed}')
                self.assertEqual(case['shape'],shapes[(i+j)%3])

if __name__=='__main__':unittest.main()
