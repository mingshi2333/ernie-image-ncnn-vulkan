import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.port_adapters import PortAdapter, Unavailable, canonical_latent, canonical_sha256, verify_pair, calibration_grid
from tools.audit_port_weights import normalized_hash, reference_inventory

class AdapterTests(unittest.TestCase):
    def test_hwc_to_chw_preserves_channel_identity(self):
        chw=np.arange(128*3*2,dtype=np.float32).reshape(128,3,2)
        raw=chw.transpose(1,2,0).copy().ravel()
        np.testing.assert_array_equal(canonical_latent(raw,'HWC',(128,3,2)),chw)
        self.assertNotEqual(canonical_sha256(raw,'CHW',(128,3,2)),canonical_sha256(raw,'HWC',(128,3,2)))

    def test_chw_identity_and_invalid_layout(self):
        x=np.arange(24,dtype=np.float32)
        np.testing.assert_array_equal(canonical_latent(x,'CHW',(2,3,4)).ravel(),x)
        for layout in ('WHC','unknown'):
            with self.assertRaises(ValueError):canonical_latent(x,layout,(2,3,4))
        with self.assertRaises(ValueError):canonical_latent(x,'CHW',(2,3,3))
        with self.assertRaises(ValueError):canonical_latent(np.array([np.nan]),'CHW',(1,1,1))

    def test_pair_fails_closed(self):
        base=dict(prompt_sha256='p',noise_sha256='n',initial_canonical_sha256='n',shape=[64,64],steps=8,cfg=1,
                  pe={'enabled':False},dtype_by_stage={s:'fp32' for s in ('text','dit','vae','scheduler')},
                  device_by_stage={s:'cpu' for s in ('text','dit','vae','scheduler')},
                  weights_canonical_sha256='w',weight_identity_status='proven')
        verify_pair(base,copy.deepcopy(base))
        for key in base:
            other=copy.deepcopy(base);other[key]=None
            with self.assertRaises(ValueError,msg=key):verify_pair(base,other)
        other=copy.deepcopy(base);other['initial_canonical_sha256']='wrong'
        with self.assertRaises(ValueError):verify_pair(other,other)

    def test_reference_command_preserves_prompt_and_disables_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prompt=b'hello\r\n';(root/'prompt.txt').write_bytes(prompt)
            noise=np.arange(128*4*4,dtype='<f4').tobytes();(root/'initial.f32').write_bytes(noise)
            case=dict(shape=[64,64],steps=8,cfg=1,noise_path='initial.f32',noise_layout='CHW',noise_dtype='<f4',
                      noise_sha256=hashlib.sha256(noise).hexdigest(),prompt_path='prompt.txt',prompt_sha256=hashlib.sha256(prompt).hexdigest(),
                      dtype_by_stage={'dit':'fp32'},pe={'enabled':False})
            adapter=PortAdapter('reference',Path('/binary'),Path('/model'),root)
            command=adapter.command(case,root/'out',True)
            self.assertEqual(command[command.index('--prompt')+1],prompt.decode())
            for flag in ('--fp32-storage','--no-pe','--input-latents','--dump-initial-latents'):self.assertIn(flag,command)
            adapter.config['precision']='fp16'
            with self.assertRaises(Unavailable):adapter.command(case,root/'out',False)
            adapter.kind='candidate';adapter.config={'threads':8}
            with self.assertRaises(Unavailable):adapter.command(case,root/'out',False)

    def test_candidate_greedy_maps_to_accepted_cli_without_changing_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prompt=b'apple';(root/'prompt.txt').write_bytes(prompt)
            noise=np.zeros(128*4*4,dtype='<f4').tobytes();(root/'initial.f32').write_bytes(noise)
            case=dict(shape=[64,64],steps=8,cfg=1,noise_path='initial.f32',noise_layout='CHW',noise_dtype='<f4',
                      noise_sha256=hashlib.sha256(noise).hexdigest(),prompt_path='prompt.txt',prompt_sha256=hashlib.sha256(prompt).hexdigest(),
                      dtype_by_stage={'dit':'fp32'},pe={'enabled':True,'sampling':'greedy','temperature':0,
                      'max_new_tokens':2048,'top_p':1})
            original=copy.deepcopy(case)
            candidate=PortAdapter('candidate',Path('/binary'),Path('/model'),root,{'pe_model':'/pe'})
            peer=PortAdapter('reference',Path('/binary'),Path('/model'),root)
            command=candidate.command(case,root/'out',False)
            self.assertIn('--pe-greedy',command)
            self.assertEqual(command[command.index('--pe-temperature')+1],'1.0')
            peer_command=peer.command(case,root/'other',False)
            self.assertEqual(float(peer_command[peer_command.index('--pe-temperature')+1]),0)
            self.assertEqual(case,original)
            modes=candidate.modes(case)
            self.assertEqual(set(modes['dtype_by_stage']),{'text','dit','vae','scheduler'})
            self.assertEqual(set(modes['device_by_stage']),{'text','dit','vae','scheduler'})
            self.assertFalse(modes['pe_temperature_used_for_sampling'])

    def test_calibration_capabilities_are_specific_to_each_port(self):
        rows=calibration_grid()
        candidate=[r for r in rows if r['port']=='candidate' and r['threads']==4]
        self.assertEqual(len(candidate),3)
        self.assertTrue(all(r['status']=='pending' for r in candidate))
        self.assertTrue(all(r['status']=='unavailable' for r in rows if r['port']=='candidate' and r['threads']==8))
        self.assertTrue(all(r['status']=='unavailable' for r in rows if r['port']=='reference' and r['precision']=='fp16'))

    def test_stream_normalization_and_transpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'weights';x=np.arange(12,dtype='<f4').reshape(3,4);p.write_bytes(x.tobytes())
            self.assertEqual(normalized_hash(p,0,12,'F32',(3,4)),hashlib.sha256(x.T.copy().tobytes()).hexdigest())
            bf=(x.ravel().view('<u4')>>16).astype('<u2');p.write_bytes(bf.tobytes())
            self.assertEqual(normalized_hash(p,0,12,'BF16'),hashlib.sha256(x.tobytes()).hexdigest())
            with self.assertRaises(ValueError):normalized_hash(p,0,13,'BF16')

    def test_unknown_weight_layer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a.ncnn.param').write_text('7767517\n1 1\nMystery a 0 1 out\n');(p/'a.ncnn.bin').write_bytes(b'1234')
            rows,gaps=reference_inventory(p);self.assertEqual(rows,[]);self.assertIn('Unsupported layer',gaps[0]['reason'])

    def test_gemm_transpose_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);x=np.arange(6,dtype='<f4').reshape(3,2)
            (p/'a.ncnn.param').write_text('7767517\n1 1\nGemm gemm_0 0 1 out 3=0 5=1 6=1 8=2 9=3 10=-1\n')
            (p/'a.ncnn.bin').write_bytes(b'\0'*4+x.tobytes())
            rows,gaps=reference_inventory(p)
            self.assertEqual(gaps,[])
            self.assertEqual(rows[0]['canonical_sha256'],hashlib.sha256(x.T.copy().tobytes()).hexdigest())
            with (p/'a.ncnn.bin').open('ab') as f:f.write(b'extra')
            _,gaps=reference_inventory(p)
            self.assertIn('Unconsumed bytes',gaps[0]['reason'])

    def test_observations_not_wall_time(self):
        a=PortAdapter('reference',Path('/a'),Path('/m'),Path('/i'))
        result=a.parse_log('DiT loaded: gpu-id=0 vulkan=1 bf16_storage=1 low_vram=1')
        self.assertIsNone(result['reported_seconds']);self.assertEqual(result['reference_runtime']['bf16_storage'],1)

if __name__=='__main__':unittest.main()
