import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.audit_port_weights import official_inventory, reference_inventory, layer_weights, audit_port_weights

REV='bc68c81e2a1730a394d5fc9fae70713dee940140'
REPO='https://huggingface.co/baidu/ERNIE-Image-Turbo'

class WeightAuditTests(unittest.TestCase):
    def official(self, root, header=None, payload=None, provenance=True):
        header=header if header is not None else {'w':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}
        payload=struct.pack('<ff',1,2) if payload is None else payload
        raw=json.dumps(header).encode();path=root/'tiny.safetensors';path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
        if provenance:(root/'tiny.manifest.json').write_text(json.dumps(dict(repository=REPO,revision=REV,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        return path
    def test_official_requires_provenance_and_pinned_revision(self):
        for mutation in ('missing','missing_hash','wrong_revision','wrong_repository'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as t:
                root=Path(t);self.official(root,provenance=mutation!='missing')
                if mutation!='missing':
                    p=root/'tiny.manifest.json';m=json.loads(p.read_text())
                    if mutation=='missing_hash':del m['sha256']
                    elif mutation=='wrong_revision':m['revision']='0'*40
                    else:m['repository']='other'
                    p.write_text(json.dumps(m))
                with self.assertRaises(ValueError):official_inventory(root)
    def test_official_rejects_invalid_offsets_and_dimensions(self):
        variants=[{'w':{'dtype':'F32','shape':[2],'data_offsets':[0,0]}},
                  {'w':{'dtype':'F32','shape':[-2],'data_offsets':[0,8]}},
                  {'w':{'dtype':'F32','shape':[True],'data_offsets':[0,4]}},
                  {'w':{'dtype':'F32','shape':[3],'data_offsets':[0,12]}},
                  {'a':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'b':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}]
        for header in variants:
            with self.subTest(header=header),tempfile.TemporaryDirectory() as t:
                root=Path(t);self.official(root,header=header)
                with self.assertRaises(ValueError):official_inventory(root)
    def test_scalar_valid_and_file_bytes_verified(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=self.official(root,{'counter':{'dtype':'I64','shape':[],'data_offsets':[0,8]}},struct.pack('<q',3))
            rows=official_inventory(root);self.assertEqual(rows[0]['canonical_sha256'],hashlib.sha256(struct.pack('<f',3)).hexdigest())
            raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
            with self.assertRaises(ValueError):official_inventory(root)
    def test_mha_loader_order_non_square_weights_and_raw_bias(self):
        # embed=2, qdim=3, kdim=4, vdim=5 distinguishes every matrix direction.
        p={'0':'2','1':'1','2':'6','3':'4','4':'5'}
        roles=[('q.weight',6),('q.bias',2),('k.weight',8),('k.bias',2),('v.weight',10),('v.bias',2),('out.weight',6),('out.bias',3)]
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);payload=b'';expected={};offset=0
            for i,(role,count) in enumerate(roles):
                values=np.arange(count,dtype='<f4')+i*100
                if role.endswith('weight'):payload+=struct.pack('<I',0);offset+=4
                expected[role]=(offset,hashlib.sha256(values.tobytes()).hexdigest());payload+=values.tobytes();offset+=4*count
            (root/'a.ncnn.param').write_text('7767517\n1 1\nMultiHeadAttention attention 1 1 in out '+' '.join(k+'='+v for k,v in p.items())+'\n')
            (root/'a.ncnn.bin').write_bytes(payload)
            rows,gaps=reference_inventory(root);self.assertEqual(gaps,[]);self.assertEqual([r['role'] for r in rows],[r[0] for r in roles])
            for r in rows:self.assertEqual((r['offset'],r['canonical_sha256']),expected[r['role']])
            self.assertEqual(rows[0]['logical_shape'],[2,3]);self.assertEqual(rows[-2]['logical_shape'],[3,2])
    def test_mha_unsupported_int8_and_invalid_dimensions_fail_closed(self):
        for p in ({'0':'2','2':'6','18':'1'},{'0':'0','2':'6'},{'0':'2','2':'5'}):
            with self.subTest(p=p),self.assertRaises(ValueError):layer_weights('MultiHeadAttention',p)
    def test_mha_missing_suffix_is_explicit_gap(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'a.ncnn.param').write_text('7767517\n1 1\nMultiHeadAttention attn 1 1 in out 0=2 2=4\n')
            (root/'a.ncnn.bin').write_bytes(struct.pack('<Iffff',0,1,2,3,4))
            rows,gaps=reference_inventory(root);self.assertEqual(len(rows),1);self.assertTrue(gaps)
    def test_named_projection_mapping_requires_instance_shape_and_value(self):
        base=dict(file='decoder.ncnn.bin',layer='attention_66',kind='MultiHeadAttention',role='q.weight',logical_shape=[2,3],canonical_sha256='a'*64,logical_target='decoder.mid_block.attentions.0.to_q.weight')
        official=dict(file='vae-decoder.safetensors',name=base['logical_target'],shape=[2,3],canonical_sha256='a'*64)
        for variation,status in [('valid','logical_projection_value_match'),('missing','official_component_missing'),('shape','logical_shape_mismatch'),('value','logical_value_mismatch'),('instance','unverified_graph_instance')]:
            with self.subTest(variation=variation),tempfile.TemporaryDirectory() as t:
                row=base.copy();target=official.copy()
                if variation=='shape':target['shape']=[3,2]
                if variation=='value':target['canonical_sha256']='b'*64
                if variation=='instance':del row['logical_target']
                with patch('tools.audit_port_weights.official_inventory',return_value=[] if variation=='missing' else [target]),patch('tools.audit_port_weights.reference_inventory',return_value=([row],[])):
                    result=audit_port_weights(Path(t),Path(t),Path(t)/'result')
                self.assertEqual(result['logical_projection_mappings'][0]['logical_mapping_status'],status)
                self.assertEqual(bool(result['gaps']),variation!='valid')
                self.assertEqual(result['status'],'unproven');self.assertFalse(result['allowed_to_close_S'])
    def test_mha_fp16_padding_before_raw_fp32_bias(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);payload=b''
            # Odd FP16 weight count requires two padding bytes before raw FP32 bias.
            for _ in range(4):payload+=struct.pack('<I',0x01306b47)+struct.pack('<e',2)+b'\0\0'+struct.pack('<f',3)
            (root/'a.ncnn.param').write_text('7767517\n1 1\nMultiHeadAttention attn 1 1 in out 0=1 2=1\n')
            (root/'a.ncnn.bin').write_bytes(payload)
            rows,gaps=reference_inventory(root);self.assertFalse(gaps);self.assertEqual(len(rows),8)
            self.assertEqual([r['offset'] for r in rows],[4,8,16,20,28,32,40,44])
            self.assertEqual(rows[0]['canonical_sha256'],hashlib.sha256(struct.pack('<f',2)).hexdigest())
            self.assertEqual(rows[1]['canonical_sha256'],hashlib.sha256(struct.pack('<f',3)).hexdigest())
if __name__=='__main__':unittest.main()
