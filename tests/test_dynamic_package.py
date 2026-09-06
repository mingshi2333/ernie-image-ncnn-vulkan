"""Small object-store corruption tests. Synthetic roots replace the trust registry
only inside mocks; production always pins real static manifest/graph hashes.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from tools import package_dynamic_model as dynamic
from tools.package_model import runtime_files,sha256

class DynamicPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.source=self.base/'source';self.source.mkdir()
        cfg=dict(packed_width=4,packed_height=4,text_bucket=32,dit_text_tokens=64,text_layers=25,dit_layers=36)
        for name in runtime_files():
            p=self.source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'small shared fixture weight')
        (self.source/'model.cfg').write_text(''.join(f'{k} {v}\n' for k,v in cfg.items()))
        self.m=dict(schema_version=2,portable=True,config=cfg,files={n:sha256(self.source/n) for n in runtime_files()},file_sizes={n:(self.source/n).stat().st_size for n in runtime_files()})
        (self.source/'manifest.json').write_text(json.dumps(self.m));digest=sha256(self.source/'manifest.json')
        self.patches=[patch.object(dynamic,'PINNED_MANIFESTS',{digest:'synthetic fixture'}),patch.object(dynamic,'audit_package'),patch.object(dynamic,'verify_package',return_value=(self.m,self.m['files'])),patch.object(dynamic,'verify_graph')]
        for p in self.patches:p.start()
        self.output=self.base/'candidate';dynamic.build_candidate([self.source],self.output)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def mutate(self,fn):
        p=self.output/'contract.json';c=json.loads(p.read_text());fn(c);p.write_text(json.dumps(c))
    def rejected(self):
        with self.assertRaises((ValueError,OSError)):dynamic.verify_candidate(self.output)
    def test_shared_store_relocation_and_offline_policy(self):
        c=dynamic.verify_candidate(self.output)
        self.assertEqual(len(c['objects']),3);self.assertFalse(c['policy']['runtime_supported']);self.assertFalse((self.output/'manifest.json').exists())
        moved=self.base/'moved';shutil.move(self.output,moved);self.assertEqual(dynamic.verify_candidate(moved),c)
    def test_final_layer_missing(self):
        self.mutate(lambda c:c['instances'][0]['runtime_bindings'].pop('dit/block-35/block.ncnn.bin'));self.rejected()
    def test_same_size_weight_corruption(self):
        path=self.output/'objects'/self.m['files']['dit/block-35/block.ncnn.bin'];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data);self.rejected()
    def test_object_size_lie(self):
        self.mutate(lambda c:c['objects'].__setitem__(self.m['files']['dit/block-35/block.ncnn.bin'],0));self.rejected()
    def test_unsafe_object_path(self):
        self.mutate(lambda c:c['instances'][0].__setitem__('source_manifest_sha256','../escape'));self.rejected()
    def test_symlink_object_rejected(self):
        path=self.output/'objects'/self.m['files']['dit/block-35/block.ncnn.bin'];path.unlink();path.symlink_to(self.source/'dit/block-35/block.ncnn.bin');self.rejected()
    def test_runtime_claim_and_numeric_policy_rejected(self):
        for value in (True,0):
            with self.subTest(value=value):
                self.mutate(lambda c:c['policy'].__setitem__('runtime_supported',value));self.rejected()
    def test_unreviewed_shape_metadata(self):
        self.mutate(lambda c:c['instances'][0]['config'].__setitem__('packed_height',8));self.rejected()
    def test_manifest_masquerade_rejected(self):
        (self.output/'manifest.json').write_text('{}');self.rejected()
    def test_unlisted_object_rejected(self):
        (self.output/'objects'/'extra').write_bytes(b'x');self.rejected()
    def test_duplicate_json_rejected(self):
        p=self.output/'contract.json';p.write_text('{"format":"x","format":"y"}');self.rejected()
    def test_independent_graph_validator_is_required(self):
        with patch.object(dynamic,'verify_graph',side_effect=ValueError('Unknown complete graph hash')):self.rejected()
    def test_unknown_source_identity(self):
        self.mutate(lambda c:c['instances'][0].__setitem__('source_manifest_sha256','0'*64));self.rejected()
    def test_cannot_overwrite_existing_candidate(self):
        with self.assertRaises(ValueError):dynamic.build_candidate([self.source],self.output)
if __name__=='__main__':unittest.main()
