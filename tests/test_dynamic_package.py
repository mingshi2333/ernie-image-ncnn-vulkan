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

class SharedRuntimePackageTests(unittest.TestCase):
    def setUp(self):
        self.fixture=DynamicPackageTests();self.fixture.setUp()
        self.output=self.fixture.output
        source=sha256(self.fixture.source/'manifest.json')
        self.contract=dynamic.shared_contract()
        self.contract['source_manifests']={source:self.fixture.m['config']}
        self.source_digest=source
        self.mock=patch.object(dynamic,'shared_contract',return_value=self.contract);self.mock.start()
        old=json.loads((self.output/'contract.json').read_text())
        self.manifest={k:self.contract[k] for k in ['schema_version','format','math','encoder','generation_quality_status']}
        self.manifest.update(instances=old['instances'],objects=old['objects'])
        (self.output/'contract.json').unlink();self.save()
    def tearDown(self):self.mock.stop();self.fixture.tearDown()
    def save(self):(self.output/'manifest.json').write_text(json.dumps(self.manifest))
    def test_runtime_resolves_complete_shared_inventory(self):
        value=dynamic.verify_shared_package(self.output)
        self.assertEqual(value['schema_version'],3)
        self.assertEqual(len(value['instances'][0]['runtime_bindings']),136)
        self.assertEqual(len(value['objects']),3)
    def test_tail_layer_missing(self):
        self.manifest['instances'][0]['runtime_bindings'].pop('dit/block-35/block.ncnn.bin');self.save()
        with self.assertRaises(ValueError):dynamic.verify_shared_package(self.output)
    def test_unreviewed_source_and_shape(self):
        self.manifest['instances'][0]['source_manifest_sha256']='0'*64;self.save()
        with self.assertRaises(ValueError):dynamic.verify_shared_package(self.output)
    def test_size_and_graph_corruption_rejected(self):
        digest=self.fixture.m['files']['vae/head.ncnn.param'];self.manifest['objects'][digest]=0;self.save()
        with self.assertRaises(ValueError):dynamic.verify_shared_package(self.output)
    def test_bn_identity_or_encoder_claim_rejected(self):
        self.manifest['math']=dict(self.manifest['math'],decoder_inverse_bn_eps=1e-4);self.save()
        with self.assertRaises(ValueError):dynamic.verify_shared_package(self.output)
    def test_shared_builder_emits_native_manifest_only(self):
        target=self.fixture.base/'runtime-new'
        value=dynamic.build_shared_package([self.fixture.source],target)
        self.assertEqual(value['schema_version'],3);self.assertFalse((target/'contract.json').exists())
    def test_reviewed_encoder_manifest_is_bound_to_source_and_cas(self):
        evidence=self.fixture.base/'encoder';evidence.mkdir();payload=b'small reviewed encoder bytes'
        (evidence/'head.ncnn.param').write_bytes(payload);(evidence/'head.ncnn.bin').write_bytes(payload)
        fixture={'width':64,'height':64,'official_revision':'rev','source_manifests':{'encoder':'em','quant':'qm','bn':'bm'},'posterior':'mode','packing':'pack','encoder_bn':{'eps':1e-4,'affine':False},'decoder_inverse_bn_eps':1e-5}
        (evidence/'fixture.json').write_text(json.dumps(fixture))
        conversion={'method':'reviewed_encoder_spatial_reshape_specialization','template_param_sha256':'75d493995616b451e51ddecc0dc352a3f200557baab3f98d742cc374ed6d0977','template_bin_sha256':sha256(evidence/'head.ncnn.bin'),'reference_fixture_sha256':sha256(evidence/'fixture.json'),'changes':[{},{}]}
        (evidence/'conversion.json').write_text(json.dumps(conversion))
        digest=sha256(evidence/'head.ncnn.param')
        self.contract['reviewed_encoders']={self.source_digest:{'status':'available','width':64,'height':64,'posterior':'mode','packing':'pack','encoder_bn_eps':1e-4,'encoder_bn_affine':False,'decoder_inverse_bn_eps':1e-5,'files':{
            'vae/encoder.ncnn.param':{'sha256':digest,'size':len(payload)},'vae/encoder.ncnn.bin':{'sha256':digest,'size':len(payload)},
            'vae/bn-mean.f32':{'sha256':self.fixture.m['files']['vae/bn-mean.f32'],'size':len(payload)},'vae/bn-variance.f32':{'sha256':self.fixture.m['files']['vae/bn-variance.f32'],'size':len(payload)}},
            'evidence':{'official_fixture_sha256':sha256(evidence/'fixture.json'),'conversion_sha256':sha256(evidence/'conversion.json'),'official_revision':'rev','official_encoder_manifest_sha256':'em','official_quant_manifest_sha256':'qm','official_bn_manifest_sha256':'bm'}}}
        target=self.fixture.base/'runtime-encoder'
        value=dynamic.build_shared_package([self.fixture.source],target,evidence)
        self.assertEqual(value['encoder']['status'],'available');self.assertEqual(value['encoder']['width'],64)
        self.assertTrue((target/'objects'/digest).is_file())
        value['encoder']['width']=32;(target/'manifest.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Unreviewed schema-3 encoder'):dynamic.verify_shared_package(target)
