import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.release_manifest import validate_manifest, load_manifest


def manifest(url=None, data=b'0123456789abcdef'):
    return {'schema_version':1,'model_revision':'a'*40,'graph_schema':'schema-3',
        'required_capabilities':['native-model-verifier'],
        'conversion':{'source':'test fixture only','script_sha256':'b'*64},
        'license':{'identifier':'test','source':'local fixture','notice':'not a published asset'},
        'files':[{'path':'weights/data.bin','url':url or 'https://example.invalid/'+'a'*40+'/data',
                  'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}


class ManifestTests(unittest.TestCase):
    def test_valid(self): self.assertEqual(validate_manifest(manifest()),manifest())
    def test_paths(self):
        for path in ['../x','/x','a/../x','a//x','a\\x','C:x','a.part','a.part.json','.download.lock','CON','a.','.']:
            with self.subTest(path=path):
                m=manifest();m['files'][0]['path']=path
                with self.assertRaises(ValueError):validate_manifest(m)
    def test_duplicate_and_parent_paths(self):
        for path in ['WEIGHTS/data.bin','weights','weights/data.bin/sub']:
            m=manifest();f=copy.deepcopy(m['files'][0]);f['path']=path;m['files'].append(f)
            with self.assertRaises(ValueError):validate_manifest(m)
    def test_identity_types_urls(self):
        for key,value in [('size',True),('size',-1),('sha256','z'*64),('url','http://example.com/'+'a'*40),('url','https://example.com/main/file')]:
            m=manifest();m['files'][0][key]=value
            with self.assertRaises(ValueError):validate_manifest(m)
        m=manifest();m['model_revision']='main'
        with self.assertRaises(ValueError):validate_manifest(m)
    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'m.json';p.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):load_manifest(p)
