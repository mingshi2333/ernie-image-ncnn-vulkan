import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from vae_reference_scope import verify_files

class RuntimeIdentityTests(unittest.TestCase):
    def test_runtime_file_hash_detects_same_size_change(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'runtime.py';p.write_bytes(b'abcd')
            rows={str(p):{'size':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}}
            verify_files(rows);p.write_bytes(b'abce')
            with self.assertRaisesRegex(ValueError,'Frozen runtime file differs'):verify_files(rows)

    def test_virtual_environment_invocation_must_not_be_realpath(self):
        import os
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'venv/bin';p.mkdir(parents=True);entry=p/'python';entry.symlink_to(sys.executable)
            self.assertNotEqual(os.path.abspath(entry),str(entry.resolve()))
            self.assertEqual(Path(os.path.abspath(entry)).parent.name,'bin')

if __name__=='__main__':unittest.main()
