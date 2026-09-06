import unittest,sys,hashlib,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_official_block_hooks import endpoint,EXPECTED,SHAPES,verify,fixed_execution_file
class OracleContractTest(unittest.TestCase):
 def test_endpoint_requires_existing_whole_oracle(self):
  endpoint(EXPECTED)
  for digest in ['0'*64,EXPECTED[:-1]+'0']:
   with self.assertRaisesRegex(ValueError,'Invalid instrumented'):endpoint(digest)
 def test_mutated_frozen_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source';p.write_bytes(b'fixed');bound={str(p):hashlib.sha256(b'fixed').hexdigest()};verify(bound);p.write_bytes(b'edited')
   with self.assertRaisesRegex(ValueError,'Changed bound'):verify(bound)
 def test_live_prepare_logs_excluded(self):
  for name in ['prepare.log','prepare-samples.jsonl','prepare-result.json']:
   self.assertFalse(fixed_execution_file(Path('execution')/name))
  self.assertTrue(fixed_execution_file(Path('execution/runner.py')))
 def test_complete_boundary_denominators(self):
  import math
  self.assertEqual({n:math.prod(s) for n,s in SHAPES.items()},{'75':17039360,'87':51118080,'88':17039360,'out0':17039360})
if __name__=='__main__':unittest.main()
