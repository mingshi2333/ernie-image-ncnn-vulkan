import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_q2_upstream_branch import read,finish
class Branch(unittest.TestCase):
 def test_reject_partial_state(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state';p.write_bytes(b'\0'*16)
   with self.assertRaisesRegex(ValueError,'Incomplete'):read(p)
 def test_nonorthogonal_response_not_variance_percentage(self):
  value=finish({'squares':{'a':4.,'b':9.},'dots':{'ab':(('a','b'),-3.)}})
  self.assertEqual(value['cosines']['ab'],-.5)
  self.assertEqual(value['a'],2.)
if __name__=='__main__':unittest.main()
