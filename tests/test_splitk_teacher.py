import unittest,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_splitk_teacher import chunks,rewrite_graph,complete
class TeacherContractTest(unittest.TestCase):
 def test_full_and_tail_coverage(self):
  for rows in [4160,4163]:
   plan=chunks(rows);self.assertEqual([r for s,n in plan for r in range(s,s+n)],list(range(rows)));self.assertLessEqual(max(n for s,n in plan),16)
 def test_reject_unknown_graph_even_with_matching_line(self):
  with self.assertRaisesRegex(ValueError,'Unreviewed graph'):rewrite_graph(b'Gemm gemm_6 1 1 87 88 10=-1 2=0 3=1 4=0 5=1 6=1 7=4160 8=4096 9=12288\n')
 def test_reject_missing_last_row(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'tensor';p.write_bytes(bytes(4159*4))
   with self.assertRaisesRegex(ValueError,'Incomplete'):complete(p,1)
 def test_reject_invalid_plan(self):
  for args in [(True,16),(4160,0),(-1,16)]:
   with self.assertRaises(ValueError):chunks(*args)
if __name__=='__main__':unittest.main()
