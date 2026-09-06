import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_mlp81_analysis import split_vectors,decomposition
class Decomposition(unittest.TestCase):
 def test_cancellation_is_not_independent_percentage(self):
  o=np.zeros((2,3),'f4');m=np.ones((2,3),'f4')*4;n=np.ones((2,3),'f4')
  d,u,e=split_vectors(n,o,m);np.testing.assert_array_equal(d,u+e)
  value=decomposition(n,o,m)
  self.assertEqual(value['elements'],6);self.assertEqual(value['propagation_local_cosine'],-1.)
  self.assertLess(value['total']['l2'],value['matched_implementation']['l2'])
 def test_reject_broadcast_or_wrong_dtype(self):
  n=np.ones((2,3),'f4')
  with self.assertRaisesRegex(ValueError,'equal shapes'):split_vectors(n,n[:1],n)
  with self.assertRaisesRegex(ValueError,'FP32'):split_vectors(n,n,n.astype('f8'))
if __name__=='__main__':unittest.main()
