import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_splitk_screen import metrics
class ScreenMetricTest(unittest.TestCase):
 def test_preserves_fp64_oracle_difference(self):
  a=np.array([2**24],dtype='f4');b=np.array([2**24+1],dtype='f8');self.assertEqual(metrics(a,b),{'l2':1.,'max':1.})
 def test_complete_output_not_first_element(self):
  a=np.zeros((5,4096),dtype='f4');b=np.zeros((5,4096),dtype='f8');a[-1,-1]=3;self.assertEqual(metrics(a,b),{'l2':3.,'max':3.})
if __name__=='__main__':unittest.main()
