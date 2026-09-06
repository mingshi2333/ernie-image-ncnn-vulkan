import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('stack',Path(__file__).resolve().parents[1]/'tools/diagnose_exact_head_stack.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class StackContractTest(unittest.TestCase):
 def fixture(self):
  exact={'tokens':4160,'inputs':{},'expected':{'sha256':'35'}};stages={'tokens':4160,'step':0,'reference_fixture_sha256':m.REFERENCE,'stages':{},'inputs':{}}
  for i in range(10):
   shape=m.SHAPE if i==0 else [1,1,4096] if i<=6 else [1,4160,128] if i<=8 else [4160,4160]
   item={'shape':shape,'dtype':'float32_le','sha256':f'in{i}'};exact['inputs'][f'in{i}']=item
   (stages['stages'] if i<=6 else stages['inputs'])[f'head-{0 if i==0 else i+1}' if i<=6 else f'in{i-4}']=copy.deepcopy(item)
  for i in range(36):stages['stages'][f'block-{i}']={'shape':m.SHAPE,'dtype':'float32_le','sha256':str(i)}
  return exact,stages
 def test_full(self):m.contract(*self.fixture())
 def test_missing_layer(self):
  a,b=self.fixture();del b['stages']['block-19']
  with self.assertRaises(ValueError):m.contract(a,b)
 def test_wrong_modulation(self):
  a,b=self.fixture();a['inputs']['in6']['sha256']='different'
  with self.assertRaises(ValueError):m.contract(a,b)
 def test_wrong_axes(self):
  a,b=self.fixture();a['inputs']['in0']['shape']=[1,4096,4160]
  with self.assertRaises(ValueError):m.contract(a,b)
 def test_metric(self):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a';b=Path(d)/'b';np.array([1,4],dtype='<f4').tofile(a);np.array([1,2],dtype='<f4').tofile(b)
   r=m.compare(a,b);self.assertEqual(r['error_l2'],2);self.assertEqual(r['max_abs_error'],2)
   np.array([float('nan'),2],dtype='<f4').tofile(a)
   with self.assertRaises(ValueError):m.compare(a,b)
if __name__=='__main__':unittest.main()
