import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_single_block_teacher import validate_inputs,command
class TeacherTest(unittest.TestCase):
 def data(self):
  b={f'in{i}':{'shape':[1,2,3],'dtype':'float32_le','sha256':str(i)} for i in range(10)};n={k:dict(v) for k,v in b.items()};n['in0']['sha256']='official14';return b,n
 def test_exact(self):b,n=self.data();validate_inputs(b,n,'official14')
 def test_conditioning_change(self):
  b,n=self.data();n['in3']['sha256']='oops'
  with self.assertRaises(ValueError):validate_inputs(b,n,'official14')
 def test_missing(self):
  b,n=self.data();del n['in9']
  with self.assertRaises(ValueError):validate_inputs(b,n,'official14')
 def test_command(self):
  c=command(Path('/o'),Path('/p'));self.assertEqual(c.count('--model'),1);self.assertEqual(c[-1],'/p/dit/block-15');self.assertNotIn('--input-head',c)
if __name__=='__main__':unittest.main()
