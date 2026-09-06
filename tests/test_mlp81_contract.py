import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_mlp81_contract import BASELINE,NATIVE,baseline_gate,tail_needed
class Contract(unittest.TestCase):
 def test_all_four_old_boundaries_required(self):
  baseline_gate(BASELINE)
  for key in BASELINE:
   for value in [None,'0'*64]:
    changed=dict(BASELINE);changed[key]=value
    with self.assertRaisesRegex(ValueError,'four old'):baseline_gate(changed)
 def test_native_replay_precedes_tail_decision(self):
  with self.assertRaisesRegex(ValueError,'teacher'):tail_needed('a'*64,'a'*64,'0'*64)
 def test_equal_81_skips_only_extra_mlp(self):
  self.assertFalse(tail_needed('a'*64,'a'*64,NATIVE))
  self.assertTrue(tail_needed('a'*64,'b'*64,NATIVE))
  with self.assertRaises(ValueError):tail_needed('', '', NATIVE)
if __name__=='__main__':unittest.main()
