import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import diagnose_boundary_pair as m
class BoundaryTest(unittest.TestCase):
 def test_wrong_final_stops_before_second_side_or_metrics(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);p={'output':str(out),'bound':{},'boundary':'75','runner':'runner','model':'model','sides':{'official':{'fixture':'fixture','expected':'wrong'},'native':{'fixture':'fixture','expected':'wrong'}}};f=out/'plan.json';f.write_text(json.dumps(p))
   def fake(command,check):
    dest=Path(command[3]);dest.mkdir();(dest/'out0.f32').write_bytes(b'1234')
   with patch.object(m.subprocess,'run',side_effect=fake) as run,patch.object(m,'compare') as compare:
    with self.assertRaisesRegex(ValueError,'Instrumented final output changed'):m.run(f)
    self.assertEqual(run.call_count,1);compare.assert_not_called();self.assertFalse((out/'result.json').exists())
 def test_unreviewed_boundary_stops_before_launch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan.json';p.write_text(json.dumps({'output':d,'bound':{},'boundary':'arbitrary'}))
   with patch.object(m.subprocess,'run') as run:
    with self.assertRaises(ValueError):m.run(p)
    run.assert_not_called()
if __name__=='__main__':unittest.main()
