import sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_up84_official import EXPECTED,endpoint
import diagnose_up84_sequence as sequence
class Up84(unittest.TestCase):
 def test_all_matched_endpoints_required(self):
  endpoint(EXPECTED)
  for key in EXPECTED:
   altered=dict(EXPECTED);altered[key]='0'*64
   with self.assertRaises(ValueError):endpoint(altered)
 def test_native_failure_stops_before_official(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'plan.json';p.write_text(json.dumps({'bound':{},'output':str(root),'native_command':['dummy']}))
   with patch.object(sequence.subprocess,'run') as call,patch.object(sequence,'native_gate',side_effect=ValueError('changed81')):
    with self.assertRaises(ValueError):sequence.run(p)
    self.assertEqual(call.call_count,1)
 def test_truncated_native_denominator(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);(out/'native').mkdir();(out/'native/81.f32').write_bytes(b'\0'*16)
   with self.assertRaisesRegex(ValueError,'denominator'):sequence.native_gate(out)
if __name__=='__main__':unittest.main()
