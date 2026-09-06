import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import diagnose_mlp81_scope_launch as launcher
class ScopeLaunch(unittest.TestCase):
 def test_tampered_guard_prevents_scope_creation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'outputs/q2-block15-mlp81-official-v5';(out/'execution').mkdir(parents=True)
   (out/'execution/diagnose_q2_scope_guard.py').write_text('disable swap guard');(out/'plan.json').write_text('{}')
   with patch.object(launcher.subprocess,'run') as run:
    with self.assertRaises(ValueError):launcher.launch(root)
    run.assert_not_called();self.assertFalse((out/'execution/launch-record.json').exists())
if __name__=='__main__':unittest.main()
