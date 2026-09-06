import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import diagnose_mlp81_launch as launcher
class Launch(unittest.TestCase):
 def test_tampered_worker_never_launches(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'outputs/q2-block15-mlp81-v2';(out/'execution').mkdir(parents=True)
   (out/'execution/worker.py').write_text('changed resource guard');(out/'plan.json').write_text('{}')
   with patch.object(launcher.subprocess,'run') as run:
    with self.assertRaisesRegex(ValueError,'worker.py'):launcher.launch(root)
    run.assert_not_called()
 def test_tampered_plan_never_launches(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'outputs/q2-block15-mlp81-v2';(out/'execution').mkdir(parents=True)
   worker=out/'execution/worker.py';worker.write_bytes(b'validated test worker');(out/'plan.json').write_text('changed plan')
   import hashlib
   with patch.object(launcher,'WORKER_SHA',hashlib.sha256(worker.read_bytes()).hexdigest()),patch.object(launcher.subprocess,'run') as run:
    with self.assertRaisesRegex(ValueError,'plan.json'):launcher.launch(root)
    run.assert_not_called()
if __name__=='__main__':unittest.main()
