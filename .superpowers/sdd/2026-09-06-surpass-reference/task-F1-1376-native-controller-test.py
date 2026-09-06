import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('controller', Path(__file__).with_name('task-F1-1376-native-controller.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Contract(unittest.TestCase):
    def test_checksum_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'input'; f.write_bytes(b'abc')
            plan = {'files': {str(f): {'size_bytes': 3, 'sha256': m.digest(f)}}}
            m.verify(plan)
            f.write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                m.verify(plan)

    def test_exact_scope_refused(self):
        with self.assertRaisesRegex(ValueError, 'Exact dedicated'):
            m.controls({'unit': 'never-the-current-scope'})

    def test_terminate_covers_detached_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cg = Path(tmp)
            (cg/'cgroup.procs').write_text(f'{os.getpid()}\n123456\n234567\n')
            killed = []
            def kill(pid, sig):
                killed.append(pid)
                if len(killed) == 2:
                    (cg/'cgroup.procs').write_text(str(os.getpid()))
            with patch.object(m.os, 'kill', side_effect=kill):
                m.terminate_scope_children(cg)
            self.assertEqual(killed, [123456,234567])

    def test_actual_small_child_failure_preserved_and_rerun_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); cg=base/'cg'; cg.mkdir()
            (cg/'memory.current').write_text('4096')
            (cg/'memory.events').write_text('max 0\noom 0\noom_kill 0\noom_group_kill 0\n')
            (cg/'cgroup.procs').write_text(str(os.getpid()))
            worker=base/'worker.py'; worker.write_text('raise SystemExit(7)\n')
            plan={'status':'prepared_not_executed','python_invocation':os.path.abspath(sys.executable),
                  'python_prefix':sys.prefix,'validator':str(worker),'candidate':str(base/'candidate'),
                  'runner':str(base/'runner'),'files':{},'unit':'fixture'}
            plan['argv']=[plan['python_invocation'],str(worker),'--model',plan['candidate'],'--runner',plan['runner'],
                          '--output',str(base/'validation'),'--cpu-only','--vae-convolution','direct']
            p=base/'plan.json';p.write_text(json.dumps(plan));sha=m.digest(p)
            with patch.object(m,'controls',return_value=(cg,{})),patch.object(m,'available',return_value=10*1024**3):
                with self.assertRaisesRegex(RuntimeError, 'Native validation failed'):
                    m.run(p,sha)
                result=json.loads((base/'execution/result.json').read_text())
                self.assertEqual(result['exit_code'],7)
                self.assertEqual(result['status'],'failed')
                self.assertTrue((base/'execution/samples.jsonl').exists())
                with self.assertRaises(FileExistsError):
                    m.run(p,sha)
                self.assertEqual(json.loads((base/'execution/result.json').read_text()),result)


if __name__ == '__main__':
    unittest.main()
