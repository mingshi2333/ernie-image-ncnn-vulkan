import sys,unittest,tempfile,json
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_q2_scope_guard import check_limits,MEMORY_MAX
class Scope(unittest.TestCase):
 def test_actual_limits_required(self):
  v={'memory.max':str(MEMORY_MAX),'memory.swap.max':'0','cpu.max':'200000 100000','memory.swap.current':'0'}
  check_limits('/app/fixed.scope',v,{0,2},'fixed')
  for key,value in [('memory.max','max'),('memory.max',str(MEMORY_MAX+1)),('memory.swap.max','max'),('cpu.max','max 100000'),('cpu.max','400000 100000'),('memory.swap.current','4096')]:
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):check_limits('/app/fixed.scope',{**v,key:value},{0,2},'fixed')
 def test_session_name_or_affinity_not_substitute_for_cgroup(self):
  v={'memory.max':str(MEMORY_MAX),'memory.swap.max':'0','cpu.max':'200000 100000','memory.swap.current':'0'}
  with self.assertRaisesRegex(ValueError,'scope'):check_limits('/app/unrelated.scope',v,{0,2},'fixed')
  with self.assertRaisesRegex(ValueError,'affinity'):check_limits('/app/fixed.scope',v,{4,6},'fixed')
 def test_scope_failure_prevents_model_spawn(self):
  import diagnose_q2_scope_guard as guard
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);plan=out/'plan.json';plan.write_text(json.dumps({'output':str(out),'scope_unit':'fixed'}))
   with patch.object(guard,'sample',side_effect=ValueError('memory.swap.max must equal zero')),patch.object(guard.subprocess,'Popen') as spawn:
    with self.assertRaisesRegex(ValueError,'memory.swap.max'):guard.run(plan)
    spawn.assert_not_called()
if __name__=='__main__':unittest.main()
