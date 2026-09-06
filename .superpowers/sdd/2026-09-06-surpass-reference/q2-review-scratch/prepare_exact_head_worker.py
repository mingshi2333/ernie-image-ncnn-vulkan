import json,hashlib,shutil
from pathlib import Path
root=Path.cwd();out=(root/'outputs/q2-chinese-step0-current-exact-head-stack-v1').resolve();archive=out/'execution'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=root/'outputs/diagnostic-chinese-step0-stages-v1';result=json.loads((old/'result.json').read_text());fixture=json.loads((old/'oracle/fixture/fixture.json').read_text())
assert sha(old/'scripts/diagnose_dit_stages.py')==fixture['source_sha256']=='5440e1cb9c6f6b7b298ff208379d8c1ed576c986c4cf9c37cb52d78c5c2e29bb'
for name,digest in result['source_snapshot'].items():assert sha(old/'scripts'/name)==digest
shutil.copytree(old/'scripts',archive/'oracle-scripts')
reference=root/'outputs/pipeline1024-chinese-s64-fp32-v1/reference';ref=json.loads((reference/'fixture.json').read_text());assert sha(reference/'fixture.json')==fixture['reference_fixture_sha256'];assert ref['prompt'].encode().endswith(b'\n') is False
provenance={'historical_oracle_execution_source_note':'Recorded __file__ digest equals saved script. Launcher executed original __file__ path, not necessarily its copied snapshot; copied Python tree is audited now. Historical imported-source trace is not a complete dynamic import log.',
'prompt':ref['prompt'],'prompt_sha256':hashlib.sha256(ref['prompt'].encode()).hexdigest(),'ids':ref['ids'],'config':ref['config'],
'input_mapping':'in0=head-0; in1..6=head-2..7; in7..9=official original in3..5. head-1 final-head temb intentionally excluded: no finalizer execution.',
'axes':'Official current transpose(0,1) saved as contiguous B=1,T=4160,H=4096. Probe read w4096 h4160; observer plain packing=false emits post-block-i row-major FP32 without channel padding.',
'oracle_fixture_sha256':sha(old/'oracle/fixture/fixture.json'),'reference_fixture_sha256':sha(reference/'fixture.json'),'oracle_source_snapshots':result['source_snapshot']}
(archive/'oracle-provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
worker=(root/'outputs/q2-chinese-step6-current-teacher-v1-execution/worker.py').read_text()
begin=worker.index('root=Path.cwd();');end=worker.index('os.sched_setaffinity')
head="root=Path.cwd(); archive=Path("+repr(str(archive))+"); output=Path("+repr(str(out))+")\n"
head+="assert hashlib.sha256((output/'plan.json').read_bytes()).hexdigest()=="+repr(sha(out/'plan.json'))+"\n"
head+="assert hashlib.sha256((archive/'diagnose_exact_head_stack.py').read_bytes()).hexdigest()=="+repr(sha(archive/'diagnose_exact_head_stack.py'))+"\n"
head+="assert hashlib.sha256((archive/'oracle-provenance.json').read_bytes()).hexdigest()=="+repr(sha(archive/'oracle-provenance.json'))+"\n"
head+="command=[str(root/'.venv/bin/python'),str(archive/'diagnose_exact_head_stack.py'),'--execute',str(output/'plan.json')]\n"
worker=worker[:begin]+head+worker[end:]
assert not (archive/'worker.py').exists();(archive/'worker.py').write_text(worker)
identity={'status':'prepared_not_executed','worker_sha256':sha(archive/'worker.py'),'plan_sha256':sha(out/'plan.json'),'tool_sha256':sha(archive/'diagnose_exact_head_stack.py'),'runner_sha256':sha(archive/'runner.snapshot'),'oracle_provenance_sha256':sha(archive/'oracle-provenance.json'),'cpu_affinity':[0,2],'gpu_run_authorized':False}
(archive/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps(identity,indent=2))
