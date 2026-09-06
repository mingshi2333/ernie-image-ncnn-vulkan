#!/usr/bin/env python3
"""Four separately scheduled stages using the already reviewed CPU scope guard."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--phase',choices=['official-input','official-output','native-input','native-output'],required=True)
p.add_argument('--plan-sha256',required=True);p.add_argument('--reference-sha256');a=p.parse_args()
base=Path(__file__).resolve().parent
# Reuse unchanged, previously executed resource primitives; authenticate before import.
import hashlib

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

if sha(base/'plan.json')!=a.plan_sha256:raise ValueError('Plan changed')
plan=json.loads((base/'plan.json').read_text())
if sha(plan['guard'])!='98770dd6dacfafd9b7296a4a8b790e2025a123e4a04213d746bc5c6aca8b1b1a':raise ValueError('Guard changed')
spec=importlib.util.spec_from_file_location('guard',plan['guard']);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
if os.path.abspath(sys.executable)!=plan['python_invocation'] or sys.prefix!=plan['python_prefix']:raise ValueError('Venv changed')
cg,controls=guard.controls(plan)
out=base/a.phase;out.mkdir()
result={'status':'failed','phase':a.phase,'controls':controls,'scope_cpu_budget':2,'native_ncnn_threads':4,
        'official_torch_threads':2,'plan_sha256':a.plan_sha256,'error':None,'exit_code':None,
        'host_min_observed':guard.available(),'peak_cgroup_memory_current':0}
process=None;start=time.monotonic_ns();result['started_monotonic_ns']=start
try:
 guard.verify(plan)
 if guard.available()<3*1024**3:raise RuntimeError('Host floor before stage')
 kind=a.phase.split('-')[1]
 if a.phase.startswith('native'):
  ref=base/('official-'+kind)/'reference'/kind
  if not a.reference_sha256 or sha(ref/'fixture.json')!=a.reference_sha256:raise ValueError('Reviewed actual reference SHA required')
  fixture=json.loads((ref/'fixture.json').read_text())
  if (fixture['component']!=kind or (fixture['height'],fixture['width'],fixture['text_tokens'])!=(48,86,64)
      or fixture['weights']!=plan['official_weights'] or fixture['official_revision']!=plan['official_revision']
      or fixture['reference_source_sha256']!=plan['transformer_source_sha256']):raise ValueError('Reference source or fixed shape differs')
  expected_count=8 if kind=='input' else 1
  if set(fixture['expected'])!={f'out{i}' for i in range(expected_count)}:raise ValueError('Reference output denominator differs')
  if fixture['gates']['fp32']!={'atol':.0002,'rtol':.0002,'nrmse':.00002}:raise ValueError('Original FP32 gates required')
  candidate=out/'candidate';candidate.mkdir()
  shutil.copy2(plan['graphs'][kind],candidate/'head.ncnn.param')
  (candidate/'head.ncnn.bin').symlink_to(plan['bins'][kind])
  shutil.copy2(ref/'fixture.json',candidate/'fixture.json')
  for entry in [*fixture['inputs'].values(),*fixture['expected'].values()]:
   if Path(entry['file']).name!=entry['file'] or sha(ref/entry['file'])!=entry['sha256']:raise ValueError('Reference tensor identity changed')
   (candidate/entry['file']).symlink_to(ref/entry['file'])
  manifest={'schema_version':1,'component':kind,'source_sha256':sha(__file__),'weights':fixture['weights'],
            'ncnn_revision':plan['ncnn_revision'],'files':{q.name:sha(q) for q in candidate.iterdir() if q.is_file()}}
  (candidate/'model.json').write_text(json.dumps(manifest,indent=2)+'\n')
  result['reference_sha256']=a.reference_sha256
  argv=[plan['python_invocation'],str(base/'source/tools/validate_dit_heads.py'),'--model',str(candidate),
        '--runner',plan['runner'],'--output',str(out/'validation'),'--cpu-only','--vae-convolution','direct']
 else:
  argv=[plan['python_invocation'],str(base/'source/tools/export_dit_heads.py'),'--output',str(out/'reference'),
        '--height','48','--width','86','--text-tokens','64','--only',kind,'--reference-only','--threads','2',
        '--official-root',plan['official_root']]
 result['argv']=argv
 with (out/'worker.log').open('xb') as log,(out/'samples.jsonl').open('x') as samples:
  process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);result['worker_pid']=process.pid
  while process.poll() is None:
   now=time.monotonic_ns();host=guard.available();memory=int((cg/'memory.current').read_text())
   result['host_min_observed']=min(result['host_min_observed'],host);result['peak_cgroup_memory_current']=max(result['peak_cgroup_memory_current'],memory)
   samples.write(json.dumps({'monotonic_ns':now,'host_available':host,'memory_current':memory})+'\n');samples.flush()
   if host<3*1024**3 or now-start>1800*10**9:raise RuntimeError('Host floor or outer timeout')
   time.sleep(.05)
  result['exit_code']=process.wait()
 if result['exit_code']:raise RuntimeError('Original worker failed')
 guard.verify(plan)
 events=dict(x.split() for x in (cg/'memory.events').read_text().splitlines())
 if any(int(events[k]) for k in ('oom','oom_kill','oom_group_kill')):raise RuntimeError('Scope OOM')
 result['outputs']={str(q.relative_to(out)):{'sha256':sha(q),'size_bytes':q.stat().st_size}
                    for q in out.rglob('*') if q.is_file() and q.name not in ('samples.jsonl','worker.log')}
 result['status']='completed_pending_independent_reference_review' if a.phase.startswith('official') else 'passed_fixed_head_pending_independent_review'
except BaseException as error:
 result['error']=str(error);guard.terminate_scope_children(cg)
 if process is not None:result['exit_code']=process.wait()
 raise
finally:
 result['finished_monotonic_ns']=time.monotonic_ns();result['wall_seconds']=(result['finished_monotonic_ns']-start)/1e9
 result['memory_events']=(cg/'memory.events').read_text();result['remaining_scope_pids']=(cg/'cgroup.procs').read_text().split()
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
