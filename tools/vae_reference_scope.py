#!/usr/bin/env python3
"""Capture an import-only runtime, or explicitly run one frozen CPU VAE reference under a parent cgroup."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from check_release import cgroup, execute
from prepare_block import sha256


def capture(output):
    if output.exists():raise ValueError('Use a new runtime snapshot')
    # Import exactly the future reference worker dependencies; no model is loaded.
    import export_vae
    files={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_file():
            path=Path(name).resolve();files[str(path)]={'size':path.stat().st_size,'sha256':sha256(path)}
    mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and parts[5].startswith('/') and Path(parts[5]).is_file():mapped.add(parts[5])
    for name in sorted(mapped):
        path=Path(name);files[name]={'size':path.stat().st_size,'sha256':sha256(path)}
    output.mkdir(parents=True);sources=output/'sources';sources.mkdir()
    for name,row in files.items():
        if name.endswith('.py'):
            dest=sources/(row['sha256']+'.py')
            if not dest.exists():shutil.copy2(name,dest)
    identity={'status':'imported_dependencies_no_model_execution','prefix':sys.prefix,
              'base_prefix':sys.base_prefix,'executable':sys.executable,
              'files':files,'mapped_files':sorted(mapped),
              'packages':{name:{'version':importlib.metadata.version(name),
                          'direct_url':importlib.metadata.distribution(name).read_text('direct_url.json')}
                          for name in ('torch','diffusers','safetensors')},
              'scope':'actual reference imports and currently mapped binaries; late lazy loads must be recorded separately'}
    (output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


def verify_files(files):
    for name,row in files.items():
        path=Path(name)
        if path.stat().st_size!=row['size'] or sha256(path)!=row['sha256']:
            raise ValueError('Frozen runtime file differs: '+name)


def run(plan_path):
    plan_path=plan_path.resolve();base=plan_path.parent;plan=json.loads(plan_path.read_text())
    if plan['status']!='prepared_not_executed':raise ValueError('Unprepared plan')
    identity=json.loads((base/'source-identity.json').read_text())
    if sha256(base/'source-identity.json')!=plan['source_identity_sha256']:raise ValueError('Source identity changed')
    runtime=json.loads((base/'runtime/identity.json').read_text())
    if sha256(base/'runtime/identity.json')!=plan['runtime_identity_sha256']:raise ValueError('Runtime identity changed')
    if sys.prefix!=runtime['prefix'] or os.path.abspath(sys.executable)!=identity['python']:
        raise ValueError('Interpreter invocation/prefix differs from frozen virtual environment')
    for name,digest in identity['files'].items():
        if sha256(base/'source'/name)!=digest:raise ValueError('Frozen worker source changed: '+name)
    if sha256(plan['input']['path'])!=plan['input']['sha256']:raise ValueError('Frozen input changed')
    verify_files(runtime['files'])
    for name,digest in identity['official_metadata_sha256'].items():
        if sha256(Path(identity['official_root'])/name)!=digest:raise ValueError('Official metadata differs')
    resource=plan['resources'];limits={'memory_max':resource['memory_max_bytes'],'swap_max':0,
                                     'host_min':resource['host_available_min_bytes'],'timeout_seconds':resource['timeout_seconds']}
    cg,controls=cgroup(limits)
    if controls['cpu.max']!='200000 100000' or sorted(os.sched_getaffinity(0))!=resource['cpu_affinity']:
        raise ValueError('CPU quota or affinity differs from plan')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be hidden for CPU reference')
    out=base/'official-execution';out.mkdir()
    argv=plan['steps'][0]['argv']
    expected=[identity['python'],str(base/'source/tools/export_vae.py'),'--output',str(base/'official-vae'),
              '--height','96','--width','172','--reference-only','--fixed-1376x768','--threads','2',
              '--official-root',identity['official_root'],'--input-f32',plan['input']['path']]
    if argv!=expected:raise ValueError('Official command differs from fixed CPU-only plan')
    start={'plan_sha256':sha256(plan_path),'runtime_sha256':plan['runtime_identity_sha256'],
           'resource_controls':controls,'cpu_affinity':sorted(os.sched_getaffinity(0)),
           'scope':'single CPU official decoder; not production quality or performance'}
    (out/'identity.json').write_text(json.dumps(start,indent=2)+'\n')
    result={'status':'running'}
    try:
        result['process']=execute(argv,out,limits,'official',cg)
        verify_files(runtime['files'])
        for name,digest in identity['official_metadata_sha256'].items():
            if sha256(Path(identity['official_root'])/name)!=digest:raise ValueError('Official metadata changed during run')
        events=dict(line.split() for line in (cg/'memory.events').read_text().splitlines())
        if any(int(events.get(k,0)) for k in ('oom','oom_kill','oom_group_kill')):raise ValueError('OOM in reference scope')
        if sha256(plan['input']['path'])!=plan['input']['sha256']:raise ValueError('Input changed during reference')
        result['status']='completed_reference_not_native_validated'
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture-runtime',type=Path);mode.add_argument('--run-plan',type=Path);a=p.parse_args()
    if a.capture_runtime:
        identity=capture(a.capture_runtime);print(json.dumps({'status':identity['status'],'files':len(identity['files'])}))
    else:run(a.run_plan)

if __name__=='__main__':main()
