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
import signal
import subprocess
import time
from check_release import cgroup, available
from prepare_block import sha256



def runtime_files():
    """Inventory actual Python modules and file-backed mappings at this boundary."""
    paths=set()
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_absolute():
            if not Path(name).is_file():
                raise ValueError('Missing loaded module: '+name)
            paths.add(str(Path(name).resolve()))
    mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and parts[5].startswith('/'):
            name=parts[5]
            if not Path(name).is_file():
                raise ValueError('Missing/deleted file-backed mapping: '+name)
            mapped.add(str(Path(name).resolve()))
    paths.update(mapped)
    return {name:{'size':Path(name).stat().st_size,'sha256':sha256(Path(name))}
            for name in sorted(paths)},sorted(mapped)


class WorkerRuntime:
    """Keep actual model-process boundary inventories; unknown files never authenticate."""
    phases=['before_model','after_model','after_first_forward','after_second_forward']

    def __init__(self,identity_path,output):
        self.identity_path=Path(identity_path);self.output=Path(output)
        self.allowed=json.loads(self.identity_path.read_text())
        self.output.mkdir(parents=True,exist_ok=False)
        self.files={};self.unknown=set();self.checkpoints=[];self.error=None
        self.process=process_identity(os.getpid())

    def checkpoint(self,phase):
        files,mapped=runtime_files()
        require_coverage(files,mapped,self.allowed)
        if phase!=self.phases[len(self.checkpoints)]:raise ValueError('Runtime phase order differs')
        for name,row in files.items():
            if name in self.files and self.files[name]!=row:
                raise ValueError('Loaded runtime changed during model execution: '+name)
            if name in self.allowed['files']:
                if self.allowed['files'][name]!=row:
                    raise ValueError('Loaded runtime differs from frozen allowlist: '+name)
            else:
                self.unknown.add(name)
                # New code is retained for review, but cannot silently extend authorization.
                archive=self.output/'objects'/row['sha256']
                archive.parent.mkdir(exist_ok=True)
                if not archive.exists():shutil.copyfile(name,archive)
                if sha256(archive)!=row['sha256'] or sha256(Path(name))!=row['sha256']:
                    raise ValueError('Runtime changed while archiving: '+name)
            self.files[name]=row
        self.checkpoints.append({'phase':phase,'process':process_identity(os.getpid()),
                                 'files':files,'mapped_files':mapped})

    def finish(self,success):
        original_error=sys.exc_info()[1]
        if original_error is not None:self.error=str(original_error)
        try:
            verify_files(self.files)
        except BaseException as error:
            self.error=str(error)
        valid=(success and self.error is None and not self.unknown
               and [x['phase'] for x in self.checkpoints]==self.phases
               and sys.prefix==self.allowed['prefix']
               and os.path.abspath(sys.executable)==os.path.abspath(self.allowed['executable']))
        report={'schema_version':1,'status':'authenticated' if valid else 'unaccepted_runtime_identity',
                'model_execution_completed':success,'authenticated':valid,'error':self.error,
                'allowlist_sha256':sha256(self.identity_path),'collector_sha256':sha256(Path(__file__)),
                'prefix':sys.prefix,'executable':sys.executable,'files':self.files,
                'process':self.process,'entry_path':str(Path(sys.argv[0]).resolve()),
                'entry_sha256':sha256(Path(sys.argv[0])),
                'unknown_files':sorted(self.unknown),'checkpoints':self.checkpoints,
                'scope':'actual model-process module and mapped-file inventories at four boundaries; '
                        'not a claim to observe transient load/unload between boundaries'}
        (self.output/'identity.json').write_text(json.dumps(report,indent=2)+'\n')
        if not valid and original_error is None:raise ValueError('Actual worker runtime identity was not authenticated; see '+str(self.output))


def require_coverage(files,mapped,runtime):
    """The preparer freezes the mandatory set; a report cannot shrink that set."""
    missing=set(runtime['required_files'])-set(files)
    missing_mappings=set(runtime['required_mapped_files'])-set(mapped)
    if missing or missing_mappings or not mapped:
        raise ValueError('Required runtime coverage missing: files='+str(len(missing))
                         +' mappings='+str(len(missing_mappings)))


def validate_worker_runtime(path,runtime,expected_sha,collector_sha,process,entry_path):
    report=json.loads(Path(path).read_text())
    if (report['status']!='authenticated' or report['authenticated'] is not True
        or report['model_execution_completed'] is not True or report['unknown_files']
        or report['allowlist_sha256']!=expected_sha or report['collector_sha256']!=collector_sha
        or report['process']!=process or report['entry_path']!=str(Path(entry_path).resolve())
        or report['entry_sha256']!=sha256(Path(entry_path))
        or report['prefix']!=runtime['prefix'] or report['executable']!=runtime['executable']
        or [x['phase'] for x in report['checkpoints']]!=WorkerRuntime.phases):
        raise ValueError('Actual worker runtime report is incomplete or unauthenticated')
    union={}
    for point in report['checkpoints']:
        require_coverage(point['files'],point['mapped_files'],runtime)
        if point['process']!=process:raise ValueError('Worker boundary process identity differs')
        if not point['files'] or not set(point['mapped_files'])<=set(point['files']):
            raise ValueError('Actual worker runtime boundary inventory is incomplete')
        for name,row in point['files'].items():
            if runtime['files'].get(name)!=row:
                raise ValueError('Actual worker runtime file was not authorized: '+name)
            union[name]=row
    if union!=report['files']:raise ValueError('Worker runtime union differs')
    verify_files(union)
    return report


def capture(output,import_reference=True):
    if output.exists():raise ValueError('Use a new runtime snapshot')
    # Import exactly the future reference worker dependencies; no model is loaded.
    if import_reference:
        import export_vae
    files,mapped=runtime_files()
    output.mkdir(parents=True);sources=output/'sources';sources.mkdir()
    for name,row in files.items():
        if name.endswith('.py'):
            dest=sources/(row['sha256']+'.py')
            if not dest.exists():shutil.copy2(name,dest)
    identity={'status':'imported_dependencies_no_model_execution','prefix':sys.prefix,
              'process':process_identity(os.getpid()),'entry_path':str(Path(sys.argv[0]).resolve()),
              'entry_sha256':sha256(Path(sys.argv[0])),'collector_sha256':sha256(Path(__file__)),
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



def process_identity(pid):
    fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'parent_pid':int(fields[1]),'start_ticks':int(fields[19])}


def execute_reference(command,path,limits,cg):
    """Same bounded polling contract as check_release.execute, with actual child PID binding."""
    started=time.monotonic()
    record={'argv':command,'exit_code':None,'failure':None,'host_min_observed':available(),
            'peak_cgroup_memory_current':0,
            'cgroup_memory_scope':'entire supervisor scope; sampled every 50 ms'}
    if record['host_min_observed']<limits['host_min']:raise RuntimeError('Host memory floor failed before execution')
    with (path/'official.log').open('xb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            record['process_identity']=process_identity(process.pid)
            if record['process_identity']['parent_pid']!=os.getpid():raise ValueError('Worker is not controller child')
            while process.poll() is None:
                record['host_min_observed']=min(record['host_min_observed'],available())
                record['peak_cgroup_memory_current']=max(record['peak_cgroup_memory_current'],int((cg/'memory.current').read_text()))
                if record['host_min_observed']<limits['host_min']:raise RuntimeError('Host memory floor crossed')
                if time.monotonic()-started>limits['timeout_seconds']:raise RuntimeError('Frozen timeout exceeded')
                time.sleep(.05)
            record['exit_code']=process.wait()
            if record['exit_code']!=0:raise RuntimeError('official returned '+str(record['exit_code']))
        except BaseException as error:
            record['failure']=str(error)
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
            record['exit_code']=process.wait()
            raise
        finally:
            record['wall_seconds']=time.monotonic()-started
            record['memory_events']=(cg/'memory.events').read_text()
            (path/'official.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def run(plan_path):
    plan_path=plan_path.resolve();base=plan_path.parent;plan=json.loads(plan_path.read_text())
    if plan['status']!='prepared_not_executed':raise ValueError('Unprepared plan')
    identity=json.loads((base/'source-identity.json').read_text())
    if sha256(base/'source-identity.json')!=plan['source_identity_sha256']:raise ValueError('Source identity changed')
    runtime=json.loads((base/'runtime/identity.json').read_text())
    if sha256(base/'runtime/identity.json')!=plan['runtime_identity_sha256']:raise ValueError('Runtime identity changed')
    probe_path=base/'worker-import-probe/identity.json'
    if sha256(probe_path)!=runtime['worker_probe_sha256']:raise ValueError('Worker import probe changed')
    probe=json.loads(probe_path.read_text())
    if (runtime['required_files']!=sorted(probe['files'])
        or runtime['required_mapped_files']!=probe['mapped_files']
        or probe['process']!=plan['worker_probe_process']
        or probe['entry_path']!=str(base/'source/tools/export_vae.py')
        or probe['entry_sha256']!=identity['files']['tools/export_vae.py']
        or probe['collector_sha256']!=identity['files']['tools/vae_reference_scope.py']):
        raise ValueError('Mandatory runtime coverage does not match actual worker import probe')
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
              '--official-root',identity['official_root'],'--input-f32',plan['input']['path'],
              '--runtime-identity',str(base/'runtime/identity.json'),'--runtime-report',str(base/'worker-runtime')]
    if argv!=expected:raise ValueError('Official command differs from fixed CPU-only plan')
    start={'plan_sha256':sha256(plan_path),'runtime_sha256':plan['runtime_identity_sha256'],
           'resource_controls':controls,'cpu_affinity':sorted(os.sched_getaffinity(0)),
           'scope':'single CPU official decoder; not production quality or performance'}
    (out/'identity.json').write_text(json.dumps(start,indent=2)+'\n')
    result={'status':'running'}
    try:
        result['process']=execute_reference(argv,out,limits,cg)
        validate_worker_runtime(base/'worker-runtime/identity.json',runtime,plan['runtime_identity_sha256'],
                                identity['files']['tools/vae_reference_scope.py'],result['process']['process_identity'],
                                base/'source/tools/export_vae.py')
        result['worker_runtime_sha256']=sha256(base/'worker-runtime/identity.json')
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
        if (base/'worker-runtime/identity.json').is_file():
            result['worker_runtime_sha256']=sha256(base/'worker-runtime/identity.json')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture-runtime',type=Path);mode.add_argument('--run-plan',type=Path);a=p.parse_args()
    if a.capture_runtime:
        identity=capture(a.capture_runtime);print(json.dumps({'status':identity['status'],'files':len(identity['files'])}))
    else:run(a.run_plan)

if __name__=='__main__':main()
