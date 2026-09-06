#!/usr/bin/env python3
"""Dedicated cgroup gate and per-PID telemetry; model payload starts only after checks."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
MEMORY_MAX=10*1024**3
RSS_MAX=9*1024**3
HOST_MIN=3*1024**3
GPU_MAX=6144

def require(ok,message):
 if not ok:raise ValueError(message)

def group_for(pid):
 rows=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
 values=[r.split(':',2)[2] for r in rows if r.startswith('0::')]
 require(len(values)==1,'Unified cgroup missing');return values[0]

def group_values(relative):
 path=Path('/sys/fs/cgroup')/relative.lstrip('/')
 names=['memory.max','memory.swap.max','cpu.max','cpuset.cpus.effective','memory.current','memory.swap.current','memory.events','memory.swap.events','cgroup.procs']
 return {name:(path/name).read_text().strip() for name in names if (path/name).exists()}

def check_limits(group,values,affinity,unit):
 require(group.endswith('/'+unit+'.scope'),'Wrong dedicated scope')
 require(values.get('memory.max')==str(MEMORY_MAX),'memory.max must equal 10GiB')
 require(values.get('memory.swap.max')=='0','memory.swap.max must equal zero')
 quota,period=values.get('cpu.max','').split()
 require(quota.isdigit() and period.isdigit() and int(period)>0 and int(quota)==2*int(period),'cpu.max must equal 200 percent')
 require(set(affinity)=={0,2},'CPU affinity must equal 0,2')
 require(values.get('memory.swap.current')=='0','Dedicated scope already has swap')

def pid_values(pid):
 status={}
 for row in Path(f'/proc/{pid}/status').read_text().splitlines():
  if ':' in row:
   key,value=row.split(':',1);status[key]=value.strip()
 return {'pid':pid,'name':status.get('Name'),'rss_bytes':int(status.get('VmRSS','0 kB').split()[0])*1024,'vm_swap_bytes':int(status.get('VmSwap','0 kB').split()[0])*1024,'cgroup':group_for(pid),'affinity':sorted(os.sched_getaffinity(pid))}

def events(values):return dict((k,int(v)) for k,v in (line.split() for line in values['memory.events'].splitlines()))

def sample(group,unit,validate=True):
 values=group_values(group)
 records=[]
 for word in values['cgroup.procs'].split():
  try:records.append(pid_values(int(word)))
  except (FileNotFoundError,ProcessLookupError):continue
 # Also collect recursively: reject a child moved to any other cgroup.
 for process in psutil.Process().children(recursive=True):
  try:
   value=pid_values(process.pid)
   if value['pid'] not in {p['pid'] for p in records}:records.append(value)
  except (FileNotFoundError,ProcessLookupError,psutil.NoSuchProcess):pass
 result={'pids':records,'rss_sum':sum(p['rss_bytes'] for p in records),'process_swap_bytes':sum(p['vm_swap_bytes'] for p in records),'cgroup':group,'cgroup_values':values,'host_available':psutil.virtual_memory().available}
 if validate:check_sample(result,unit)
 return result

def check_sample(result,unit):
 check_limits(result['cgroup'],result['cgroup_values'],os.sched_getaffinity(0),unit)
 require(all(p['cgroup']==result['cgroup'] for p in result['pids']),'Process escaped dedicated scope')
 require(all(set(p['affinity'])=={0,2} for p in result['pids']),'Scope process changed affinity')

def gpu_memory():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--id=0'],capture_output=True,text=True,timeout=5,check=True)
 require(p.stdout.strip().isdigit(),'GPU monitor unavailable');return int(p.stdout.strip())

def run(plan_path):
 from diagnose_official_block_hooks import sha,verify
 p=json.loads(plan_path.read_text());out=Path(p['output']);unit=p['scope_unit'];group=group_for(os.getpid())
 require(not (out/'attempt.json').exists(),'Attempt already started')
 first=sample(group,unit,validate=False)
 (out/'execution/preflight-cgroup.json').write_text(json.dumps(first,indent=2)+'\n')
 check_sample(first,unit);require(first['host_available']>=HOST_MIN,'Host floor preflight');require(first['rss_sum']<=RSS_MAX,'RSS preflight');require(gpu_memory()<=GPU_MAX,'GPU preflight')
 verify(p['bound'])
 first=sample(group,unit);require(first['host_available']>=HOST_MIN,'Host floor after identity checks')
 (out/'attempt.json').write_text(json.dumps({'plan_sha256':sha(plan_path),'preflight':first,'worker_pid':os.getpid()},indent=2)+'\n')
 begin=time.monotonic();initial_events=events(first['cgroup_values']);reason=None;child=None;peak=0;gpupeak=0
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 with (out/'execution/worker.log').open('w') as log,(out/'execution/samples.jsonl').open('w') as samples:
  try:
   child=subprocess.Popen([sys.executable,str(out/'execution/diagnose_mlp81_official_resume.py'),str(plan_path)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   while True:
    observation=sample(group,unit,validate=False);gpu=gpu_memory();observation.update(seconds=time.monotonic()-begin,gpu_mib=gpu)
    samples.write(json.dumps(observation)+'\n');samples.flush();peak=max(peak,observation['rss_sum']);gpupeak=max(gpupeak,gpu)
    check_sample(observation,unit)
    require(observation['rss_sum']<=RSS_MAX,'process_tree_rss_limit')
    require(observation['process_swap_bytes']==0,'process_swap_detected')
    require(observation['host_available']>=HOST_MIN,'host_available_limit')
    require(gpu<=GPU_MAX,'whole_gpu_limit')
    require(all(events(observation['cgroup_values']).get(k,0)==initial_events.get(k,0) for k in ['max','oom','oom_kill','oom_group_kill']),'cgroup_memory_event')
    require(time.monotonic()-begin<=2400,'wall_timeout')
    if child.poll() is not None:break
    time.sleep(.5)
  except Exception as exc:
   reason=str(exc)
  finally:
   # A moved descendant is also killed; never kill this guard before its report persists.
   descendants=psutil.Process().children(recursive=True)
   for proc in reversed(descendants):
    try:proc.kill()
    except psutil.Error:pass
   if child is not None:rc=child.wait()
   else:rc=None
 result={'status':'completed' if rc==0 and reason is None else 'guard_or_child_failure','return_code':rc,'stop_reason':reason,'elapsed_seconds':time.monotonic()-begin,'peak_rss_sum_bytes':peak,'peak_whole_gpu_mib':gpupeak,'scope':group,'memory_max_bytes':MEMORY_MAX,'rss_limit_bytes':RSS_MAX,'swap_max_bytes':0,'cpu_quota_percent':200,'cpu_affinity':[0,2],'minimum_host_available_bytes':HOST_MIN,'gpu_limit_mib':GPU_MAX,'native_forwards_this_attempt':0,'formal_acceptance':False}
 (out/'execution/worker-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 require(result['status']=='completed','Guarded attempt failed')

if __name__=='__main__':
 if sys.argv[1]=='probe':
  group=group_for(os.getpid());record=sample(group,sys.argv[3]);Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
 else:run(Path(sys.argv[1]))
