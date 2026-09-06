#!/usr/bin/env python3
"""No-model Vulkan discovery supervisor; unknown mappings are evidence, never model authorization."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run(out):
 plan=json.loads((out/'plan.json').read_text())
 for name,digest in plan['bound'].items():
  if sha(name)!=digest:raise ValueError('Changed discovery input '+name)
 group=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));cg=Path('/sys/fs/cgroup')/group.lstrip('/')
 controls={x:(cg/x).read_text().strip() for x in ['memory.max','memory.swap.max','cpu.max']}
 assert group.endswith('/ernie-q2-vk-discovery-v2.scope') and controls=={'memory.max':str(4*1024**3),'memory.swap.max':'0','cpu.max':'200000 100000'} and set(os.sched_getaffinity(0))=={0,2}
 assert psutil.virtual_memory().available>=3*1024**3
 (out/'preflight.json').write_text(json.dumps({'group':group,'controls':controls,'affinity':sorted(os.sched_getaffinity(0))})+'\n')
 records=[];files={};error=None;begin=time.monotonic();child=None
 try:
  with (out/'run.log').open('x') as log:
   child=subprocess.Popen([str(out/'runner')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while True:
    pids=[]
    for pid in map(int,(cg/'cgroup.procs').read_text().split()):
     try:
      proc=psutil.Process(pid);paths=set()
      for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
       fields=line.split(maxsplit=5)
       if len(fields)==6 and fields[5].startswith('/') and Path(fields[5]).is_file():paths.add(str(Path(fields[5]).resolve()))
      status=Path(f'/proc/{pid}/status').read_text();swap=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmSwap:')))*1024
      pg=next(x[3:] for x in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'));assert pg==group and proc.cpu_affinity()==[0,2]
      pids.append({'pid':pid,'name':proc.name(),'rss':proc.memory_info().rss,'swap':swap,'paths':sorted(paths),'cgroup':pg,'affinity':proc.cpu_affinity()})
      for name in paths:
       if name not in files:files[name]={'sha256':sha(name),'size':Path(name).stat().st_size}
     except (FileNotFoundError,ProcessLookupError,psutil.NoSuchProcess):continue
    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--id=0'],text=True).strip());row={'seconds':time.monotonic()-begin,'pids':pids,'rss':sum(p['rss'] for p in pids),'swap':sum(p['swap'] for p in pids),'gpu_mib':gpu,'host':psutil.virtual_memory().available,'memory_current':int((cg/'memory.current').read_text()),'events':(cg/'memory.events').read_text()};records.append(row)
    assert row['rss']<=4*1024**3 and row['swap']==0 and gpu<=6144 and row['host']>=3*1024**3 and row['seconds']<=30,'discovery resource limit'
    assert all(int(x.split()[1])==0 for x in row['events'].splitlines()),'memory event'
    if child.poll() is not None:break
    time.sleep(.05)
   assert child.wait()==0,'probe exit'
   assert (out/'run.log').read_text().splitlines()[-1]=='DESTROYED','missing lifecycle completion'
 except BaseException as e:error=str(e)
 finally:
  if child and child.poll() is None:child.kill()
  if child:child.wait()
 for name,row in files.items():
  if sha(name)!=row['sha256']:error='mapped file changed '+name
 result={'status':'discovery_completed' if error is None else 'discovery_failed','error':error,'exit':child.returncode if child else None,'wall':time.monotonic()-begin,'mapped_files':files,'samples':records,'model_forwards':0,'scope':group,'formal_acceptance':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('mapped_files','samples')}));return error is None
if __name__=='__main__':raise SystemExit(0 if run(Path(sys.argv[1]).resolve()) else 1)
