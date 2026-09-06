#!/usr/bin/env python3
import os,json,hashlib,time,subprocess,signal
from pathlib import Path
O=Path(__file__).parent;p=json.loads((O/'plan.json').read_text());start=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(__file__)==p['worker_sha256']
for path,v in p['inputs'].items():assert Path(path).stat().st_size==v['size_bytes'] and sha(path)==v['sha256'],path
scope=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));cg=Path('/sys/fs/cgroup')/scope.lstrip('/')
def sample():
 d={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max','memory.current','memory.swap.current','memory.events']};d['seconds']=time.monotonic()-start;d['host_available']=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;return d
pre=sample();assert pre['memory.max']=='4294967296' and pre['memory.swap.max']=='0' and pre['cpu.max']=='200000 100000';assert sorted(os.sched_getaffinity(0))==[8,10]
(O/'preflight.json').write_text(json.dumps({'scope':scope,'values':pre,'affinity':sorted(os.sched_getaffinity(0))},indent=2)+'\n')
samples=[];failure=None
with (O/'link.log').open('w') as log:
 proc=subprocess.Popen(p['argv'],cwd=p['cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while proc.poll() is None:
  s=sample();samples.append(s)
  if s['host_available']<3*1024**3 or time.monotonic()-start>180 or int(s['memory.swap.current']) or s['memory.events']!=pre['memory.events']:
   failure='resource_guard';os.killpg(proc.pid,signal.SIGKILL);break
  time.sleep(.1)
 code=proc.wait()
samples.append(sample());(O/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
post={}
for path,v in p['inputs'].items():
 digest=sha(path);post[path]=digest;assert digest==v['sha256'],'Input changed '+path
result={'status':'link_failed' if code or failure else 'linked_pending_identity','return_code':code,'failure':failure,'wall_seconds':time.monotonic()-start,'inputs_unchanged':True,'post_sha256':post,'memory_max':4294967296,'scope':scope,'max_sampled_memory_current':max(int(s['memory.current']) for s in samples),'min_host_available':min(s['host_available'] for s in samples),'outputs':{},'distributable':False,'licenses_complete':False}
for name in ['ernie-image.relinked','link.map','link.d','link.log','samples.json','preflight.json']:
 q=O/name
 if q.is_file():result['outputs'][name]={'sha256':sha(q),'size_bytes':q.stat().st_size}
if code==0 and failure is None:
 result['same_binary_as_d3']=sha(O/'ernie-image.relinked')==p['original_binary_sha256']
 result['status']='passed_identical_binary_link_map' if result['same_binary_as_d3'] else 'link_output_identity_changed'
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['status']=='passed_identical_binary_link_map' else 1)
