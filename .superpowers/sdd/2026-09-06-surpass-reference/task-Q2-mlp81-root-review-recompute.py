#!/usr/bin/env python3
import hashlib,json,math,os,resource,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
OUT=ROOT/'outputs/q2-block15-mlp81-official-v5'
DEST=ROOT/'outputs/q2-mlp81-root-review-v1.json'
assert not DEST.exists()
start=time.monotonic();cache={};verified=0
scope=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
cg=Path('/sys/fs/cgroup')/scope.lstrip('/')
def cgread():
 return {k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.current','memory.swap.current','memory.events','cpu.max']}
control=cgread()
assert control['memory.max']=='4294967296' and control['memory.swap.max']=='0' and control['cpu.max']=='200000 100000'
assert sorted(os.sched_getaffinity(0))==[8,10]
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
def sha(path):
 path=Path(path);s=path.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with path.open('rb') as f:
   for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
  cache[key]=h.hexdigest()
 return cache[key]
def check(path,digest,size=None):
 global verified
 assert sha(path)==digest,str(path)
 if size is not None:assert Path(path).stat().st_size==size,str(path)
 verified+=1
report=ROOT/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-mlp81-official-execution-evidence.json'
e=json.loads(report.read_text());plan=json.loads((OUT/'plan.json').read_text())
for path,digest in e['file_sha256'].items():check(ROOT/path,digest)
for path,digest in plan['bound'].items():check(path,digest)
r=json.loads((OUT/'result.json').read_text());w=json.loads((OUT/'execution/worker-result.json').read_text())
assert r==e['result'] and w==e['worker']
assert r['status']==w['status']=='completed' and w['return_code']==0 and w['stop_reason'] is None
assert r['forwards']=={'native_full_block':0,'official_full_block':1,'official_mlp_only':1}
assert r['native_replay_reused'] is True and r['tail_skipped_81_exact'] is False
assert r['baseline_hook_counts']=={k:1 for k in ['75','81','87','88']} and r['tail_hook_counts']=={'87':1,'88':1}
assert r['formal_acceptance'] is False
old=Path(plan['old_official'])
oldhash={'75':'09189225db6307daa124c88dce40dfc257e8af49d9cea125b44c6c64072b69af','87':'384328e7033248ef5c43d1a9494d4f93c7e13556bd0913bdda786cfc1f1f10dd','88':'3a1eb976b4772659e9461745bb21ec9fa92582adfd98d90f728aab83d8375ab8','out0':'a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197'}
for k,v in oldhash.items():
 assert r['baseline_sha256'][k]==v
 check(old/'actual'/f'{k}.f32',v);check(OUT/'official'/f'{k}.f32',v)
assert json.loads((OUT/'execution/torch-config.json').read_text())==json.loads((old/'execution/torch-config.json').read_text())
runtime_counts={}
for label in ['before','after']:
 rt=json.loads((OUT/f'execution/gpu-{label}-forward-runtime.json').read_text());runtime_counts[label]=len(rt['files'])
 for path,d in rt['files'].items():
  check(path,d['sha256'],d['bytes']);check(d['snapshot'],d['sha256'],d['bytes'])
rows=[json.loads(s) for s in (OUT/'execution/samples.jsonl').read_text().splitlines()]
pre=json.loads((OUT/'execution/preflight-cgroup.json').read_text())
for s in [pre]+rows:
 c=s['cgroup_values']
 assert c['memory.max']=='10737418240' and c['memory.swap.max']=='0' and c['cpu.max']=='200000 100000'
 assert c['memory.swap.current']=='0' and s['process_swap_bytes']==0
 assert s['cgroup']==w['scope'] and s['host_available']>=3221225472
 for event in ['memory.events','memory.swap.events']:
  assert all(int(line.split()[1])==0 for line in c[event].splitlines())
 assert sum(p['rss_bytes'] for p in s['pids'])==s['rss_sum'] and s['rss_sum']<9663676416
 for p in s['pids']:
  assert p['vm_swap_bytes']==0 and p['cgroup']==w['scope'] and p['affinity']==[0,2]
 if 'gpu_mib' in s:assert s['gpu_mib']<=6144
assert len(rows)==71
assert max(s['rss_sum'] for s in rows)==w['peak_rss_sum_bytes']
assert max(s['gpu_mib'] for s in rows)==w['peak_whole_gpu_mib']
assert json.loads((OUT/'execution/scope-exit.json').read_text())=={'return_code':0}
launch=json.loads((OUT/'execution/launch-record.json').read_text())
assert launch['plan_sha256']==sha(OUT/'plan.json') and launch['guard_sha256']==sha(OUT/'execution/diagnose_q2_scope_guard.py')
arrays={};ids={}
def array(label,path,digest,width):
 check(path,digest,4160*width*4)
 a=np.memmap(path,dtype='<f4',mode='r',shape=(4160,width))
 for i in range(0,4160,31):assert np.isfinite(a[i:i+31]).all()
 arrays[label]=a;ids[str(path)]={'sha256':digest,'elements':int(a.size),'shape':[4160,width]}
for k in ['75','81','87','88','out0']:
 assert r['shapes'][k]==[1,4160,12288 if k=='87' else 4096]
 array('O'+k,OUT/'official'/f'{k}.f32',r['baseline_sha256'][k],12288 if k=='87' else 4096)
for k in ['87','88']:array('M'+k,OUT/'official'/f'native81-{k}.f32',r['tail_sha256'][k],12288 if k=='87' else 4096)
array('N81',OUT/'native/boundary.f32','da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54',4096)
array('Nout0',OUT/'native/out0.f32','175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3',4096)
for k,folder,h in [('75','q2-block15-boundary-v1','762941a013ce27957c6b123985eb7ef1dd36fafe7c79c7f08cb7569ac154c1bd'),('87','q2-block15-boundary87-v1','a951b1ed772d5a53c0b30ba8a1d1882c52491f6c564cb9f70b09ec6ef4cd3c46'),('88','q2-block15-boundary88-v1','b1d26a4acc2c44d7864c25ba59c821ded6fb8e8c34575d7b35d0ca9586b95cbf')]:
 array('N'+k,ROOT/'outputs'/folder/'official/boundary.f32',h,12288 if k=='87' else 4096)
comparison={};split={}
for k in ['75','81','87','88','out0']:
 a,b=arrays['N'+k],arrays['O'+k];sq=ref=0.;maximum=0.;changed=0;rowsq=[]
 for i in range(0,4160,31):
  x=a[i:i+31].astype(np.float64);y=b[i:i+31].astype(np.float64);d=x-y
  sq+=float(np.sum(d*d,dtype=np.float64));ref+=float(np.sum(y*y,dtype=np.float64));maximum=max(maximum,float(np.max(np.abs(d))));changed+=int(np.count_nonzero(d));rowsq.extend(np.sum(d*d,axis=1).tolist())
 v={'elements':int(a.size),'l2':math.sqrt(sq),'max_abs':maximum,'nrmse':math.sqrt(sq/ref),'changed_elements':changed,'top_l2_rows':np.argsort(rowsq)[-8:][::-1].tolist()}
 oldv=e['analysis']['comparisons'][k]
 for f in ['l2','max_abs','nrmse']:assert math.isclose(v[f],oldv[f],rel_tol=1e-12,abs_tol=1e-15),(k,f,v[f],oldv[f])
 for f in ['elements','changed_elements','top_l2_rows']:assert v[f]==oldv[f],(k,f)
 comparison[k]=v
for k in ['87','88']:
 a,b,m=arrays['N'+k],arrays['O'+k],arrays['M'+k];sums=np.zeros(3,dtype='f8');maxima=np.zeros(3,dtype='f8');dot=0.;identity_max=0.;rowsq=[]
 for i in range(0,4160,31):
  n=a[i:i+31].astype('f8');o=b[i:i+31].astype('f8');q=m[i:i+31].astype('f8');d=n-o;u=q-o;local=n-q
  identity_max=max(identity_max,float(np.max(np.abs(d-(u+local)))))
  for j,vec in enumerate([d,u,local]):sums[j]+=np.sum(vec*vec,dtype=np.float64);maxima[j]=max(maxima[j],float(np.max(np.abs(vec))))
  dot+=float(np.sum(u*local,dtype=np.float64));rowsq.extend(np.sum(local*local,axis=1).tolist())
 v={'elements':int(a.size),'total':{'l2':float(np.sqrt(sums[0])),'max_abs':float(maxima[0])},'propagated_input':{'l2':float(np.sqrt(sums[1])),'max_abs':float(maxima[1])},'matched_implementation':{'l2':float(np.sqrt(sums[2])),'max_abs':float(maxima[2])},'propagation_local_cosine':float(dot/np.sqrt(sums[1]*sums[2])),'sum_identity_max_abs':identity_max,'squared_norm_identity_residual':float(sums[0]-sums[1]-sums[2]-2*dot),'largest_local_rows':np.argsort(rowsq)[-8:][::-1].tolist()}
 oldv=e['analysis']['decomposition'][k]
 assert identity_max<=1e-12
 for f in ['total','propagated_input','matched_implementation']:
  for metric in ['l2','max_abs']:assert math.isclose(v[f][metric],oldv[f][metric],rel_tol=1e-12,abs_tol=1e-15),(k,f,metric)
 assert math.isclose(v['propagation_local_cosine'],oldv['propagation_local_cosine'],rel_tol=1e-12,abs_tol=1e-15)
 assert v['elements']==oldv['elements'] and v['largest_local_rows']==oldv['largest_local_rows']
 split[k]=v
result={'status':'independently_verified_fixed_diagnostic','scope':'Read-only CPU reconstruction; no model forward or GPU call','script_sha256':sha(__file__),'evidence_sha256':sha(report),'plan_sha256':sha(OUT/'plan.json'),'verified_hash_entries':verified,'unique_inodes_hashed':len(cache),'bound_entries':len(plan['bound']),'runtime_counts':runtime_counts,'input_identity':ids,'comparisons':comparison,'decomposition':split,'actual_execution_resource_audit':{'samples':len(rows),'peak_rss_sum_bytes':max(s['rss_sum'] for s in rows),'peak_cgroup_current_bytes':max(int(s['cgroup_values']['memory.current']) for s in rows),'whole_gpu_mib':max(s['gpu_mib'] for s in rows),'min_host_available':min(s['host_available'] for s in rows),'all_swap_and_memory_events_zero':True},'review_process':{'scope':scope,'affinity':sorted(os.sched_getaffinity(0)),'cgroup_before':control,'cgroup_after':cgread(),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.monotonic()-start},'formal_acceptance':False}
DEST.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','bound_entries','runtime_counts','decomposition','actual_execution_resource_audit','review_process']},indent=2))
