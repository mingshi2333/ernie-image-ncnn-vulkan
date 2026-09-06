import os,json,hashlib,time
from pathlib import Path
R=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference');O=R/'outputs/q2-up84-v1';D=R/'outputs/q2-up84-root-review-v1.json';assert not D.exists()
start=time.monotonic();cache={}
def sha(p):
 p=Path(p);s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
scope=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));cg=Path('/sys/fs/cgroup')/scope.lstrip('/')
assert (cg/'memory.max').read_text().strip()=='4294967296' and (cg/'memory.swap.max').read_text().strip()=='0' and (cg/'cpu.max').read_text().strip()=='200000 100000'
assert sorted(os.sched_getaffinity(0))==[8,10]
e=json.loads((R/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-up84-preparation-evidence.json').read_text());plan=json.loads((O/'plan.json').read_text())
identity={}
for name,v in e['identity'].items():
 if isinstance(v,str):assert sha(O/name)==v,name;identity[name]=v
for name,h in plan['bound'].items():assert sha(name)==h,name
assert len(plan['bound'])==6133
old=R/'outputs/q2-block15-mlp81-v2/execution'
# The original small observer stores source beside the frozen executable.
source=O/'execution/source';references={}
for f in source.iterdir():
 if not f.is_file() or f.name=='diagnose_block_boundary.cpp':continue
 candidates=[Path(k) for k,h in plan['bound'].items() if Path(k).name==f.name and 'q2-block15' in k and h==sha(f)]
 assert candidates,f.name
 assert all(sha(q)==sha(f) for q in candidates)
 references[f.name]=[str(q) for q in candidates]
lib=O/'execution/libncnn.a';libs=[Path(k) for k,h in plan['bound'].items() if Path(k).name=='libncnn.a' and 'q2-block15' in k and h==sha(lib)]
assert libs and all(sha(q)==sha(lib) for q in libs)
assert sha(source/'diagnose_block_boundary.cpp')==sha(R/'tools/diagnose_up84_boundary.cpp')
assert plan['max_forwards']=={'native_full_block':1,'official_full_block':0,'official_mlp_only':1}
oldplan=json.loads((R/'outputs/q2-block15-mlp81-official-v5/plan.json').read_text());assert plan['weights']==oldplan['weights'] and plan['exact_graph_tail']==oldplan['exact_graph_tail']
for k in ['param_sha256','binary_sha256','official_component_sha256']:assert plan[k]==oldplan[k]
probe=json.loads((O/'execution/cpu-scope-probe.json').read_text());c=probe['cgroup_values'];assert c['memory.max']=='10737418240' and c['memory.swap.max']=='0' and c['cpu.max']=='200000 100000'
for p in probe['pids']:
 assert p['affinity']==[0,2] and p['vm_swap_bytes']==0 and p['cgroup']==probe['cgroup']
 for f in p['mapped_library_paths']:q=str(Path(f).resolve());assert q in plan['bound'] and sha(q)==plan['bound'][q]
cl=json.loads((O/'execution/compile-limits.json').read_text());cr=json.loads((O/'execution/compile-result.json').read_text())
assert cl==e['compile_limits'] and cr==e['compile_result'] and cr['exit_code']==0
assert cl['memory.max']=='4294967296' and cl['memory.swap.max']=='0' and cl['cpu.max']=='200000 100000' and cl['affinity']==[0,2]
result={'status':'implementation_and_bound_files_verified_pending_plan_metadata_correction','reviewed_identity':identity,'bound_entries':len(plan['bound']),'unique_inodes_hashed':len(cache),'math_source_matches':references,'libncnn_matches':[str(q) for q in libs],'compile_limits':cl,'compile_result':cr,'probe_mapped_libraries_all_bound':True,'scope':scope,'review_events':(cg/'memory.events').read_text().strip(),'review_swap':(cg/'memory.swap.current').read_text().strip(),'wall_seconds':time.monotonic()-start,'new_model_forwards':0,'formal_acceptance':False}
D.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','bound_entries','unique_inodes_hashed','wall_seconds','review_events','review_swap']}))
