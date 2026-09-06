from pathlib import Path
import json,hashlib,os,importlib.util,subprocess,sys
w=Path.cwd();old=w/'outputs/q2-up84-v2';new=w/'outputs/q2-up84-v3'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a=json.loads((old/'plan.json').read_text());b=json.loads((new/'plan.json').read_text())
assert sha(new/'plan.json')=='b69caa2b2e1d33025e2c76088d80876bb2a331e284de9b7367cab7099765c82a'
assert sha(new/'launcher.py')=='cf6aad76cdb199c24cec4ca831d91f748e875eb3d769fb892ce9427a57b125f6'
assert sha(new/'execution/diagnose_up84_scope_guard.py')=='86b50e9d9024ea2f533d5b2ccb1a437bdfb43ab6594ec7bbf01f52e3d07b0193'
assert set(a['bound']) <= set(b['bound']) and len(b['bound'])==6251
assert all(b['bound'][p]==s for p,s in a['bound'].items())
extra={p:s for p,s in b['bound'].items() if p not in a['bound']};assert len(extra)==53
for p,s in extra.items():assert sha(p)==s,p
changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
assert changed=={'bound','loader_inventory','native_command','output','previous_attempt','scope_unit'}
assert [p.replace('/q2-up84-v2/','/q2-up84-v3/') for p in a['native_command']]==b['native_command']
added=" from diagnose_vulkan_inventory import verify_inventory\n verify_inventory(json.loads(Path(p['loader_inventory']).read_text()),dict(os.environ))\n"
assert (new/'execution/diagnose_up84_scope_guard.py').read_text()==(old/'execution/diagnose_up84_scope_guard.py').read_text()
for name in ['runner','libncnn.a','diagnose_up84_official.py','diagnose_up84_sequence.py','diagnose_official_block_hooks.py']:
 assert sha(new/'execution'/name)==sha(old/'execution'/name),name
for p in (old/'execution/source').iterdir():
 if p.is_file():assert sha(p)==sha(new/'execution/source'/p.name),p
spec=importlib.util.spec_from_file_location('inventory_review',new/'execution/diagnose_vulkan_inventory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=json.loads(Path(b['loader_inventory']).read_text());m.verify_inventory(c,dict(os.environ))
assert c['bound']['/usr/lib64/amdvlk64.so']==b['bound']['/usr/lib64/amdvlk64.so']==sha('/usr/lib64/amdvlk64.so')
for p in ['/etc/vulkan/icd.d/amd_icd64.json','/etc/vulkan/implicit_layer.d/amd_icd64.json']:
 assert c['manifests'][p][0]['path']=='/usr/lib64/amdvlk64.so'
assert len(c['unresolved'])==1 and 'ShaderDebugger' in c['unresolved'][0]['error']
assert (new/'execution/diagnose_vulkan_inventory.py').read_bytes()==(w/'tools/diagnose_vulkan_inventory.py').read_bytes()
probe=json.loads((new/'python-startup-probe.json').read_text());summary=json.loads((new/'startup-summary.json').read_text())
assert probe['returncode']==0 and probe['sample_count']==len(probe['samples'])==47
assert probe['model_forwards']==probe['vulkan_calls']==0
assert len(probe['observed_files'])==29
assert probe['observed_files']['/etc/ld.so.cache']==b['bound']['/etc/ld.so.cache']==sha('/etc/ld.so.cache')
for path,digest in probe['observed_files'].items():assert digest==b['bound'][path]==sha(path)
for row in probe['samples']:
 v=row['cgroup_values'];assert v['memory.max']=='10737418240' and v['memory.swap.max']=='0' and v['cpu.max']=='200000 100000'
 assert row['process_swap_bytes']==0 and row['host_available']>=3221225472
 for item in row['pids']:
  assert item['affinity']==[0,2]
assert summary['samples']==47 and summary['observed_file_count']==29
assert summary['peak_rss']==max(r['rss_sum'] for r in probe['samples'])
assert summary['host_min']==min(r['host_available'] for r in probe['samples'])
unit=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1];cg=Path('/sys/fs/cgroup')/unit.lstrip('/')
limits={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.swap.current','cpu.max','memory.events']}
assert limits['memory.max']=='4294967296' and limits['memory.swap.max']=='0' and limits['memory.swap.current']=='0' and limits['cpu.max']=='200000 100000'
assert sorted(os.sched_getaffinity(0))==[8,10]
tests=[]
for name in ['test_vulkan_inventory.py','test_up84_contract.py']:
 r=subprocess.run([sys.executable,str(w/'tests'/name)],capture_output=True,text=True)
 tests.append({'file':name,'return_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0,tests[-1]
result={'status':'prepared_independently_reviewed_not_executed','plan_sha256':sha(new/'plan.json'),'launcher_sha256':sha(new/'launcher.py'),'guard_sha256':sha(new/'execution/diagnose_up84_scope_guard.py'),'previous_review_commit':'a8daffe','previous_bound_entries_unchanged':6198,'additional_entries_fully_verified':extra,'bound_count':len(b['bound']),'changed_top_level_keys':sorted(changed),'guard_byte_identical_to_v2':True,'math_runner_inputs_sequence_official_unchanged':True,'catalog_sha256':sha(Path(b['loader_inventory'])),'catalog_bound_files':len(c['bound']),'catalog_manifest_files':len(c['manifests']),'loader_configuration_files':len(c['loader_configuration']['bound']),'actual_cpu_probe_summary':summary,'actual_cpu_probe_sha256':sha(new/'python-startup-probe.json'),'unresolved':c['unresolved'],'tests':tests,'scope':unit,'limits':limits,'final_events':(cg/'memory.events').read_text().strip(),'new_model_forwards':0,'formal_acceptance':False,'limitations':['Directory discovery is a frozen conservative superset, not exact loader selection emulation.','Optional unresolved shader-debugger remains explicit; any actual unknown mapping still causes failure.','Runtime mapped-library observation remains sampled, not an exhaustive transient-load proof.']}
(w/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-up84-v3-root-preparation-review-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','bound_count','previous_bound_entries_unchanged','catalog_bound_files','catalog_manifest_files','new_model_forwards','final_events']},indent=2))
