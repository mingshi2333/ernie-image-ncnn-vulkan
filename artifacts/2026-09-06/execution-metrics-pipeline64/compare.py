from pathlib import Path
import hashlib,json,sys
b=Path(__file__).resolve().parent;p=json.loads((b/'plan.json').read_text())
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
# External execution constraints are evidence, not assumed.
for side in ('on','off'):
 s=json.loads((b/side/'process.json').read_text());g=p['guard']
 assert s['complete'] and s['return_code']==0 and s['failure'] is None and s['cgroup_seen']
 assert s['memory_max_observed']==str(g['memory_max_bytes']) and s['memory_swap_max_observed']==str(g['memory_swap_max_bytes'])
 assert s['minimum_host_available']>=g['minimum_host_available_bytes'] and s['peak_memory_current']<=g['memory_max_bytes']
 events={line.split()[0]:int(line.split()[1]) for line in s['memory_events'].splitlines()}
 assert 'oom' in events and 'oom_kill' in events and events['oom']==0 and events['oom_kill']==0
# Exact fixed denominator: no missing or unexpected trace entries.
expected=set(p['expected']['trace_files']);rows=[]
for side in ('on','off'):
 got={str(x.relative_to(b/side/'trace')) for x in (b/side/'trace').rglob('*') if x.is_file()}
 assert got==expected,(side,sorted(expected-got),sorted(got-expected))
assert len(expected)==27 and sum(n.endswith('.f32') for n in expected)==25
for n in sorted(expected):
 a=b/'on/trace'/n;c=b/'off/trace'/n
 assert a.stat().st_size==p['expected']['trace_sizes'][n] and c.stat().st_size==p['expected']['trace_sizes'][n],n
 assert a.read_bytes()==c.read_bytes(),n
 if n.endswith('.f32'):
  import array,math
  values=array.array('f');values.frombytes(a.read_bytes());assert all(math.isfinite(x) for x in values),n
 rows.append({'file':n,'sha256':sha(a),'size':a.stat().st_size,'bitwise_equal':True})
png=(b/'on/native.png').read_bytes();assert png==(b/'off/native.png').read_bytes()
assert png[:8]==b'\x89PNG\r\n\x1a\n'
import zlib
pos=8;chunks=[]
while pos<len(png):
 assert pos+12<=len(png);length=int.from_bytes(png[pos:pos+4],'big');kind=png[pos+4:pos+8];end=pos+12+length;assert end<=len(png)
 payload=png[pos+8:pos+8+length];crc=int.from_bytes(png[pos+8+length:end],'big');assert (zlib.crc32(kind+payload)&0xffffffff)==crc
 chunks.append((kind,payload));pos=end
assert pos==len(png) and chunks[0][0]==b'IHDR' and len(chunks[0][1])==13 and int.from_bytes(chunks[0][1][:4],'big')==64 and int.from_bytes(chunks[0][1][4:8],'big')==64
assert chunks[-1]==(b'IEND',b'') and sum(kind==b'IEND' for kind,_ in chunks)==1 and any(kind==b'IDAT' for kind,_ in chunks)
assert (b/'on/trace/initial.f32').read_bytes()==(b/'initial.f32').read_bytes()
assert (b/'on/trace/prompt.txt').read_bytes()==(b/'prompt.txt').read_bytes()
assert (b/'on/trace/ids.txt').read_bytes()==(b/'historical-ids.txt').read_bytes()
m=json.loads((b/'on/metrics.json').read_text());assert m['valid'] and m['coverage_complete'] and m['execution_finished_successfully'] and m['run_status']=='success'
assert not m['formal_speed_eligible'] and not m['formal_memory_eligible'] and m['trace_enabled']
assert m['allocation_domain']=='vk_device_memory' and m['coverage_scope']=='observed_ncnn_allocator_lifetime'
assert m['total']['allocations']>0 and m['total']['peak_bytes']>0 and m['total']['live_bytes']==0
assert m['devices'] and any(d['index']==0 for d in m['devices'])
assert m['allocators'] and all(not a['active'] and a['live_handles']==0 for a in m['allocators'])
assert m['stage_coverage']=='partial_known_intervals'
assert m['stage_time_scope'].startswith('non_overlapping host intervals;')
assert m['gpu_time'] is None and m['cpu_rss'] is None
st=m['stage_times'];assert st and st['upload_bytes'] is None and st['download_bytes'] is None and st['submissions'] is not None
for name in ('verify','read_prepare','upload','compute','download'):
 assert st[name] is not None and st[name]['host_nanoseconds']>=0 and st[name]['samples']>0 and st[name]['gpu_nanoseconds'] is None
assert sum(v['host_nanoseconds'] for k,v in st.items() if isinstance(v,dict))<=m['host_nanoseconds']
result={'status':'passed','scope':p['scope'],'trace_count':len(rows),'f32_count':sum(x['file'].endswith('.f32') for x in rows),'traces':rows,'png_sha256':sha(b/'on/native.png'),'metrics':m,'plan_sha256':sha(b/'plan.json'),'formal_speed_eligible':False,'formal_memory_eligible':False}
(b/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','trace_count':len(rows),'png_sha256':result['png_sha256']}))
