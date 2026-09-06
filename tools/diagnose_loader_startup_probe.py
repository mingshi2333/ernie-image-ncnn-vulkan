#!/usr/bin/env python3
"""Actual Python startup plus explicit loader-cache mmap; no Vulkan/model imports."""
import json,os,subprocess,sys,time
from pathlib import Path

def run(out):
 sys.path.insert(0,str(out/'execution'))
 import diagnose_up84_scope_guard as g
 from diagnose_official_block_hooks import sha
 from diagnose_vulkan_inventory import verify_inventory
 p=json.loads((out/'plan.json').read_text());verify_inventory(json.loads(Path(p['loader_inventory']).read_text()),dict(os.environ))
 group=g.group_for(os.getpid());unit=group.rsplit('/',1)[1].removesuffix('.scope');rows=[];seen={}
 # Explicit mmap makes the cache mapping observable deterministically; startup
 # mappings alone may disappear before a sampling interval. This is not Vulkan.
 code="import mmap,time; f=open('/etc/ld.so.cache','rb'); m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); time.sleep(1)"
 child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
 try:
  while child.poll() is None:
   row=g.sample(group,unit);rows.append(row)
   for item in row['pids']:
    for name in item['mapped_library_paths']:
     resolved=str(Path(name).resolve());g.require(resolved in p['bound'],'Unbound mapped runtime '+resolved)
     if resolved not in seen:g.require(sha(resolved)==p['bound'][resolved],'Changed mapped runtime '+resolved);seen[resolved]=p['bound'][resolved]
   g.require(row['process_swap_bytes']==0 and row['rss_sum']<=g.RSS_MAX and row['host_available']>=g.HOST_MIN,'CPU resource guard');time.sleep(.01)
  g.require(child.returncode==0,'Python startup failure');g.require('/etc/ld.so.cache' in seen,'Probe did not observe explicit cache mapping')
 finally:
  if child.poll() is None:child.kill()
  child.wait()
 result={'status':'CPU_python_startup_and_cache_mapping_pass','scope':group,'child_pid':child.pid,'returncode':child.returncode,'model_forwards':0,'vulkan_calls':0,'sample_count':len(rows),'observed_files':seen,'samples':rows}
 (out/'python-startup-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','observed_files')}))
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
