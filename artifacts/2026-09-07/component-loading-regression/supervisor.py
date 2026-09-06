import json,os,pathlib,subprocess,sys,time,uuid
b=pathlib.Path(__file__).resolve().parent;side=sys.argv[1] if len(sys.argv)==2 else ''
if side != 'on':raise SystemExit('usage: supervisor.py on')
p=json.loads((b/'plan.json').read_text());g=p['guard'];out=b/side
uid=os.getuid();unit='ernie-o2-component64-'+side+'-'+uuid.uuid4().hex
cg=pathlib.Path(f'/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope')
env={**os.environ,'XDG_RUNTIME_DIR':f'/run/user/{uid}','DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{uid}/bus','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
cmd=['systemd-run','--user','--scope','--quiet','--unit='+unit,f"--property=MemoryMax={g['memory_max_bytes']}",f"--property=MemorySwapMax={g['memory_swap_max_bytes']}",'--property=CPUQuota=200%','taskset','-c',','.join(map(str,g['cpu_affinity'])),sys.executable,str(b/'worker.py'),side]
def available():
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise RuntimeError('no MemAvailable')
state={'side':side,'supervised_command':cmd,'guard':g,'complete':False,'cgroup_seen':False,'peak_memory_current':0,'minimum_host_available':available(),'failure':None};start=time.monotonic()
for name in ('supervisor.log','samples.jsonl','process.json'):
 if (out/name).exists():raise SystemExit('refuse overwrite: '+str(out/name))
if state['minimum_host_available']<g['minimum_host_available_bytes']:raise SystemExit('host memory guard failed before launch')
with (out/'supervisor.log').open('wb') as log,(out/'samples.jsonl').open('w') as samples:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:
  while proc.poll() is None:
   av=available();state['minimum_host_available']=min(state['minimum_host_available'],av)
   if av<g['minimum_host_available_bytes']:raise RuntimeError('host MemAvailable below floor')
   if time.monotonic()-start>g['timeout_seconds']:raise RuntimeError('timeout')
   if cg.exists():
    state['cgroup_seen']=True;state['cpu_max_observed']=(cg/'cpu.max').read_text().strip();state['memory_max_observed']=(cg/'memory.max').read_text().strip();state['memory_swap_max_observed']=(cg/'memory.swap.max').read_text().strip();state['peak_memory_current']=max(state['peak_memory_current'],int((cg/'memory.current').read_text()));state['memory_events']=(cg/'memory.events').read_text()
    if state['cpu_max_observed']!='200000 100000' or state['memory_max_observed']!=str(g['memory_max_bytes']) or state['memory_swap_max_observed']!='0':raise RuntimeError('actual scope controls differ')
   samples.write(json.dumps({'seconds':time.monotonic()-start,'available':av,'memory_current':state['peak_memory_current']})+'\n');samples.flush();time.sleep(g['sample_seconds'])
  state['return_code']=proc.wait();state['complete']=state['return_code']==0 and state['cgroup_seen']
 except BaseException as e:
  state['failure']=str(e);subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=KILL',unit+'.scope'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);proc.wait();state['return_code']=proc.returncode
state['wall_seconds']=time.monotonic()-start;(out/'process.json').write_text(json.dumps(state,indent=2)+'\n')
raise SystemExit(0 if state['complete'] else 1)
