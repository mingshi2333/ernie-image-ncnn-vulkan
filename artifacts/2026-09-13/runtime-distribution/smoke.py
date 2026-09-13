"""One installed Linux default-path image with the project's existing resource bounds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path(__file__).resolve().parent
out = base/'evidence/linux-image'
plan = json.loads((out/'plan.json').read_text())

def save(name, value):
    (out/name).write_text(json.dumps(value,indent=2)+'\n')

def available():
    return next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))

def gpu():
    return int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip())

if '--worker' in sys.argv:
    start = time.monotonic()
    with (out/'native.log').open('w') as log:
        result = subprocess.run(plan['command'],cwd=plan['cwd'],stdout=log,stderr=subprocess.STDOUT)
    cg = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    save('worker.json',{'return_code':result.returncode,'seconds':time.monotonic()-start,
                        'memory_events':(cg/'memory.events').read_text(),'memory_peak_bytes':int((cg/'memory.peak').read_text()),
                        'limits':{k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max']}})
    raise SystemExit(result.returncode)

assert not (out/'result.json').exists(), 'Preserve the existing run; do not overwrite it.'
unit='ernie-hf-runtime-20260913'
env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
     'XDG_RUNTIME_DIR':f'/run/user/{os.getuid()}','DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{os.getuid()}/bus'}
for k in ['LD_LIBRARY_PATH','DYLD_LIBRARY_PATH','VK_INSTANCE_LAYERS','VK_LAYER_PATH']:
    env.pop(k,None)
command=['systemd-run','--user','--scope','--quiet','--unit='+unit,
         '--property=MemoryMax=17179869184','--property=MemorySwapMax=0','--property=CPUQuota=200%',
         'taskset','-c','4,6',sys.executable,str(Path(__file__).resolve()),'--worker']
record={'scope_command':command,'minimum_host_available_bytes':available(),'gpu_whole_device_peak_mib':gpu(),
        'complete':False,'scope':'Default FP16 512x512 first-use execution; no official parity or speed comparison'}
assert record['minimum_host_available_bytes']>=3*1024**3 and record['gpu_whole_device_peak_mib']<=6144
start=time.monotonic()
with (out/'scope.log').open('w') as log, (out/'resources.jsonl').open('w') as resources:
    p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while p.poll() is None:
            host,used=available(),gpu()
            record['minimum_host_available_bytes']=min(host,record['minimum_host_available_bytes'])
            record['gpu_whole_device_peak_mib']=max(used,record['gpu_whole_device_peak_mib'])
            resources.write(json.dumps({'seconds':time.monotonic()-start,'host_available_bytes':host,'whole_gpu_mib':used})+'\n');resources.flush()
            if host<3*1024**3 or used>6144 or time.monotonic()-start>1800:
                raise RuntimeError('Existing host/GPU/time resource guard reached')
            time.sleep(1)
        record['return_code']=p.wait()
        record['complete']=record['return_code']==0
    except BaseException as error:
        record['failure']=str(error)
        subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=KILL',unit+'.scope'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        record['return_code']=p.wait()
record['wall_seconds']=time.monotonic()-start
if (out/'worker.json').exists():
    worker=json.loads((out/'worker.json').read_text())
    record['worker']=worker
    events={k:int(v) for k,v in (line.split() for line in worker['memory_events'].splitlines())}
    record['complete'] &= events['oom']==events['oom_kill']==0
if (out/'apple.png').is_file():
    with (out/'apple.png').open('rb') as stream:
        record['image_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
save('result.json',record)
print(json.dumps(record,indent=2),flush=True)
raise SystemExit(0 if record['complete'] else 1)
