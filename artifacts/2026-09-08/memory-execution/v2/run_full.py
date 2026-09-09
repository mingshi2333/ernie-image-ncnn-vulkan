"""Serial frozen-binary memory execution regression with existing resource bounds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path(__file__).resolve().parent
plan = json.loads((base / 'full-plan.json').read_text())
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')
def available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def gpu():
    return int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip())

if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    case = sys.argv[2]
    out = base / 'full' / case
    command = json.loads((out / 'command.json').read_text())
    with (out / 'native.log').open('w') as log:
        start = time.monotonic()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    record = {'return_code':result.returncode,'native_wall_seconds':time.monotonic()-start,
        'command':command,'memory_events':(cgroup/'memory.events').read_text(),
        'memory_peak_bytes':int((cgroup/'memory.peak').read_text()),
        'observed_limits':{k:(cgroup/k).read_text().strip() for k in ('memory.max','memory.swap.max','cpu.max')}}
    save(out/'worker-result.json',record)
    raise SystemExit(result.returncode)

limits = plan['resource_limits']
env = {**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','HF_HUB_OFFLINE':'1',
       'TRANSFORMERS_OFFLINE':'1', 'VK_INSTANCE_LAYERS':'VK_LAYER_KHRONOS_validation','PYTHONDONTWRITEBYTECODE':'1', 'XDG_RUNTIME_DIR':f'/run/user/{os.getuid()}',
       'DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{os.getuid()}/bus'}
sdk = Path(plan['validation']['runtime'])
env['VK_LAYER_PATH'] = str(sdk/'share/vulkan/explicit_layer.d')
env['LD_LIBRARY_PATH'] = str(sdk/'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
progress = {'status':'running','pid':os.getpid(),'completed':[],'plan_sha256':sha(base/'full-plan.json'),
    'supervisor_sha256':sha(__file__)}
save(base/'full-progress.json',progress)
for version, precision in plan['cases']:
    case = version+'-'+precision
    out = base/'full'/case
    out.mkdir(parents=True)
    binary = base/'frozen-bin'/'ernie-image'
    assert sha(binary)==plan['bindings'][str(binary)]['sha256']
    command = [str(binary),'--model',plan['model'],'--prompt',plan['prompt'],'--output',str(out/'native.png'),
        '--trace-dir',str(out/'trace'),'--width','512','--height','512','--latent',plan['initial'],
        '--steps','8','--threads','2','--device','vulkan','--precision',precision,'--text-down-vector',
        '--vae-device','cpu','--vae-convolution','direct','--dit-weights','host','--gpu-reserve-mib','512',
        '--gpu','0','--dit-cache-mib','0','--ram-reserve-mib','3072','--model-loading','stdio',
        '--report-json',str(out/'generation.json')]
    for flag,value in zip(plan['case_options'][version][::2],plan['case_options'][version][1::2]):
        if flag in command: command[command.index(flag)+1]=value
        else: command.extend([flag,value])
    save(out/'command.json',command)
    unit = 'ernie-memory-exec-v2-'+case
    scope = ['systemd-run','--user','--scope','--quiet','--unit='+unit,
        '--property=MemoryMax='+str(limits['memory_max_bytes']),'--property=MemorySwapMax=0',
        '--property=CPUQuota=200%','taskset','-c','4,6',sys.executable,str(Path(__file__).resolve()),'--worker',case]
    record = {'case':case,'command':command,'scope_command':scope,'complete':False,'minimum_host_available_bytes':available(),
        'sampled_gpu_whole_device_peak_mib':gpu()}
    assert record['minimum_host_available_bytes']>=limits['host_available_min_bytes']
    assert record['sampled_gpu_whole_device_peak_mib']<=limits['gpu_whole_device_max_mib']
    progress['current']=case
    save(base/'full-progress.json',progress)
    print('Starting',case,flush=True)
    start=time.monotonic()
    with (out/'scope.log').open('w') as log, (out/'resources.jsonl').open('w') as samples:
        process=subprocess.Popen(scope,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while process.poll() is None:
                host, used=available(),gpu()
                record['minimum_host_available_bytes']=min(record['minimum_host_available_bytes'],host)
                record['sampled_gpu_whole_device_peak_mib']=max(record['sampled_gpu_whole_device_peak_mib'],used)
                samples.write(json.dumps({'elapsed_seconds':time.monotonic()-start,'host_available_bytes':host,'gpu_whole_device_mib':used})+'\n');samples.flush()
                if host<limits['host_available_min_bytes']:raise RuntimeError('host available below existing 3 GiB guard')
                if used>limits['gpu_whole_device_max_mib']:raise RuntimeError('whole GPU above existing 6144 MiB guard')
                if time.monotonic()-start>limits['timeout_seconds_per_phase']:raise RuntimeError('existing 1800 s timeout')
                time.sleep(1)
            record['return_code']=process.wait()
            record['complete']=record['return_code']==0
        except BaseException as error:
            record['failure']=str(error)
            subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=KILL',unit+'.scope'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            record['return_code']=process.wait()
    record['wall_seconds']=time.monotonic()-start
    save(out/'process.json',record)
    if not record['complete']:
        progress['status']='execution_failed';save(base/'full-progress.json',progress)
        raise SystemExit(1)
    worker=json.loads((out/'worker-result.json').read_text())
    events=dict((k,int(v)) for k,v in (line.split() for line in worker['memory_events'].splitlines()))
    assert events['oom']==events['oom_kill']==0
    log_text=(out/'native.log').read_text()
    if 'VUID-' in log_text or 'Validation Error' in log_text:
        progress['status']='validation_failed';save(base/'full-progress.json',progress)
        raise SystemExit(1)
    progress['completed'].append(case)
    save(base/'full-progress.json',progress)
    print('Finished',case,'seconds',round(record['wall_seconds'],2),flush=True)
progress['status']='complete';progress.pop('current',None)
save(base/'full-progress.json',progress)
