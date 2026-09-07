"""Serial fixed-image worker with the existing cgroup/host/GPU bounds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in ('official', 'native')
plan_path = BASE / 'plan.json'
plan = json.loads(plan_path.read_text())
with plan_path.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
limits = plan['resource_limits']
out = BASE / phase
out.mkdir()
uid = os.getuid()
unit = 'ernie-blocks17-19-768-v1-' + phase
cg = Path(f'/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope')
env = {**os.environ, 'XDG_RUNTIME_DIR': f'/run/user/{uid}', 'DBUS_SESSION_BUS_ADDRESS': f'unix:path=/run/user/{uid}/bus',
       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
command = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + unit,
           '--property=MemoryMax=' + str(limits['memory_max_bytes']), '--property=MemorySwapMax=0', '--property=CPUQuota=200%',
           'taskset', '-c', '4,6', sys.executable, str(BASE / 'run.py'), phase, digest]
def available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def gpu():
    return max(int(v.strip()) for v in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines())
result = {'phase': phase, 'command': command, 'plan_sha256': digest, 'limits': limits,
          'peak_memory_current': 0, 'minimum_host_available': available(), 'peak_gpu_whole_device_mib': gpu(), 'complete': False}
assert result['minimum_host_available'] >= limits['host_available_min_bytes']
assert result['peak_gpu_whole_device_mib'] <= limits['gpu_whole_device_max_mib']
start = time.monotonic()
next_gpu = start
with (out / 'runner.log').open('wb') as log, (out / 'samples.jsonl').open('w') as samples:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            now = time.monotonic()
            host = available()
            result['minimum_host_available'] = min(result['minimum_host_available'], host)
            if host < limits['host_available_min_bytes']:
                raise RuntimeError('Host available memory below 3 GiB')
            if now - start > limits['timeout_seconds_per_phase']:
                raise RuntimeError('Phase exceeded 1800 seconds')
            sample = {'monotonic': now, 'host_available_bytes': host}
            if cg.exists():
                result['cgroup_seen'] = True
                sample['memory_current'] = int((cg / 'memory.current').read_text())
                result['peak_memory_current'] = max(result['peak_memory_current'], sample['memory_current'])
                result['observed_limits'] = {name: (cg / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
                result['memory_events'] = (cg / 'memory.events').read_text()
            if now >= next_gpu:
                sample['whole_device_gpu_mib'] = gpu()
                owned = set((cg / 'cgroup.procs').read_text().split()) if cg.exists() else set()
                raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=3)
                processes = []
                for line in raw.splitlines():
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == 2 and values[0].isdigit() and values[1].isdigit():
                        processes.append({'pid': int(values[0]), 'mib': int(values[1]), 'owned': values[0] in owned})
                sample['compute_processes'] = processes
                result['last_compute_processes'] = processes
                result['peak_owned_compute_mib'] = max(result.get('peak_owned_compute_mib', 0), sum(p['mib'] for p in processes if p['owned']))
                result['peak_gpu_whole_device_mib'] = max(result['peak_gpu_whole_device_mib'], sample['whole_device_gpu_mib'])
                if sample['whole_device_gpu_mib'] > limits['gpu_whole_device_max_mib']:
                    samples.write(json.dumps(sample) + '\n'); samples.flush()
                    raise RuntimeError('Whole-device GPU memory exceeds 6144 MiB')
                next_gpu = now + .5
            samples.write(json.dumps(sample) + '\n')
            samples.flush()
            time.sleep(.05)
        result['return_code'] = process.wait()
        result['complete'] = result['return_code'] == 0 and result.get('cgroup_seen', False)
    except BaseException as error:
        result['failure'] = str(error)
        subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=KILL', unit + '.scope'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result['return_code'] = process.wait()
result['wall_seconds'] = time.monotonic() - start
(out / 'process.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['complete'] else 1)
