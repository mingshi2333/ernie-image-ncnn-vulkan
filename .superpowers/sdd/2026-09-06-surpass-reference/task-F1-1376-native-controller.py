#!/usr/bin/env python3
"""One explicitly scheduled, fixed CPU decoder validation; never prepare or promote."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(plan):
    for name, row in plan['files'].items():
        p = Path(name)
        if p.stat().st_size != row['size_bytes'] or digest(p) != row['sha256']:
            raise ValueError('Frozen file changed: ' + name)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('Host available memory missing')


def controls(plan):
    relative = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / relative.lstrip('/')
    actual = {key: (cg / key).read_text().strip() for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
    if (actual != {'memory.max': str(16 * 1024**3), 'memory.swap.max': '0', 'cpu.max': '200000 100000'}
        or sorted(os.sched_getaffinity(0)) != [12, 14]
        or cg.name != plan['unit'] + '.scope'):
        raise ValueError('Exact dedicated memory/swap/CPU scope required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden')
    return cg, actual


def terminate_scope_children(cg):
    # Validator and /usr/bin/time/native runner create separate sessions.
    # Kill all other processes in this dedicated scope, not merely validator's PGID.
    for _ in range(20):
        children = [int(x) for x in (cg / 'cgroup.procs').read_text().split() if int(x) != os.getpid()]
        if not children:
            return
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(.05)


def run(path, expected_sha):
    path = Path(path).resolve()
    if digest(path) != expected_sha:
        raise ValueError('Plan SHA differs from externally reviewed identity')
    plan = json.loads(path.read_text())
    base = path.parent
    expected = [plan['python_invocation'], plan['validator'], '--model', plan['candidate'],
                '--runner', plan['runner'], '--output', str(base / 'validation'),
                '--cpu-only', '--vae-convolution', 'direct']
    if plan['argv'] != expected or plan['status'] != 'prepared_not_executed':
        raise ValueError('Fixed validation argv/status differs')
    if os.path.abspath(sys.executable) != plan['python_invocation'] or sys.prefix != plan['python_prefix']:
        raise ValueError('Venv invocation or prefix differs')
    cg, actual = controls(plan)
    out = base / 'execution'
    if (base / 'validation').exists():
        raise ValueError('Validation output already exists')
    out.mkdir()  # Reuse is forbidden even after a failed attempt.
    result = {'status': 'failed', 'plan_sha256': expected_sha, 'controls': actual,
              'affinity': sorted(os.sched_getaffinity(0)), 'argv': expected,
              'native_ncnn_threads': 4, 'scope_cpu_budget': 2, 'native_acceptance_eligible': False,
              'formal_performance_eligible': False, 'peak_cgroup_memory_current': 0,
              'host_min_observed': available(), 'exit_code': None, 'failure': None}
    process = None
    start = time.monotonic_ns()
    result['started_monotonic_ns'] = start
    try:
        verify(plan)
        if available() < 3 * 1024**3:
            raise RuntimeError('Host floor before model execution')
        with (out / 'validator.log').open('xb') as log, (out / 'samples.jsonl').open('x') as samples:
            process = subprocess.Popen(expected, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['validator_pid'] = process.pid
            result['controller_pid'] = os.getpid()
            result['validator_proc_stat'] = Path(f'/proc/{process.pid}/stat').read_text()
            while process.poll() is None:
                now = time.monotonic_ns()
                host = available()
                memory = int((cg / 'memory.current').read_text())
                result['host_min_observed'] = min(result['host_min_observed'], host)
                result['peak_cgroup_memory_current'] = max(result['peak_cgroup_memory_current'], memory)
                samples.write(json.dumps({'monotonic_ns': now, 'host_available': host, 'memory_current': memory}) + '\n')
                samples.flush()
                if host < 3 * 1024**3:
                    raise RuntimeError('Host available floor crossed')
                if now - start > 1800 * 10**9:
                    raise RuntimeError('1800-second outer timeout exceeded')
                time.sleep(.05)
            result['exit_code'] = process.wait()
        if result['exit_code'] != 0:
            raise RuntimeError('Native validation failed; preserve original fixed gates and logs')
        matrix = json.loads((base / 'validation/matrix.json').read_text())
        if (len(matrix) != 1 or not matrix[0]['passed'] or matrix[0]['backend'] != 'cpu'
            or matrix[0]['precision'] != 'fp32' or set(matrix[0]['outputs']) != {'out0'}):
            raise ValueError('Fixed CPU FP32 output denominator/status differs')
        verify(plan)
        if digest(base / 'validation/ernie-head-runner.snapshot') != plan['files'][plan['runner']]['sha256']:
            raise ValueError('Actually copied runner differs')
        events = dict(x.split() for x in (cg / 'memory.events').read_text().splitlines())
        if any(int(events[k]) for k in ('oom', 'oom_kill', 'oom_group_kill')):
            raise ValueError('Scope OOM observed')
        result['status'] = 'passed_fixed_component_pending_independent_review'
    except BaseException as error:
        result['failure'] = str(error)
        terminate_scope_children(cg)
        if process is not None:
            result['exit_code'] = process.wait()
        raise
    finally:
        result['finished_monotonic_ns'] = time.monotonic_ns()
        result['wall_seconds'] = (result['finished_monotonic_ns'] - start) / 1e9
        result['memory_events'] = (cg / 'memory.events').read_text()
        result['remaining_scope_pids'] = [int(x) for x in (cg / 'cgroup.procs').read_text().split()]
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    args = p.parse_args()
    run(args.plan, args.plan_sha256)
