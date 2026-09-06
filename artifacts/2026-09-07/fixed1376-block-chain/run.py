#!/usr/bin/env python3
"""Fixed 4192-token block then connected-chain experiment, using existing guards."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def reference(path, expected):
    if not expected or sha(path / 'fixture.json') != expected:
        raise ValueError('Previously reviewed reference required')
    fixture = json.loads((path / 'fixture.json').read_text())
    for entry in [*fixture['inputs'].values(), fixture['expected']]:
        if Path(entry['file']).name != entry['file'] or sha(path / entry['file']) != entry['sha256']:
            raise ValueError('Reference tensor changed')
    if fixture['tokens'] != 4192:
        raise ValueError('Wrong fixed token count')
    return fixture


def execute(args, plan):
    sys.path.insert(0, str(BASE / 'source/tools'))
    out = BASE / args.phase
    if args.phase == 'official-single':
        sys.argv = [str(BASE / 'source/tools/export_dit_block.py'), '--block', '0',
                    '--weights', plan['blocks'][0]['official_weights'], '--output', str(out / 'reference'),
                    '--height', '48', '--width', '86', '--text-tokens', '64', '--valid-text', '62',
                    '--seed', '20260905', '--fixture-only']
        runpy.run_path(sys.argv[0], run_name='__main__')
        return
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    if args.phase == 'official-chain':
        single = BASE / 'official-single/reference'
        original = reference(single, args.reference_sha256)
        if not json.loads((BASE / 'native-single/validation/result.json').read_text())['passed']:
            raise ValueError('The actual first block must pass before the full chain')
        from export_dit_block import load_block, make_inputs, save_tensor
        inputs, freqs = make_inputs(48, 86, 64, 62, 20260905)
        ref = out / 'reference'
        ref.mkdir()
        entries = {f'in{i}': save_tensor(ref / f'in{i}.f32', value) for i, value in enumerate(inputs)}
        if any(entries[k]['sha256'] != v['sha256'] for k, v in original['inputs'].items()):
            raise ValueError('Synthetic inputs must reproduce the saved first-block fixture exactly')
        state = {'tokens': 4192, 'grid': [48, 86], 'text_tokens': 64, 'valid_text': 62,
                 'blocks': list(range(36)), 'inputs': entries, 'gates': plan['chain_gates'],
                 'scope': 'Connected 36 real-weight blocks; synthetic hidden state and AdaLN, no heads/Euler/VAE',
                 'first_block_reference_sha256': args.reference_sha256, 'reference_stages': [],
                 'first_block_reused': True, 'new_official_block_forwards': 35,
                 'official_revision': plan['official_revision'],
                 'reference_source_sha256': original['reference_source_sha256']}
        current = torch.from_numpy(np.fromfile(single / original['expected']['file'], '<f4')
                                   .reshape(original['expected']['shape'])).transpose(0, 1)
        for index in range(36):
            start = time.monotonic()
            if index:
                block, manifest = load_block(Path(plan['blocks'][index]['official_weights']), index)
                if manifest['sha256'] != plan['blocks'][index]['official_sha256'] or manifest['revision'] != plan['official_revision']:
                    raise ValueError('Official block weight identity differs')
                current = block(current, freqs, list(inputs[1:7]), attention_mask=inputs[-1][None, None])
                del block
            entry = save_tensor(ref / f'after-block-{index:02d}.f32', current.transpose(0, 1))
            if not np.isfinite(current.numpy()).all():
                raise ValueError('Non-finite official chain')
            state['reference_stages'].append({'block': index, 'output': entry,
                                              'elapsed_seconds': time.monotonic() - start})
            state['expected'] = entry
            (ref / 'fixture.json').write_text(json.dumps(state, indent=2) + '\n')
            print(json.dumps({'official_block': index, 'sha256': entry['sha256']}), flush=True)
        return
    from validate_block_sequence import run
    is_single = args.phase == 'native-single'
    ref = BASE / ('official-single' if is_single else 'official-chain') / 'reference'
    fixture = reference(ref, args.reference_sha256)
    count = 1 if is_single else 36
    if not is_single and len(fixture['reference_stages']) != 36:
        raise ValueError('Incomplete official chain')
    models = [Path(x['candidate']) for x in plan['blocks'][:count]]
    result = run(models, ref, fixture, out / 'validation', Path(plan['runner']), 'vulkan', 'fp32', 'stream',
                 ['--trace-dir', str(out / 'trace')])
    if not result['passed']:
        raise RuntimeError('Original final-output gate failed; preserve the negative result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['official-single', 'native-single', 'official-chain', 'native-chain'], required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if sha(BASE / 'plan.json') != args.plan_sha256:
        raise ValueError('Frozen plan changed')
    plan = json.loads((BASE / 'plan.json').read_text())
    if args.execute:
        execute(args, plan)
        return
    if sha(plan['guard']) != '98770dd6dacfafd9b7296a4a8b790e2025a123e4a04213d746bc5c6aca8b1b1a':
        raise ValueError('Existing resource guard changed')
    spec = importlib.util.spec_from_file_location('guard', plan['guard'])
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if os.path.abspath(sys.executable) != plan['python_invocation'] or sys.prefix != plan['python_prefix']:
        raise ValueError('Venv invocation changed')
    cg, controls = guard.controls(plan)
    out = BASE / args.phase
    out.mkdir()
    files = {**plan['files'], **plan['phase_files'][args.phase]}
    result = {'status': 'failed', 'phase': args.phase, 'plan_sha256': args.plan_sha256, 'controls': controls,
              'peak_cgroup_memory_current': 0, 'host_min_observed': guard.available(), 'error': None,
              'scope_cpu_budget': 2, 'native_ncnn_threads': 4,
              'official_torch_threads': 4 if args.phase == 'official-single' else 2,
              'reference_sha256': args.reference_sha256, 'sampled_whole_gpu_mib': None}
    process = sampler = None
    start = time.monotonic()
    try:
        guard.verify({'files': files})
        argv = [plan['python_invocation'], str(BASE / 'run.py'), *sys.argv[1:], '--execute']
        result['command'] = argv
        with (out / 'worker.log').open('xb') as log, (out / 'samples.jsonl').open('x') as samples, (out / 'gpu-memory.log').open('xb') as gpu:
            if args.phase.startswith('native'):
                sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '--id=0', '--loop-ms=500'], stdout=gpu, stderr=subprocess.DEVNULL)
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                now = time.monotonic();host = guard.available();memory = int((cg / 'memory.current').read_text())
                result['host_min_observed'] = min(result['host_min_observed'], host)
                result['peak_cgroup_memory_current'] = max(result['peak_cgroup_memory_current'], memory)
                samples.write(json.dumps({'seconds': now - start, 'host_available': host, 'memory_current': memory}) + '\n');samples.flush()
                readings = [int(s) for s in (out / 'gpu-memory.log').read_text().splitlines() if s.isdigit()]
                if readings:
                    result['sampled_whole_gpu_mib'] = max(readings)
                    if max(readings) > 6144:raise RuntimeError('Whole GPU 6144 MiB guard exceeded')
                if host < 3 * 1024**3 or now - start > 1800:raise RuntimeError('Host floor or 1800-second timeout')
                time.sleep(.05)
            result['exit_code'] = process.wait()
            if sampler is not None:
                sampler.terminate();sampler.wait(timeout=5);sampler = None
        if result['exit_code']:raise RuntimeError('Worker failed; keep original output and gates')
        guard.verify({'files': files})
        result['status'] = 'actual_completed_pending_full_result_review'
    except BaseException as error:
        result['error'] = str(error)
        guard.terminate_scope_children(cg)
        if process is not None:result['exit_code'] = process.wait()
        raise
    finally:
        if sampler is not None:
            sampler.terminate();sampler.wait(timeout=5)
        result['wall_seconds'] = time.monotonic() - start
        result['memory_events'] = (cg / 'memory.events').read_text()
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
