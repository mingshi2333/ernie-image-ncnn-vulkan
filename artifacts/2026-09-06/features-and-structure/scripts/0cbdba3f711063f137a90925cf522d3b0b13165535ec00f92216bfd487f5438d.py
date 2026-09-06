#!/usr/bin/env python3
"""Build a static resolution/text variant from pinned weights and validate its new graphs.

Run in the reference Python environment after the baseline components have been
prepared. Exports and validation run sequentially; all logs and failed stages
remain in --work. The resulting --output package uses local weight links.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def dimensions(width, height, tokens):
    if any(type(v) is not int for v in (width, height, tokens)):
        raise ValueError('Dimensions and tokens must be integers')
    if (not 16 <= width <= 1024 or not 16 <= height <= 1024 or width % 16 or height % 16
            or not 1 <= tokens <= 2048):
        raise ValueError('Use multiples of 16 in [16,1024] and text tokens in [1,2048]')
    w, h = width//16, height//16
    if w*h + tokens > 6144:
        raise ValueError('Complete attention length exceeds 6144 tokens')
    return w, h, w*h + tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width', type=int, required=True, help='Output pixels')
    p.add_argument('--height', type=int, required=True, help='Output pixels')
    p.add_argument('--text-tokens', type=int, required=True, help='Capacity including BOS, up to 2048')
    p.add_argument('--text-source', type=Path, default=ROOT/'models/text-s64-v2')
    p.add_argument('--dit-source', type=Path, default=ROOT/'models/dit-s4160-residual-v1')
    p.add_argument('--embedding-package', type=Path, default=ROOT/'models/text-s32-v1')
    p.add_argument('--vae-template', type=Path, default=ROOT/'models/vae-8x8-v1')
    p.add_argument('--text-template', type=Path, help='Reuse an independently exported block zero of this exact capacity')
    p.add_argument('--dit-template', type=Path, help='Reuse an independently exported runtime block zero at this exact shape')
    p.add_argument('--runner-dir', type=Path, default=ROOT/'build')
    p.add_argument('--work', type=Path, required=True, help='New directory retaining conversion and validation evidence')
    p.add_argument('--output', type=Path, required=True, help='New model directory')
    args = p.parse_args()
    try:
        w, h, total = dimensions(args.width, args.height, args.text_tokens)
    except ValueError as error:
        p.error(str(error))
    args.work, args.output = args.work.resolve(), args.output.resolve()
    if args.work.exists() or args.output.exists() or args.work == args.output:
        p.error('Use distinct new work and output directories')
    for path in (args.text_source, args.dit_source, args.embedding_package, args.vae_template):
        if not path.is_dir():
            p.error(f'Missing prepared baseline component: {path}')
    for name in ('ernie-text-runner', 'ernie-block-runner', 'ernie-head-runner'):
        if not (args.runner_dir/name).is_file():
            p.error(f'Build {name} first')
    args.work.mkdir(parents=True)
    runners = args.work/'runners'
    runners.mkdir()
    for name in ('ernie-text-runner', 'ernie-block-runner', 'ernie-head-runner'):
        shutil.copy2(args.runner_dir/name, runners/name)
    from package_model import sha256, package_model
    record = {'complete': False, 'resolution': [args.width, args.height],
              'text_tokens': args.text_tokens, 'total_tokens': total,
              'builder_sha256': sha256(__file__), 'stages': [],
              'scope': 'Static component FP32 validation; full prompt-to-image quality is a separate test'}

    def save():
        (args.work/'build.json').write_text(json.dumps(record, indent=2)+'\n')

    def stage(name, script, *arguments):
        command = [sys.executable, str(ROOT/'tools'/script), *map(str, arguments)]
        entry = {'name': name, 'command': command, 'script_sha256': sha256(ROOT/'tools'/script),
                 'complete': False}
        record['stages'].append(entry)
        save()
        print(json.dumps({'stage': name, 'status': 'running'}), flush=True)
        start = time.monotonic()
        with (args.work/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(return_code=result.returncode, seconds=time.monotonic()-start,
                     complete=result.returncode == 0)
        save()
        if result.returncode:
            raise RuntimeError(f'{name} failed; retained log: {args.work/(name+".log")}')

    text_template = args.text_template or args.work/'text-template'
    if args.text_template:
        from rebucket_text import verified
        meta = verified(text_template)
        if meta['block'] != 0 or meta['tokens'] != args.text_tokens or 'fixture.json' not in meta['files']:
            raise ValueError('Require independent block-zero export at the requested capacity')
        record['text_template_manifest_sha256'] = sha256(text_template/'model.json')
    else:
        stage('export-text', 'export_text_block.py', '--tokens', args.text_tokens, '--output', text_template)
    stage('validate-text-template', 'validate_text.py', '--model', text_template,
          '--fixture', text_template, '--cpu-only', '--runner', runners/'ernie-text-runner',
          '--output', args.work/'text-template-validation')
    stage('reuse-text-weights', 'rebucket_text.py', '--source', args.text_source,
          '--template', text_template, '--output', args.work/'text')
    dit_template = args.dit_template or args.work/'dit-template/runtime'
    if args.dit_template:
        from validate_dit_block import verify
        meta, fixture = verify(dit_template, dit_template)
        if meta['block'] != 0 or fixture['grid'] != [h, w] or fixture['text_tokens'] != args.text_tokens:
            raise ValueError('Require independent block-zero DiT export at the requested shape')
        record['dit_template_manifest_sha256'] = sha256(dit_template/'model.json')
    else:
        stage('export-dit', 'export_dit_block.py', '--width', w, '--height', h,
              '--text-tokens', args.text_tokens, '--valid-text', min(6, args.text_tokens), '--output', dit_template.parent)
    stage('validate-dit-template', 'validate_dit_block.py', '--model', dit_template,
          '--fixture', dit_template, '--backend', 'cpu', '--precision', 'fp32',
          '--runner', runners/'ernie-block-runner', '--output', args.work/'dit-template-validation')
    dit_args = [value for i in range(36) for value in ('--model', args.dit_source/f'block-{i:02d}')]
    stage('reuse-dit-weights', 'rebucket_dit.py', *dit_args, '--template', dit_template,
          '--fp32-residual', '--output', args.work/'dit')
    stage('export-heads', 'export_dit_heads.py', '--width', w, '--height', h,
          '--text-tokens', args.text_tokens, '--output', args.work/'heads')
    for kind in ('input', 'output'):
        stage('validate-'+kind, 'validate_dit_heads.py', '--model', args.work/'heads'/kind,
              '--output', args.work/(kind+'-validation'), '--cpu-only', '--runner', runners/'ernie-head-runner')
    stage('reference-vae', 'export_vae.py', '--width', w*2, '--height', h*2,
          '--reference-only', '--output', args.work/'vae-reference')
    stage('specialize-vae', 'specialize_vae.py', '--template', args.vae_template,
          '--reference', args.work/'vae-reference', '--output', args.work/'vae')
    stage('validate-vae', 'validate_dit_heads.py', '--model', args.work/'vae',
          '--output', args.work/'vae-validation', '--cpu-only', '--vae-convolution', 'direct',
          '--runner', runners/'ernie-head-runner')
    component_args = [value for kind, count in (('text', 25), ('dit', 36))
                      for i in range(count) for value in ('--'+kind, args.work/kind/f'block-{i:02d}')]
    stage('assemble', 'prepare_pipeline.py', *component_args, '--input-head', args.work/'heads/input',
          '--output-head', args.work/'heads/output', '--vae', args.work/'vae',
          '--embedding-package', args.embedding_package, '--output', args.work/'assembled')
    package_model(args.work/'assembled', args.output, link=True)
    record.update(complete=True, package=str(args.output), manifest_sha256=sha256(args.output/'manifest.json'))
    save()
    print(json.dumps({'complete': True, 'package': str(args.output), 'evidence': str(args.work/'build.json')}), flush=True)


if __name__ == '__main__':
    main()
