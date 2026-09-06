"""Assemble the previously exercised fixed graphs without changing schema-3."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from package_model import sha256, verify_package, runtime_files
from source_inventory import source_files

source = ROOT / 'models/turbo1024-s64-portable'
candidate = ROOT / 'outputs/f1-shape1376-s64-plan-v5'
shape_plan = json.loads((candidate / 'plan.json').read_text())
assert sha256(source / 'manifest.json') == shape_plan['source_manifest_sha256']
manifest, files = verify_package(source)
graphs = {item['path']: item for item in shape_plan['graphs']}
assert len(graphs) == 64
model = BASE / 'model'
model.mkdir()
for name in runtime_files():
    dest = model / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if name in graphs:
        assert sha256(source / name) == graphs[name]['source_sha256']
        assert sha256(candidate / 'graphs' / name) == graphs[name]['candidate_sha256']
        shutil.copy2(candidate / 'graphs' / name, dest)
    elif name == 'model.cfg':
        dest.write_text(''.join(f'{k} {v}\n' for k, v in shape_plan['target_config'].items()))
    else:
        dest.symlink_to((source / name).resolve())
manifest.update(portable=False, scope='Fixed 1376x768/s64 development package; linked original weights; full-image validation pending',
                config=shape_plan['target_config'],
                files={name: sha256(model / name) for name in runtime_files()},
                file_sizes={name: (model / name).stat().st_size for name in runtime_files()},
                provenance={'source_manifest_sha256': sha256(source / 'manifest.json'),
                            'shape_plan_sha256': sha256(candidate / 'plan.json'),
                            'assembler_sha256': sha256(__file__),
                            'schema3_registry_unchanged': True})
(model / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
verify_package(model)

bindings = {}
def bind(path):
    path = Path(path).absolute()
    bindings[str(path)] = {'sha256': sha256(path), 'bytes': path.stat().st_size}

snapshot = BASE / 'source'
paths = source_files(ROOT)
for path in paths:
    dest = snapshot / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    assert sha256(dest) == sha256(path)
    bind(dest)
(snapshot / 'models').symlink_to((ROOT / 'models').resolve(), target_is_directory=True)
original_run = ROOT / 'outputs/execution-metrics-component64-o2-v1'
runner = BASE / 'ernie-image.snapshot'
shutil.copy2(original_run / 'ernie-image-on', runner)
assert sha256(runner) == 'a336a60620489a890c7c5d32a594af5480b268a6eb9c51179fc19bee30bb147a'
bind(runner)
for name in ('plan.json', 'ncnn-base.json', 'ncnn-derived-on.json', 'cmake-cache-on.txt'):
    dest = BASE / ('runner-' + name)
    shutil.copy2(original_run / name, dest)
    bind(dest)
for name in [*runtime_files(), 'manifest.json']:
    bind(model / name)
noise = ROOT / 'outputs/f1-heads1376-preparation-v1/official-input/reference/input/in0.f32'
assert sha256(noise) == 'b2a384ece486abdaa5cd3a110307f8a9b1b934c591e4a8bbcaa82902876ffe20'
shutil.copy2(noise, BASE / 'initial.f32')
(BASE / 'prompt.txt').write_text('A red apple on a wooden table, soft daylight, realistic photo.', encoding='utf-8')
bind(BASE / 'initial.f32')
bind(BASE / 'prompt.txt')
official = ROOT / 'models/official'
stems = [*(f'text-block-{i:02d}' for i in range(25)), *(f'dit-block-{i:02d}' for i in range(36)),
         'text-embed', *(f'dit-{s}' for s in ('x_embedder', 'text_proj', 'time_embedding', 'adaLN_modulation', 'final_norm', 'final_linear')),
         'vae-decoder', 'vae-post-quant']
for stem in stems:
    meta_path = official / (stem + '.manifest.json')
    meta = json.loads(meta_path.read_text())
    path = official / (stem + '.safetensors')
    assert sha256(path) == meta['sha256']
    assert meta['revision'] == manifest['official_model_revision']
    bind(path)
    bind(meta_path)
for name in ('text_encoder-config.json', 'vae-config.json'):
    bind(official / name)
for path in (ROOT / 'models/tokenizer').iterdir():
    if path.is_file():
        bind(path)
runtime = json.loads((ROOT / 'outputs/f2-positive05-1024x1024-v1/official-plan/runtime-identity.json').read_text())
runtime['scope'] = 'Selected actual official implementation files and installed versions; not a sealed Python/CUDA dependency closure'
site = ROOT / '.venv/lib/python3.14/site-packages'
for name, path in [('dit', site / 'diffusers/models/transformers/transformer_ernie_image.py'),
                   ('text', site / 'transformers/models/mistral/modeling_mistral.py')]:
    runtime['sources'][name] = {'path': str(path.resolve()), 'sha256': sha256(path)}
for item in runtime['sources'].values():
    assert sha256(item['path']) == item['sha256']
    bind(item['path'])
(BASE / 'runtime-identity.json').write_text(json.dumps(runtime, indent=2) + '\n')
bind(BASE / 'runtime-identity.json')
for name in ('prepare.py', 'run.py', 'supervisor.py'):
    bind(BASE / name)
python = str(ROOT / '.venv/bin/python')
common = [python, str(snapshot / 'tools/validate_pipeline.py'), '--model', str(model),
          '--runner', str(runner), '--prompt-file', str(BASE / 'prompt.txt'), '--precision', 'fp32',
          '--width', '1376', '--height', '768', '--steps', '8', '--latent', str(BASE / 'initial.f32')]
plan = {'schema_version': 1, 'scope': 'One complete fixed 1376x768/s64 native prompt-to-PNG development fixture versus official FP32 modules',
        'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_file_count': len(paths), 'model_file_count': 136, 'model_manifest_sha256': sha256(model / 'manifest.json'),
        'source_manifest_sha256': shape_plan['source_manifest_sha256'],
        'runner_sha256': sha256(runner), 'runner_provenance': str(original_run),
        'mapped_loading': False, 'text_down_reduction': 'vector', 'pe': False, 'cfg': 1,
        'source_inventory_scope': 'All current repository source/build directories; compiled runner separately bound to its preserved O2 build evidence',
        'initial_noise_provenance': str(noise), 'config': manifest['config'],
        'gates': {'fp32': {'nrmse': .003, 'global_rtol': .01, 'atol': .0002, 'pixel_mae': .1, 'pixel_max': 2},
                  'conditioning': {'nrmse': .0002, 'global_rtol': .0002, 'atol': .0002}},
        'resource_limits': {'cpu_affinity': [4, 6], 'cpu_quota_percent': 200, 'memory_max_bytes': 16 * 1024**3,
                            'memory_swap_max_bytes': 0, 'host_available_min_bytes': 3 * 1024**3,
                            'gpu_whole_device_max_mib': 6144, 'timeout_seconds_per_phase': 1800,
                            'host_poll_seconds': .05, 'gpu_poll_seconds': .5},
        'commands': {'official': [*common, '--output', str(BASE / 'official/validation'), '--reference-device', 'cuda', '--reference-only'],
                     'native': [*common, '--output', str(BASE / 'native/validation'), '--reference', str(BASE / 'official/validation/reference'), '--text-down-vector']},
        'bindings': bindings}
(BASE / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'prepared': True, 'source_files': len(paths), 'bindings': len(bindings), 'plan_sha256': sha256(BASE / 'plan.json')}), flush=True)
