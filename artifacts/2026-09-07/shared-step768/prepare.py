import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base = Path(__file__).resolve().parent
previous = Path('/var/tmp/ernie-runtime-squares-v1')
case = previous / '768'
sys.path.insert(0, str(root / 'tools'))
from source_inventory import source_files
from pipeline_reference import full_reference_contract, reviewed_shared_reference
from pipeline_package import select_shared_instance

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

source_before = json.loads((base / 'source-before.json').read_text())
for name, item in source_before.items():
    assert (root / name).stat().st_size == item['bytes'] and sha(root / name) == item['sha256'], name
assert sha(base / 'native-time.f32') == '5919ea0805bf18fa0de6fcb1ebf080cd7e62af72ec411a57f5f9dce71cd604c1'
for name in ('src/dit.cpp', 'src/block_sequence.cpp', 'src/denoiser.cpp', 'src/ernie_attention.cpp',
             'src/component_files.cpp', 'src/shape_graph.cpp', 'src/latent_ops.cpp'):
    assert sha(root / name) == sha(previous / 'source' / name), name
shared_path = root / 'models/turbo-shared-v2/manifest.json'
shared = json.loads(shared_path.read_text())
ref_path = case / 'official/reference/fixture.json'
ref = json.loads(ref_path.read_text())
review = json.loads((previous / 'review768.json').read_text())
assert review['complete_execution_both'] and review['passed_tensors'] == 18
selected, target = select_shared_instance(shared['instances'], 768, 768, len(ref['ids']))
assert full_reference_contract(ref, selected['config'], ref['prompt'], 8) == 25
binding = dict(source_manifest_sha256=selected['source_manifest_sha256'],
               shared_manifest_sha256=sha(shared_path), runtime_bindings=selected['runtime_bindings'],
               runtime_target=target)
reviewed_shared_reference(ref_path, binding, shared_path)
assert len(ref['ids']) == 15 and selected['config']['text_bucket'] == 32

torch.set_num_threads(2)
torch.set_grad_enabled(False)
schedule = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.)
schedule.set_timesteps(sigmas=torch.linspace(1., 0., 9)[:-1], device='cpu')
timestep = schedule.timesteps[6].reshape(1)
features = get_timestep_embedding(timestep, 4096, flip_sin_to_cos=False, downscale_freq_shift=0)
assert timestep.item() == json.loads((base / 'native-time.json').read_text())['timestep']
features.numpy().astype('<f4').tofile(base / 'official-time.f32')

for path in source_files(root):
    dest = base / 'source' / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    dest.chmod(0o444)
shutil.copy2(root / 'build-dev/ernie-block-sequence-runner', base / 'runner.snapshot')
(base / 'runner.snapshot').chmod(0o555)

native = case / 'native/trace'
official = case / 'official/reference'
tensor_shapes = {'in0': [1,128,48,48], 'in1': [1,64,3072], 'in2': [1,4096],
                 'in3': [1,2368,128], 'in4': [1,2368,128], 'in5': [2368,2368],
                 'expected': [1,128,48,48]}
native_hashes = {row['name']: row['native_sha256'] for row in json.loads((case / 'comparison.json').read_text())['comparisons']}
mapping = {'in0': 'step-5', 'in1': 'padded-text', 'in3': 'constant-0',
           'in4': 'constant-1', 'in5': 'constant-2', 'expected': 'prediction-6'}
commands, fixtures = {}, {}
for phase, directory in (('native', native), ('official', official)):
    fixture_dir = base / (phase + '-fixture')
    fixture_dir.mkdir()
    entries = {}
    for name, shape in tensor_shapes.items():
        source = base / (phase + '-time.f32') if name == 'in2' else directory / (mapping[name] + '.f32')
        raw = source.read_bytes()
        assert len(raw) == int(np.prod(shape)) * 4 and np.isfinite(np.frombuffer(raw, '<f4')).all()
        if name != 'in2':
            if phase == 'native': expected_hash = native_hashes[mapping[name]]
            elif name == 'in0': expected_hash = ref['outputs'][5]['step']['sha256']
            elif name == 'expected': expected_hash = ref['outputs'][6]['prediction']['sha256']
            else: expected_hash = ref['inputs'][mapping[name]]['sha256']
            assert sha(source) == expected_hash
        dest = fixture_dir / (name + '.f32'); dest.write_bytes(raw); dest.chmod(0o444)
        entries[name] = dict(file=dest.name, shape=shape, sha256=sha(dest), source=str(source))
    fixtures[phase] = dict(inputs={k:v for k,v in entries.items() if k != 'expected'}, expected=entries['expected'])
    (fixture_dir / 'fixture.json').write_text(json.dumps(fixtures[phase], indent=2) + '\n')
    commands[phase] = [[str(base / 'runner.snapshot'), '--package', str(shared_path.parent),
        '--fixture', str(fixture_dir), '--output', str(base / phase / 'actual.f32'),
        '--width', '48', '--height', '48', '--text-tokens', '64', '--tokens', '2368',
        '--valid-text-tokens', '15', '--threads', '2', '--backend', 'vulkan',
        '--precision', 'fp32', '--policy', 'stream', '--host-weights']]

# Reuse the preceding guarded single-command worker; phase names here refer to
# saved input origin. Both phases execute the same native Vulkan probe.
worker = (case / 'run.py').read_text()
(base / 'run.py').write_text(worker)
supervisor = (case / 'supervisor.py').read_text().replace('ernie-square-768-v1-', 'ernie-step768-v1-')
(base / 'supervisor.py').write_text(supervisor)
old_plan = json.loads((case / 'plan.json').read_text())
bound = [*list((base / 'source').rglob('*')), base / 'runner.snapshot', base / 'time-features.snapshot',
         base / 'prepare.py', base / 'run.py', base / 'supervisor.py', base / 'source-before.json',
         base / 'native-time.json', base / 'native-time.f32', base / 'official-time.f32',
         shared_path, ref_path, case / 'comparison.json', previous / 'review768.json', case / 'plan.json',
         Path(inspect.getsourcefile(get_timestep_embedding)), Path(inspect.getsourcefile(FlowMatchEulerDiscreteScheduler))]
for phase in commands:
    bound.extend((base / (phase + '-fixture')).iterdir())
bound = sorted(set(path for path in bound if path.is_file()))
plan = dict(protocol='shared-768-step6-replay-and-teacher-v1',
            scope='Two single native Vulkan FP32 predictions using saved native versus official inputs. Teacher execution requires exact native replay. No free-running acceptance, performance or OOM recovery claim.',
            source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            native_acceptance_eligible=False, step=6, steps=8, runtime_size=[768,768],
            source_config=selected['config'], reference_fixture_sha256=sha(ref_path),
            original_full_runner_sha256=sha(previous / 'ernie-image.snapshot'), runner_sha256=sha(base / 'runner.snapshot'),
            resource_limits=old_plan['resource_limits'], commands=commands, fixtures=fixtures,
            criteria=dict(native='Exact saved native prediction bytes', official=old_plan['gates']['fp32']),
            bindings={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in bound})
(base / 'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'runner':plan['runner_sha256'],'plan':sha(base/'plan.json'),'bindings':len(bound),
                  'native_time':sha(base/'native-time.f32'),'official_time':sha(base/'official-time.f32')}))
