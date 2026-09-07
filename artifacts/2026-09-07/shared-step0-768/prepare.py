"""Freeze a first-prediction replay and the shared-package diagnostic CLI."""
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding

root=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base=Path(__file__).resolve().parent
previous=Path('/var/tmp/ernie-runtime-squares-v1')
case=previous/'768'
sys.path.insert(0,str(root/'tools'))
from source_inventory import source_files
from pipeline_reference import full_reference_contract,reviewed_shared_reference
from pipeline_package import select_shared_instance

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

assert json.loads((base/'tests.json').read_text())['return_code']==0
assert json.loads((base/'feature-build-result.json').read_text())['return_code']==0
assert sha(base/'native-time.f32')=='0f8858bd249998b7665bbffed0b0669df9e64b9fe857fb49152cd37ac0977c22'
for name in ('src/dit.cpp','src/block_sequence.cpp','src/denoiser.cpp','src/ernie_attention.cpp',
             'src/component_files.cpp','src/shape_graph.cpp','src/latent_ops.cpp'):
    assert sha(root/name)==sha(previous/'source'/name),name
shared_path=root/'models/turbo-shared-v2/manifest.json'
shared=json.loads(shared_path.read_text())
reference=case/'official/reference'
saved=json.loads((reference/'fixture.json').read_text())
selected,target=select_shared_instance(shared['instances'],768,768,len(saved['ids']))
binding=dict(source_manifest_sha256=selected['source_manifest_sha256'],
             shared_manifest_sha256=sha(shared_path),runtime_bindings=selected['runtime_bindings'],runtime_target=target)
assert full_reference_contract(saved,selected['config'],saved['prompt'],8)==25
reviewed_shared_reference(reference/'fixture.json',binding,shared_path)
assert len(saved['ids'])==15 and selected['config']['text_bucket']==32

sources={}
for path in source_files(root):
    relative=path.relative_to(root)
    destination=base/'source'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,destination);destination.chmod(0o444)
    sources[str(relative)]=dict(bytes=path.stat().st_size,sha256=sha(path))
(base/'source-before.json').write_text(json.dumps(sources,indent=2)+'\n')
shutil.copy2(root/'build-dev/ernie-block-sequence-runner',base/'runner.snapshot')
(base/'runner.snapshot').chmod(0o555)
assert sha(base/'runner.snapshot')=='1d754f49534434cf8e3882b271714060394f5233b6a13eb3a0cf2c7e7bcb665a'

native=case/'native/trace'
old_comparison=json.loads((case/'comparison.json').read_text())
native_hashes={row['name']:row['native_sha256'] for row in old_comparison['comparisons']}
mapping={'in0':'initial','in1':'padded-text','in3':'constant-0','in4':'constant-1','in5':'constant-2','expected':'prediction-0'}
shapes={'in0':[1,128,48,48],'in1':[1,64,3072],'in2':[1,4096],
        'in3':[1,2368,128],'in4':[1,2368,128],'in5':[2368,2368],'expected':[1,128,48,48]}
fixture_dir=base/'native-fixture';fixture_dir.mkdir()
entries={}
for key,shape in shapes.items():
    source=base/'native-time.f32' if key=='in2' else native/(mapping[key]+'.f32')
    raw=source.read_bytes()
    assert len(raw)==int(np.prod(shape))*4 and np.isfinite(np.frombuffer(raw,'<f4')).all()
    if key!='in2':assert sha(source)==native_hashes[mapping[key]],key
    destination=fixture_dir/(key+'.f32');destination.write_bytes(raw);destination.chmod(0o444)
    entries[key]=dict(file=destination.name,shape=shape,sha256=sha(destination),source=str(source))
(fixture_dir/'fixture.json').write_text(json.dumps(entries,indent=2)+'\n')

shutil.copyfile(Path('/var/tmp/ernie-step768-v1/run.py'),base/'run.py')
supervisor=Path('/var/tmp/ernie-step768-v1/supervisor.py').read_text().replace('ernie-step768-v1-','ernie-step0-768-v1-')
(base/'supervisor.py').write_text(supervisor)
commands={
 'native':[[str(base/'runner.snapshot'),'--package',str(shared_path.parent),'--fixture',str(fixture_dir),
            '--output',str(base/'native/actual.f32'),'--width','48','--height','48','--text-tokens','64',
            '--tokens','2368','--valid-text-tokens','15','--threads','2','--backend','vulkan',
            '--precision','fp32','--policy','stream','--host-weights']],
 'official':[[str(root/'.venv/bin/python'),str(base/'source/tools/diagnose_pipeline_step.py'),
              '--model',str(shared_path.parent),'--reference',str(reference),'--step','0',
              '--output',str(base/'official/diagnostic'),'--runner',str(base/'runner.snapshot'),
              '--threads','2','--host-weights','--precision','fp32']]}

bound=[p for p in (base/'source').rglob('*') if p.is_file()]
bound.extend(fixture_dir.iterdir())
bound.extend(base/name for name in ('prepare.py','run.py','supervisor.py','runner.snapshot',
 'time-features.snapshot','native-time.f32','native-time.json','source-before.json','tests.json','feature-checks.json'))
bound.extend([shared_path,reference/'fixture.json',case/'comparison.json',previous/'review768.json',case/'plan.json'])
bound.extend(reference/entry['file'] for entry in saved['inputs'].values())
bound.extend(reference/entry['file'] for group in saved['outputs'] for entry in group.values())
bound.extend(reference/entry['file'] for entry in saved['final'].values())
bound.extend(Path(inspect.getsourcefile(obj)) for obj in (get_timestep_embedding,FlowMatchEulerDiscreteScheduler))
bound=sorted(set(bound))
old_plan=json.loads((case/'plan.json').read_text())
plan=dict(protocol='shared-768-step0-cli-and-replay-v1',step=0,steps=8,
 scope='Two first-prediction native Vulkan FP32 calls. Replay must exactly match the old native output before the all-official-input diagnostic CLI runs. No complete quality, speed or memory-recovery acceptance.',
 native_acceptance_eligible=False,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 source_count=len(sources),runner_sha256=sha(base/'runner.snapshot'),package_binding=binding,
 reference_fixture_sha256=sha(reference/'fixture.json'),native_fixture=entries,
 resource_limits=old_plan['resource_limits'],commands=commands,
 criteria=dict(native='Byte-exact saved native prediction-0',official=old_plan['gates']['fp32']),
 bindings={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in bound})
(base/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'plan_sha256':sha(base/'plan.json'),'sources':len(sources),'bindings':len(bound),'runner_sha256':plan['runner_sha256']}))
