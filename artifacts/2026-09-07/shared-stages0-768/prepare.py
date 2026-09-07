"""Freeze the existing stage command with a shared 768 source and old anchors."""
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModel

base=Path(__file__).resolve().parent
root=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
previous=Path('/var/tmp/ernie-step0-768-v1')
case=Path('/var/tmp/ernie-runtime-squares-v1/768')
sys.path.insert(0,str(root/'tools'))
from source_inventory import source_files
from pipeline_package import select_shared_instance
from pipeline_reference import full_reference_contract,reviewed_shared_reference
from diagnose_dit_stages import stage_weight_source

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert json.loads((base/'tests.json').read_text())['return_code']==0
shared=root/'models/turbo-shared-v2/manifest.json'
manifest=json.loads(shared.read_text())
reference=case/'official/reference'
saved=json.loads((reference/'fixture.json').read_text())
selected,target=select_shared_instance(manifest['instances'],768,768,len(saved['ids']))
binding=dict(source_manifest_sha256=selected['source_manifest_sha256'],shared_manifest_sha256=sha(shared),
             runtime_bindings=selected['runtime_bindings'],runtime_target=target)
assert full_reference_contract(saved,selected['config'],saved['prompt'],8)==25
reviewed_shared_reference(reference/'fixture.json',binding,shared)
weights,provenance=stage_weight_source(shared.parent,saved,binding)
assert provenance['identical_runtime_assets']==71 and len(weights)==36
sources={}
for path in source_files(root):
    name=path.relative_to(root);destination=base/'source'/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,destination);destination.chmod(0o444)
    sources[str(name)]=dict(bytes=path.stat().st_size,sha256=sha(path))
(base/'source-before.json').write_text(json.dumps(sources,indent=2)+'\n')
shutil.copyfile(previous/'runner.snapshot',base/'runner.snapshot');(base/'runner.snapshot').chmod(0o555)
assert sha(base/'runner.snapshot')=='1d754f49534434cf8e3882b271714060394f5233b6a13eb3a0cf2c7e7bcb665a'
shutil.copyfile(previous/'run.py',base/'run.py')
(base/'supervisor.py').write_text((previous/'supervisor.py').read_text().replace('ernie-step0-768-v1-','ernie-stages0-768-v1-'))
command=[str(root/'.venv/bin/python'),str(base/'source/tools/diagnose_dit_stages.py'),
 '--model',str(shared.parent),'--reference',str(reference),'--step','0',
 '--output',str(base/'native/diagnostic'),'--runner',str(base/'runner.snapshot'),
 '--official-root',str((root/'models/official').resolve()),'--threads','2','--host-weights',
 '--precision','fp32','--reference-device','cuda']
bound=[p for p in (base/'source').rglob('*') if p.is_file()]
bound.extend(base/name for name in ('prepare.py','run.py','supervisor.py','source-before.json','runner.snapshot','tests.json'))
bound.extend([shared,reference/'fixture.json',previous/'official-comparison.json',previous/'official/diagnostic/native/actual.f32',case/'plan.json'])
bound.extend(reference/e['file'] for e in saved['inputs'].values())
bound.extend(reference/e['file'] for group in saved['outputs'] for e in group.values())
bound.extend(reference/e['file'] for e in saved['final'].values())
bound.extend(Path(inspect.getsourcefile(obj)) for obj in (get_timestep_embedding,FlowMatchEulerDiscreteScheduler,ErnieImageTransformer2DModel))
old_plan=json.loads((case/'plan.json').read_text())
original_manifests={name:item for name,item in old_plan['bindings'].items() if '/models/official/dit-' in name}
assert len(original_manifests)==42
for name,item in original_manifests.items():
    path=Path(name);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
    bound.append(path)
for digest in (binding['source_manifest_sha256'],provenance['source_manifest_sha256']):
    bound.append(shared.parent/'objects'/digest)
bound=sorted(set(bound))
plan=dict(protocol='shared-768-step0-stage-trace-v1',step=0,steps=8,source_count=len(sources),
 source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 scope='Actual existing stage command: official CPU heads/CUDA streamed blocks exits before native Vulkan trace. No text/Euler/VAE/full generation. Preserve old free-running18/25.',
 native_acceptance_eligible=False,package_binding=binding,official_weight_provenance=provenance,
 reference_fixture_sha256=sha(reference/'fixture.json'),runner_sha256=sha(base/'runner.snapshot'),
 commands={'native':[command]},resource_limits=old_plan['resource_limits'],
 criteria=dict(stage_count=44,finite_and_shape_checked=True,
  official_prediction_replay_sha256=saved['outputs'][0]['prediction']['sha256'],
  native_prediction_replay_sha256=sha(previous/'official/diagnostic/native/actual.f32'),
  prediction_gate=old_plan['gates']['fp32'],
  stage_metrics='Descriptive per-stage/image/text differences; no new stage gate or additive causal attribution'),
 bindings={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in bound})
(base/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'plan_sha256':sha(base/'plan.json'),'bindings':len(bound),'source_count':len(sources),'official_manifests':len(original_manifests)}))
