"""Read back current sources, executed inputs, and the completed CLI result."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base=Path(__file__).resolve().parent
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
sources=json.loads((base/'source-before.json').read_text())
for name,expected in sources.items():
    assert (root/name).stat().st_size==expected['bytes'] and sha(root/name)==expected['sha256'],name
plan=json.loads((base/'plan.json').read_text())
assert sha(base/'plan.json')=='d223454f05f387e336e5b37718a61c13b9dcf8339fab5671ceb760c1a5a7d5df'
diagnostic=base/'official/diagnostic'
fixture=json.loads((diagnostic/'fixture/fixture.json').read_text())
original=Path('/var/tmp/ernie-runtime-squares-v1/768/official/reference')
saved=json.loads((original/'fixture.json').read_text())
mapping={'in0':saved['inputs']['initial'],'in1':saved['inputs']['padded-text'],
         'in3':saved['inputs']['constant-0'],'in4':saved['inputs']['constant-1'],
         'in5':saved['inputs']['constant-2'],'expected':saved['outputs'][0]['prediction']}
for name,entry in mapping.items():
    actual=fixture['expected'] if name=='expected' else fixture['inputs'][name]
    assert actual['sha256']==entry['sha256']==sha(diagnostic/'fixture'/actual['file'])==sha(original/entry['file'])
for name,expected in plan['package_binding'].items():
    assert fixture['package_binding'][name]==expected,name
for name,digest in fixture['source_snapshot'].items():
    assert sha(diagnostic/'scripts'/name)==digest==sha(base/'source/tools'/name),name
sys.path.insert(0,str(root/'tools'))
from pipeline_reference import full_reference_contract,reviewed_shared_reference
assert full_reference_contract(saved,fixture['config'],saved['prompt'],8)==25
reviewed_shared_reference(original/'fixture.json',fixture['package_binding'],root/'models/turbo-shared-v2/manifest.json')
assert sha(root/'build-dev/ernie-block-sequence-runner')==plan['runner_sha256']
ncnn=root.parents[1]/'third_party/ncnn'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ncnn,text=True).strip()=='6a1bf000f363714839a36793addc8c879d3d899e'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ncnn,text=True).strip()
processes={phase:json.loads((base/phase/'process.json').read_text()) for phase in ('native','official')}
assert all(p['complete'] and p['return_code']==0 for p in processes.values())
report=dict(plan_sha256=sha(base/'plan.json'),source_files_unchanged=len(sources),
 exact_official_copied_inputs=len(mapping)-1,exact_saved_expected=True,
 official_time_features_sha256=fixture['inputs']['in2']['sha256'],
 native_time_features_sha256=sha(base/'native-time.f32'),
 package_source_and_runtime_binding_authenticated=True,
 copied_python_sources_unchanged=len(fixture['source_snapshot']),
 runner_sha256=plan['runner_sha256'],ncnn_clean_pinned=True,
 all_model_commands_complete=True,
 limitation='This authenticates existing records and selected inputs; it does not repeat full model inference or claim complete native quality acceptance.')
with (base/'audit.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
