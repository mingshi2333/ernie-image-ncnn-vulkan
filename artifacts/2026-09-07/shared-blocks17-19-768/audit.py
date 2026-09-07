"""Recompute every selected-block result independently with NumPy FP64."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

B=Path(__file__).resolve().parent
R=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def identity(path):
    path=Path(path);return {'bytes':path.stat().st_size,'sha256':sha(path)}
def read(item):
    path=Path(item['file'])
    assert identity(path)=={k:item[k] for k in ('bytes','sha256')}
    a=np.fromfile(path,'<f4')
    assert a.size==math.prod(item['shape']) and np.isfinite(a).all()
    return a.astype(np.float64)
def metrics(expected,actual):
    assert expected.shape==actual.shape
    delta=actual-expected
    return {'max_abs_error':float(np.abs(delta).max()),'nrmse':float(np.sqrt(np.dot(delta,delta)/max(np.dot(expected,expected),1e-30))),
            'reference_max_abs':float(np.abs(expected).max())}
p=json.loads((B/'plan.json').read_text())
assert sha(B/'plan.json')=='f5a8b08f09cb31b3cb3629ced64c83f9518f3f5baf940193bf501754f2d2075c'
assert p['native_acceptance_eligible'] is False and p['official_gate_applied'] is False and p['indices']==[17,18,19]
for path,item in p['bindings'].items():assert identity(path)==item,path
sources=json.loads((B/'source-before.json').read_text())
for relative,item in sources.items():
    assert identity(R/relative)==identity(B/'source'/relative)==item,relative
processes={}
for phase in ('official','native'):
    process=json.loads((B/phase/'process.json').read_text())
    worker=json.loads((B/phase/'worker-result.json').read_text())
    assert process['complete'] and process['return_code']==0 and process['plan_sha256']==sha(B/'plan.json')
    assert len(worker)==1 and worker[0]['return_code']==0 and worker[0]['command']==p['commands'][phase][0]
    events=dict(line.split() for line in process['memory_events'].splitlines())
    assert all(events[key]=='0' for key in ('max','oom','oom_kill'))
    processes[phase]=process
official=json.loads((B/'official/results/result.json').read_text())
assert [row['block'] for row in official]==p['indices']
for row in official:
    index=row['block'];path=B/'official/results'/f'block-{index}.f32'
    assert row['exact_replay'] and sha(path)==p['tensors'][f'expected-official-{index}']['sha256']==row['output']['sha256']
    assert np.fromfile(path,'<f4').size==2368*4096 and np.isfinite(np.fromfile(path,'<f4')).all()
rows=json.loads((B/'native/results.json').read_text())
assert len(rows)==len(p['cases'])==9
recomputed=[]
for case,row in zip(p['cases'],rows):
    index=case['block'];variant=case['variant'];out=B/'native'/f'block-{index}-{variant}'
    assert (index,variant)==(row['block'],row['variant']) and row['return_code']==0
    assert json.loads((out/'fixture/fixture.json').read_text())==case
    assert set(case['inputs'])=={f'in{i}' for i in range(10)}
    for name,item in case['inputs'].items():
        assert identity(out/'fixture'/(name+'.f32'))=={k:item[k] for k in ('bytes','sha256')}
    runtime=row['runtime']
    assert runtime['block_index']==index and runtime['blocks']==1 and not runtime['dit_heads']
    assert runtime['package_mode'] and runtime['threads']==2 and runtime['host_weights'] and runtime['text_bucket']==32 and runtime['valid_text_tokens']==15
    assert runtime['precision']=='fp32' and runtime['backend']=='vulkan' and runtime['policy']=='stream'
    assert len(runtime['load_seconds'])==len(runtime['compute_seconds'])==1
    assert {f.name for f in (out/'trace').iterdir()}=={f'block-{index}.f32'}
    assert sha(out/'actual.f32')==sha(out/'trace'/f'block-{index}.f32')==row['actual']['sha256']
    expected=read(case['expected']);actual=read(row['actual']);m=metrics(expected,actual)
    assert all(math.isclose(value,row['official_difference'][key],rel_tol=1e-9,abs_tol=1e-12) for key,value in m.items())
    if variant=='native-replay':assert row['native_anchor_exact'] and sha(out/'actual.f32')==case['replay_anchor']['sha256']
    recomputed.append({'block':index,'variant':variant,'values':actual.size,'actual_sha256':sha(out/'actual.f32'),**m})
comparisons=[]
for index in p['indices']:
    family={r['variant']:r for r in recomputed if r['block']==index}
    a=family['native-replay'];b=family['official-cpu-rope'];c=family['official-cuda-rope']
    cpu_path=B/'native'/f'block-{index}-official-cpu-rope'/'actual.f32'
    cuda_path=B/'native'/f'block-{index}-official-cuda-rope'/'actual.f32'
    cpu=np.fromfile(cpu_path,'<f4').astype(np.float64);cuda=np.fromfile(cuda_path,'<f4').astype(np.float64)
    comparisons.append({'block':index,'accumulated':a,'teacher_cpu_rope':b,'teacher_cuda_rope':c,
        'accumulated_to_aligned_nrmse_ratio':a['nrmse']/max(c['nrmse'],1e-30),
        'cpu_to_cuda_rope_output_difference':metrics(cuda,cpu)})
position={}
for name in ('cos','sin'):
    cpu=read(p['tensors'][f'{name}-cpu']);cuda=read(p['tensors'][f'{name}-cuda'])
    position[name]={'values':cpu.size,'different_values':int(np.count_nonzero(cpu!=cuda)),**metrics(cuda,cpu)}
ncnn=R.parents[1]/'third_party/ncnn'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ncnn,text=True).strip()=='6a1bf000f363714839a36793addc8c879d3d899e'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ncnn,text=True).strip()
report={'protocol':p['protocol'],'plan_sha256':sha(B/'plan.json'),'verified_bindings':len(p['bindings']),
    'source_files_current_and_frozen':len(sources),'complete_execution':True,'official_exact_replays':3,'native_exact_replays':3,
    'native_output_values':sum(row['values'] for row in recomputed),'official_output_values':3*2368*4096,
    'native_acceptance_eligible':False,'official_gate_applied':False,'rows':recomputed,'comparisons':comparisons,'position_tables':position,
    'processes':processes,'limitations':'Aligned blocks replace incoming hidden and modulation; these observations do not allocate additive causal shares or prove full-trajectory quality. Old full76818/25 remains unchanged.'}
(B/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'verified_bindings':len(p['bindings']),'native_outputs':report['native_output_values'],
                  'comparisons':comparisons,'position_tables':position},indent=2))
