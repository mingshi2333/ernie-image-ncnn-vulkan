"""Bounded same-input checks of shared 768 first-step blocks 17, 18 and 19."""
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
OLD = Path('/var/tmp/ernie-stages0-768-v1')
REFERENCE = Path('/var/tmp/ernie-runtime-squares-v1/768/official/reference')
INDICES = (17, 18, 19)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def identity(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def imports():
    sys.path.insert(0, str(BASE/'source/tools'))
    import numpy as np
    import torch
    from export_dit_block import make_inputs, load_block, save_tensor
    assert Path(inspect.getsourcefile(make_inputs)).resolve() == BASE/'source/tools/export_dit_block.py'
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return np, torch, make_inputs, load_block, save_tensor

def checked_tensor(np, torch, item):
    path = Path(item['file'])
    assert identity(path) == {k:item[k] for k in ('bytes','sha256')}
    assert path.stat().st_size == int(np.prod(item['shape']))*4
    array = np.fromfile(path, '<f4').reshape(item['shape'])
    assert np.isfinite(array).all()
    return torch.from_numpy(array.copy())

def bound_tensor(path, shape):
    return {'file':str(path), 'shape':list(shape), 'dtype':'float32_le', **identity(path)}

def verify(plan):
    for name, expected in plan['bindings'].items():
        assert identity(name) == expected, name

def prepare():
    assert not (BASE/'plan.json').exists()
    sys.path.insert(0,str(ROOT/'tools'))
    from source_inventory import source_files
    sources = {p.relative_to(ROOT).as_posix():identity(p) for p in source_files(ROOT)}
    for relative in sources:
        target = BASE/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
        target.chmod(0o444)
    write(BASE/'source-before.json',sources)
    np,torch,make_inputs,_,save_tensor=imports()
    from diagnose_dit_stages import reference_package, stage_weight_source
    model=ROOT/'models/turbo-shared-v2'
    saved,cfg,binding=reference_package(model,REFERENCE,0)
    weights,provenance=stage_weight_source(model,saved,binding)
    old_fixture=OLD/'native/diagnostic/oracle/fixture'
    f=json.loads((old_fixture/'fixture.json').read_text())
    old_audit=json.loads((OLD/'audit.json').read_text())
    archived=ROOT/'artifacts/2026-09-07/shared-stages0-768'
    inventory=json.loads((archived/'inventory.json').read_text())
    assert identity(old_fixture/'fixture.json')==inventory['native/diagnostic/oracle/fixture/fixture.json']
    assert identity(OLD/'audit.json')==inventory['audit.json']
    assert cfg==f['config'] and f['step']==0 and f['threads']==2 and f['valid_text_tokens']==15
    assert f['reference_fixture_sha256']==sha(REFERENCE/'fixture.json')
    assert weights==f['block_weight_sha256'] and old_audit['official_prediction_exact_replay'] and old_audit['native_prediction_exact_replay']
    stage_rows={row['stage']:row for row in old_audit['stages']}
    inp=BASE/'inputs';inp.mkdir()
    tensors={}
    def copy(name,path,shape,expected):
        assert sha(path)==expected,str(path)
        target=inp/(name+'.f32');shutil.copyfile(path,target)
        tensors[name]=bound_tensor(target,shape)
        checked_tensor(np,torch,tensors[name])
    for index in INDICES:
        for role,number in [('hidden',index-1),('expected',index)]:
            name=f'block-{number}'
            for side in ('official','native'):
                path=(old_fixture if side=='official' else OLD/'native/diagnostic/trace')/(name+'.f32')
                digest=f['stages'][name]['sha256'] if side=='official' else stage_rows[name]['actual_sha256']
                copy(f'{role}-{side}-{index}',path,f['stages'][name]['shape'],digest)
    for number in range(1,7):
        name=f'head-{number+1}'
        for side in ('official','native'):
            path=(old_fixture if side=='official' else OLD/'native/diagnostic/trace')/(name+'.f32')
            digest=f['stages'][name]['sha256'] if side=='official' else stage_rows[name]['actual_sha256']
            copy(f'mod-{side}-{number}',path,[1,1,4096],digest)
    generated,frequencies=make_inputs(48,48,64,15,20260905)
    for offset,name in enumerate(('cos-cpu','sin-cpu','mask')):
        entry=f['inputs'][f'in{offset+3}']
        copy(name,old_fixture/entry['file'],entry['shape'],entry['sha256'])
        assert torch.equal(checked_tensor(np,torch,tensors[name]),generated[offset+7])
    save_tensor(inp/'frequencies.f32',frequencies)
    tensors['frequencies']=bound_tensor(inp/'frequencies.f32',frequencies.shape)
    del generated
    gpu_frequencies=frequencies.to('cuda')
    for name,value in [('cos-cuda',torch.cos(gpu_frequencies)),('sin-cuda',torch.sin(gpu_frequencies))]:
        save_tensor(inp/(name+'.f32'),value[:,:,0,:])
        tensors[name]=bound_tensor(inp/(name+'.f32'),[1,2368,128])
    torch.cuda.synchronize()
    write(BASE/'position-preparation.json',{'scope':'Only position-table evaluation; no model forward',
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),'threads':torch.get_num_threads(),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'cpu_cos_sha256':tensors['cos-cpu']['sha256'],'cuda_cos_sha256':tensors['cos-cuda']['sha256'],
        'cpu_sin_sha256':tensors['sin-cpu']['sha256'],'cuda_sin_sha256':tensors['sin-cuda']['sha256']})
    del gpu_frequencies,frequencies,value
    torch.cuda.empty_cache()
    cases=[]
    for variant in ('native-replay','official-cpu-rope','official-cuda-rope'):
        for index in INDICES:
            side='native' if variant=='native-replay' else 'official'
            rope='cuda' if variant=='official-cuda-rope' else 'cpu'
            names=[f'hidden-{side}-{index}',*[f'mod-{side}-{i}' for i in range(1,7)],f'cos-{rope}',f'sin-{rope}','mask']
            cases.append({'block':index,'variant':variant,'inputs':{f'in{i}':tensors[name] for i,name in enumerate(names)},
                'expected':tensors[f'expected-official-{index}'],'replay_anchor':tensors[f'expected-native-{index}']})
    previous_plan=json.loads((OLD/'plan.json').read_text())
    bound=[BASE/'source'/relative for relative in sources]
    bound.extend(BASE/name for name in ('experiment.py','run.py','supervisor.py','source-before.json','runner.snapshot','build-record.json','cli-checks.json','position-preparation.json'))
    bound.extend(Path(item['file']) for item in tensors.values())
    bound.extend([OLD/'plan.json',OLD/'audit.json',old_fixture/'fixture.json',REFERENCE/'fixture.json',model/'manifest.json'])
    bound.extend(model/'objects'/digest for digest in (binding['source_manifest_sha256'],provenance['source_manifest_sha256']))
    bound.extend(Path(name) for name in previous_plan['bindings'] if '/site-packages/' in name)
    for index in INDICES:
        path=ROOT/f'models/official/dit-block-{index:02d}.safetensors'
        assert sha(path)==weights[index]
        bound.extend([path,path.with_suffix('.manifest.json')])
    plan={'protocol':'shared-768-step0-blocks17-19-same-input-v1','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_count':len(sources),'indices':list(INDICES),'model':str(model),'config':cfg,'package_binding':binding,
        'official_weight_provenance':provenance,'block_weight_sha256':weights,'runner_sha256':sha(BASE/'runner.snapshot'),
        'native_acceptance_eligible':False,'official_gate_applied':False,'cases':cases,'tensors':tensors,
        'scope':'Three official same-input block replays and nine native single-block calls; no heads/text/Euler/VAE/free generation.',
        'criteria':{'official_replay':'All three outputs must exactly match saved official stages.',
                    'native_replay':'All three outputs must exactly match saved native stages before teachers.',
                    'same_inputs':'Official CUDA cos/sin, official incoming hidden/modulation and mask are byte-identical inputs to the compared block implementations.',
                    'metrics':'Descriptive only. CPU-RoPE teachers retain trigonometric differences; no new quality gate or additive causal allocation.'},
        'commands':{phase:[[str(ROOT/'.venv/bin/python'),str(BASE/'experiment.py'),phase]] for phase in ('official','native')},
        'resource_limits':previous_plan['resource_limits'],
        'bindings':{str(path):identity(path) for path in sorted(set(bound))}}
    write(BASE/'plan.json',plan)
    print(json.dumps({'prepared':True,'sources':len(sources),'bindings':len(plan['bindings']),'plan_sha256':sha(BASE/'plan.json')}),flush=True)

def official(plan):
    np,torch,_,load_block,save_tensor=imports()
    t=plan['tensors'];out=BASE/'official/results';out.mkdir()
    frequencies=checked_tensor(np,torch,t['frequencies']).to('cuda')
    assert torch.equal(torch.cos(frequencies)[:,:,0,:].cpu(),checked_tensor(np,torch,t['cos-cuda']))
    assert torch.equal(torch.sin(frequencies)[:,:,0,:].cpu(),checked_tensor(np,torch,t['sin-cuda']))
    mask=checked_tensor(np,torch,t['mask'])[None,None].to('cuda')
    modulation=[checked_tensor(np,torch,t[f'mod-official-{i}']).to('cuda') for i in range(1,7)]
    rows=[]
    for index in INDICES:
        hidden=checked_tensor(np,torch,t[f'hidden-official-{index}']).transpose(0,1).to('cuda')
        block,manifest=load_block(ROOT/f'models/official/dit-block-{index:02d}.safetensors',index)
        assert manifest['sha256']==plan['block_weight_sha256'][index]
        assert block.self_attention.processor._attention_backend is None and torch.get_float32_matmul_precision()=='highest'
        result=block.to('cuda')(hidden,frequencies,modulation,attention_mask=mask)
        entry=save_tensor(out/f'block-{index}.f32',result.transpose(0,1))
        row={'block':index,'output':entry,'exact_replay':entry['sha256']==t[f'expected-official-{index}']['sha256']}
        rows.append(row);write(out/'result.json',rows);print(json.dumps(row),flush=True)
        assert row['exact_replay'], 'Selected official block does not replay its saved stage'
        del block,result,hidden
        torch.cuda.empty_cache()

def native(plan):
    np,torch,_,_,_=imports()
    from export_dit_block import metrics
    assert all(row['exact_replay'] for row in json.loads((BASE/'official/results/result.json').read_text()))
    summaries=[]
    for case in plan['cases']:
        index=case['block'];out=BASE/'native'/f"block-{index}-{case['variant']}";out.mkdir()
        fixture=out/'fixture';fixture.mkdir()
        for name,item in case['inputs'].items():
            checked_tensor(np,torch,item)
            shutil.copyfile(item['file'],fixture/(name+'.f32'))
        write(fixture/'fixture.json',case)
        command=[str(BASE/'runner.snapshot'),'--package',plan['model'],'--block-index',str(index),
            '--fixture',str(fixture),'--output',str(out/'actual.f32'),'--tokens','2368','--width','48','--height','48',
            '--text-tokens','64','--valid-text-tokens','15','--threads','2','--backend','vulkan','--precision','fp32',
            '--policy','stream','--host-weights','--trace-dir',str(out/'trace')]
        with (out/'runner.log').open('x') as log:
            process=subprocess.run(['/usr/bin/time','-v','-o',str(out/'resources.log'),*command],stdout=log,stderr=subprocess.STDOUT)
        row={'block':index,'variant':case['variant'],'command':command,'return_code':process.returncode}
        write(out/'result.json',row)
        assert process.returncode==0
        runtime=[json.loads(line) for line in (out/'runner.log').read_text().splitlines() if line.startswith('{')][-1]
        assert runtime['block_index']==index and runtime['blocks']==1 and runtime['package_mode'] and not runtime['dit_heads']
        assert runtime['host_weights'] and runtime['threads']==2 and runtime['valid_text_tokens']==15 and runtime['text_bucket']==32
        assert {p.name for p in (out/'trace').iterdir()}=={f'block-{index}.f32'}
        assert sha(out/'actual.f32')==sha(out/f'trace/block-{index}.f32')
        actual_item=bound_tensor(out/'actual.f32',[1,2368,4096])
        actual=checked_tensor(np,torch,actual_item)
        expected=checked_tensor(np,torch,case['expected'])
        row.update(actual=actual_item,runtime=runtime,official_difference=metrics(expected,actual),
                   native_anchor_exact=actual_item['sha256']==case['replay_anchor']['sha256'],
                   native_acceptance_eligible=False,official_gate_applied=False)
        write(out/'result.json',row);summaries.append(row);write(BASE/'native/results.json',summaries)
        print(json.dumps({k:row[k] for k in ('block','variant','return_code','official_difference','native_anchor_exact')}),flush=True)
        if case['variant']=='native-replay':
            assert row['native_anchor_exact'],'Selected native block does not replay its saved stage'

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    else:
        plan=json.loads((BASE/'plan.json').read_text());verify(plan)
        assert plan['indices']==list(INDICES) and not plan['native_acceptance_eligible']
        {'official':official,'native':native}[mode](plan)
        verify(plan)
