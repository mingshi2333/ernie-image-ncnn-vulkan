#!/usr/bin/env python3
"""Bounded block-15 teacher call using the already validated a4a stack runner."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnose_exact_head_stack import sha, checked, compare, require, SHAPE, RUNNER

BASE_PLAN='3aba6aa5ec1bec491a36187bc0f62c5332549d26df396655010d1966db4201f7'
BASE_RESULT='bc8ddc450ebce74400c3b2f3ec4c1a6e748f5323ceeb314521aae65e54089b63'
PROTOCOL='chinese-step0-block15-official-input-v1'


def validate_inputs(base, inputs, official14):
    require(set(inputs)==set(base)=={f'in{i}' for i in range(10)},'Ten inputs required')
    for name,item in inputs.items():
        require(item['shape']==base[name]['shape'] and item['dtype']=='float32_le','Changed axes/dtype')
        require(item['sha256']==(official14 if name=='in0' else base[name]['sha256']),'Changed conditioning or wrong hidden input')


def command(out, package):
    return [str(out/'execution/runner.snapshot'),'--fixture',str(out/'fixture'),'--tokens','4160','--output',str(out/'actual.f32'),
            '--backend','vulkan','--precision','fp32','--policy','stream','--trace-dir',str(out/'trace'),'--model',str(package/'dit/block-15')]


def prepare(basefile, out):
    basefile=basefile.resolve();out=out.resolve();require(not out.exists(),'Use fresh output')
    require(sha(basefile)==BASE_PLAN and sha(basefile.parent/'result.json')==BASE_RESULT,'Unreviewed baseline')
    base=json.loads(basefile.read_text());baseline=json.loads((basefile.parent/'result.json').read_text())
    official14=base['denominator'][14];oracle=base['denominator'][15];require(official14['layer']==14 and oracle['layer']==15,'Layer mapping differs')
    inputs={name:dict(item) for name,item in base['inputs'].items()};inputs['in0']=dict(official14,source=official14['file'])
    validate_inputs(base['inputs'],inputs,official14['sha256'])
    (out/'fixture').mkdir(parents=True);(out/'execution').mkdir()
    for name,item in inputs.items():
        checked(item['file'],item);src=item['file'];dest=out/'fixture'/f'{name}.f32';shutil.copy2(src,dest);item.update(file=str(dest),source=src)
    checked(oracle['file'],oracle)
    for script in [Path(__file__),Path(__file__).with_name('diagnose_exact_head_stack.py')]:shutil.copy2(script,out/'execution'/script.name)
    shutil.copy2(basefile.parent/'execution/runner.snapshot',out/'execution/runner.snapshot')
    require(sha(out/'execution/runner.snapshot')==RUNNER,'Different runner')
    native={'file':str(basefile.parent/'trace/block-15.f32'),'shape':SHAPE,'dtype':'float32_le','sha256':baseline['rows'][15]['actual_sha256']};checked(native['file'],native)
    package=Path(base['package']);bound={**base['bound_sha256'],str(basefile):BASE_PLAN,str(basefile.parent/'result.json'):BASE_RESULT}
    for file in (out/'execution').iterdir():bound[str(file)]=sha(file)
    models={name:digest for name,digest in base['model_files_sha256'].items() if '/block-15/' in name};require(len(models)==2,'Wrong model denominator')
    plan={'protocol':PROTOCOL,'status':'prepared_not_executed','native_acceptance_eligible':False,'official_gate_applied':False,
          'base_plan':str(basefile),'output':str(out),'package':str(package),'inputs':inputs,'official14_sha256':official14['sha256'],
          'expected':oracle,'baseline_stack_block15':native,'bound_sha256':bound,'model_files_sha256':models,'command':command(out,package),
          'layer_semantics':'only model block-15; runner observer local block-0 maps to global block 15; raw out0 complete [4160,4096]',
          'scope':'Same-input complete single-block implementation difference; descriptive only, no image/prediction gate'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');return plan


def execute(path):
    plan=json.loads(path.read_text());out=Path(plan['output']);base=json.loads(Path(plan['base_plan']).read_text())
    require(sha(plan['base_plan'])==BASE_PLAN and plan['protocol']==PROTOCOL,'Wrong baseline/protocol')
    require(plan['official_gate_applied'] is False and plan['native_acceptance_eligible'] is False,'Wrong scope')
    require(plan['command']==command(out,Path(plan['package'])),'Changed command')
    validate_inputs(base['inputs'],plan['inputs'],plan['official14_sha256'])
    require(plan['official14_sha256']==base['denominator'][14]['sha256'] and plan['expected']==base['denominator'][15],'Wrong layer mapping')
    require(not (out/'actual.f32').exists() and not (out/'trace').exists(),'Already executed')
    for file,digest in {**plan['bound_sha256'],**plan['model_files_sha256']}.items():require(sha(file)==digest,'Changed bound bytes')
    for item in [*plan['inputs'].values(),plan['expected'],plan['baseline_stack_block15']]:checked(item['file'],item)
    subprocess.run(plan['command'],check=True)
    require({x.name for x in (out/'trace').iterdir()}=={'block-0.f32'},'Wrong output denominator')
    actual=out/'actual.f32';require(actual.stat().st_size==4160*4096*4 and sha(actual)==sha(out/'trace/block-0.f32'),'Wrong output bytes/layout')
    result={'protocol':PROTOCOL,'status':'completed_descriptive_only','plan_sha256':sha(path),'actual_sha256':sha(actual),
            'official_gate_applied':False,'native_acceptance_eligible':False,'global_block':15,'values':4160*4096,
            'same_input_official_difference':compare(actual,plan['expected']['file']),
            'difference_from_accumulated_stack':compare(actual,plan['baseline_stack_block15']['file']),
            'accumulated_stack_official_difference':compare(plan['baseline_stack_block15']['file'],plan['expected']['file'])}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--base',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    require(bool(a.prepare)!=bool(a.execute),'Choose prepare/execute');print(json.dumps(prepare(a.base,a.prepare) if a.prepare else execute(a.execute),indent=2))
