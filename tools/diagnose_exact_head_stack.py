#!/usr/bin/env python3
"""One reviewed step-0 exact-head replay; descriptive 36-boundary diagnostic only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np

EXACT = '2b864c6904f05348699ecb853692e476c32cdd684f9f2e62efbbea4549e61509'
STAGES = '768a1f496b2537e3bda25053e6bb6456dea9a66d9106e0c5416d552c4d4b0d18'
REFERENCE = '81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7'
RUNNER = 'a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc'
PACKAGE = '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1'
PROTOCOL = 'chinese-step0-exact-head-stack-v1'
SHAPE = [1, 4160, 4096]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def checked(path, item):
    require(item['dtype'] == 'float32_le', 'Wrong dtype')
    require(Path(path).stat().st_size == math.prod(item['shape'])*4, 'Wrong tensor size')
    require(sha(path) == item['sha256'], 'Wrong tensor hash')
    a = np.memmap(path, dtype='<f4', mode='r')
    for start in range(0, a.size, 2**20):
        require(np.isfinite(a[start:start+2**20]).all(), 'Nonfinite tensor')


def contract(exact, stages):
    require(exact['tokens'] == stages['tokens'] == 4160 and stages['step'] == 0, 'Wrong step/tokens')
    require(stages['reference_fixture_sha256'] == REFERENCE, 'Wrong full reference')
    require(set(exact['inputs']) == {f'in{i}' for i in range(10)}, 'Ten-input denominator required')
    for i in range(10):
        item = exact['inputs'][f'in{i}']
        expected = SHAPE if i == 0 else [1,1,4096] if i <= 6 else [1,4160,128] if i <= 8 else [4160,4160]
        origin = stages['stages'][f'head-{0 if i == 0 else i+1}'] if i <= 6 else stages['inputs'][f'in{i-4}']
        require(item['shape'] == expected and item['dtype'] == 'float32_le', 'Input axes/dtype mismatch')
        require(item['sha256'] == origin['sha256'], 'Exact head is not the bound official boundary')
    blocks = {k for k in stages['stages'] if k.startswith('block-')}
    require(blocks == {f'block-{i}' for i in range(36)}, '36-layer denominator required')
    for i in range(36):
        item = stages['stages'][f'block-{i}']
        require(item['shape'] == SHAPE and item['dtype'] == 'float32_le', 'Output axes/dtype mismatch')
    require(exact['expected']['sha256'] == stages['stages']['block-35']['sha256'], 'Final boundary mismatch')


def compare(actual, reference):
    x = np.memmap(actual, '<f4', mode='r'); y = np.memmap(reference, '<f4', mode='r')
    require(x.shape == y.shape and x.size > 0, 'Comparison denominator mismatch')
    ss = yy = 0.; maximum = 0.
    for start in range(0, x.size, 2**20):
        a=x[start:start+2**20].astype('f8'); b=y[start:start+2**20].astype('f8')
        require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite comparison')
        delta=a-b;ss+=float(np.sum(delta*delta));yy+=float(np.sum(b*b));maximum=max(maximum,float(abs(delta).max()))
    return {'nrmse': math.sqrt(ss/yy) if yy else (0. if not ss else None), 'error_l2':math.sqrt(ss), 'reference_l2':math.sqrt(yy), 'max_abs_error':maximum}


def command(out, package):
    return [str(out/'execution/runner.snapshot'),'--fixture',str(out/'fixture'),'--tokens','4160',
            '--output',str(out/'actual.f32'),'--backend','vulkan','--precision','fp32','--policy','stream',
            '--trace-dir',str(out/'trace'),*[a for i in range(36) for a in ['--model',str(package/f'dit/block-{i:02}')]]]


def prepare(root, out):
    root=root.resolve();out=out.resolve();require(not out.exists(), 'Use a fresh output directory')
    old=root/'outputs/diagnostic-chinese-exact-heads-v1/fixture';oracle=root/'outputs/diagnostic-chinese-step0-stages-v1/oracle/fixture'
    source=root/'outputs/q2-chinese-step6-current-teacher-v1-execution';package=root/'models/turbo1024-s64-portable'
    require(sha(old/'fixture.json') == EXACT and sha(oracle/'fixture.json') == STAGES, 'Unreviewed fixture')
    require(sha(source/'runner.snapshot') == RUNNER and sha(package/'manifest.json') == PACKAGE, 'Unreviewed binary/package')
    exact=json.loads((old/'fixture.json').read_text());stages=json.loads((oracle/'fixture.json').read_text());contract(exact, stages)
    snapshot=json.loads((source/'snapshot.json').read_text())
    for name,digest in snapshot['files'].items(): require(sha(source/name)==digest, 'Predictor source changed')
    (out/'fixture').mkdir(parents=True); (out/'execution/source').mkdir(parents=True)
    bound={str(old/'fixture.json'):EXACT,str(oracle/'fixture.json'):STAGES,str(package/'manifest.json'):PACKAGE}
    for name,digest in snapshot['files'].items():
        dest=out/'execution/source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
        bound[str(dest)]=digest
    shutil.copy2(source/'snapshot.json',out/'execution/source-snapshot.json')
    bound[str(out/'execution/source-snapshot.json')]=sha(source/'snapshot.json')
    shutil.copy2(source/'runner.snapshot',out/'execution/runner.snapshot');bound[str(out/'execution/runner.snapshot')]=RUNNER
    shutil.copy2(__file__,out/'execution/diagnose_exact_head_stack.py');bound[str(out/'execution/diagnose_exact_head_stack.py')]=sha(__file__)
    inputs={}
    for name,item in exact['inputs'].items():
        origin=old/item['file'];checked(origin,item);dest=out/'fixture'/f'{name}.f32';shutil.copy2(origin,dest)
        inputs[name]={**item,'file':str(dest),'source':str(origin)}
    denominator=[]
    for i in range(36):
        item=stages['stages'][f'block-{i}'];path=oracle/item['file'];checked(path,item)
        denominator.append({**item,'file':str(path),'layer':i,'trace':f'block-{i}.f32'})
    manifest=json.loads((package/'manifest.json').read_text());weights={}
    for i in range(36):
        for suffix in ['bin','param']:
            name=f'dit/block-{i:02}/block.ncnn.{suffix}';p=package/name
            require(p.stat().st_size==manifest['file_sizes'][name], 'Model file size differs')
            weights[str(p)]=manifest['files'][name]
    plan={'protocol':PROTOCOL,'status':'prepared_not_executed','native_acceptance_eligible':False,'official_gate_applied':False,
          'output':str(out),'package':str(package),'command':command(out,package),'inputs':inputs,'denominator':denominator,
          'bound_sha256':bound,'model_files_sha256':weights,'model_bytes_rehash':'required immediately before execute; preparation checks fixed manifest and sizes',
          'axes':'little-endian contiguous FP32 [token,hidden]; singleton batch stripped; all 4160 tokens and 4096 features; zero-based post-block traces',
          'scope':'Exact official projected state + six modulation tensors + cos/sin/mask; 36 blocks only, no heads/Euler/VAE; descriptive errors, no acceptance gate',
          'guard':{'rss_bytes':9*1024**3,'gpu_mib':6144,'host_available_bytes':3*1024**3,'cpu_affinity':[0,2],'native_threads':4,'disk_trace_bytes':36*4160*4096*4}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');return plan


def execute(path):
    p=json.loads(path.read_text());out=Path(p['output'])
    require(p['protocol']==PROTOCOL and p['native_acceptance_eligible'] is False and p['official_gate_applied'] is False,'Wrong protocol')
    require(p['command']==command(out,Path(p['package'])),'Changed command')
    require([v['layer'] for v in p['denominator']]==list(range(36)),'Wrong denominator')
    require(not (out/'actual.f32').exists() and not (out/'trace').exists(),'Existing execution')
    for name,digest in {**p['bound_sha256'],**p['model_files_sha256']}.items():require(sha(name)==digest,'Bound file changed: '+name)
    for item in p['inputs'].values(): checked(item['file'],item)
    for item in p['denominator']: checked(item['file'],item)
    require(shutil.disk_usage(out).free > p['guard']['disk_trace_bytes']+1024**3,'Insufficient trace disk budget')
    result=subprocess.run(p['command'],check=False)
    require(result.returncode==0,'Native execution failed')
    require({q.name for q in (out/'trace').iterdir()}=={f'block-{i}.f32' for i in range(36)},'Actual trace denominator differs')
    rows=[]
    for item in p['denominator']:
        actual=out/'trace'/item['trace'];require(actual.stat().st_size==math.prod(SHAPE)*4,'Wrong trace shape')
        rows.append({'layer':item['layer'],'actual_sha256':sha(actual),'reference_sha256':item['sha256'],**compare(actual,item['file'])})
    require(sha(out/'actual.f32')==rows[-1]['actual_sha256'],'Final result is not block 35')
    r={'protocol':PROTOCOL,'status':'completed_descriptive_only','plan_sha256':sha(path),'official_gate_applied':False,
       'native_acceptance_eligible':False,'denominator':36,'rows':rows,'final_sha256':sha(out/'actual.f32')}
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');return r


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path);parser.add_argument('--execute',type=Path);parser.add_argument('--root',type=Path,default=Path.cwd());args=parser.parse_args()
    require(bool(args.prepare)!=bool(args.execute),'Choose prepare or execute')
    print(json.dumps(prepare(args.root,args.prepare) if args.prepare else execute(args.execute),indent=2))
