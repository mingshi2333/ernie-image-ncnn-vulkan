"""Freeze four explicit RAM-weight configurations before any model execution."""
import datetime,hashlib,json,os,random,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
BASE=Path('/var/tmp/ernie-memory-grid512-v1')
PRIOR=Path('/var/tmp/ernie-benchmark-runtime512-v1')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
from source_inventory import source_files

def identity(path):
    with path.open('rb') as stream:return {'bytes':path.stat().st_size,'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def git(*args,root=ROOT):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()

def main():
    assert not git('status','--porcelain=v1'), 'Origin must be clean before snapshot'
    ncnn=ROOT.parent.parent/'third_party/ncnn'
    assert not git('status','--porcelain=v1',root=ncnn)
    assert git('rev-parse','HEAD',root=ncnn)=='6a1bf000f363714839a36793addc8c879d3d899e'
    native=json.loads((PRIOR/'native/benchmark/generation.json').read_text())
    settings={**native['request'],'dit_weights':'host','gpu_reserve_mib':512}
    for key in ('dit_cache_mib','model_loading'):settings.pop(key)
    configurations={name:{'model_loading':loading,'dit_cache_mib':cache} for name,loading,cache in (
        ('stdio-off','stdio',0),('stdio-cache','stdio',6144),('mapped-off','mapped',0),('mapped-cache','mapped',6144))}
    seed=20260907
    order=list(configurations);random.Random(seed).shuffle(order)
    schedule=[('warmup',-1,order)]
    # Including warmups, each configuration occupies every order position once.
    schedule += [('measured',r,order[r+1:]+order[:r+1]) for r in range(3)]
    BASE.mkdir()
    (BASE/'runtime-source').mkdir()
    sources=source_files(ROOT)
    for source in sources:
        target=BASE/'runtime-source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert identity(target)==identity(source)
    for source,name in ((ROOT/'build-dev/ernie-image','ernie-image.snapshot'),(PRIOR/'initial.f32','initial.f32'),
        (Path('/var/tmp/ernie-runtime512-native-v1/native/native.png'),'baseline.png')):
        shutil.copy2(source,BASE/name)
    for name in ('analyze.py','master.py','prepare.py','test_contract.py','contract-check.log'):
        shutil.copy2(HERE/name,BASE/name)
    (BASE/'prompt.txt').write_text(native['prompt'],encoding='utf-8')
    model=(ROOT/'models/turbo-shared-v2').resolve()
    build={'head':git('rev-parse','HEAD'),'origin_clean':True,'ncnn_head':git('rev-parse','HEAD',root=ncnn),
        'binary':identity(BASE/'ernie-image.snapshot'),'settings':[v for v in (ROOT/'build-dev/CMakeCache.txt').read_text().splitlines()
        if v.startswith(('ERNIE_','CMAKE_BUILD_TYPE:','CMAKE_CXX_COMPILER:'))]}
    assert build['binary']['sha256']=='ff2f934f317a602f3d098bf6d74cd7c2e2a6de725ed8fcf04d8d1e2bcbaa3e1c'
    assert 'ERNIE_ENABLE_ALLOCATION_METRICS:BOOL=OFF' in build['settings']
    write(BASE/'build-identity.json',build)
    write(BASE/'host-before.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'meminfo':Path('/proc/meminfo').read_text(),'cpuinfo':Path('/proc/cpuinfo').read_text(),
        'uname':list(os.uname()),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.used','--format=csv'],text=True)})
    limits=json.loads((PRIOR/'plan.json').read_text())['resource_limits']
    run=(PRIOR/'run.py').read_text()
    run=run.replace("    for name,item in plan['bindings'].items():", "    entries=dict(plan['bindings'])\n    for name,item in entries.items():\n        path=Path(name)\n        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:\n            raise ValueError('Frozen binding changed: '+name)\n    entries.update(json.loads(Path(plan['shared_bindings']).read_text()))\n    for name,item in entries.items():")
    assert 'entries.update' in run
    supervisor=(PRIOR/'supervisor.py').read_text()
    paths=[BASE/'runtime-source'/p.relative_to(ROOT) for p in sources]
    paths += [BASE/name for name in ('ernie-image.snapshot','initial.f32','baseline.png','prompt.txt','analyze.py','master.py','prepare.py',
        'test_contract.py','contract-check.log','build-identity.json','host-before.json')]
    paths.append(model/'manifest.json')
    trials=[]
    for phase,repeat,sequence in schedule:
        for config in sequence:
            index=len(trials);name=f'{index:02d}-{phase}-{config}'
            directory=BASE/name;directory.mkdir()
            for child in ('model','temporary','cache'):(directory/child).mkdir()
            (directory/'run.py').write_text(run)
            (directory/'supervisor.py').write_text(supervisor.replace('ernie-benchmark-runtime512-v1-',f'ernie-memory-grid512-v1-{index:02d}-'))
            paths.extend((directory/'run.py',directory/'supervisor.py'))
            c=configurations[config]
            command=['bwrap','--die-with-parent','--unshare-net','--ro-bind','/','/', '--bind',str(directory),str(directory),
                '--ro-bind',str(model),str(directory/'model'),'--tmpfs',str(ROOT.parent.parent),'--dev-bind','/dev','/dev','--proc','/proc',
                '--chdir',str(directory),'--setenv','TMPDIR',str(directory/'temporary'),'--setenv','XDG_CACHE_HOME',str(directory/'cache'),'--',
                sys.executable,str(BASE/'runtime-source/tools/benchmark_pipeline.py'),'--model',str(directory/'model'),
                '--runner',str(BASE/'ernie-image.snapshot'),'--output',str(directory/'native/benchmark'),'--prompt-file',str(BASE/'prompt.txt'),
                '--width','512','--height','512','--latent',str(BASE/'initial.f32'),'--noise-sha256',identity(BASE/'initial.f32')['sha256'],
                '--steps','8','--seed','42','--threads','2','--device','vulkan','--precision','fp32','--text-device','cpu','--text-down-vector',
                '--vae-device','cpu','--vae-convolution','direct','--dit-weights','host','--gpu-reserve-mib','512','--gpu','0',
                '--dit-cache-mib',str(c['dit_cache_mib']),'--ram-reserve-mib','3072','--model-loading',c['model_loading'],'--timeout','1500']
            trials.append({'id':name,'phase':phase,'repeat':repeat,'config':config,'order':index,'command':command})
    write(BASE/'bindings.json',{str(p):identity(p) for p in paths})
    for trial in trials:
        directory=BASE/trial['id']
        subplan={'commands':{'native':[trial.pop('command')]},'resource_limits':limits,
            'shared_bindings':str(BASE/'bindings.json'),'bindings':{str(BASE/'bindings.json'):identity(BASE/'bindings.json')}}
        write(directory/'plan.json',subplan);trial['plan_sha256']=identity(directory/'plan.json')['sha256']
    grid={'schema_version':1,'scope':'Four-setting within-project repeated 512 development comparison; not the formal peer S/M or quality protocol.',
        'source_commit':build['head'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'random_seed':seed,
        'order_method':'Seeded shuffle for warmups, followed by three cyclic rotations for measured rounds; fresh process every trial.',
        'binary_sha256':build['binary']['sha256'],'noise_sha256':identity(BASE/'initial.f32')['sha256'],
        'image_sha256':identity(BASE/'baseline.png')['sha256'],'package_manifest_sha256':identity(model/'manifest.json')['sha256'],
        'prompt':native['prompt'],'prompt_sha256':identity(BASE/'prompt.txt')['sha256'],'token_ids':native['token_ids'],
        'common_settings':settings,'configurations':configurations,'trials':trials,'resource_limits':limits,
        'bindings_sha256':identity(BASE/'bindings.json')['sha256'],
        'comparisons':[['stdio-off','mapped-off'],['stdio-cache','mapped-cache'],['stdio-off','stdio-cache'],['mapped-off','mapped-cache']],
        'acceptance':'Every planned run must exit successfully with the frozen identities, exact native request, trace/instrumentation OFF, completed eight steps and byte-exact baseline PNG. Preserve any failure and stop without retry or default promotion. Aggregate only after all sixteen pass.',
        'measurement_scope':'Native process launch through exit, including native verification, image/report close. Python verifies all files before timing, warming file cache without controlling it. Fresh process and shader-cache directory each trial. No cold-cache or desktop-isolation claim. RSS, whole-device sampled GPU, cgroup including file-cache pressure are distinct; no allocator-complete VRAM claim.',
        'decision_scope':'Three measured runs per configuration; report medians and within-round ratios, cache hits/evictions and file-cache pressure. No significance or broad default recommendation from this single development fixture.',
        'placement_scope':'All runs explicitly request RAM weights. This compares loading/cache, not auto placement, actual GPU OOM, activation spilling, or failure recovery.'}
    write(BASE/'grid.json',grid)
    assert not git('status','--porcelain=v1')
    print(json.dumps({'base':str(BASE),'source_commit':build['head'],'grid_sha256':identity(BASE/'grid.json')['sha256'],
        'bindings':len(paths),'trials':[{k:t[k] for k in ('id','phase','repeat','config','order')} for t in trials]},indent=2))

if __name__=='__main__':main()
