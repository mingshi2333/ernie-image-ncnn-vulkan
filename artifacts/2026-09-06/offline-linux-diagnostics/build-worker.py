import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path.cwd();kind=sys.argv[1]
assert kind in ('cpu','vulkan')
source=(root/'outputs/d3-runtime-source-v3').resolve()
out=root/('outputs/d3-runtime-build-'+kind+'-v3');out.mkdir(exist_ok=False)
build=source/'build'/('linux-'+kind)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
meta=source/'source-identity.json';snapshot=json.loads(meta.read_text())
def changes():return [p for p,s in snapshot['files'].items() if sha(source/p)!=s]
assert not changes()
ncnn=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/third_party/ncnn')
def dirty():return subprocess.check_output(['git','-C',str(ncnn),'status','--porcelain','--untracked-files=no'],text=True)
assert not dirty()
revision=subprocess.check_output(['git','-C',str(ncnn),'rev-parse','HEAD'],text=True).strip()
assert revision=='6a1bf000f363714839a36793addc8c879d3d899e'
commands=[['cmake','--preset','linux-'+kind,'-DERNIE_NCNN_SOURCE_DIR='+str(ncnn),'-DCMAKE_C_COMPILER=/usr/bin/clang','-DCMAKE_CXX_COMPILER=/usr/bin/clang++'],
          ['cmake','--build','--preset','linux-'+kind],
          ['ctest','--test-dir',str(build),'-R','^(pipeline_api_contract|request_validation_contract|gpu_context_contract|gpu_context_missing_driver|cli_contract|installed_cpp_consumer)$','--output-on-failure','--no-tests=error']]
cg=Path('/sys/fs/cgroup')/next(x.split('::',1)[1].lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
limits={k:(cg/k).read_text().strip() for k in ('memory.max','memory.swap.max','cpu.max')}
assert limits=={'memory.max':'4294967296','memory.swap.max':'0','cpu.max':'200000 100000'},limits
identity={'commands':commands,'source_identity_sha256':sha(meta),'worker_sha256':sha(__file__),'ncnn_revision':revision,
          'ncnn_submodules':subprocess.check_output(['git','-C',str(ncnn),'submodule','status','--recursive'],text=True),
          'resource_controls':limits,'cpu_affinity':sorted(os.sched_getaffinity(0))}
for compiler in ('clang++','rustc','cargo','cmake'):
    identity[compiler]=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
results=[];code=0
for label,command in zip(('configure','build','test'),commands):
    t=time.monotonic()
    with (out/(label+'.log')).open('xb') as log:
        result=subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
    results.append({'name':label,'exit_code':result.returncode,'wall_seconds':time.monotonic()-t,'log_sha256':sha(out/(label+'.log'))})
    print(label,result.returncode,flush=True)
    if result.returncode:code=result.returncode;break
binaries={p.name:sha(p) for p in sorted(build.glob('ernie-*')) if p.is_file()}
result={'exit_code':code,'commands':results,'source_changes_after_build':changes(),'ncnn_changes_after':dirty(),'binaries':binaries,'memory_events':(cg/'memory.events').read_text()}
result['exit_code']=code or int(bool(result['source_changes_after_build'] or result['ncnn_changes_after']))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True);raise SystemExit(result['exit_code'])
