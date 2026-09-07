"""Freeze the new benchmark wrapper, native producer and unchanged 512 input."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base=Path('/var/tmp/ernie-benchmark-runtime512-v1')
prior=Path('/var/tmp/ernie-cache-file-lru512-v1')
sys.path.insert(0,str(root/'tools'))
from source_inventory import source_files

def identity(path):
    with path.open('rb') as stream:return {'bytes':path.stat().st_size,'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
base.mkdir()
for name in ('model','temporary','cache','runtime-source'):(base/name).mkdir()
shutil.copy2(root/'build-dev/ernie-image',base/'ernie-image.snapshot')
shutil.copy2(prior/'initial.f32',base/'initial.f32')
shutil.copy2(prior/'run.py',base/'run.py')
supervisor=(prior/'supervisor.py').read_text().replace('ernie-cache-file-lru512-v1-', 'ernie-benchmark-runtime512-v1-')
supervisor=supervisor.replace("native_binary = (BASE / 'ernie-image.snapshot').resolve()", "native_binary = (BASE / 'native/benchmark/ernie-image.snapshot').resolve()")
(base/'supervisor.py').write_text(supervisor)
sources=source_files(root)
for path in sources:
    target=base/'runtime-source'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
model=(root/'models/turbo-shared-v2').resolve()
command=['bwrap','--die-with-parent','--unshare-net','--ro-bind','/','/', '--bind',str(base),str(base),
    '--ro-bind',str(model),str(base/'model'),'--ro-bind',str(base/'runtime-source'),str(base/'runtime-source'),
    '--tmpfs',str(root.parent.parent),'--dev-bind','/dev','/dev','--proc','/proc','--chdir',str(base),
    '--setenv','TMPDIR',str(base/'temporary'),'--setenv','XDG_CACHE_HOME',str(base/'cache'),'--',
    sys.executable,str(base/'runtime-source/tools/benchmark_pipeline.py'),'--model',str(base/'model'),
    '--runner',str(base/'ernie-image.snapshot'),'--output',str(base/'native/benchmark'),
    '--width','512','--height','512','--latent',str(base/'initial.f32'),'--noise-sha256',identity(base/'initial.f32')['sha256'],
    '--steps','8','--threads','2','--device','vulkan','--precision','fp32','--text-down-vector',
    '--vae-device','cpu','--vae-convolution','direct','--dit-weights','auto','--gpu-reserve-mib','8192','--gpu','0',
    '--dit-cache-mib','6144','--ram-reserve-mib','3072','--model-loading','mapped','--timeout','1500']
build={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'uncommitted_implementation':True,'binary':identity(base/'ernie-image.snapshot'),
    'ncnn_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent.parent/'third_party/ncnn',text=True).strip(),
    'settings':[v for v in (root/'build-dev/CMakeCache.txt').read_text().splitlines() if v.startswith(('ERNIE_','CMAKE_BUILD_TYPE:','CMAKE_CXX_COMPILER:'))]}
assert 'ERNIE_ENABLE_ALLOCATION_METRICS:BOOL=OFF' in build['settings']
(base/'build-identity.json').write_text(json.dumps(build,indent=2)+'\n')
baseline=Path('/var/tmp/ernie-runtime512-native-v1/native/native.png')
paths=sources+[base/'runtime-source'/p.relative_to(root) for p in sources]
paths += [base/name for name in ('ernie-image.snapshot','initial.f32','run.py','supervisor.py','build-identity.json')]
paths += [model/'manifest.json',baseline,root/'outputs/benchmark-runtime-v1/compare.py',Path(__file__).resolve()]
plan={'scope':'Single real shared512 benchmark integration; native trace and allocation instrumentation OFF; same cache/loading settings as the last traced regression. No paired speed claim.',
      'commands':{'native':[command]},'resource_limits':json.loads((prior/'plan.json').read_text())['resource_limits'],
      'bindings':{str(p):identity(p) for p in paths},'baseline_image':str(baseline),
      'package_manifest_sha256':identity(model/'manifest.json')['sha256'],
      'measurement_scope':'Wrapper verifies all files before native timing and may warm page cache; native timer includes native verification, report/image close. Supervisor includes Python verification and snapshots; whole GPU includes desktop; cgroup includes file cache.'}
(base/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'base':str(base),'bindings':len(plan['bindings']),'binary':build['binary'],'plan':identity(base/'plan.json')},indent=2))
