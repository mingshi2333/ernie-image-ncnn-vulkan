import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path.cwd();source=(root/'outputs/d1-preset-source-v5').resolve();build=source/'build/linux-vulkan'
out=root/'outputs/d1-preset-vulkan-test-v5';out.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
meta=source/'source-identity.json';identity=json.loads(meta.read_text())
def changes():return [p for p,s in identity['files'].items() if sha(source/p)!=s]
assert not changes()
assert json.loads((root/'outputs/d1-preset-vulkan-v5/result.json').read_text())['exit_code']==0
binaries={p.name:sha(p) for p in sorted(build.glob('ernie-*')) if p.is_file()}
command=['ctest','--preset','linux-vulkan','--no-tests=error']
record={'command':command,'source_identity_sha256':sha(meta),'build_identity_sha256':sha(root/'outputs/d1-preset-vulkan-v5/identity.json'),'binaries':binaries,'worker_sha256':sha(Path(__file__))}
record['gpu_observation_before']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used','--format=csv,noheader'],text=True).strip()
(out/'identity.json').write_text(json.dumps(record,indent=2)+'\n')
start=time.monotonic()
with (out/'test.log').open('w') as log:completed=subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
shutil.copyfile(build/'Testing/Temporary/LastTest.log',out/'LastTest.log')
result={'exit_code':completed.returncode,'wall_seconds':time.monotonic()-start,'source_changes_after':changes(),'binary_changes_after':[name for name,s in binaries.items() if sha(build/name)!=s]}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
raise SystemExit(completed.returncode or bool(result['source_changes_after']) or bool(result['binary_changes_after']))
