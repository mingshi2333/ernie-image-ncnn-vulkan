from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, time
root=Path.cwd(); out=root/'outputs/windows-cpu-v1'; build=root/'build-windows-cpu-v1'
base=Path('/var/tmp/ernie-windows-cpu-v1'); base.mkdir()
runtime=base/'runtime'; runtime.mkdir()
search=Path('/usr/x86_64-w64-mingw32/sys-root/mingw/bin')
known={p.name.lower():p for p in search.glob('*.dll')}
imports={}; copied={}; pending=list(build.glob('*.exe'))
while pending:
 file=pending.pop()
 if str(file) in imports: continue
 text=subprocess.check_output(['/usr/bin/x86_64-w64-mingw32-objdump','-p',str(file)],text=True)
 names=re.findall(r'DLL Name: (\S+)',text); imports[str(file)]=names
 for name in names:
  key=name.lower()
  if key in known and key not in copied:
   source=known[key]; target=runtime/source.name; shutil.copy2(source,target)
   copied[key]={'source':str(source),'bytes':source.stat().st_size,'sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest()}
   pending.append(source)
(out/'windows-imports.json').write_text(json.dumps({'imports':imports,'runtime_copies':copied,'other_dlls':'Windows system imports resolved by Wine for this diagnostic; not redistribution approval.'},indent=2)+'\n')
env=os.environ.copy()
for key in ('DISPLAY','WAYLAND_DISPLAY'): env.pop(key,None)
env.update(WINEPREFIX=str(base/'wine-prefix'),WINEARCH='win64',WINEDEBUG='-all',WINEDLLOVERRIDES='winemenubuilder.exe=d',WINEPATH='Z:'+str(runtime).replace('/','\\'),OMP_NUM_THREADS='2')
records=[]
def run(command,name):
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  p=subprocess.run(command,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 records.append({'name':name,'argv':list(map(str,command)),'return_code':p.returncode,'seconds':time.monotonic()-start})
 (out/'wine-results.json').write_text(json.dumps({'scope':'Windows MinGW CPU binaries under Wine on Linux; not native Windows or GPU evidence.','commands':records,'environment_overrides':{k:env[k] for k in ('WINEPREFIX','WINEARCH','WINEDEBUG','WINEDLLOVERRIDES','WINEPATH','OMP_NUM_THREADS')}},indent=2)+'\n')
 return p.returncode
assert run(['/usr/bin/wineboot','-u'],'wineboot')==0
assert run(['/usr/bin/wine',str(build/'ernie-image.exe'),'--help'],'help')==0
assert run(['/usr/bin/wine',str(build/'ernie-image.exe'),'--diagnose'],'diagnose')==0
inventory=json.loads((out/'ctest-inventory.json').read_text())['tests']
native=[t['name'] for t in inventory if t['command'][0]=='/usr/bin/wine']
excluded=[t['name'] for t in inventory if t['name'] not in native]
assert native
(out/'selected-contracts.json').write_text(json.dumps({'native_windows_ctests':native,'python_harness_tests_not_run':excluded},indent=2)+'\n')
code=run(['ctest','--test-dir',str(build),'-R','^('+'|'.join(map(re.escape,native))+')$','--parallel','1','--output-on-failure','--output-junit',str(out/'ctest-windows.xml')],'ctest')
raise SystemExit(code)
