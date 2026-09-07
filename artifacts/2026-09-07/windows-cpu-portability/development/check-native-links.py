"""Windows API creates these links; original POSIX/Wine failures stay recorded."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
root=Path.cwd();out=root/'outputs/windows-cpu-v1';base=Path('/var/tmp/ernie-windows-native-links-v1')
base.mkdir(exist_ok=False);sys.path.insert(0,str(root))
from tests.test_package import PackageTests
from tests.test_pe_package import PePackageTests
env=os.environ.copy()
for key in ('DISPLAY','WAYLAND_DISPLAY'):env.pop(key,None)
env.update(WINEPREFIX='/var/tmp/ernie-windows-cpu-v1/wine-prefix',WINEARCH='win64',WINEDEBUG='-all',
 WINEDLLOVERRIDES='winemenubuilder.exe=d',WINEPATH='Z:\\var\\tmp\\ernie-windows-cpu-v1\\runtime')
result={'scope':'Actual Windows Rust symlink creation and package verification under Wine; model-free, not native Windows acceptance','status':'running','commands':[]}
def run(cmd,name,expected=0,contains=None):
 start=time.monotonic();p=subprocess.run(list(map(str,cmd)),env=env,capture_output=True,text=True,timeout=90)
 (base/(name+'.log')).write_text(p.stdout+p.stderr)
 result['commands'].append({'name':name,'argv':list(map(str,cmd)),'return_code':p.returncode,'seconds':time.monotonic()-start})
 if p.returncode!=expected or contains and contains not in p.stdout+p.stderr:raise RuntimeError(name+': '+p.stdout+p.stderr)
try:
 helper=base/'link-semantics.exe'
 run(['/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/rustc',out/'link-semantics.rs','--edition=2024','--target=x86_64-pc-windows-gnu','-C','linker=/usr/bin/x86_64-w64-mingw32-gcc','-o',helper],'build-helper')
 for name,kind,relative in [('image-file','file','vae/bn-mean.f32'),('image-directory','directory','vae'),('pe-file','file','block-00/pe.ncnn.bin')]:
  case=base/name;case.mkdir();fixture=(PePackageTests if name=='pe-file' else PackageTests)();fixture.setUp()
  try:
   (fixture.root/'manifest.json').write_text(json.dumps(fixture.manifest));shutil.copytree(fixture.root,case/'model')
  finally:fixture.tearDown()
  flag='--pe-model' if name=='pe-file' else '--model';cli=root/'build-windows-cpu-v1/ernie-image.exe'
  run(['/usr/bin/wine',cli,flag,case/'model','--verify-model'],name+'-before',contains='Model verified')
  path=case/'model'/relative;target=case/'external';path.rename(target)
  run(['/usr/bin/wine',helper,'create-'+kind,target,path],name+'-create',contains='symlink=true')
  run(['/usr/bin/wine',cli,flag,case/'model','--verify-model'],name+'-reject',1,'contains a symlink')
 result['status']='passed'
except BaseException as error:
 result.update(status='failed',error=str(error));raise
finally:(base/'result.json').write_text(json.dumps(result,indent=2)+'\n')
