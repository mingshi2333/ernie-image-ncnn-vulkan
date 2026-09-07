from pathlib import Path
import json, os, subprocess
root=Path.cwd();out=root/'outputs/windows-cpu-v1';base=Path('/var/tmp/ernie-wine-link-semantics-v1')
base.mkdir(exist_ok=False)
(base/'target').write_text('known content\n');(base/'target-directory').mkdir()
(base/'posix-file-link').symlink_to(base/'target')
(base/'posix-directory-link').symlink_to(base/'target-directory',target_is_directory=True)
rows=[]
def run(command,name,env=None):
 p=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True,timeout=120)
 (out/(name+'.log')).write_text(p.stdout+p.stderr)
 rows.append({'name':name,'argv':list(map(str,command)),'return_code':p.returncode})
 (out/'link-semantics-results.json').write_text(json.dumps({'scope':'Windows Rust filesystem observation under Wine on Linux-created symbolic links','commands':rows,'linux':[{'name':p.name,'is_symlink':p.is_symlink(),'resolved':str(p.resolve())} for p in sorted(base.iterdir())]},indent=2)+'\n')
 if p.returncode:raise RuntimeError(name)
run(['/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/rustc',out/'link-semantics.rs','--edition=2024','--target=x86_64-pc-windows-gnu','-C','linker=/usr/bin/x86_64-w64-mingw32-gcc','-o',base/'link-semantics.exe'],'link-semantics-build')
env=os.environ.copy()
for key in ('DISPLAY','WAYLAND_DISPLAY'):env.pop(key,None)
env.update(WINEPREFIX='/var/tmp/ernie-windows-cpu-v1/wine-prefix',WINEARCH='win64',WINEDEBUG='-all',WINEDLLOVERRIDES='winemenubuilder.exe=d')
run(['/usr/bin/wine',base/'link-semantics.exe',base],'link-semantics-windows',env)
