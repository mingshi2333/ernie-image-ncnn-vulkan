from pathlib import Path
import json,os,subprocess,sys,time
platform=sys.argv[1];root=Path.cwd();out=root/'outputs/windows-cpu-v1'
env=os.environ.copy()
if platform=='windows':
 build=root/'build-windows-cpu-v1'
 env.update(RUSTC='/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/rustc',
   CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER='/usr/bin/x86_64-w64-mingw32-gcc')
else:build=root/'build-install-cpu'
cmd=['cmake','--build',str(build),'--parallel','2'];start=time.monotonic()
with (out/(platform+'-utf8-build.log')).open('w') as log:
 p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
(out/(platform+'-utf8-build.json')).write_text(json.dumps({'platform':platform,'argv':cmd,'return_code':p.returncode,'seconds':time.monotonic()-start},indent=2)+'\n')
raise SystemExit(p.returncode)
