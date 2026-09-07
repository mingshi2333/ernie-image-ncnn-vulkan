from pathlib import Path
import json, os, subprocess, time
root=Path.cwd(); out=root/'outputs/windows-cpu-v1'; build=root/'build-windows-cpu-v1'
env=os.environ.copy()
env['RUSTC']='/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/rustc'
env['CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER']='/usr/bin/x86_64-w64-mingw32-gcc'
records=[]
def run(args, name, expected=0):
 started=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  result=subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
 records.append({'name':name,'argv':args,'return_code':result.returncode,'seconds':time.monotonic()-started,'expected':expected})
 (out/'build-commands.json').write_text(json.dumps({'commands':records,'environment_overrides':{k:env[k] for k in ('RUSTC','CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER')}},indent=2)+'\n')
 if result.returncode!=expected: raise SystemExit(f'{name}: {result.returncode}, expected {expected}')
args=['cmake','-S',str(root),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_TOOLCHAIN_FILE='+str(out/'toolchain.cmake'),'-DERNIE_NCNN_SOURCE_DIR=/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/third_party/ncnn','-DERNIE_ENABLE_VULKAN=OFF','-DERNIE_BUILD_TOKENIZER=ON','-DERNIE_BUILD_GENERATOR=ON','-DERNIE_INSTALL_SDK=ON','-DERNIE_EXPERIMENT_COMPACT_MODEL_READER=OFF','-DNCNN_INT8=OFF','-DNCNN_WEIGHT_QUANT=OFF','-DCARGO_EXECUTABLE=/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/cargo']
run(args,'configure-missing-rust-target',1)
assert 'Cross-compiling the tokenizer requires ERNIE_RUST_TARGET' in (out/'configure-missing-rust-target.log').read_text()
run(args+['-DERNIE_RUST_TARGET=x86_64-pc-windows-gnu'],'configure')
run(['cmake','--build',str(build),'--parallel','2'],'build')
