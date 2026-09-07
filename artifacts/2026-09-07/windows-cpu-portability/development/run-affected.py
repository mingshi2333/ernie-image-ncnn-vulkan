from pathlib import Path
import json,os,subprocess,time
root=Path.cwd();out=root/'outputs/windows-cpu-v1';env=os.environ.copy()
for key in ('DISPLAY','WAYLAND_DISPLAY'):env.pop(key,None)
env.update(WINEPREFIX='/var/tmp/ernie-windows-cpu-v1/wine-prefix',WINEARCH='win64',WINEDEBUG='-all',WINEDLLOVERRIDES='winemenubuilder.exe=d',WINEPATH='Z:\\var\\tmp\\ernie-windows-cpu-v1\\runtime',OMP_NUM_THREADS='2',ERNIE_TEST_RUNNER=str(out/'image-via-wine'),ERNIE_REPORT_RUNNER=str(out/'report-via-wine'))
commands=[]
def run(args,name):
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 commands.append({'name':name,'argv':args,'return_code':result.returncode,'seconds':time.monotonic()-start})
 (out/'affected-results.json').write_text(json.dumps({'scope':'Cross-compiled Windows CPU executables under Wine, including Linux Python fixtures using explicit Wine wrappers.','commands':commands},indent=2)+'\n')
 return result.returncode
failed=run(['ctest','--test-dir',str(root/'build-windows-cpu-v1'),'-R','^(generation_report_writer|text_down_contract_cpu)$','--output-on-failure','--output-junit',str(out/'ctest-affected.xml')],'ctest-affected')
failed=run(['/usr/bin/python3','-m','unittest','tests.test_generation_report','tests.test_cli','tests.test_package','tests.test_pe_package','-v'],'python-windows') or failed
raise SystemExit(failed)
