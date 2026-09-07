from pathlib import Path
import json,os,subprocess,sys,time
root=Path.cwd();out=root/'outputs/windows-cpu-v1';platform=sys.argv[1];env=os.environ.copy()
commands=[]
def run(args,name,required=True):
 start=time.monotonic()
 with (out/(name+'.log')).open('wb') as log:
  p=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 commands.append({'name':name,'argv':args,'return_code':p.returncode,'required':required,'seconds':time.monotonic()-start})
 (out/(platform+'-utf8-validation.json')).write_text(json.dumps({'scope':platform+' CPU; Windows executables run under Wine','commands':commands},indent=2)+'\n')
 return p.returncode if required else 0
if platform=='windows':
 for key in ('DISPLAY','WAYLAND_DISPLAY'):env.pop(key,None)
 env.update(WINEPREFIX='/var/tmp/ernie-windows-cpu-v1/wine-prefix',WINEARCH='win64',WINEDEBUG='-all',
  WINEDLLOVERRIDES='winemenubuilder.exe=d',WINEPATH='Z:\\var\\tmp\\ernie-windows-cpu-v1\\runtime',OMP_NUM_THREADS='2',
  ERNIE_TEST_RUNNER=str(out/'image-via-wine'),ERNIE_REPORT_RUNNER=str(out/'report-via-wine'))
 # This retained server is confined to the service and owned prefix. Its exit
 # status and all test timings are retained; these are not performance data.
 run(['/usr/bin/wineserver','-p60'],'wine-persistence-utf8',False)
 build=root/'build-windows-cpu-v1'
 inventory=json.loads(subprocess.check_output(['ctest','--test-dir',str(build),'--show-only=json-v1'],env=env))
 selected=[t['name'] for t in inventory['tests'] if t.get('command',[None])[0]=='/usr/bin/wine']
 (out/'ctest-utf8-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 regex='^('+'|'.join(selected)+')$'
 failed=run(['ctest','--test-dir',str(build),'-R',regex,'--output-on-failure','--output-junit',str(out/'ctest-utf8-windows.xml')],'ctest-utf8-windows')
 failed=run(['/usr/bin/python3','-m','unittest','tests.test_cli','tests.test_generation_report','-v'],'python-utf8-windows') or failed
 failed=run(['/usr/bin/python3',str(out/'check-native-links.py')],'native-windows-links') or failed
else:
 build=root/'build-install-cpu'
 failed=run(['ctest','--test-dir',str(build),'--parallel','2','--output-on-failure','--output-junit',str(out/'ctest-utf8-linux.xml')],'ctest-utf8-linux')
raise SystemExit(failed)
