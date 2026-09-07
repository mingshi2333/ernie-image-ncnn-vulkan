import hashlib
import json
from pathlib import Path
import shutil
import subprocess
base=Path(__file__).resolve().parent
root=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
results=[]
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def run(name,command):
 with (base/(name+'.log')).open('xb') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 results.append({'name':name,'command':command,'return_code':result.returncode})
 (base/'build-results.json').write_text(json.dumps(results,indent=2)+'\n')
 if result.returncode:raise RuntimeError(name+' failed')
try:
 run('configure-on',['cmake','-S',str(root),'-B',str(root/'build-dev'),'-DERNIE_EXPERIMENT_MAPPED_MODEL_LOADING=ON'])
 run('build-on',['cmake','--build',str(root/'build-dev'),'--target','ernie-image','--parallel','2'])
 shutil.copy2(root/'build-dev/ernie-image',base/'ernie-image.on.snapshot')
finally:
 run('configure-off',['cmake','-S',str(root),'-B',str(root/'build-dev'),'-DERNIE_EXPERIMENT_MAPPED_MODEL_LOADING=OFF'])
 run('build-off',['cmake','--build',str(root/'build-dev'),'--target','ernie-image','--parallel','2'])
result={'baseline':sha(base/'ernie-image.off.snapshot'),'mapped':sha(base/'ernie-image.on.snapshot'),'restored':sha(root/'build-dev/ernie-image')}
result['restored_exact']=result['baseline']==result['restored']
(base/'binary-identity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert result['restored_exact']
