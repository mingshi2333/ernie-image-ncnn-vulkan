"""Verify frozen identities and the actual execution namespaces without models."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parent
ROOT=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan')
SIZES=((1376,768),(768,1376),(2048,1024),(1024,2048))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
records=[]
bindings=0
for name,item in json.loads((BASE/'batch-identity.json').read_text()).items():
    path=Path(name);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
for width,height in SIZES:
    label=f'{width}x{height}';case=BASE/label
    plan=json.loads((case/'plan.json').read_text())
    request=json.loads((case/'request.json').read_text())
    assert request['runtime_size']==[width,height]
    assert (case/'initial.f32').stat().st_size==128*(width//16)*(height//16)*4
    for name,item in plan['bindings'].items():
        path=Path(name);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
        bindings+=1
    for path in case.glob('*.py'):compile(path.read_text(),str(path),'exec')
    for phase in ('native','official'):
        command=plan['commands'][phase][0];prefix=command[:command.index('--')+1]
        if phase=='native':
            assert command[command.index('--width')+1]==str(width)
            assert command[command.index('--height')+1]==str(height)
            assert command[command.index('--dit-weights')+1]=='host'
            assert command[command.index('--dit-cache-mib')+1]=='0'
            assert command[command.index('--model-loading')+1]=='stdio'
            for name,args in (('hidden-original-source',['/usr/bin/test','!','-e',str(ROOT/'src')]),
                              ('native-help',[str(BASE/'ernie-image.snapshot'),'--help'])):
                result=subprocess.run([*prefix,*args],capture_output=True,text=True,timeout=15)
                assert result.returncode==0,(label,name,result.stderr)
                records.append({'case':label,'check':name,'return_code':result.returncode})
        result=subprocess.run([*prefix,'/usr/bin/readlink','/proc/self/ns/net'],capture_output=True,text=True,timeout=15)
        assert result.returncode==0 and result.stdout.strip()!=os.readlink('/proc/self/ns/net')
        records.append({'case':label,'check':phase+'-isolated-network','return_code':0})
records.append({'check':'frozen-bindings-and-case-syntax','bindings':bindings,'passed':True})
with (BASE/'preflight.json').open('x') as stream:stream.write(json.dumps(records,indent=2)+'\n')
print(json.dumps({'preflight_checks':len(records),'bindings':bindings,'passed':True}),flush=True)
