#!/usr/bin/env python3
"""Approved one-way entry: launcher -> guard/plan -> sealed official-only payload."""
import hashlib,json,os,subprocess
from pathlib import Path
PLAN_SHA='621ef5ab8cb3c800f65306e416b37d94defa54e39761d1312190d966f5238e53'
GUARD_SHA='90c4fd6d9556e7675806d58fe5634f6f20bc6c6d856fb31ed930cc9b098ad1cc'
UNIT='ernie-q2-up84-v1'

def launch(root):
 out=root/'outputs/q2-up84-v1';guard=out/'execution/diagnose_up84_scope_guard.py';plan=out/'plan.json'
 for file,digest in [(guard,GUARD_SHA),(plan,PLAN_SHA)]:
  if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise ValueError('Changed approved input '+str(file))
 env={**os.environ,'XDG_RUNTIME_DIR':f'/run/user/{os.getuid()}','DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{os.getuid()}/bus'}
 command=['systemd-run','--user','--scope','--unit='+UNIT,'-p','MemoryMax=10G','-p','MemorySwapMax=0','-p','CPUQuota=200%','-p','AllowedCPUs=0,2','taskset','-c','0,2',str(root/'.venv/bin/python'),str(guard),str(plan)]
 # Preserve the launch even when the bus/scope fails before the guard can run.
 record=out/'execution/launch-record.json'
 with record.open('x') as f:json.dump({'command':command,'guard_sha256':GUARD_SHA,'plan_sha256':PLAN_SHA,'env':{k:env[k] for k in ['XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS']}},f,indent=2)
 with (out/'execution/scope-launch.log').open('x') as log:
  result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 (out/'execution/scope-exit.json').write_text(json.dumps({'return_code':result.returncode})+'\n')
 return result.returncode
if __name__=='__main__':raise SystemExit(launch(Path.cwd()))
