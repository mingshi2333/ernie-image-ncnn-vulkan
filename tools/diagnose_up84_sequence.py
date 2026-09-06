#!/usr/bin/env python3
import json,subprocess,sys
from pathlib import Path
from diagnose_official_block_hooks import sha,verify,require
from diagnose_up84_official import N81
NOUT='175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3'
def native_gate(out):
 import numpy as np
 for key,width,expected in [('81',4096,N81),('out0',4096,NOUT),('84',12288,None)]:
  f=out/'native'/f'{key}.f32';require(f.stat().st_size==4160*width*4,'Wrong native full denominator')
  require(np.isfinite(np.memmap(f,dtype='<f4',mode='r')).all(),'Nonfinite native '+key)
  if expected:require(sha(f)==expected,'Native observer changed '+key)
 require((out/'native.log').read_text().splitlines()[-1]=='UP84_COMPLETE_LAYOUT_OK','Native layout proof missing')
def run(path):
 p=json.loads(path.read_text());verify(p['bound']);out=Path(p['output'])
 require(not (out/'sequence-started.json').exists(),'Already attempted');(out/'sequence-started.json').write_text(json.dumps({'plan_sha256':sha(path)})+'\n')
 with (out/'native.log').open('w') as log:subprocess.run(p['native_command'],stdout=log,stderr=subprocess.STDOUT,check=True)
 native_gate(out);verify(p['bound'])
 subprocess.run([sys.executable,str(out/'execution/diagnose_up84_official.py'),str(path)],check=True)
if __name__=='__main__':run(Path(sys.argv[1]))
