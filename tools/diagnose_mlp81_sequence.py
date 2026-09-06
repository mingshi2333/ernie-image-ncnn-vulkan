#!/usr/bin/env python3
"""One native observer followed by one official baseline/optional MLP, no retries."""
import json,subprocess,sys
from pathlib import Path
from diagnose_official_block_hooks import require,verify,sha
from diagnose_mlp81_contract import NATIVE

def run(plan):
 p=json.loads(plan.read_text());verify(p['bound']);out=Path(p['output'])
 require(not (out/'started.json').exists(),'Experiment already attempted; no retries')
 (out/'started.json').write_text(json.dumps({'plan_sha256':sha(plan),'scope':'native81 observer -> official baseline -> optional matched MLP'},indent=2)+'\n')
 with (out/'native.log').open('w') as log:
  subprocess.run(p['native_command'],stdout=log,stderr=subprocess.STDOUT,check=True)
 native=out/'native';layout=json.loads((out/'native.log').read_text().splitlines()[-1])
 require(layout=={'blob':'81','boundary_whdc':[4096,4160,1,1],'final_whdc':[4096,4160,1,1]},'Invalid observer layout')
 require((native/'boundary.f32').stat().st_size==17039360*4 and (native/'out0.f32').stat().st_size==17039360*4,'Incomplete observer denominator')
 require(sha(native/'out0.f32')==NATIVE,'Observer changed old native teacher')
 verify(p['bound'])
 subprocess.run([sys.executable,str(out/'execution/diagnose_mlp81_official.py'),str(plan)],check=True)
if __name__=='__main__':run(Path(sys.argv[1]))
