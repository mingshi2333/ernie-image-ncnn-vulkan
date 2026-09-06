#!/usr/bin/env python3
"""Independent CPU-only full-denominator arithmetic reconstruction check."""
import json,sys
from pathlib import Path
import numpy as np
from diagnose_official_block_hooks import sha,require

def run(root,dest):
 require(not dest.exists(),'New output required');out=root/'outputs/q2-block15-mlp81-official-v5';src=out/'analysis-v2.json'
 require(sha(src)=='d19c2b2900f2355f87348aa1e052cb77cfd143bae1e1efd469fca8ba569d6814','Analysis anchor')
 a=json.loads(src.read_text());result={}
 for name,width in [('87',12288),('88',4096)]:
  paths=[root/f'outputs/q2-block15-boundary{name}-v1/official/boundary.f32',out/f'official/{name}.f32',out/f'official/native81-{name}.f32']
  tensors=[]
  for p in paths:
   require(sha(p)==a['identity'][str(p)] and p.stat().st_size==4160*width*4,'Tensor identity/denominator')
   tensors.append(np.memmap(p,dtype='<f4',mode='r',shape=(4160,width)))
  max_error=0.;nonzero=0;count=0
  for row in range(0,4160,64):
   n,o,m=[x[row:row+64].astype(np.float64) for x in tensors]
   require(all(np.isfinite(x).all() for x in [n,o,m]),'Nonfinite source')
   residual=(n-o)-((m-o)+(n-m));max_error=max(max_error,float(np.abs(residual).max()));nonzero+=int(np.count_nonzero(residual));count+=residual.size
  result[name]={'elements':count,'max_abs_reconstruction_residual':max_error,'nonzero_residual_elements':nonzero}
 payload={'source_sha256':sha(__file__),'analysis_sha256':sha(src),'arithmetic':'Load saved little-endian FP32 bytes. Independently cast N,O,M to FP64 before subtraction. Compute D=N-O, U=M-O, E=N-M and R=D-(U+E), no fitted coefficients. Finite checks and all rows, chunks64. No model execution.','results':result,'scope':'Arithmetic reconstruction, not causal contribution percentages','formal_acceptance':False}
 dest.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':run(Path.cwd().resolve(),Path(sys.argv[1]).resolve())
