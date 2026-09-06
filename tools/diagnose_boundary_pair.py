#!/usr/bin/env python3
"""Execute two frozen conditional inputs; reject boundary evidence unless out0 is exact."""
import json,subprocess,sys
from pathlib import Path
from diagnose_exact_head_stack import sha,require,compare

def run(path):
 p=json.loads(path.read_text());out=Path(p['output'])
 for f,h in p['bound'].items():require(sha(f)==h,'Changed bound bytes '+f)
 require(p['boundary'] in {'75','88','87'},'Only reviewed residual or MLP-down boundary')
 rows=[]
 for side in ['official','native']:
  item=p['sides'][side];dest=out/side
  subprocess.run([p['runner'],p['model'],item['fixture'],str(dest),p['boundary']],check=True)
  actual=sha(dest/'out0.f32');record={'side':side,'out0_sha256':actual,'expected_out0_sha256':item['expected'],'out0_bitwise_equal':actual==item['expected']};rows.append(record)
  require(actual==item['expected'],'Instrumented final output changed; boundary evidence invalid')
  require((dest/'boundary.f32').stat().st_size==4160*(12288 if p['boundary']=='87' else 4096)*4,'Wrong boundary shape')
  record['boundary_sha256']=sha(dest/'boundary.f32')
 result={'scope':'Conditional propagation; not official rounding error or image quality gate','status':'valid_instrumented_pair','plan_sha256':sha(path),'out0_checks':rows,'boundary':p['boundary'],
         'boundary_difference':compare(out/'native/boundary.f32',out/'official/boundary.f32'),'final_difference':compare(out/'native/out0.f32',out/'official/out0.f32'),
         'native_acceptance_eligible':False,'official_gate_applied':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(run(Path(sys.argv[1])),indent=2))
