import json,hashlib
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'outputs/q2-block15-splitk-teacher-v1';shape=(4160,4096)
def tensor(p):return np.memmap(p,'<f4','r',shape=shape)
def norm(a):return float(np.linalg.norm(a))
r=tensor(root/'outputs/q2-block15-boundary-v1/official/boundary.f32');old=tensor(root/'outputs/q2-block15-boundary88-v1/official/boundary.f32');new=tensor(out/'candidate/boundary.f32');gate=np.fromfile(out/'fixture/in6.f32','<f4');a=tensor(out/'candidate/out0.f32');b=tensor(root/'outputs/q2-block15-boundary87-v1/official/out0.f32')
u=(old*gate).astype('f4');v=(new*gate).astype('f4')
assert np.array_equal((r+u).astype('f4').view('u4'),b.view('u4'));assert np.array_equal((r+v).astype('f4').view('u4'),a.view('u4'))
down=new.astype('f8')-old;update=v.astype('f8')-u;final=a.astype('f8')-b
result={'scope':'Complete same-residual conditional update decomposition; not independent official internal ground truth','elements':a.size,'both_final_residuals_bitwise_reconstructed':True,'down_delta_l2':norm(down),'fp32_gated_update_delta_l2':norm(update),'final_delta_l2':norm(final),'final_add_rounding_delta_l2':norm(final-update),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_identity':'All tensors are bound by plan.json and result.json of the same frozen teacher; no inferred reference down tensor'}
(out/'residual-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
