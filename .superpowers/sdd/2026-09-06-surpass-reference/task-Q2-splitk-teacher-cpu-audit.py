import json,hashlib
from pathlib import Path
import numpy as np
root=Path.cwd();o=root/'outputs/q2-block15-splitk-teacher-v1';p=json.loads((o/'plan.json').read_text());f=p['fields'];shape=(4160,4096)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as a:
  for chunk in iter(lambda:a.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path,shape):return np.memmap(path,'<f4','r',shape=shape)
def metric(a,b):
 d=a.astype('f8')-b;return {'l2':float(np.linalg.norm(d)),'max':float(abs(d).max())}
a=read(o/'candidate/out0.f32',shape);b=read(f['baseline'],shape);official=read(f['oracle'],shape)
e=a.astype('f8')-official;row,col=np.unravel_index(np.argmax(abs(e)),shape);row2=int(np.argmax(np.linalg.norm(e,axis=1)));rows=sorted(set([int(row),row2]));del e
x=read(o/'candidate/upstream87.f32',(4160,12288))[rows].astype('f8');w=read(f['screen_weight'],(12288,4096));precise=x@w.astype('f8')
down=read(o/'candidate/boundary.f32',shape)[rows];old=read(root/'outputs/q2-block15-boundary88-v1/official/boundary.f32',shape)[rows]
base_error=b.astype('f8')-official;change=a.astype('f8')-b
result={'scope':'Post-result CPU FP64 spot audit, selected by observed complete out0 errors; not predeclared screen or full dot denominator','rows':rows,'selected_row_reasons':{'global_max_abs_error_row':int(row),'global_max_abs_error_column':int(col),'largest_row_error_l2':row2},'selected_elements':len(rows)*4096,'candidate_down_vs_same_input_fp64':metric(down,precise),'baseline_down_vs_same_input_fp64':metric(old,precise),'error_change_cosine':float(np.sum(base_error*change)/(np.linalg.norm(base_error)*np.linalg.norm(change))),'oracle_norm':float(np.linalg.norm(official.astype('f8'))),'files':{str(q.resolve()):sha(q) for q in [o/'result.json',o/'plan.json',o/'candidate/out0.f32',o/'candidate/boundary.f32',o/'candidate/upstream87.f32',Path(f['screen_weight']),Path(f['oracle']),Path(f['baseline']),Path(__file__)]},'interpretation':'Complete teacher worsened despite selected same-input dot accuracy; cannot distinguish upstream/reference cancellation without independent official internal boundaries'}
(o/'post-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
