import hashlib,json,re
from pathlib import Path
import numpy as np
from PIL import Image
b=Path(__file__).resolve().parent
plan=json.loads((b/'plan.json').read_text()); baseline=Path(plan['baseline_run'])
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,item in plan['bindings'].items():
 path=Path(name)
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
process=json.loads((b/'native/process.json').read_text())
old_process=json.loads((baseline/'native/process.json').read_text())
assert process['complete'] and old_process['complete']
assert sha(baseline/'native/native.png')=='bf87214d21461bfda96529830b28a5b7630ef0dcf307c0172f1666fc64044e28'
old_dir=baseline/'native/trace';new_dir=b/'native/trace'
old_files={p.relative_to(old_dir) for p in old_dir.rglob('*') if p.is_file()}
new_files={p.relative_to(new_dir) for p in new_dir.rglob('*') if p.is_file()}
assert old_files==new_files
rows=[]
for rel in sorted(old_files):
 a=old_dir/rel;c=new_dir/rel;row={'file':str(rel),'bytes':a.stat().st_size,'baseline_sha256':sha(a),'mapped_sha256':sha(c)}
 row['bitwise_equal']=row['baseline_sha256']==row['mapped_sha256']
 if rel.suffix=='.f32':
  x=np.fromfile(a,'<f4');y=np.fromfile(c,'<f4')
  row['elements']=int(x.size);row['finite']=bool(np.isfinite(x).all() and np.isfinite(y).all());row['same_elements']=x.size==y.size
 rows.append(row)
floats=[r for r in rows if 'elements' in r];assert len(floats)==25
image_old=baseline/'native/native.png';image_new=b/'native/native.png'
a=np.array(Image.open(image_old));c=np.array(Image.open(image_new));assert a.shape==c.shape==(512,512,3)
png={'baseline_sha256':sha(image_old),'mapped_sha256':sha(image_new),'pixels_equal':bool(np.array_equal(a,c))}
png['bitwise_equal']=png['baseline_sha256']==png['mapped_sha256']
mappings=process['observed_model_mappings'];assert mappings
assert all(item['permissions']==['r--p'] for item in mappings.values()),mappings
for name,item in mappings.items():
 prefix=str(b/'model')+'/'
 if name.startswith(prefix):name='/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/models/turbo-shared-v2/'+name[len(prefix):]
 item['file_bytes']=Path(name).stat().st_size
log_old=(baseline/'native/command-0.log').read_text();log_new=(b/'native/command-0.log').read_text()
def times(log):
 return {'model_verify_seconds':float(re.search(r'Model verified: ([0-9.]+) s',log).group(1)),
         'denoise_seconds':[float(x) for x in re.findall(r'Denoise [1-8]/8: ([0-9.]+) s',log)]}
result={'scope':'Native runtime512 mapped-loading versus exact prior baseline; all saved tensors and PNG checked byte-for-byte; prior baseline has full official comparison. Single diagnostic timings with uncontrolled cache state, not a paired performance benchmark.',
        'build_identity':json.loads(Path(plan['build_identity']).read_text()),'files':rows,'float_tensors':len(floats),'total_elements':sum(r['elements'] for r in floats),
        'png':png,'mapped_objects_observed':len(mappings),'all_observed_mappings_read_only':True,
        'timings':{'baseline_scope_seconds':old_process['wall_seconds'],'mapped_scope_seconds':process['wall_seconds'],'baseline':times(log_old),'mapped':times(log_new)},
        'passed':all(r['bitwise_equal'] and r.get('finite',True) and r.get('same_elements',True) for r in rows) and png['bitwise_equal']}
(b/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('float_tensors','total_elements','png','mapped_objects_observed','timings','passed')},indent=2))
raise SystemExit(0 if result['passed'] else 1)
