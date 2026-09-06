import json,sys
from pathlib import Path
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p/'source/tools'))
from diagnose_exact_head_stack import compare,sha
d=json.loads((p/'plan.json').read_text());r={}
for mode in ['off','on']:
 a=p/mode/'actual.f32';assert a.stat().st_size==4160*4096*4;assert sha(a)==sha(p/mode/'trace/block-0.f32')
 r[mode]={'sha256':sha(a),**compare(a,d['expected']['file'])}
 records=[]
 for i,expected_count in enumerate([1,4,18]):
  parsed=[json.loads(line) for line in (p/mode/f'command-{i}.log').read_text().splitlines() if line.startswith('{')]
  if i==2:
   meta=parsed.pop(0)
   assert meta=={'ncnn_revision':'6a1bf000f363714839a36793addc8c879d3d899e','q_heads':32,'kv_heads':8,'head_dim':128,'backend':'vulkan','runtime_layer':True}
  assert len(parsed)==expected_count and all(x['passed'] is True for x in parsed)
  records.extend(parsed)
 r[mode]['operator_cases']=len(records)
r['values']=4160*4096;r['baseline_exact']=r['off']['sha256']==d['baseline']['sha256']
r['advance_full_image']=r['on']['error_l2']<r['off']['error_l2'] and r['on']['max_abs_error']<r['off']['max_abs_error']
r['initial_review_note']='Two parser errors included the non-case cache metadata record in23 expected cases. Fixed explicit metadata/case enumeration; no model rerun.'
(p/'comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
