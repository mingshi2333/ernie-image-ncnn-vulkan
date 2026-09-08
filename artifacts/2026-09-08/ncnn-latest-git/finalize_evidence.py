"""Archive only complete, rechecked evidence; retain numerical failures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

base=Path(__file__).resolve().parent
work=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
archive=work/'artifacts/2026-09-08/ncnn-latest-git'
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
progress=read(base/'full-progress.json')
assert progress['status']=='complete' and len(progress['completed'])==5
plan=read(base/'full-plan.json')
checked=[]
for path,record in plan['bindings'].items():
    assert Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'],path
    checked.append(path)
assert sha(base/'full-plan.json')==progress['plan_sha256']
assert sha(base/'run_full.py')==progress['supervisor_sha256']
ctest=ET.parse(base/'ctest.xml').getroot()
assert all(ctest.attrib[key]==value for key,value in {'tests':'57','failures':'0','skipped':'0','disabled':'0'}.items())
blocks=read(base/'block-results.json')
assert len(blocks)==9 and all(row['bitwise_equal'] for row in blocks)
full=read(base/'full-comparison.json')
assert len(full['results'])==5 and sha(base/'compare_full.py')==full['comparator_sha256']
assert full['plan_sha256']==progress['plan_sha256']
summary={'project_revision':read(base/'frozen-identity.json')['project_head'],
    'baseline_ncnn':plan['base_ncnn'],'candidate_ncnn':plan['candidate_ncnn'],
    'complete':True,'platform':'Linux / NVIDIA RTX 4060 Laptop 8GB',
    'ctest':{key:int(ctest.attrib[key]) for key in ['tests','failures','skipped','disabled']},
    'matched_real_weight_cases':len(blocks),'real_weight_executions':2*len(blocks),
    'bitwise_equal_real_weight_cases':sum(row['bitwise_equal'] for row in blocks),
    'real_weight_elements_per_version':sum(row['elements'] for row in blocks),
    'post_run_rechecked_file_bindings':len(checked),'full_runs':[],
    'formal_performance_claim':False,'production_pin_changed':False,'new_remote_ci_run':False}
for result in full['results']:
    case=result['case'];out=base/'full'/case
    worker=read(out/'worker-result.json')
    events=dict((key,int(value)) for key,value in (line.split() for line in worker['memory_events'].splitlines()))
    assert result['process']['complete'] and worker['return_code']==0 and events['oom']==events['oom_kill']==0
    entry={'case':case,'official_passed_tensors':result['passed_tensors'],'official_total_tensors':result['total_tensors'],
        'official_numerical_gate_passed':result['passed'],'failed_tensors':[r['name'] for r in result['tensors'] if not r['passed']],
        'official_png':result['png'],'compared_elements':result['total_elements'],
        'scope_wall_seconds':result['process']['wall_seconds'],'cgroup_memory_peak_bytes':worker['memory_peak_bytes'],
        'whole_gpu_sampled_peak_mib':result['process']['sampled_gpu_whole_device_peak_mib'],
        'memory_events':events}
    if 'old_new' in result:
        pair=result['old_new']
        assert len(pair['tensors'])==25
        entry['old_new']={k:v for k,v in pair.items() if k!='tensors'}
    summary['full_runs'].append(entry)
    dest=archive/'full'/case;dest.mkdir(parents=True,exist_ok=True)
    for name in ['command.json','process.json','worker-result.json','generation.json','scope.log']:
        shutil.copy2(out/name,dest/name)
    if case.startswith('candidate-'):
        shutil.copy2(out/'native.png',work/'outputs/ncnn-latest-git-v1'/('apple-'+case.removeprefix('candidate-')+'.png'))
for version in ('baseline','candidate'):
    rows=[json.loads(line) for line in (base/f'precision-{version}-v3.jsonl').read_text().splitlines()]
    assert len(rows)==22
    summary[version+'_precision_probe']={kind:{'passed':sum(row['passed'] for row in rows if row['kind']==kind),
        'total':sum(row['kind']==kind for row in rows)} for kind in ('conversion','reduction')}
summary['candidate_compatibility_regression_passed']=all(
    row['old_new']['bitwise_equal_tensors']==25 and row['old_new']['png_bitwise_equal']
    for row in summary['full_runs'] if row['case'].startswith('candidate-'))
for name in ['full-progress.json','full-comparison.json','compare_full.py','historical-fp32-anchor.json','finalize_evidence.py']:
    shutil.copy2(base/name,archive/name)
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(base/'summary.json',archive/'summary.json')
print(json.dumps(summary,indent=2),flush=True)
