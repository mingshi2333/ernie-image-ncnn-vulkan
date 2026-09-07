"""Verify frozen trace-OFF benchmark integration; no speedup or new tensor gate."""
import hashlib
import json
from pathlib import Path
from PIL import Image

root=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base=Path('/var/tmp/ernie-benchmark-runtime512-v1')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
plan=json.loads((base/'plan.json').read_text())
for name,binding in plan['bindings'].items():
    path=Path(name)
    assert path.stat().st_size==binding['bytes'] and sha(path)==binding['sha256'],name
process=json.loads((base/'native/process.json').read_text())
worker=json.loads((base/'native/worker-result.json').read_text())
job=base/'native/benchmark'
r=json.loads((job/'result.json').read_text());native=json.loads((job/'generation.json').read_text())
assert process['complete'] and process['return_code']==0 and all(c['return_code']==0 for c in worker)
assert r['passed'] and r['status']=='ok' and r['selection_status']=='verified_native_report'
assert r['native_report']==native and r['native_report_sha256']==sha(job/'generation.json')
assert r['formal_comparison_eligible'] and not native['trace_enabled'] and not native['allocation_instrumentation']
assert not (job/'trace').exists()
assert r['runner_sha256']==sha(base/'ernie-image.snapshot')==sha(job/'ernie-image.snapshot')
assert r['noise_sha256']==sha(base/'initial.f32')==sha(job/'initial.f32')
assert r['package_manifest_sha256']==plan['package_manifest_sha256']
assert native['shape']==[512,512] and native['model']=={'schema_version':3,'source_width':1024,'source_height':1024,'text_bucket':32,'dit_text_tokens':64}
assert len(native['token_ids'])==15 and len(r['denoise_seconds'])==8
assert native['request']['dit_cache_mib']==6144 and native['request']['gpu_reserve_mib']==8192
assert native['model_loading_requested']=='mapped'
assert native['placement_requests']['gpu']==0 and native['placement_requests']['budget_unavailable']==0
assert native['weight_cache']['hits']+native['weight_cache']['loads']==288
assert native['placement_requests']['ram']==native['weight_cache']['loads']+16
assert native['weight_cache']['peak_charged_bytes']<=6144*1024*1024
baseline=Path(plan['baseline_image'])
assert (job/'native.png').read_bytes()==baseline.read_bytes()
with Image.open(job/'native.png') as image:
    image.load();assert image.size==(512,512) and image.mode=='RGB'
limits=plan['resource_limits']
assert process['minimum_host_available']>=limits['host_available_min_bytes']
assert process['peak_gpu_whole_device_mib']<=limits['gpu_whole_device_max_mib']
events={k:int(v) for k,v in (line.split() for line in process['memory_events'].splitlines())}
assert events.get('oom',0)==events.get('oom_kill',0)==0
for pid in process['observed_native_pids']:assert not Path(f'/proc/{pid}').exists()
result={'passed':True,'scope':'Actual shared-package trace-OFF timing-tool integration and exact PNG regression; one run, not formal paired speed or new full-tensor validation',
        'source_bindings':len(plan['bindings']),'plan_sha256':sha(base/'plan.json'),
        'binary_sha256':r['runner_sha256'],'image_sha256':sha(job/'native.png'),
        'wrapper_native_process_seconds':r['wall_seconds'],'native_report_total_seconds':r['total_seconds'],
        'supervisor_seconds':process['wall_seconds'],'max_rss_kib':r.get('max_rss_kib'),
        'native_memory_observed':process.get('peak_sampled_native_memory'),
        'whole_gpu_peak_mib':process['peak_gpu_whole_device_mib'],
        'cgroup_peak_including_file_cache_bytes':process['peak_memory_current'],'memory_events':events,
        'placement_requests':native['placement_requests'],'cache':native['weight_cache'],
        'official_quality_revalidated':False,'formal_speed_comparison_complete':False}
(base/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
