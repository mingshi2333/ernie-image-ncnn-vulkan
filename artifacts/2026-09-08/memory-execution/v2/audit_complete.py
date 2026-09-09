"""Read complete run records and verify the declared execution mechanisms."""
import hashlib, json
from pathlib import Path

base = Path(__file__).resolve().parent
plan = json.loads((base/'full-plan.json').read_text())
comparison = json.loads((base/'full-comparison.json').read_text())
preflight = json.loads((base/'preflight.json').read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert comparison['plan_sha256'] == preflight['plan_sha256'] == sha(base/'full-plan.json')
assert preflight['all_passed'] and preflight['source_head'] == plan['execution_source_head']
assert sha(base/'frozen-bin/ernie-image') == preflight['binary']['sha256']
assert comparison['comparator_sha256'] == plan['bindings'][str(base/'compare_full.py')]['sha256']
assert len(comparison['results']) == len(plan['cases']) == 3
progress = json.loads((base/'full-progress.json').read_text())
assert progress['status'] == 'complete'
results = []
for result in comparison['results']:
    case = result['case']
    path = base/'full'/case
    report = json.loads((path/'generation.json').read_text())
    worker = json.loads((path/'worker-result.json').read_text())
    process = result['process']
    command = json.loads((path/'command.json').read_text())
    assert process['complete'] and process['return_code'] == worker['return_code'] == 0
    assert process['command'] == worker['command'] == command
    assert command[0] == str(base/'frozen-bin/ernie-image')
    for field, value in {'gpu_memory':'auto', 'gpu_spill_mib':2048, 'oom_retries':3,
                         'ram_reserve_mib':3072, 'dit_weights':'host', 'model_loading':'stdio',
                         'threads':2, 'steps':8, 'device':'vulkan', 'vae_device':'cpu'}.items():
        assert report['request'][field] == value, (case, field)
    limits = plan['resource_limits']
    assert process['minimum_host_available_bytes'] >= limits['host_available_min_bytes']
    assert process['sampled_gpu_whole_device_peak_mib'] <= limits['gpu_whole_device_max_mib']
    assert process['wall_seconds'] <= limits['timeout_seconds_per_phase']
    assert worker['observed_limits']['memory.max'] == str(limits['memory_max_bytes'])
    assert worker['observed_limits']['memory.swap.max'] == '0'
    events = dict((k, int(v)) for k, v in (line.split() for line in worker['memory_events'].splitlines()))
    assert events['oom'] == events['oom_kill'] == 0
    text = (path/'native.log').read_text()
    assert 'VUID-' not in text and 'Validation Error' not in text
    memory, prefetch, recovery = (report[k] for k in ['gpu_memory', 'weight_prefetch', 'memory_recovery'])
    assert memory['host_peak_bytes'] <= 2048*1024*1024
    assert memory['host_allocations'] == memory['host_device_local_allocations']+memory['host_non_device_local_allocations']
    assert 0 <= recovery['retries'] <= 3
    checks = {'complete_and_valid': True}
    if case == 'mixed-prefetch-fp32':
        assert report['request']['gpu_reserve_mib'] == 5400
        assert report['request']['dit_cache_mib'] == report['request']['dit_prefetch_mib'] == 1024
        checks.update({
            'actual_device_and_host_buffers': memory['device_allocations'] > 0 and memory['host_allocations'] > 0,
            'host_buffers_use_independent_ram': memory['host_non_device_local_allocations'] > 0 and memory['host_device_local_allocations'] == 0,
            'prefetch_used': prefetch['used'] > 0,
            'bounded_prefetch': prefetch['peak_charged_bytes'] <= 1024*1024*1024,
            'exact_baseline_tensors': result['old_new']['bitwise_equal_tensors'] == 25,
            'exact_baseline_png': result['old_new']['png_bitwise_equal']})
    elif case == 'normal-fp32':
        checks.update({'prefetch_disabled': prefetch['started'] == 0,
                       'exact_baseline_tensors': result['old_new']['bitwise_equal_tensors'] == 25,
                       'exact_baseline_png': result['old_new']['png_bitwise_equal']})
    results.append({'case': case, 'mechanism_checks': checks, 'all_mechanism_checks_pass': all(checks.values()),
                    'gpu_memory': memory, 'weight_prefetch': prefetch, 'weight_cache': report['weight_cache'],
                    'memory_recovery': recovery, 'memory_events': events,
                    'cgroup_peak_bytes': worker['memory_peak_bytes'],
                    'process': process, 'official_tensors_passed': result['passed_tensors'],
                    'official_numerical_pass': result['passed'], 'png': result['png'],
                    'generation_report_sha256': sha(path/'generation.json')})
record = {'scope':'Same-source512x512 diagnostic, actual mixed-placement observations and unchanged official numerical gates',
          'source_head': plan['execution_source_head'], 'plan_sha256': sha(base/'full-plan.json'),
          'audit_sha256': sha(__file__), 'results': results,
          'note':'Controlled budget choices do not prove physical GPU exhaustion; no speed acceptance or full-model forced-retry claim'}
(base/'execution-audit.json').write_text(json.dumps(record, indent=2)+'\n')
for item in results:
    print(item['case'], 'mechanisms', item['all_mechanism_checks_pass'], 'official', item['official_numerical_pass'])
