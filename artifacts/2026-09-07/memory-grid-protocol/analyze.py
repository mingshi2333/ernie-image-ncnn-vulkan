"""Summarize the predeclared four-setting development grid without dropping failures."""
import hashlib,json,math,os,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_bindings():
    expected=os.environ.get('ERNIE_GRID_SHA256')
    if expected and digest(BASE/'grid.json')!=expected:raise ValueError('Frozen grid identity changed')
    plan=json.loads((BASE/'grid.json').read_text())
    trials=plan['trials']
    assert len(trials)==16 and len({t['id'] for t in trials})==16
    assert [t['order'] for t in trials]==list(range(16))
    assert len(plan['configurations'])==4
    for config in plan['configurations']:
        selected=[t for t in trials if t['config']==config]
        assert sorted((t['phase'],t['repeat']) for t in selected)==[('measured',0),('measured',1),('measured',2),('warmup',-1)]
    assert digest(BASE/'bindings.json')==plan['bindings_sha256']
    for name,binding in json.loads((BASE/'bindings.json').read_text()).items():
        p=Path(name)
        if p.stat().st_size!=binding['bytes'] or digest(p)!=binding['sha256']:
            raise ValueError('Frozen experiment identity changed: '+name)
    for trial in plan['trials']:
        assert digest(BASE/trial['id']/'plan.json')==trial['plan_sha256']
    return plan

def result_for(plan, trial):
    directory=BASE/trial['id'];job=directory/'native/benchmark'
    record={k:trial[k] for k in ('id','phase','repeat','config','order')}
    process_path=directory/'native/process.json'
    if not process_path.exists():
        failed=(directory/'supervisor-exit.json').exists()
        return {**record,'status':'failed' if failed else 'pending',**({'failure':'Supervisor ended without a process record'} if failed else {})}
    process={}
    try:
        process=json.loads(process_path.read_text())
        assert process['complete'] and process['return_code']==0,process.get('failure','Native process failed')
        r=json.loads((job/'result.json').read_text());native=json.loads((job/'generation.json').read_text())
        assert r['passed'] and r['status']=='ok' and r['selection_status']=='verified_native_report'
        assert r['formal_comparison_eligible'] and not native['trace_enabled'] and not native['allocation_instrumentation']
        assert native==r['native_report'] and digest(job/'generation.json')==r['native_report_sha256']
        assert r['runner_sha256']==plan['binary_sha256']==digest(job/'ernie-image.snapshot')
        assert r['noise_sha256']==plan['noise_sha256']==digest(job/'initial.f32')
        assert r['package_manifest_sha256']==plan['package_manifest_sha256']
        assert r['prompt_sha256']==plan['prompt_sha256'] and native['prompt']==plan['prompt']
        assert native['token_ids']==plan['token_ids']
        assert native['model']=={'schema_version':3,'source_width':1024,'source_height':1024,'text_bucket':32,'dit_text_tokens':64}
        assert native['vulkan_gpu_index']==0 and native['shape']==[512,512] and len(r['denoise_seconds'])==8
        settings=plan['common_settings']|plan['configurations'][trial['config']]
        assert native['request']==settings
        assert r['requested_settings']==settings and not native['pe']['enabled']
        assert native['model_loading_requested']==settings['model_loading']
        assert native['progress_scope']=='verify is cumulative from generation start; text is the text stage; denoise is per step'
        assert digest(job/'native.png')==plan['image_sha256'], 'PNG differs from the verified baseline'
        assert r['image']['mode']=='RGB' and r['image']['size']==[512,512]
        assert not (job/'trace').exists()
        assert native['placement_requests']['gpu']==0 and native['placement_requests']['budget_unavailable']==0
        cache=native['weight_cache'];placement=native['placement_requests']
        if settings['dit_cache_mib']:
            assert cache['hits']+cache['loads']==288 and placement['ram']==cache['loads']+16
            assert cache['peak_charged_bytes']<=settings['dit_cache_mib']*1024*1024
        else:assert placement['ram']==304 and not any(cache.values())
        assert isinstance(r.get('max_rss_kib'),int) and r['max_rss_kib']>0
        assert math.isclose(r['wall_seconds'],(r['wall_finished_monotonic_ns']-r['wall_started_monotonic_ns'])/1e9,rel_tol=1e-9)
        assert r['wall_seconds']>0 and math.isfinite(r['wall_seconds'])
        limits=plan['resource_limits']
        assert process['minimum_host_available']>=limits['host_available_min_bytes']
        assert process['peak_gpu_whole_device_mib']<=limits['gpu_whole_device_max_mib']
        assert process['observed_limits']['memory.max']==str(limits['memory_max_bytes'])
        assert process['observed_limits']['memory.swap.max']=='0'
        assert process['observed_limits']['cpu.max']=='200000 100000'
        events={k:int(v) for k,v in (line.split() for line in process['memory_events'].splitlines())}
        assert not events.get('oom',0) and not events.get('oom_kill',0)
        assert process['observed_native_pids'], 'No native process was observed'
        for pid in process['observed_native_pids']:
            assert Path(f'/proc/{pid}/exe').resolve()!=job/'ernie-image.snapshot', 'Native process remains live'
        return {**record,'status':'ok','wall_seconds':r['wall_seconds'],'native_seconds':r['total_seconds'],
            'supervisor_seconds':process['wall_seconds'],'max_rss_kib':r['max_rss_kib'],
            'whole_gpu_sampled_peak_mib':r.get('gpu_device_total_mib',{}).get('sampled_peak'),
            'cgroup_peak_including_file_cache_bytes':process['peak_memory_current'],'memory_events':events,
            'host_available_min_bytes':process['minimum_host_available'],
            'cache':cache,'placement_requests':placement,'image_sha256':digest(job/'native.png'),
            'native_vm_swap_peak_bytes':process.get('peak_sampled_native_memory',{}).get('VmSwap'),
            'package_preverification':r['package_preverification'],
            'records_sha256':{str(p.relative_to(directory)):digest(p) for p in
                (process_path,directory/'native/worker-result.json',job/'request.json',job/'result.json',job/'generation.json',job/'resources.log')}}
    except (AssertionError,OSError,KeyError,TypeError,ValueError) as error:
        return {**record,'status':'failed','failure':str(error),'process':process}

def analyze():
    plan=verify_bindings();records=[result_for(plan,t) for t in plan['trials']]
    complete=all(r['status']=='ok' for r in records)
    result={'schema_version':1,'scope':plan['scope'],'status':'complete' if complete else 'incomplete',
        'expected_runs':len(records),'completed_runs':sum(r['status']=='ok' for r in records),
        'failed_runs':sum(r['status']=='failed' for r in records),'records':records,
        'configuration_summaries':None,'paired_development_ratios':None,
        'formal_peer_speed_complete':False,'formal_memory_complete':False,
        'full_quality_complete':False,'recommended_default_change':None}
    if complete:
        summaries={}
        for config in plan['configurations']:
            measured=[r for r in records if r['phase']=='measured' and r['config']==config]
            assert len(measured)==3 and {r['repeat'] for r in measured}=={0,1,2}
            summaries[config]={'runs':3,'wall_seconds':[r['wall_seconds'] for r in measured],
                'median_wall_seconds':statistics.median(r['wall_seconds'] for r in measured),
                'minimum_wall_seconds':min(r['wall_seconds'] for r in measured),
                'maximum_wall_seconds':max(r['wall_seconds'] for r in measured),
                'median_rss_kib':statistics.median(r['max_rss_kib'] for r in measured),
                'maximum_rss_kib':max(r['max_rss_kib'] for r in measured),
                'maximum_cgroup_bytes':max(r['cgroup_peak_including_file_cache_bytes'] for r in measured),
                'cache_hits':[r['cache']['hits'] for r in measured],
                'pressure_evictions':[r['cache']['evictions'] for r in measured],
                'max_events':[r['memory_events']['max'] for r in measured]}
        comparisons={}
        for left,right in plan['comparisons']:
            pairs=[]
            for repeat in range(3):
                a=next(r for r in records if r['phase']=='measured' and r['repeat']==repeat and r['config']==left)
                b=next(r for r in records if r['phase']=='measured' and r['repeat']==repeat and r['config']==right)
                pairs.append({'repeat':repeat,'left_over_right_wall_ratio':a['wall_seconds']/b['wall_seconds'],
                    'left_over_right_rss_ratio':a['max_rss_kib']/b['max_rss_kib']})
            comparisons[left+'/'+right]={'pairs':pairs,'median_wall_ratio':statistics.median(p['left_over_right_wall_ratio'] for p in pairs),
                'median_rss_ratio':statistics.median(p['left_over_right_rss_ratio'] for p in pairs)}
        result['configuration_summaries']=summaries;result['paired_development_ratios']=comparisons
    temporary=BASE/'results.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(BASE/'results.json')
    return result

if __name__=='__main__':
    result=analyze();print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    raise SystemExit(0 if result['status']=='complete' else 2)
