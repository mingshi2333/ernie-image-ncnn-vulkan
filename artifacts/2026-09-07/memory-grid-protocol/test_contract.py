"""Synthetic orchestration checks only; no new model or performance evidence."""
import copy,importlib.util,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('grid_analyze',HERE/'analyze.py')
analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
OLD=Path('/var/tmp/ernie-benchmark-runtime512-v1/native/benchmark/result.json')

def write(path,value):path.write_text(json.dumps(value)+'\n')

class GridContract(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='ernie-grid-contract-')
        self.base=Path(self.temporary.name);analyzer.BASE=self.base
        self.old=copy.deepcopy(json.loads(OLD.read_text()))
        configurations={n:{'model_loading':mode,'dit_cache_mib':cache} for n,mode,cache in (
            ('stdio-off','stdio',0),('stdio-cache','stdio',6144),('mapped-off','mapped',0),('mapped-cache','mapped',6144))}
        settings={**self.old['requested_settings'],'dit_weights':'host','gpu_reserve_mib':512}
        for key in ('model_loading','dit_cache_mib'):settings.pop(key)
        self.plan={'scope':'SYNTHETIC CONTRACT FIXTURE','configurations':configurations,'common_settings':settings,
            'resource_limits':{'memory_max_bytes':16*1024**3,'host_available_min_bytes':3*1024**3,'gpu_whole_device_max_mib':6144},
            'package_manifest_sha256':'synthetic-model','prompt_sha256':self.old['prompt_sha256'],
            'prompt':self.old['native_report']['prompt'],'token_ids':self.old['native_report']['token_ids'],
            'trials':[],'comparisons':[['stdio-off','mapped-off']]}
        for name in ('binary','noise','image'):
            p=self.base/name;p.write_bytes(('synthetic-'+name).encode());self.plan[name+'_sha256']=analyzer.digest(p)
        self.sentinel=self.base/'source.txt';self.sentinel.write_text('frozen')
        write(self.base/'bindings.json',{str(self.sentinel):{'bytes':6,'sha256':analyzer.digest(self.sentinel)}})
        self.plan['bindings_sha256']=analyzer.digest(self.base/'bindings.json')
        for phase,repeat in [('warmup',-1)]+[('measured',i) for i in range(3)]:
            for number,(name,cfg) in enumerate(configurations.items()):
                index=len(self.plan['trials']);directory=self.base/f'{index:02d}';job=directory/'native/benchmark';job.mkdir(parents=True)
                write(directory/'plan.json',{})
                trial={'id':directory.name,'phase':phase,'repeat':repeat,'config':name,'order':index,
                    'plan_sha256':analyzer.digest(directory/'plan.json')};self.plan['trials'].append(trial)
                for name,file in (('binary','ernie-image.snapshot'),('noise','initial.f32'),('image','native.png')):
                    shutil.copy2(self.base/name,job/file)
                r=copy.deepcopy(self.old);native=r['native_report']
                native['request']=settings|cfg;r['requested_settings']=native['request']
                native['model_loading_requested']=cfg['model_loading']
                native['progress_scope']='verify is cumulative from generation start; text is the text stage; denoise is per step'
                if not cfg['dit_cache_mib']:
                    native['weight_cache']={k:0 for k in native['weight_cache']};native['placement_requests']['ram']=304
                r['runner_sha256']=self.plan['binary_sha256'];r['noise_sha256']=self.plan['noise_sha256']
                r['package_manifest_sha256']=self.plan['package_manifest_sha256']
                r['wall_seconds']=[20,30,10,15][number];r['wall_started_monotonic_ns']=0
                r['wall_finished_monotonic_ns']=r['wall_seconds']*10**9
                write(job/'generation.json',native);r['native_report_sha256']=analyzer.digest(job/'generation.json');write(job/'result.json',r)
                for p in (job/'request.json',job/'resources.log',directory/'native/worker-result.json'):p.write_text('{}')
                process={'complete':True,'return_code':0,'minimum_host_available':5*1024**3,'peak_gpu_whole_device_mib':2048,
                    'observed_limits':{'memory.max':str(16*1024**3),'memory.swap.max':'0','cpu.max':'200000 100000'},
                    'memory_events':'max 0\noom 0\noom_kill 0','observed_native_pids':[2**31-1],
                    'wall_seconds':42,'peak_memory_current':9*1024**3}
                write(directory/'native/process.json',process)
        write(self.base/'grid.json',self.plan)

    def tearDown(self):self.temporary.cleanup()

    def test_complete_uses_only_three_measurements_and_keeps_formal_open(self):
        result=analyzer.analyze();self.assertEqual(result['completed_runs'],16)
        self.assertEqual(result['configuration_summaries']['stdio-off']['runs'],3)
        self.assertEqual(result['paired_development_ratios']['stdio-off/mapped-off']['median_wall_ratio'],2)
        self.assertFalse(result['formal_peer_speed_complete']);self.assertFalse(result['formal_memory_complete'])
        self.assertIsNone(result['recommended_default_change'])

    def test_failure_never_drops_a_run_or_produces_aggregate(self):
        write(self.base/'00/native/process.json',{'complete':False,'return_code':-9,'failure':'synthetic failure'})
        result=analyzer.analyze();self.assertEqual((result['completed_runs'],result['failed_runs']),(15,1))
        self.assertEqual(len(result['records']),16);self.assertIsNone(result['configuration_summaries'])

    def test_missing_and_malformed_terminal_records_are_failures(self):
        p=self.base/'00/native/process.json';p.unlink()
        self.assertEqual(analyzer.analyze()['records'][0]['status'],'pending')
        write(self.base/'00/supervisor-exit.json',{'return_code':1})
        self.assertEqual(analyzer.analyze()['failed_runs'],1)
        p.write_text('{');self.assertEqual(analyzer.analyze()['failed_runs'],1)

    def test_changed_image_is_a_failed_trial(self):
        (self.base/'00/native/benchmark/native.png').write_bytes(b'wrong')
        self.assertEqual(analyzer.analyze()['failed_runs'],1)

    def test_changed_source_and_duplicate_schedule_rejected(self):
        self.sentinel.write_text('changed')
        with self.assertRaises(ValueError):analyzer.verify_bindings()
        self.sentinel.write_text('frozen')
        self.plan['trials'][1]['id']=self.plan['trials'][0]['id'];write(self.base/'grid.json',self.plan)
        with self.assertRaises(AssertionError):analyzer.verify_bindings()

    def test_master_stops_after_first_failed_supervisor_and_rejects_restart(self):
        for name in ('master.py','analyze.py'):shutil.copy2(HERE/name,self.base/name)
        for trial in self.plan['trials']:
            directory=self.base/trial['id'];shutil.rmtree(directory/'native')
            (directory/'supervisor.py').write_text('raise SystemExit(9)\n')
        command=[sys.executable,str(self.base/'master.py'),analyzer.digest(self.base/'grid.json')]
        first=subprocess.run(command,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(first.returncode,0)
        state=json.loads((self.base/'progress.json').read_text())
        self.assertEqual(state['status'],'stopped_after_failure');self.assertEqual(len(state['finished_trials']),1)
        self.assertFalse((self.base/'01/supervisor.log').exists())
        saved=(self.base/'progress.json').read_bytes()
        second=subprocess.run(command,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(second.returncode,0);self.assertIn('already started',second.stderr)
        self.assertEqual(saved,(self.base/'progress.json').read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
