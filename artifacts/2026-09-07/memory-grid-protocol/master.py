"""Execute the frozen grid once, serially, preserving every planned trial."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from analyze import BASE,analyze,verify_bindings

def main():
    os.environ['ERNIE_GRID_SHA256']=sys.argv[1]
    lock=(BASE/'master.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (BASE/'progress.json').exists():raise RuntimeError('Grid already started; inspect its owned process, do not restart')
    plan=verify_bindings()
    state={'status':'running','master_pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'started_monotonic_ns':time.monotonic_ns(),'expected_runs':len(plan['trials']),'finished_trials':[]}
    def save():
        state['updated_monotonic_ns']=time.monotonic_ns()
        temporary=BASE/'progress.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(BASE/'progress.json')
    save()
    try:
        for trial in plan['trials']:
            directory=BASE/trial['id']
            state['current_trial']=trial['id'];save();print(json.dumps({'event':'start','trial':trial}),flush=True)
            with (directory/'supervisor.log').open('xb') as log:
                process=subprocess.Popen([sys.executable,str(directory/'supervisor.py'),'native'],stdout=log,stderr=subprocess.STDOUT)
                state['supervisor_pid']=process.pid;save();code=process.wait()
            (directory/'supervisor-exit.json').write_text(json.dumps({'return_code':code,'finished_monotonic_ns':time.monotonic_ns()})+'\n')
            summary=analyze();record=next(r for r in summary['records'] if r['id']==trial['id'])
            state['finished_trials'].append(record);state['last_return_code']=code;save()
            print(json.dumps({'event':'finished','record':record,'completed':summary['completed_runs'],'expected':summary['expected_runs']}),flush=True)
            if code or record['status']!='ok':raise RuntimeError('Stopped after trial failure: '+trial['id'])
        state['status']='complete'
    except BaseException as error:
        state['status']='stopped_after_failure';state['failure']=repr(error)
        raise
    finally:
        state['finished_monotonic_ns']=time.monotonic_ns();save()
    print(json.dumps({'event':'complete','results':str(BASE/'results.json')}),flush=True)

if __name__=='__main__':main()
