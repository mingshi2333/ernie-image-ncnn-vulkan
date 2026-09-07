"""Execute the frozen size matrix serially; retain independent numeric failures."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def classify(phase,code,directory,log):
    path=directory/('comparison.json' if phase=='compare' else phase+'/process.json')
    try:
        value=json.loads(path.read_text())
    except (OSError,ValueError):
        return 'execution_failure'
    if not isinstance(value,dict):return 'execution_failure'
    if phase!='compare':
        if code!=0:return 'execution_failure'
        return 'completed' if value.get('complete') is True and value.get('return_code')==0 else 'execution_failure'
    if code not in (0,1) or 'Traceback (most recent call last)' in log:
        return 'execution_failure'
    rows=value.get('comparisons',[]);png=value.get('png',{})
    if not isinstance(rows,list) or len(rows)!=25 or not isinstance(png,dict):return 'execution_failure'
    if any(not isinstance(row,dict) or type(row.get('passed')) is not bool for row in rows) or type(png.get('passed')) is not bool:
        return 'execution_failure'
    passed=png['passed'] and all(row['passed'] for row in rows)
    if value.get('passed') is not passed or code!=int(not passed):return 'execution_failure'
    return 'completed' if passed else 'numerical_failure'

def main():
    identity=json.loads((BASE/'batch-identity.json').read_text())
    for name,item in identity.items():
        path=Path(name);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
    commands=json.loads((BASE/'commands.json').read_text());results=[]
    assert not (BASE/'results.json').exists(),'Do not restart or overwrite an existing batch'
    for step in commands:
        print(json.dumps({'starting':step['case']+' '+step['phase']}),flush=True)
        started=time.monotonic();log_path=BASE/(step['case']+'-'+step['phase']+'.log')
        with log_path.open('x') as log:
            process=subprocess.Popen(step['argv'],stdout=log,stderr=subprocess.STDOUT)
            (BASE/'progress.json').write_text(json.dumps({'status':'running','case':step['case'],'phase':step['phase'],
                'child_pid':process.pid,'started_monotonic':started,'completed_commands':len(results)},indent=2)+'\n')
            code=process.wait()
        status=classify(step['phase'],code,BASE/step['case'],log_path.read_text())
        results.append({**step,'return_code':code,'status':status,'wall_seconds':time.monotonic()-started})
        (BASE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
        if status=='execution_failure':
            (BASE/'progress.json').write_text(json.dumps({'status':status,'case':step['case'],'phase':step['phase'],'completed_commands':len(results)},indent=2)+'\n')
            raise SystemExit(code or 2)
    numeric=sum(row['status']=='numerical_failure' for row in results)
    for name,item in identity.items():assert Path(name).stat().st_size==item['bytes'] and sha(name)==item['sha256']
    (BASE/'progress.json').write_text(json.dumps({'status':'complete_with_numeric_failures' if numeric else 'complete',
        'completed_commands':len(results),'numerical_failures':numeric,'execution_failures':0},indent=2)+'\n')
    raise SystemExit(1 if numeric else 0)

if __name__=='__main__':main()
