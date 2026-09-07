"""Small execution-policy checks; no models or GPU allocations."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import run_batch

BASE=Path(__file__).resolve().parent
def comparison(passed=True):
    return {'passed':passed,'png':{'passed':True},
            'comparisons':[{'name':str(i),'passed':passed if i==24 else True} for i in range(25)]}

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def check_compare(self,value,code,expected,log=''):
        (self.base/'comparison.json').write_text(json.dumps(value))
        self.assertEqual(run_batch.classify('compare',code,self.base,log),expected)
    def test_pass(self):self.check_compare(comparison(),0,'completed')
    def test_numeric_failure(self):self.check_compare(comparison(False),1,'numerical_failure')
    def test_wrong_exit(self):self.check_compare(comparison(False),0,'execution_failure')
    def test_contradictory_result(self):
        value=comparison(False);value['passed']=True
        self.check_compare(value,1,'execution_failure')
    def test_missing_tensor(self):
        value=comparison();value['comparisons'].pop()
        self.check_compare(value,0,'execution_failure')
    def test_non_boolean_tensor(self):
        value=comparison();value['comparisons'][0]['passed']=1
        self.check_compare(value,0,'execution_failure')
    def test_traceback(self):self.check_compare(comparison(False),1,'execution_failure','Traceback (most recent call last)')
    def test_malformed_json(self):
        (self.base/'comparison.json').write_text('{')
        self.assertEqual(run_batch.classify('compare',1,self.base,''),'execution_failure')
    def test_wrong_container(self):self.check_compare([],1,'execution_failure')
    def test_missing_file(self):
        self.assertEqual(run_batch.classify('compare',1,self.base,''),'execution_failure')
    def test_native_contract(self):
        (self.base/'native').mkdir();path=self.base/'native/process.json'
        for value,code,expected in (({'complete':True,'return_code':0},0,'completed'),
            ({'complete':False,'return_code':0},0,'execution_failure'),
            ({'complete':True,'return_code':0},1,'execution_failure')):
            path.write_text(json.dumps(value))
            self.assertEqual(run_batch.classify('native',code,self.base,''),expected)
    def run_serial(self,first_code):
        shutil.copyfile(BASE/'run_batch.py',self.base/'run_batch.py')
        commands=[]
        for label,code in (('a',first_code),('b',0)):
            case=self.base/label;case.mkdir()
            code_text='import json;from pathlib import Path;'+\
                'Path('+repr(str(case/'comparison.json'))+').write_text('+repr(json.dumps(comparison(code==0)))+');'+\
                'raise SystemExit('+str(code)+')'
            commands.append({'case':label,'phase':'compare','argv':[sys.executable,'-c',code_text]})
        (self.base/'commands.json').write_text(json.dumps(commands))
        identity={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in (self.base/'commands.json',self.base/'run_batch.py')}
        (self.base/'batch-identity.json').write_text(json.dumps(identity))
        proc=subprocess.run([sys.executable,str(self.base/'run_batch.py')],capture_output=True,text=True,timeout=15)
        return proc,json.loads((self.base/'progress.json').read_text()),json.loads((self.base/'results.json').read_text())
    def test_serial_continues_numeric_failure(self):
        proc,progress,results=self.run_serial(1)
        self.assertEqual(proc.returncode,1,proc.stderr)
        self.assertEqual(progress['status'],'complete_with_numeric_failures')
        self.assertEqual([v['status'] for v in results],['numerical_failure','completed'])
    def test_serial_stops_execution_failure(self):
        proc,progress,results=self.run_serial(2)
        self.assertEqual(proc.returncode,2,proc.stderr)
        self.assertEqual(progress['status'],'execution_failure')
        self.assertEqual(len(results),1)
        self.assertFalse((self.base/'b/comparison.json').exists())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}
    with (BASE/'policy-checks.json').open('x') as out:out.write(json.dumps(summary,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
