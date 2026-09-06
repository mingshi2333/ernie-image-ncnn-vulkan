import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
SPEC=importlib.util.spec_from_file_location('time_swap', Path(__file__).parents[1]/'tools/diagnose_time_feature_swap.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

class SwapTests(unittest.TestCase):
    def fixture(self, root):
        base=root/'base';base.mkdir();entries={}
        for name in sorted(m.INPUTS|{'expected'}):
            x=np.arange(4096 if name=='in2' else 8,dtype='<f4');p=base/(name+'.f32');x.tofile(p)
            entries[name]={'file':p.name,'shape':list(x.shape),'sha256':m.sha(p)}
        fixture={'inputs':{k:v for k,v in entries.items() if k!='expected'},'expected':entries['expected'],
                 'gates':{'fp32':{'atol':.0002,'rtol':.01,'nrmse':.003}}}
        (base/'fixture.json').write_text(json.dumps(fixture))
        native=root/'native.f32';np.zeros(4096,dtype='<f4').tofile(native)
        return base,native,fixture
    def test_only_feature_changes_gates_remain(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,native,f=self.fixture(root);out=root/'out'
            result=m.swap_inputs(base,native,out)
            self.assertEqual(result['gates'],f['gates']);self.assertFalse(result['native_acceptance_eligible'])
            for k in m.INPUTS-{'in2'}|{'expected'}:self.assertEqual(m.sha(base/(k+'.f32')),m.sha(out/(k+'.f32')))
            self.assertEqual(m.sha(native),m.sha(out/'in2.f32'))
            with self.assertRaises(ValueError):m.swap_inputs(base,native,out)
    def test_mutated_baseline_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,native,_=self.fixture(root);(base/'in0.f32').write_bytes(b'x'*32)
            with self.assertRaises(ValueError):m.swap_inputs(base,native,root/'out')
            self.assertFalse((root/'out').exists())
    def test_invalid_feature_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,native,_=self.fixture(root)
            for values in [np.zeros(4095,dtype='<f4'),np.full(4096,np.nan,dtype='<f4')]:
                values.tofile(native)
                with self.assertRaises(ValueError):m.swap_inputs(base,native,root/'out')
    def test_missing_input_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,native,f=self.fixture(root)
            f['inputs']['in0']['file']='../native.f32'
            (base/'fixture.json').write_text(json.dumps(f))
            with self.assertRaises(ValueError):m.swap_inputs(base,native,root/'out')
            del f['inputs']['in0'];(base/'fixture.json').write_text(json.dumps(f))
            with self.assertRaises(ValueError):m.swap_inputs(base,native,root/'out')
if __name__=='__main__':unittest.main()
