import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
SPEC=importlib.util.spec_from_file_location('replay',Path(__file__).parents[1]/'tools/diagnose_native_step_replay.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
class ReplayTests(unittest.TestCase):
    def test_exact_and_changed_output_no_official_gate(self):
        a=np.array([1.,2.],dtype='<f4');r=m.compare(a,a.copy());self.assertTrue(r['bitwise_equal'])
        b=a.copy();b[1]=np.nextafter(b[1],np.float32(3.));r=m.compare(b,a)
        self.assertFalse(r['bitwise_equal']);self.assertEqual(r['different_float_count'],1)
        self.assertNotIn('passed',r)
    def test_invalid_output_rejected(self):
        for a,b in [(np.ones(2),np.ones(3)),(np.array([np.nan]),np.ones(1))]:
            with self.assertRaises(ValueError):m.compare(a,b)
    def test_bound_tensor_size_shape_sha_and_finite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';np.ones(4,dtype='<f4').tofile(p);digest=m.sha(p)
            self.assertEqual(m.checked(p,[2,2],digest).size,4)
            for shape,sha in [([True,4],digest),([3,2],digest),([4],'0'*64)]:
                with self.assertRaises(ValueError):m.checked(p,shape,sha)
            np.full(4,np.nan,dtype='<f4').tofile(p)
            with self.assertRaises(ValueError):m.checked(p,[4],m.sha(p))
    def test_command_pins_heads_layers_layout_and_precision(self):
        cmd=m.native_command(Path('/execution'),Path('/output'),Path('/package'))
        self.assertEqual(cmd.count('--model'),36)
        for key,value in [('--tokens','4160'),('--width','64'),('--height','64'),('--text-tokens','64'),('--precision','fp32'),('--policy','stream')]:
            self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertNotIn('--host-weights',cmd)
        self.assertNotIn('--isolated-pipeline-cache',cmd)
if __name__=='__main__':unittest.main()
