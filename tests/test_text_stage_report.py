import os
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.diagnose_text_stages import compare,summarize_layers,verify_blob_map,TYPES,derive_vector_down_graph

class TextStageTests(unittest.TestCase):
    def test_vector_down_contract_is_explicit_and_narrow(self):
        original='Gemm gemm_6 1 1 72 73 10=-1 2=0 3=1 4=0 5=1 6=1 7=2048 8=3072 9=9216'
        self.assertEqual(derive_vector_down_graph('unchanged\n'+original).splitlines()[0],'unchanged')
        self.assertIn('DiagnosticVectorDown',derive_vector_down_graph(original))
        for bad in (original.replace('3=1','3=0'),original.replace('7=2048','7=256'),original+' 0=2',original+'\n'+original,''):
            with self.assertRaises(ValueError):derive_vector_down_graph(bad)
    def test_signed_zero_is_not_bitwise_equal(self):
        result=compare(np.array([0.],dtype='<f4'),np.array([-0.],dtype='<f4'))
        self.assertFalse(result['bitwise_equal']);self.assertEqual(result['max_abs_error'],0)
    def test_fp64_difference_not_lost_in_bitwise_check(self):
        result=compare(np.array([1.0+1e-12],dtype='<f8'),np.array([1.0],dtype='<f4'))
        self.assertFalse(result['bitwise_equal']);self.assertGreater(result['max_abs_error'],0)
    def test_shape_finite_guards(self):
        for bad in (np.zeros(3),np.array([np.inf,np.nan])):
            with self.assertRaises(ValueError):compare(np.zeros(2),bad)
    def test_local_error_and_free_running_separate(self):
        rows=[dict(layer=0,free_running=dict(nrmse=.1),teacher_forced=dict(nrmse=.01)),dict(layer=1,free_running=dict(nrmse=.4),teacher_forced=dict(nrmse=.001))]
        summary=summarize_layers(rows)
        self.assertEqual(summary['largest_nrmse_increase']['layer'],1)
        self.assertEqual(summary['layers'][1]['teacher_forced']['nrmse'],.001)
        with self.assertRaises(ValueError):summarize_layers(rows[::-1])
    def test_blob_map_checks_producers(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'text.param';p.write_text('7767517\n17 17\n'+'\n'.join(f'{kind} n{i} 0 1 {blob}' for i,(blob,kind) in enumerate(TYPES.items())))
            self.assertEqual(verify_blob_map(p)['norm'],'6')
            p.write_text(p.read_text().replace('RMSNorm','Gemm'))
            with self.assertRaises(ValueError):verify_blob_map(p)
@unittest.skipUnless(os.environ.get('ERNIE_TEXT_GEMM_DIAGNOSTIC'), 'set diagnostic runner for C++ integration')
class TextVectorRunnerTests(unittest.TestCase):
    def test_exact_rows_and_failure_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);runner=os.environ['ERNIE_TEXT_GEMM_DIAGNOSTIC']
            (p/'graph').write_text('7767517\n2 2\nInput input 0 1 in0\nInnerProduct down 1 1 in0 out0 0=8 1=0 2=128\n')
            w=np.eye(8,16,dtype='<f4');(p/'weights').write_bytes(struct.pack('<I',0)+w.tobytes())
            x=np.arange(48,dtype='<f4').reshape(3,16);x.tofile(p/'input')
            cmd=[runner,str(p/'graph'),str(p/'weights'),str(p/'input'),str(p/'output'),'3','16','8']
            good=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            np.testing.assert_array_equal(np.fromfile(p/'output',dtype='<f4').reshape(3,8),x[:,:8])
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            cmd[4]=str(p/'invalid');cmd[5]='4'
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertFalse((p/'invalid').exists())
            cmd[5]='3';x[0,0]=np.nan;x.tofile(p/'input')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

if __name__=='__main__':unittest.main()
