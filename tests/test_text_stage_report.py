import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.diagnose_text_stages import compare,summarize_layers,verify_blob_map,TYPES

class TextStageTests(unittest.TestCase):
    def test_signed_zero_is_not_bitwise_equal(self):
        result=compare(np.array([0.],dtype='<f4'),np.array([-0.],dtype='<f4'))
        self.assertFalse(result['bitwise_equal']);self.assertEqual(result['max_abs_error'],0)
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
if __name__=='__main__':unittest.main()
