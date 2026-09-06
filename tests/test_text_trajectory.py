import tempfile
from pathlib import Path
import unittest
import numpy as np
from tools.diagnose_text_trajectory import update_diagnostics,schedule,read_tensor,sha

class TextTrajectoryTests(unittest.TestCase):
    def test_exact_update_separates_injection_from_rounding(self):
        ref=np.array([1.,2.]);before=ref+np.array([.1,-.1]);pred=np.array([3.,4.]);rp=np.array([2.,4.]);delta=-.25
        r=update_diagnostics(before,before+delta*pred,pred,ref,ref+delta*rp,rp,delta)
        self.assertLess(r['update_residual_l2'],1e-14);self.assertAlmostEqual(r['prediction_injection_l2'],.25)
    def test_rounding_residual_is_measured_not_assigned_to_dit(self):
        z=np.zeros(3);after=z.copy();after[1]=.125
        r=update_diagnostics(z,after,z,z,z,z,-.5)
        self.assertEqual(r['prediction_injection_l2'],0);self.assertEqual(r['update_residual_max'],.125);self.assertIsNone(r['incoming_injection_cosine'])
    def test_nonfinite_mismatched_or_invalid_schedule_rejected(self):
        z=np.zeros(2)
        for bad in [np.zeros(3),np.array([0,np.nan])]:
            with self.assertRaises(ValueError):update_diagnostics(z,bad,z,z,z,z,-.5)
        for n in [True,0,7,9]:
            with self.assertRaises(ValueError):schedule(n)
        self.assertAlmostEqual(float(schedule(8).sum()),-1)
    def test_tensor_hash_shape_and_nonfinite_are_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(np.array([1,2],dtype='<f4').tobytes());digest=sha(p)
            self.assertEqual(read_tensor(p,digest,[2]).size,2)
            for h,s in [('0'*64,[2]),(digest,[3]),(digest,[True,2])]:
                with self.assertRaises(ValueError):read_tensor(p,h,s)
            p.write_bytes(np.array([np.inf,2],dtype='<f4').tobytes())
            with self.assertRaises(ValueError):read_tensor(p,sha(p),[2])
if __name__=='__main__':unittest.main()
