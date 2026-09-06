import copy
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_native_step_sensitivity import validate_factor, describe, FACTORS

class SensitivityTest(unittest.TestCase):
    def setUp(self):
        self.base={f'in{i}':{'shape':[1], 'dtype':'float32_le','sha256':str(i)*64} for i in range(6)}
    def candidate(self,factor):
        result=copy.deepcopy(self.base);key,digest=FACTORS[factor];result[key]['sha256']=digest;return result
    def test_each_factor_and_untouched_denominator(self):
        for factor in FACTORS:validate_factor(self.base,self.candidate(factor),factor)
    def test_second_change_or_missing_input_rejected(self):
        for factor in FACTORS:
            value=self.candidate(factor);value['in3']['sha256']='f'*64
            with self.assertRaises(ValueError):validate_factor(self.base,value,factor)
            value=self.candidate(factor);del value['in5']
            with self.assertRaises(ValueError):validate_factor(self.base,value,factor)
    def test_wrong_replacement_shape_dtype_rejected(self):
        for field,value in [('sha256','f'*64),('shape',[2]),('dtype','float16_le')]:
            candidate=self.candidate('text');candidate['in1'][field]=value
            with self.assertRaises(ValueError):validate_factor(self.base,candidate,'text')
    def test_descriptive_distances_have_no_gate(self):
        result=describe(np.array([2],dtype='f4'),np.array([1],dtype='f4'),np.array([4],dtype='f4'))
        self.assertEqual(result['difference_from_native_replay']['max_abs_error'],1)
        self.assertEqual(result['descriptive_distance_from_official_prediction']['max_abs_error'],2)
        self.assertFalse(result['native_acceptance_eligible']);self.assertFalse(result['official_gate_applied'])
        self.assertNotIn('passed',str(result))

if __name__=='__main__':unittest.main()
