import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.audit_port_relations import require_pin, finalizer_contract, derived_metrics, tensor_range
import json,struct

class RelationsTests(unittest.TestCase):
    def test_wrong_source_pin_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source';p.write_bytes(b'changed')
            with self.assertRaises(ValueError):require_pin(p,'0'*64)
    def test_scale_shift_edge_and_dimensions_rejected(self):
        graph=Path('outputs/reference-port-v1/assets/dit/finalizer.ncnn.param').read_text()
        self.assertEqual(finalizer_contract(graph)['gemm_0'],'scale_rows_0_4096')
        for bad in [graph.replace('8 5 9','8 4 9'),graph.replace('8=4096','8=8192'),graph.replace('3=1','3=0')]:
            with self.assertRaises(ValueError):finalizer_contract(bad)
    def test_ulp_difference_preserved(self):
        a=np.ones(4,dtype='<f4');b=a.copy();b[1]=np.nextafter(b[1],np.float32(2))
        r=derived_metrics(a,b);self.assertFalse(r['bitwise_equal']);self.assertEqual(r['different_values'],1)
        with self.assertRaises(ValueError):derived_metrics(a,np.array([np.nan],dtype='f4'))
    def test_wrong_official_shape_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'tensor';h=json.dumps({'x':{'dtype':'BF16','shape':[8],'data_offsets':[0,16]}}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+bytes(16))
            self.assertEqual(tensor_range(p,'x',[8]),8+len(h))
            with self.assertRaises(ValueError):tensor_range(p,'x',[4,2])

if __name__=='__main__':unittest.main()
