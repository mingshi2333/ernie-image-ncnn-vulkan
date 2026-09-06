import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from vae_shape_contract import validate_decoder_shape, specialize_decoder_graph

GRAPH='7767517\n3 4\nInput input 0 1 x\nReshape reshape_99 1 1 x y 0=64 1=512\nReshape reshape_100 1 1 y out 0=8 1=8 2=512\n'

class DecoderShapeTests(unittest.TestCase):
    def test_fixed_shape_requires_explicit_opt_in_and_cannot_export_pnnx(self):
        validate_decoder_shape(96,172,fixed=True,reference_only=True)
        for h,w,fixed,reference in [(96,172,False,True),(172,96,True,True),
                                  (96,171,True,True),(96,172,True,False),
                                  (129,8,False,True),(64,128,False,False)]:
            with self.subTest(shape=(h,w),fixed=fixed,reference=reference),self.assertRaises(ValueError):
                validate_decoder_shape(h,w,fixed=fixed,reference_only=reference)
        validate_decoder_shape(8,8,reference_only=False)
        validate_decoder_shape(128,128,reference_only=True)

    def test_two_spatial_reshapes_and_nonshape_bytes(self):
        result,changes=specialize_decoder_graph(GRAPH,96,172,fixed=True)
        self.assertEqual(result,GRAPH.replace('0=64 1=512','0=16512 1=512').replace('0=8 1=8 2=512','0=172 1=96 2=512'))
        self.assertEqual(len(changes),2)
        self.assertEqual(result.splitlines()[:3],GRAPH.splitlines()[:3])

    def test_reject_changed_missing_duplicate_or_unknown_reshape(self):
        for bad in (GRAPH.replace('0=64','0=65'),GRAPH.replace('Reshape reshape_99','Reshape other'),
                    GRAPH+GRAPH.splitlines()[3]+'\n',GRAPH.replace(GRAPH.splitlines()[4]+'\n',''),
                    GRAPH.replace('reshape_99 1 1','reshape_99 2 1')):
            with self.subTest(graph=bad),self.assertRaises(ValueError):
                specialize_decoder_graph(bad,96,172,fixed=True)

if __name__=='__main__':unittest.main()
