import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_shape_1376 as plan

SOURCE=dict(packed_width=64,packed_height=64,text_bucket=64,dit_text_tokens=64,text_layers=25,dit_layers=36)
GRAPH='7767517\n2 3\nReshape reshape_99 1 1 a b 0=16384\nReshape reshape_100 1 1 b c 0=128 1=128\n'
CANONICAL='7767517\n2 3\nReshape reshape_99 1 1 a b 0=VAE_PIXELS\nReshape reshape_100 1 1 b c 0=VAE_WIDTH 1=VAE_HEIGHT\n'

class FixedPlanTests(unittest.TestCase):
    def test_only_enumerated_candidate_fields_change(self):
        digest=hashlib.sha256(CANONICAL.encode()).hexdigest()
        with patch.dict(plan.CONTRACT_HASHES,{'vae':digest}):
            candidate=plan.candidate_graph(GRAPH,'vae',SOURCE)
            self.assertEqual(candidate,GRAPH.replace('0=16384','0=16512').replace('0=128 1=128','0=172 1=96'))
            with self.assertRaisesRegex(ValueError,'complete source graph'):
                plan.candidate_graph(GRAPH.replace('a b','changed b'),'vae',SOURCE)
            with self.assertRaises(ValueError):
                plan.candidate_graph(GRAPH.replace('0=16384','0=16383'),'vae',SOURCE)

    def test_plan_never_accepts_unpinned_source(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'reviewed 1024/s64'):
                plan.prepare(root,root/'output',root,root/'runner',sys.executable)
            self.assertFalse((root/'output').exists())

    def test_target_stays_s64_and_within_existing_sequence_cap(self):
        self.assertEqual(plan.TARGET['text_bucket'],64)
        self.assertEqual(plan.TARGET['packed_width']*plan.TARGET['packed_height']+plan.TARGET['dit_text_tokens'],4192)

if __name__=='__main__':unittest.main()
