import json
from pathlib import Path
import tempfile
import unittest
from tools.audit_shape_contract import normalize_graph, audit_package, dimensions, RULES

class ShapeContractAuditTests(unittest.TestCase):
    config=dict(packed_width=4,packed_height=3,text_bucket=32,dit_text_tokens=32,text_layers=25,dit_layers=36)
    def graph(self,kind):
        values=dimensions(self.config)
        rows=['7767517','99 99']
        for node,params in RULES[kind].items():
            rows.append('Reshape '+node+' 1 1 in out '+' '.join(k+'='+str(values[v]) for k,v in params.items()))
        return '\n'.join(rows)+'\n'
    def test_all_explicit_fields_recorded(self):
        for kind in RULES:
            with self.subTest(kind=kind):
                digest,fields=normalize_graph(self.graph(kind),kind,self.config)
                self.assertEqual(len(digest),64)
                self.assertEqual(len(fields),sum(len(x) for x in RULES[kind].values()))
    def test_missing_duplicate_or_changed_shape_rejected(self):
        g=self.graph('input')
        for changed in (g.replace('0=12','0=13'),g.replace('0=12','0=12 0=12'),g+'Reshape reshape_7 1 1 in out 0=12\n',g.replace('Reshape reshape_7 1 1 in out 0=12\n','')):
            with self.subTest(graph=changed),self.assertRaises(ValueError):normalize_graph(changed,'input',self.config)
    def test_unlisted_parameters_and_topology_are_hash_bound(self):
        g=self.graph('input');digest,_=normalize_graph(g,'input',self.config)
        for changed in (g.replace('0=12','0=12 99=5'),g.replace('Reshape reshape_7','Permute reshape_7'),g+'Reshape unreviewed 1 1 x y 0=7\n'):
            self.assertNotEqual(normalize_graph(changed,'input',self.config)[0],digest)
    def test_different_shapes_normalize_same_only_at_listed_fields(self):
        g=self.graph('input');new=dict(self.config,packed_width=8,packed_height=6,dit_text_tokens=64)
        changed=g.replace('0=12','0=48').replace('7=32','7=64')
        self.assertEqual(normalize_graph(g,'input',self.config)[0],normalize_graph(changed,'input',new)[0])
    def test_unknown_manifest_rejected_even_if_self_consistent(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'manifest.json').write_text(json.dumps({'config':self.config,'files':{}}))
            with self.assertRaisesRegex(ValueError,'Unknown static package'):audit_package(root)
    def test_unknown_bucket_or_kind_rejected(self):
        with self.assertRaises(ValueError):dimensions(dict(self.config,text_bucket=128))
        with self.assertRaises(ValueError):normalize_graph('', 'unknown',self.config)
if __name__=='__main__':unittest.main()
