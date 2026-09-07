"""Stage tracing must retain the authenticated runtime weight relationship."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from diagnose_dit_stages import stage_weight_source


class StageWeightSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'objects').mkdir()
        self.blocks = [f'{index:064x}' for index in range(36)]
        self.source = {'schema_version': 2, 'portable': True, 'official_model_revision': 'pinned',
                       'files': {'model.cfg': 'config32', 'block.param': 'graph32', 'block.bin': 'weights'}}
        self.donor = {**self.source, 'files': dict(self.source['files'], **{'model.cfg': 'config64'}),
                      'source_weights': {'dit': self.blocks}}

    def save_object(self, data):
        content = json.dumps(data).encode()
        digest = hashlib.sha256(content).hexdigest()
        (self.root/'objects'/digest).write_bytes(content)
        return digest

    def binding(self):
        source = self.save_object(self.source)
        donor = self.save_object(self.donor)
        provenance = {'source_manifest_sha256': donor, 'identical_runtime_assets': 1}
        return ({'runtime_source': {'official_weight_provenance': provenance}},
                {'source_manifest_sha256': source})

    def test_shared_donor_can_change_graph_metadata_but_preserves_weights(self):
        saved, binding = self.binding()
        weights, provenance = stage_weight_source(self.root, saved, binding)
        self.assertEqual(weights, self.blocks)
        self.assertEqual(provenance, saved['runtime_source']['official_weight_provenance'])

    def test_changed_runtime_weight_cannot_supply_provenance(self):
        self.donor['files']['block.bin'] = 'other-weights'
        saved, binding = self.binding()
        with self.assertRaisesRegex(ValueError, 'different runtime assets'):
            stage_weight_source(self.root, saved, binding)

    def test_changed_cas_object_is_rejected(self):
        saved, binding = self.binding()
        path = self.root/'objects'/binding['source_manifest_sha256']
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'object differs'):
            stage_weight_source(self.root, saved, binding)

    def test_missing_donor_or_incomplete_weight_list_is_rejected(self):
        saved, binding = self.binding()
        with self.assertRaisesRegex(ValueError, 'provenance digest'):
            stage_weight_source(self.root, {}, binding)
        self.donor['source_weights']['dit'] = self.blocks[:-1]
        saved, binding = self.binding()
        with self.assertRaisesRegex(ValueError, 'all 36'):
            stage_weight_source(self.root, saved, binding)

    def test_fixed_package_keeps_its_own_official_weight_list(self):
        (self.root/'manifest.json').write_text(json.dumps(self.donor))
        self.assertEqual(stage_weight_source(self.root, {}, None), (self.blocks, None))


if __name__ == '__main__':
    unittest.main()
