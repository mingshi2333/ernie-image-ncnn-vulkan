"""Reject ambiguous or mismatched package/oracle configuration before a run."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import pipeline_package as package


class ValidationPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'packed_width': 32, 'packed_height': 24, 'text_bucket': 2048}
        self.instance = {'config': self.config, 'source_manifest_sha256': 'a' * 64,
                         'runtime_bindings': {'text/a': 'b' * 64}}

    def manifest(self, value):
        (self.root / 'manifest.json').write_text(json.dumps(value))

    def test_legacy_selection_stays_compatible_and_rejects_mismatch(self):
        self.manifest({'schema_version': 2, 'config': self.config})
        self.assertEqual(package.validation_package(self.root), (self.config, None))
        for dimensions in ((384, 512), (512, None), (True, 384), (511, 384)):
            with self.assertRaises(ValueError):
                package.validation_package(self.root, *dimensions)

    def test_shared_requires_existing_oracle(self):
        self.manifest({'schema_version': 3})
        for options in ({}, {'reference': self.root, 'reference_only': True}):
            with patch.object(package, 'verify_shared_package') as verifier:
                with self.assertRaisesRegex(ValueError, 'existing verified'):
                    package.validation_package(self.root, **options)
                verifier.assert_not_called()

    def test_shared_authenticates_then_selects_exact_instance(self):
        self.manifest({'schema_version': 3})
        second = {**self.instance, 'config': {**self.config, 'packed_width': 64, 'packed_height': 64}}
        with patch.object(package, 'verify_shared_package', return_value={'instances': [self.instance, second]}) as verifier:
            for dimensions in ((None, None), (384, 512)):
                with self.assertRaises(ValueError):
                    package.validation_package(self.root, *dimensions, reference=self.root)
            config, binding = package.validation_package(self.root, 512, 384, reference=self.root)
            self.assertEqual(config, self.config)
            self.assertEqual(binding['runtime_bindings'], self.instance['runtime_bindings'])
            self.assertEqual(binding['source_manifest_sha256'], 'a' * 64)
            self.assertEqual(verifier.call_count, 3)

    def test_shared_verification_failure_cannot_be_bypassed_by_matching_shape(self):
        self.manifest({'schema_version': 3})
        with patch.object(package, 'verify_shared_package', side_effect=ValueError('Object checksum mismatch')):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                package.validation_package(self.root, 512, 384, reference=self.root)


if __name__ == '__main__':
    unittest.main()
