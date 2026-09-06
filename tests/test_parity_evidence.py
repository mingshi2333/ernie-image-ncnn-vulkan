"""Historical evidence must remain verifiable without hiding damaged new runs."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import collect_parity_evidence as evidence


class ValidatorSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'outputs' / 'run'
        self.run.mkdir(parents=True)
        self.payload = b'# saved validator version\n'
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.result = {'validator_sha256': self.digest}
        self.legacy = self.root / 'outputs' / 'validator-source-variants' / (self.digest + '.py')
        self.legacy.parent.mkdir()
        self.legacy.write_bytes(self.payload)
        self.root_patch = patch.object(evidence, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_legacy_validator_is_recovered_by_exact_digest(self):
        self.assertEqual(evidence.validator_snapshot(self.run, self.result), self.legacy)

    def test_corrupted_legacy_snapshot_is_rejected(self):
        self.legacy.write_bytes(b'# changed validator\n')
        with self.assertRaisesRegex(ValueError, 'Checksum differs'):
            evidence.validator_snapshot(self.run, self.result)

    def test_missing_modern_snapshot_cannot_use_legacy_fallback(self):
        self.result['source_snapshot'] = {'validate_pipeline.py': self.digest}
        with self.assertRaises(FileNotFoundError):
            evidence.validator_snapshot(self.run, self.result)

    def test_present_but_corrupted_snapshot_cannot_use_legacy_fallback(self):
        scripts = self.run / 'scripts'
        scripts.mkdir()
        (scripts / 'validate_pipeline.py').write_bytes(b'# damaged local snapshot\n')
        with self.assertRaisesRegex(ValueError, 'Checksum differs'):
            evidence.validator_snapshot(self.run, self.result)

    def test_modern_snapshot_keeps_its_original_location(self):
        scripts = self.run / 'scripts'
        scripts.mkdir()
        source = scripts / 'validate_pipeline.py'
        source.write_bytes(self.payload)
        self.result['source_snapshot'] = {'validate_pipeline.py': self.digest}
        self.assertEqual(evidence.validator_snapshot(self.run, self.result), source)


if __name__ == '__main__':
    unittest.main()
