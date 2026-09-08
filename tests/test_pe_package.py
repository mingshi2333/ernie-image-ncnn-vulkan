"""Native and Python verification must detect damaged optional PE packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from pe_package import CONFIG, runtime_files, sha256, verify_pe_package


class PePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'pe'; self.root.mkdir()
        for name in runtime_files():
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture '+name.encode())
        shutil.copyfile(ROOT/'tests/fixtures/pe-chat-template.jinja', self.root/'tokenizer/chat_template.jinja')
        (self.root/'pe.cfg').write_text(''.join(f'{k} {v}\n' for k, v in CONFIG.items()))
        lock = json.loads((ROOT/'sources.lock.json').read_text())
        self.manifest = dict(schema_version=1, kind='prompt_enhancer', portable=True, config=CONFIG.copy(),
                             official_model_revision=lock['official_model']['revision'], ncnn_revision=lock['ncnn']['revision'],
                             files={n: sha256(self.root/n) for n in runtime_files()},
                             file_sizes={n: (self.root/n).stat().st_size for n in runtime_files()})

    def tearDown(self): self.temporary.cleanup()

    def check(self, valid):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        if valid: verify_pe_package(self.root)
        else:
            with self.assertRaises((ValueError, OSError)): verify_pe_package(self.root)
        runner = Path(os.environ.get('ERNIE_TEST_RUNNER', ROOT/'build/ernie-image'))
        if runner.is_file():
            run = subprocess.run([str(runner), '--pe-model', str(self.root), '--verify-model'],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode == 0, valid, run.stdout+run.stderr)

    def test_complete_package(self): self.check(True)

    def test_reviewed_old_revision_remains_compatible(self):
        self.manifest['ncnn_revision'] = '6a1bf000f363714839a36793addc8c879d3d899e'
        self.check(True)

    def test_unreviewed_or_malformed_revisions_still_reject(self):
        for revision in ('f6f734f44d66f469fefee9ee401fd1cb5e3d573e',
                         'f'*40, None, True, [], {}):
            with self.subTest(revision=revision):
                self.manifest['ncnn_revision'] = revision
                self.check(False)

    def test_tail_block_corruption(self):
        p = self.root/'block-25/pe.ncnn.bin'
        data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
        self.check(False)

    def test_missing_head(self):
        (self.root/'head.ncnn.bin').unlink(); self.check(False)

    def test_omitted_tail_checksum(self):
        del self.manifest['files']['block-25/pe.ncnn.bin']; self.check(False)

    def test_revision_mismatch(self):
        self.manifest['ncnn_revision'] = 'unreviewed'; self.check(False)

    def test_resealed_wrong_capacity(self):
        self.manifest['config']['capacity'] = 8192; self.check(False)

    def test_resealed_configuration_disagreement(self):
        path = self.root/'pe.cfg'; path.write_text(path.read_text().replace('capacity 4096', 'capacity 8192'))
        self.manifest['files']['pe.cfg'] = sha256(path); self.check(False)

    def test_resealed_changed_chat_template(self):
        path = self.root/'tokenizer/chat_template.jinja'; path.write_bytes(b'changed template')
        self.manifest['files']['tokenizer/chat_template.jinja'] = sha256(path)
        self.manifest['file_sizes']['tokenizer/chat_template.jinja'] = path.stat().st_size
        self.check(False)

    def test_portable_package_rejects_external_link(self):
        path = self.root/'block-00/pe.ncnn.bin'
        outside = Path(self.temporary.name)/'external'; path.rename(outside); path.symlink_to(outside)
        self.check(False)

    def test_move_without_source(self):
        moved = Path(self.temporary.name)/'relocated'; self.root.rename(moved); self.root = moved
        self.check(True)


if __name__ == '__main__': unittest.main()
