"""Archive, identity and real small namespace tests; no ERNIE model is run."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
import zlib

from tools import check_release as check


class ReleaseSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ernie-release-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'runtime.tar.gz'
        self.inventory = self.root / 'files.json'
        self.payload = {'bin/ernie-image': b'installed CLI fixture'}
        selected = [{'path': p, 'size': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                    for p, b in self.payload.items()]
        self.payload['release.json'] = json.dumps({'selected_install_files': selected,
                                                  'distributable': False}).encode()
        self.write_archive()

    def write_archive(self, extras=(), omit=(), contents=None):
        data = contents or self.payload
        self.inventory.write_text(json.dumps([{'path': p, 'size': len(b),
            'sha256': hashlib.sha256(b).hexdigest()} for p, b in self.payload.items()]))
        with tarfile.open(self.archive, 'w:gz') as tar:
            for p, b in data.items():
                if p in omit:
                    continue
                member = tarfile.TarInfo('ernie-runtime/' + p)
                member.size = len(b)
                tar.addfile(member, io.BytesIO(b))
            for member, data in extras:
                tar.addfile(member, io.BytesIO(data) if data is not None else None)

    def extract(self):
        return check.extract_verified(self.archive, self.inventory, self.root / 'prefix')

    def test_complete_payload_and_relocation(self):
        result = self.extract()
        self.assertEqual(set(result), set(self.payload))
        (self.root / 'prefix').rename(self.root / '移动 installation')
        self.assertEqual((self.root / '移动 installation/bin/ernie-image').read_bytes(), self.payload['bin/ernie-image'])
        self.assertEqual((self.root / '移动 installation/bin/ernie-image').stat().st_mode & 0o777, 0o755)

    def test_missing_and_unknown_members_reject(self):
        self.write_archive(omit={'bin/ernie-image'})
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.extract()
        shutil.rmtree(self.root / 'prefix')
        member = tarfile.TarInfo('ernie-runtime/unlisted')
        self.write_archive(extras=[(member, b'')])
        with self.assertRaisesRegex(ValueError, 'set/size'):
            self.extract()

    def test_duplicate_and_link_members_reject(self):
        duplicate = tarfile.TarInfo('ernie-runtime/bin/ernie-image')
        duplicate.size = len(self.payload['bin/ernie-image'])
        self.write_archive(extras=[(duplicate, self.payload['bin/ernie-image'])])
        with self.assertRaisesRegex(ValueError, 'set/size'):
            self.extract()
        shutil.rmtree(self.root / 'prefix')
        link = tarfile.TarInfo('ernie-runtime/link')
        link.type = tarfile.SYMTYPE
        link.linkname = '/etc/passwd'
        self.write_archive(extras=[(link, None)])
        with self.assertRaisesRegex(ValueError, 'type'):
            self.extract()

    def test_traversal_rejects_without_external_write(self):
        member = tarfile.TarInfo('ernie-runtime/../../escape')
        member.size = 1
        self.write_archive(extras=[(member, b'x')])
        with self.assertRaisesRegex(ValueError, 'Noncanonical'):
            self.extract()
        self.assertFalse((self.root / 'escape').exists())
        for path in ['/absolute', 'a//b', 'a/./b', 'a\\b', '../x']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                check.member_path(path)

    def test_corruption_and_false_install_identity_reject(self):
        wrong = dict(self.payload)
        wrong['bin/ernie-image'] = b'x' * len(wrong['bin/ernie-image'])
        self.write_archive(contents=wrong)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.extract()
        shutil.rmtree(self.root / 'prefix')
        self.payload['release.json'] = b'{"selected_install_files":[{"path":"bin/ernie-image","size":1,"sha256":"wrong"}]}'
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'installation identity'):
            self.extract()

    def test_inventory_duplicate_boolean_and_size_reject(self):
        items = json.loads(self.inventory.read_text())
        for bad in [items + [items[0]], [{**items[0], 'size': True}, items[1]],
                    [{**items[0], 'size': check.MAX_FILE_BYTES + 1}, items[1]]]:
            self.inventory.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                check.read_inventory(self.inventory)

    def case(self):
        model = self.root / 'source' / 'model'
        model.mkdir(parents=True)
        (model / 'manifest.json').write_text('{}')
        prompt = self.root / 'prompt.txt'
        prompt.write_bytes(b'A red apple.\n')
        return {'schema_version': 1, 'scope': 'development_linux_offline_delivery',
            'archive': check.identity(self.archive), 'inventory': check.identity(self.inventory),
            'model': {'path': str(model), 'manifest': check.identity(model / 'manifest.json')},
            'inputs': {'prompt': check.identity(prompt), 'latent': check.identity(prompt)},
            'expected_png': check.identity(prompt), 'hidden_paths': [str(model.parent)],
            'request': {'kind': 'text-to-image', 'width': 1024, 'height': 1024, 'steps': 8,
                        'strength': None, 'threads': 2, 'text_down_vector': True},
            'resources': {'memory_max': 10 * check.GIB, 'swap_max': 0,
                          'host_min': 3 * check.GIB, 'timeout_seconds': 1800}}

    def test_case_endpoints_and_invalid_resource_or_request_reject(self):
        case = self.case()
        check.validate_case(case)
        for field, value in [('threads', True), ('steps', 4), ('width', 768), ('strength', .5)]:
            bad = copy.deepcopy(case)
            bad['request'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check.validate_case(bad)
        for field, value in [('memory_max', 21 * check.GIB), ('swap_max', 1), ('host_min', 0), ('timeout_seconds', True)]:
            bad = copy.deepcopy(case)
            bad['resources'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check.validate_case(bad)
        bad = copy.deepcopy(case)
        bad['request'].update(kind='img2img', strength=.51)
        with self.assertRaises(ValueError):
            check.validate_case(bad)

    def test_changed_input_and_model_manifest_identity_reject(self):
        case = self.case()
        record = case['inputs']['prompt']
        Path(record['path']).write_bytes(b'A red apple!\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            check.checked_record(record)
        case['model']['manifest']['path'] = record['path']
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            check.validate_case(case)

    def test_hidden_roots_must_cover_real_repository(self):
        case = self.case()
        path = self.root / 'case.json'
        path.write_text(json.dumps(case))
        with self.assertRaisesRegex(ValueError, 'entire source repository'):
            check.prepare(path, self.root / 'prepared')

    def test_metadata_expansion_is_bounded_before_tarfile(self):
        with tarfile.open(self.archive, 'w:gz', pax_headers={'comment': 'x' * 1048576}) as tar:
            for p, b in self.payload.items():
                member = tarfile.TarInfo('ernie-runtime/' + p)
                member.size = len(b)
                tar.addfile(member, io.BytesIO(b))
        self.assertLess(self.archive.stat().st_size, 10000)
        with self.assertRaisesRegex(ValueError, 'metadata exceeds'):
            self.extract()
        self.assertFalse((self.root / 'prefix').exists())

    def test_sparse_and_contiguous_tar_types_are_rejected(self):
        for kind in (tarfile.GNUTYPE_SPARSE, tarfile.CONTTYPE):
            with self.subTest(kind=kind):
                member = tarfile.TarInfo('ernie-runtime/extension')
                member.type = kind
                self.write_archive(extras=[(member, b'')])
                with self.assertRaisesRegex(ValueError, 'member type'):
                    self.extract()

    def test_valid_long_unicode_path_uses_bounded_pax(self):
        name = 'licenses/' + '中文' * 25 + '/LICENSE'
        self.payload[name] = b'long path notice'
        self.write_archive()
        self.extract()
        self.assertEqual((self.root / 'prefix' / name).read_bytes(), b'long path notice')

    def test_prepared_mapping_cannot_redirect_to_equal_external_bytes(self):
        case = self.case()
        path = self.root / 'case.json'
        path.write_text(json.dumps(case))
        output = self.root / 'prepared'
        with mock.patch.object(check.subprocess, 'check_output', return_value=str(self.root / 'source/.git')):
            check.prepare(path, output)
        preparation = output / 'preparation.json'
        value = json.loads(preparation.read_text())
        value['inputs']['prompt']['path'] = case['inputs']['prompt']['path']
        preparation.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'canonical identity'):
            check.run(output)

    def test_prepared_inventory_cannot_omit_runtime_file(self):
        case = self.case()
        path = self.root / 'case.json'
        path.write_text(json.dumps(case))
        output = self.root / 'prepared'
        with mock.patch.object(check.subprocess, 'check_output', return_value=str(self.root / 'source/.git')):
            check.prepare(path, output)
        preparation = output / 'preparation.json'
        value = json.loads(preparation.read_text())
        del value['files']['移动 installation/bin/ernie-image']
        preparation.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'file/input set'):
            check.run(output)

    def test_png_crc_dimensions_and_truncation(self):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
        png += chunk(b'IDAT', zlib.compress(b'\x00\xff\x00\x00')) + chunk(b'IEND', b'')
        path = self.root / 'image.png'
        path.write_bytes(png)
        check.png_container(path, 1, 1)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            check.png_container(path, 2, 1)
        path.write_bytes(png[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            check.png_container(path, 1, 1)
        broken = bytearray(png)
        broken[45] ^= 1
        path.write_bytes(broken)
        with self.assertRaisesRegex(ValueError, 'CRC'):
            check.png_container(path, 1, 1)

    @unittest.skipUnless(os.name == 'posix' and shutil.which('bwrap'), 'Requires actual Linux bubblewrap')
    def test_actual_namespace_relocation_and_environment(self):
        original = self.root / 'source checkout'
        model = original / 'model'
        model.mkdir(parents=True)
        (model / 'sentinel.txt').write_text('bound model')
        output = self.root / '独立 install'
        (output / 'results').mkdir(parents=True)
        (output / '模型 shared').mkdir()
        wrapper = check.sandbox(output, model, [str(original)])
        script = 'test -z "${VIRTUAL_ENV+x}" && test -z "${PYTHONPATH+x}" && test -z "$(ls -A -- "$1")" && cat "$2/sentinel.txt" && readlink /proc/self/ns/net'
        proc = subprocess.run([*wrapper, '/bin/sh', '-c', script, 'test', str(original), str(output / '模型 shared')],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, 'VIRTUAL_ENV': '/should/be/absent', 'PYTHONPATH': '/should/be/absent'})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('bound model', proc.stdout)
        self.assertNotIn(os.readlink('/proc/self/ns/net'), proc.stdout)
        self.assertEqual((model / 'sentinel.txt').read_text(), 'bound model')


if __name__ == '__main__':
    unittest.main()
