import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
import shutil
import subprocess
from tools.release_dependencies import archive_crates, member_hash
from tools.release_dependencies import closure, elf_identity, loader_paths, crate_notices, sha


class DependencyEvidenceTests(unittest.TestCase):
    def test_closure_excludes_dev_and_requires_complete_graph(self):
        def node(key, deps):
            return {'id': key, 'features': [], 'deps': [{'pkg': k, 'dep_kinds': [{'kind': kind}]} for k,kind in deps]}
        m={'packages':[{'id':v} for v in ['root','normal','build','dev']],
           'resolve':{'root':'root','nodes':[node('root',[('normal',None),('build','build'),('dev','dev')]),node('normal',[]),node('build',[]),node('dev',[])]}}
        self.assertEqual({p['id'] for p in closure(m)},{'root','normal','build'})
        m['resolve']['nodes'].pop(1)
        with self.assertRaisesRegex(ValueError,'Incomplete'):closure(m)

    def test_loader_rejects_missing_and_unknown_lines(self):
        self.assertEqual(loader_paths('linux-vdso.so.1 (0x0)\nlibc.so.6 => /lib64/libc.so.6 (0xab)\n/lib64/ld-linux-x86-64.so.2 (0xac)'),[Path('/lib64/ld-linux-x86-64.so.2'),Path('/lib64/libc.so.6')])
        for text in ['libbad.so => not found', 'surprising output', '']:
            with self.assertRaises(ValueError):loader_paths(text)

    def test_elf_distinguishes_i686_and_x86_64(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'elf';h=bytearray(20);h[:4]=b'\x7fELF';h[4:6]=bytes([2,1]);h[18]=62;p.write_bytes(h)
            self.assertEqual(elf_identity(p),{'class':64,'endian':1,'machine':62})
            h[4]=1;h[18]=3;p.write_bytes(h);self.assertEqual(elf_identity(p)['class'],32)
            p.write_bytes(b'bad')
            with self.assertRaises(ValueError):elf_identity(p)

    def test_nested_native_notice_authentication_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache/index';cache.mkdir(parents=True);archive=cache/'demo-1.0.crate'
            with tarfile.open(archive,'w:gz') as tar:
                for name in ['demo-1.0/LICENSE','demo-1.0/vendor/native/COPYING']:
                    m=tarfile.TarInfo(name);m.size=6;tar.addfile(m,io.BytesIO(b'notice'))
            lock={'package':[{'name':'demo','version':'1.0','checksum':sha(archive)}]}
            p={'name':'demo','version':'1.0'};r=crate_notices(p,lock,root/'cache',root/'out')
            self.assertEqual(len(r['notices']),2);self.assertIn('vendor/native/COPYING',r['notices'][1]['path'])
            archive.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'authenticated'):crate_notices(p,lock,root/'cache',root/'bad')

    def test_unsafe_archive_member_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'cache/index').mkdir(parents=True);archive=root/'cache/index/demo-1.0.crate'
            with tarfile.open(archive,'w:gz') as tar:
                m=tarfile.TarInfo('demo-1.0/../LICENSE');m.size=1;tar.addfile(m,io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                crate_notices({'name':'demo','version':'1.0'},{'package':[{'name':'demo','version':'1.0','checksum':sha(archive)}]},root/'cache',root/'out')

class ArchiveEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ar'), 'binutils ar unavailable')
    def test_member_id_requires_matching_dep_info(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);name='demo-0123456789abcdef.o';obj=root/name;obj.write_bytes(b'fixed object bytes')
            archive=root/'lib.a';subprocess.run(['ar','rcs',str(archive),str(obj)],check=True)
            package={'id':'demo@1','name':'demo','version':'1','manifest_path':str(root/'registry/demo-1/Cargo.toml')}
            deps=root/'deps';deps.mkdir()
            missing=archive_crates(archive,deps,[package]);self.assertEqual(len(missing['unassigned_crates']),1)
            (deps/'demo-0123456789abcdef.d').write_text(str(root/'registry/demo-1/src/lib.rs'))
            matched=archive_crates(archive,deps,[package]);self.assertEqual(matched['matched_project_crates'][0]['id'],'demo@1')
            self.assertEqual(member_hash(archive,name),sha(obj))

if __name__=='__main__':unittest.main()
