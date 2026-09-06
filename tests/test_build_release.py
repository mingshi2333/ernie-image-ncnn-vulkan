import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from tools.build_release import build_release, collect_notices, sha256


class ReleaseDraftTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.install=self.root/'install'
        (self.install/'bin').mkdir(parents=True)
        (self.install/'bin/ernie-image').write_bytes(b'fixture executable only')
        shared=self.install/'share/ernie-image';shared.mkdir(parents=True)
        for name in ['LICENSE','RUNNING.md','sources.lock.json']:(shared/name).write_text('local test')
    def build(self,name='out',**kw):return build_release(self.install,self.root/name,**kw)
    def test_runtime_allowlist_and_fail_closed_draft(self):
        (self.install/'models').mkdir();(self.install/'models/weight.bin').write_bytes(b'NEVERCOPY')
        (self.install/'.env').write_text('SECRET')
        (self.install/'lib').mkdir();(self.install/'lib/libprivate.a').write_text('SDK')
        r=self.build(platform_label='Windows verified')
        self.assertFalse(r['distributable']);self.assertFalse(r['published']);self.assertFalse(r['platform']['verified'])
        self.assertFalse(r['licenses']['licenses_complete']);self.assertEqual(len(r['selected_install_files']),4)
        with tarfile.open(self.root/'out/ernie-runtime-draft.tar.gz') as tar:
            names=tar.getnames();self.assertFalse(any('/models/' in n or n.endswith('.env') or 'libprivate' in n for n in names))
            self.assertTrue(all(n.startswith('ernie-runtime/') for n in names))
        files=json.loads((self.root/'out/files.json').read_text())
        for f in files:
            p=self.root/'out/ernie-runtime'/f['path'];self.assertEqual(sha256(p),f['sha256']);self.assertEqual(p.stat().st_size,f['size'])
    def test_sdk_explicit(self):
        lib=self.install/'lib/cmake/Ernie';lib.mkdir(parents=True);(lib/'ErnieConfig.cmake').write_text('# fixture')
        (self.install/'lib/libernie.a').write_bytes(b'archive')
        r=self.build(include_sdk=True)
        self.assertTrue(r['include_sdk']);self.assertTrue(any(f['path'].endswith('libernie.a') for f in r['selected_install_files']))
    def test_missing_sdk_rejected(self):
        with self.assertRaisesRegex(ValueError,'SDK missing'):self.build(include_sdk=True)
    def test_symlink_and_existing_output_rejected(self):
        self.build()
        with self.assertRaises(FileExistsError):self.build()
        p=self.install/'bin/ernie-image';p.unlink();p.symlink_to(self.install/'share/ernie-image/LICENSE')
        with self.assertRaisesRegex(ValueError,'Symlink'):self.build('other')
    def test_unsafe_output_rejected(self):
        with self.assertRaises(ValueError):build_release(self.install,self.install/'new')
    def test_reproducible_archive(self):
        self.build('one');self.build('two')
        self.assertEqual(sha256(self.root/'one/ernie-runtime-draft.tar.gz'),sha256(self.root/'two/ernie-runtime-draft.tar.gz'))
    def test_wrong_install_evidence_cannot_claim_verified(self):
        e=self.root/'evidence.json';e.write_text(json.dumps({'status':'passed','installed_files':[]}))
        r=self.build(install_evidence=e)
        self.assertFalse(r['evidence']['installation']['selected_files_match']);self.assertFalse(r['distributable'])
        self.assertIn('Installation evidence not bound/passed',r['blockers'])
    def test_real_install_evidence_byte_field(self):
        known=[]
        for name in ['bin/ernie-image','share/ernie-image/LICENSE','share/ernie-image/RUNNING.md','share/ernie-image/sources.lock.json']:
            p=self.install/name;known.append({'path':name,'bytes':p.stat().st_size,'sha256':sha256(p)})
        e=self.root/'e.json';e.write_text(json.dumps({'status':'passed','installed_files':known}))
        r=self.build(install_evidence=e)
        self.assertTrue(r['evidence']['installation']['selected_files_match']);self.assertFalse(r['distributable'])
    def test_authenticated_crate_archive_notice(self):
        import io
        source=self.root/'source';(source/'tokenizer').mkdir(parents=True)
        registry=self.root/'registry/src';registry.mkdir(parents=True)
        archive=self.root/'registry/cache/cache/demo-1.0.crate';archive.parent.mkdir(parents=True)
        with tarfile.open(archive,'w:gz') as tar:
            for name,data in {'Cargo.toml':b'[package]\nname="demo"\nversion="1.0"\nlicense="MIT"\n','LICENSE':b'fixture notice'}.items():
                info=tarfile.TarInfo('demo-1.0/'+name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        (source/'tokenizer/Cargo.lock').write_text('[[package]]\nname="demo"\nversion="1.0"\nsource="registry+https://example.invalid"\nchecksum="'+sha256(archive)+'"\n')
        r=collect_notices(source,self.root/'notices',registry)
        self.assertTrue(any(x.get('archive_sha256')==sha256(archive) for x in r['entries']))
        archive.write_bytes(b'corrupt')
        r=collect_notices(source,self.root/'notices2',registry)
        self.assertFalse(any(x['component']=='demo-1.0' for x in r['entries']))
    def test_source_and_build_mismatch_reported(self):
        source=self.root/'source';source.mkdir();(source/'a').write_text('changed')
        (source/'source-identity.json').write_text(json.dumps({'files':{'a':'a'*64}}))
        e=self.root/'build';e.mkdir();(e/'identity.json').write_text(json.dumps({'source_identity_sha256':'b'*64}));(e/'result.json').write_text(json.dumps({'exit_code':0,'source_changes_after_build':[]}))
        r=self.build(source=source,build_evidence=e)
        self.assertIn('Frozen source identity mismatch',r['blockers']);self.assertIn('Build/source identity not linked',r['blockers'])
    def test_crate_notice_requires_fixed_cache_checksum(self):
        source=self.root/'source';(source/'tokenizer').mkdir(parents=True)
        (source/'tokenizer/Cargo.lock').write_text('[[package]]\nname="demo"\nversion="1.0"\nsource="registry+https://example.invalid"\nchecksum="'+'a'*64+'"\n')
        registry=self.root/'registry';crate=registry/'cache/demo-1.0';crate.mkdir(parents=True)
        cargo=crate/'Cargo.toml';cargo.write_text('[package]\nname="demo"\nversion="1.0"\nlicense="MIT"\n')
        license=crate/'LICENSE';license.write_text('fixture notice')
        check={'package':'a'*64,'files':{'Cargo.toml':sha256(cargo),'LICENSE':sha256(license)}}
        (crate/'.cargo-checksum.json').write_text(json.dumps(check))
        r=collect_notices(source,self.root/'notices',registry)
        self.assertTrue(any(e['component']=='demo-1.0' for e in r['entries']))
        license.write_text('corrupt')
        r=collect_notices(source,self.root/'notices2',registry)
        self.assertFalse(any(e['component']=='demo-1.0' for e in r['entries']));self.assertTrue(any('checksum' in g for g in r['gaps']))
