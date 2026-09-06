import json
import os
import types
from unittest import mock
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from vae_reference_scope import verify_files, runtime_files, WorkerRuntime, validate_worker_runtime, process_identity, execute_reference

class RuntimeIdentityTests(unittest.TestCase):
    def test_runtime_file_hash_detects_same_size_change(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'runtime.py';p.write_bytes(b'abcd')
            rows={str(p):{'size':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}}
            verify_files(rows);p.write_bytes(b'abce')
            with self.assertRaisesRegex(ValueError,'Frozen runtime file differs'):verify_files(rows)

    def test_virtual_environment_invocation_must_not_be_realpath(self):
        import os
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'venv/bin';p.mkdir(parents=True);entry=p/'python';entry.symlink_to(sys.executable)
            self.assertNotEqual(os.path.abspath(entry),str(entry.resolve()))
            self.assertEqual(Path(os.path.abspath(entry)).parent.name,'bin')

class WorkerRuntimeTests(unittest.TestCase):
    def setUp(self):
        patch=mock.patch.object(sys,'argv',[__file__]);patch.start();self.addCleanup(patch.stop)

    def fixture(self,root):
        source=root/'module.py';source.write_text('value=1\n')
        row={'size':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        identity=root/'allow.json'
        identity.write_text(json.dumps({'files':{str(source):row},'mapped_files':[str(source)],'required_files':[str(source)],'required_mapped_files':[str(source)],'prefix':sys.prefix,'executable':sys.executable}))
        return source,row,identity

    def test_actual_new_module_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'late.py';path.write_text('x=1\n')
            module=types.ModuleType('vae_review_late');module.__file__=str(path)
            with mock.patch.dict(sys.modules,{'vae_review_late':module}):
                files,_=runtime_files()
            self.assertIn(str(path),files)
            self.assertEqual(files[str(path)]['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_four_authenticated_boundaries_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,row,identity=self.fixture(root);collector=WorkerRuntime(identity,root/'actual')
            with mock.patch('vae_reference_scope.runtime_files',return_value=({str(source):row},[str(source)])):
                for phase in collector.phases:collector.checkpoint(phase)
            collector.finish(True)
            report=root/'actual/identity.json';runtime=json.loads(identity.read_text())
            allow_sha=hashlib.sha256(identity.read_bytes()).hexdigest()
            import vae_reference_scope
            collector_sha=hashlib.sha256(Path(vae_reference_scope.__file__).read_bytes()).hexdigest()
            validate_worker_runtime(report,runtime,allow_sha,collector_sha,process_identity(os.getpid()),sys.argv[0])
            original=json.loads(report.read_text())
            # A self-consistent truncated union with identities unchanged must fail.
            data=json.loads(report.read_text())
            for point in data['checkpoints']:point['mapped_files']=[]
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Required runtime coverage missing'):
                validate_worker_runtime(report,runtime,allow_sha,collector_sha,process_identity(os.getpid()),sys.argv[0])
            extra=root/'second.py';extra.write_text('x=2')
            runtime['files'][str(extra)]={'size':3,'sha256':hashlib.sha256(extra.read_bytes()).hexdigest()}
            runtime['required_files'].append(str(extra))
            report.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'Required runtime coverage missing'):
                validate_worker_runtime(report,runtime,allow_sha,collector_sha,process_identity(os.getpid()),sys.argv[0])
            del runtime['files'][str(extra)];runtime['required_files'].remove(str(extra))
            wrong_process={**process_identity(os.getpid()),'pid':os.getpid()+1}
            with self.assertRaisesRegex(ValueError,'incomplete or unauthenticated'):
                validate_worker_runtime(report,runtime,allow_sha,collector_sha,wrong_process,sys.argv[0])
            data=original;data['checkpoints'].pop();report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'incomplete or unauthenticated'):
                validate_worker_runtime(report,runtime,allow_sha,collector_sha,process_identity(os.getpid()),sys.argv[0])

    def test_unknown_lazy_dependency_archived_but_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,row,identity=self.fixture(root);collector=WorkerRuntime(identity,root/'actual')
            late=root/'late.so';late.write_bytes(b'not a real binary, only an inventory fixture')
            late_row={'size':late.stat().st_size,'sha256':hashlib.sha256(late.read_bytes()).hexdigest()}
            for index,phase in enumerate(collector.phases):
                files={str(source):row}
                if index:files[str(late)]=late_row
                with mock.patch('vae_reference_scope.runtime_files',return_value=(files,list(files))):collector.checkpoint(phase)
            with self.assertRaisesRegex(ValueError,'not authenticated'):collector.finish(True)
            report=json.loads((root/'actual/identity.json').read_text())
            self.assertFalse(report['authenticated']);self.assertEqual(report['unknown_files'],[str(late)])
            self.assertEqual((root/'actual/objects'/late_row['sha256']).read_bytes(),late.read_bytes())

    def test_controller_binds_real_child_pid_and_start_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cg=root/'cg';cg.mkdir()
            (cg/'memory.current').write_text('0');(cg/'memory.events').write_text('oom 0\n')
            output=root/'child.json'
            code=("import os,pathlib,json,time; p=os.getpid(); "
                  "s=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split(); "
                  "pathlib.Path("+repr(str(output))+").write_text(json.dumps({'pid':p,'parent_pid':os.getppid(),'start_ticks':int(s[19])})); time.sleep(.05)")
            record=execute_reference([sys.executable,'-c',code],root,{'host_min':0,'timeout_seconds':10},cg)
            self.assertEqual(record['process_identity'],json.loads(output.read_text()))
            self.assertEqual(record['process_identity']['parent_pid'],os.getpid())

    def test_changed_loaded_file_rejected_and_failure_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,row,identity=self.fixture(root);collector=WorkerRuntime(identity,root/'actual')
            with mock.patch('vae_reference_scope.runtime_files',return_value=({str(source):row},[str(source)])):
                collector.checkpoint('before_model')
            source.write_text('value=2\n')
            altered={'size':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
            try:
                with mock.patch('vae_reference_scope.runtime_files',return_value=({str(source):altered},[str(source)])):
                    collector.checkpoint('after_model')
            except ValueError as error:
                self.assertIn('changed during model execution',str(error))
                collector.finish(False)
            else:self.fail('changed file accepted')
            self.assertFalse(json.loads((root/'actual/identity.json').read_text())['authenticated'])

if __name__=='__main__':unittest.main()
