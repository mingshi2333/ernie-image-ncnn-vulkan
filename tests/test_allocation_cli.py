"""Actual CLI control flow, fake bounded inference; no GPU/model invocation."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
ON=Path(os.environ.get('ERNIE_ALLOCATION_TEST_ON','/nonexistent/allocation-on'))
OFF=Path(os.environ.get('ERNIE_ALLOCATION_TEST_OFF','/nonexistent/allocation-off'))
@unittest.skipUnless(ON.is_file() and OFF.is_file(),'build dedicated allocation CLI contracts first')
class AllocationCliTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def run_case(self,model='success',image='image.png',extra=(),runner=ON):
  report=self.p/'metrics.json';env=dict(os.environ,ERNIE_TEST_REPORT_PATH=str(report))
  args=[str(runner),'--model',model,'--prompt','cat','--device','cpu','--precision','fp32','--output',str(self.p/image),'--metrics-json',str(report),*extra]
  return subprocess.run(args,text=True,capture_output=True,env=env,timeout=10),report
 def test_success_lifetime_and_json(self):
  r,p=self.run_case();self.assertEqual(r.returncode,0,r.stderr);v=json.loads(p.read_text())
  self.assertEqual(v['total'],dict(live_bytes=0,peak_bytes=35,allocations=1));self.assertTrue(v['coverage_complete'])
  self.assertTrue(v['valid']);self.assertFalse(v['trace_enabled']);self.assertFalse(v['formal_speed_eligible']);self.assertFalse(v['formal_memory_eligible'])
  self.assertEqual(v['stage_times']['verify']['host_nanoseconds'],10);self.assertEqual(v['stage_times']['submissions'],3)
  self.assertIsNone(v['stage_times']['upload_bytes']);self.assertIsNone(v['stage_times']['verify']['gpu_nanoseconds'])
  self.assertEqual(v['devices'][0]['name'],'fake "device"\n');self.assertEqual(v['run_status'],'success');self.assertNotIn('schema_version',r.stdout)
  self.assertTrue(v['execution_finished_successfully'])
 def test_generation_error_survives_cleanup(self):
  r,p=self.run_case('generation-failure');self.assertEqual(r.returncode,1);self.assertIn('original generation failure',r.stderr)
  v=json.loads(p.read_text());self.assertEqual(v['run_status'],'generation_failed');self.assertTrue(v['coverage_complete']);self.assertEqual(v['total']['live_bytes'],0)
  self.assertFalse(v['execution_finished_successfully']);self.assertEqual(v['stage_times']['verify']['host_nanoseconds'],10)
 def test_image_error_is_distinct(self):
  r,p=self.run_case(image='image-failure.png');self.assertEqual(r.returncode,1);self.assertIn('original image failure',r.stderr)
  self.assertEqual(json.loads(p.read_text())['run_status'],'image_write_failed')
 def test_report_collision_preserves_image_and_file(self):
  r,p=self.run_case('report-failure');self.assertEqual(r.returncode,1);self.assertIn('Cannot create allocation report',r.stderr)
  self.assertEqual(p.read_text(),'preserve-existing-report');self.assertEqual((self.p/'image.png').read_bytes(),bytes([2,4,6]))
 def test_double_error_preserves_primary(self):
  r,p=self.run_case('generation-report-failure');self.assertEqual(r.returncode,1)
  self.assertIn('original generation failure\nAllocation report also failed:',r.stderr);self.assertIn('Allocation report also failed:',r.stderr);self.assertEqual(p.read_text(),'preserve-existing-report')
 def test_off_rejects_before_generation(self):
  r,p=self.run_case('generation-failure',runner=OFF);self.assertEqual(r.returncode,1);self.assertIn('no allocation instrumentation',r.stderr);self.assertFalse(p.exists())
 def test_trace_is_marked(self):
  r,p=self.run_case(extra=('--trace-dir',str(self.p/'trace')));self.assertEqual(r.returncode,0);self.assertTrue(json.loads(p.read_text())['trace_enabled'])
 def test_missing_device_identity_is_incomplete(self):
  r,p=self.run_case('missing-identity');self.assertEqual(r.returncode,0)
  v=json.loads(p.read_text());self.assertFalse(v['device_identity_complete']);self.assertFalse(v['coverage_complete'])
 def test_conflicting_device_identity_invalidates_measurement(self):
  r,p=self.run_case('conflicting-identity');self.assertEqual(r.returncode,0)
  v=json.loads(p.read_text());self.assertFalse(v['valid']);self.assertFalse(v['coverage_complete'])
 def test_live_allocation_is_incomplete(self):
  r,p=self.run_case('live-allocation');self.assertEqual(r.returncode,0)
  v=json.loads(p.read_text());self.assertFalse(v['coverage_complete']);self.assertEqual(v['total']['live_bytes'],35)
 def test_borrowed_instance_is_incomplete(self):
  r,p=self.run_case('borrowed-instance');self.assertEqual(r.returncode,0)
  v=json.loads(p.read_text());self.assertTrue(v['initial_instance_present']);self.assertFalse(v['coverage_complete'])
 def test_no_allocation_is_unavailable(self):
  r,p=self.run_case('no-allocation');self.assertEqual(r.returncode,0)
  v=json.loads(p.read_text());self.assertIsNone(v['total']);self.assertFalse(v['coverage_complete'])
 def test_existing_report_is_not_overwritten(self):
  p=self.p/'metrics.json';p.write_text('keep');r,_=self.run_case();self.assertEqual(r.returncode,1);self.assertEqual(p.read_text(),'keep')
if __name__=='__main__':unittest.main()
