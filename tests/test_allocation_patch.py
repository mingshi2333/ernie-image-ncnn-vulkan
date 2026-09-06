import importlib.util,os,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('derive',Path(__file__).resolve().parents[1]/'cmake/derive_allocation_ncnn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AllocationPatchTests(unittest.TestCase):
 def test_unknown_source_rejected(self):
  with self.assertRaises(ValueError):m.transform('unreviewed allocator')
 @unittest.skipUnless(os.environ.get('NCNN_AUDIT_SOURCE'),'explicit pinned checkout required')
 def test_fixed_patch_preserves_original_calls_and_free_order(self):
  root=Path(os.environ['NCNN_AUDIT_SOURCE']);original=(root/'src/allocator.cpp').read_text();modified=m.transform(original)
  self.assertEqual(modified.count('vkAllocateMemory('),original.count('vkAllocateMemory('))
  self.assertEqual(modified.count('vkFreeMemory('),original.count('vkFreeMemory('))
  self.assertEqual(modified.count('ernie_metric_alloc(this,'),3)
  self.assertEqual(modified.count('ernie::allocation_memory_destroyed('),10)
  lines=modified.splitlines()
  for i,line in enumerate(lines):
   if 'ernie::allocation_memory_destroyed(' in line:self.assertIn('vkFreeMemory(',lines[i-1])
  self.assertIn(m.sha((root/'src/allocator.cpp').read_bytes()),{a for _,a in m.PINS.values()})
 @unittest.skipUnless(os.environ.get('NCNN_AUDIT_SOURCE'),'explicit pinned checkout required')
 def test_full_inventory_identity(self):
  import json
  inventory=m.inventory(Path(os.environ['NCNN_AUDIT_SOURCE']))
  self.assertIn(m.sha(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()),{d for d,_ in m.PINS.values()})
if __name__=='__main__':unittest.main()
