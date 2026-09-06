import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_vulkan_inventory import search_roots,inspect_manifest,resolve_library,manifests,verify_inventory
class InventoryTests(unittest.TestCase):
 def test_etc_and_non_arch_filename(self):
  self.assertIn('/etc/vulkan/icd.d',search_roots({'HOME':'/fake'}))
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'amd_icd64.json').write_text('{}');self.assertEqual(len(manifests([t])[1]),1)
 def test_override_and_settings_roots(self):
  roots=search_roots({'HOME':'/fake','VK_IMPLICIT_LAYER_PATH':'/implicit','VK_DRIVER_FILES':'/driver/custom.json','XDG_CONFIG_DIRS':'/custom-config'})
  self.assertIn('/implicit',roots);self.assertIn('/driver/custom.json',roots);self.assertIn('/custom-config/vulkan/icd.d',roots);self.assertIn('/etc/vulkan/loader_settings.d',roots)
 def test_absolute_relative_basename_and_elf32(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);lib=p/'lib64.so';lib.write_bytes(b'\x7fELF\x02'+bytes(15));small=p/'lib32.so';small.write_bytes(b'\x7fELF\x01'+bytes(15));m=p/'amd.json'
   for name in (str(lib),'./lib64.so','lib64.so'):
    m.write_text(json.dumps({'ICD':{'library_path':name}}));self.assertEqual(inspect_manifest(m,{'lib64.so':[str(lib)]},{})[0]['path'],str(lib))
   self.assertIsNone(resolve_library(m,str(small),{},{}))
 def test_unresolved_and_membership_fail_closed(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)
   with self.assertRaisesRegex(ValueError,'Unresolved'):resolve_library(p/'m.json','missing.so',{}, {})
   c={'environment':{},'roots':[t],'directory_membership':manifests([t])[0],'unresolved':[{'missing':True}]}
   with self.assertRaisesRegex(ValueError,'Unresolved'):verify_inventory(c,{},require_resolved=True)
   c['unresolved']=[];(p/'added.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'membership'):verify_inventory(c,{})
if __name__=='__main__':unittest.main()
