#!/usr/bin/env python3
"""Export authenticated tracked ncnn sources; patch allocator observation only.
No original checkout writes. The full file inventory is SHA256-bound, not just
an allocator snippet. Optional uninitialized pybind11 remains excluded (Python OFF).
"""
import argparse,hashlib,json,re,subprocess,shutil
from pathlib import Path
PIN='6a1bf000f363714839a36793addc8c879d3d899e'
FULL_SHA='e5e8d449ddb09e2183faca8be4ab535899ff8bc463665fea209c9b05604ef4d5'
ALLOCATOR_SHA='601d69dab40823366fa6aa2be00c8e37bb0f1e960fe96323c36daed887c4fda2'
PINS={PIN:(FULL_SHA,ALLOCATOR_SHA),'f6f734f44d66f469fefee9ee401fd1cb5e3d573e':('a3340d10902b5102ee19fe1af2d1c317947b111595e7a92884ccf5236a0013af','fe8f6bc2fcee095ed172763445ddcd6c8e39070a561eaf5048fdb433246dd6b0')}
def sha(data):return hashlib.sha256(data).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
def inventory(root,prefix=''):
 result={}
 for row in git(root,'ls-files','--stage','-z').split(b'\0'):
  if not row:continue
  meta,name=row.split(b'\t');mode,blob,stage=meta.decode().split();name=name.decode();p=root/name;key=prefix+name
  if stage!='0':raise ValueError('Unmerged source')
  if mode=='160000':
   result[key+'/GITLINK']=blob
   if name=='glslang':
    if git(p,'rev-parse','HEAD').decode().strip()!=blob:raise ValueError('Submodule pin mismatch')
    result.update(inventory(p,key+'/'))
   elif name!='python/pybind11':raise ValueError('Unexpected submodule')
   continue
  if mode not in ('100644','100755') or p.is_symlink():raise ValueError('Unsupported source entry')
  data=p.read_bytes()
  if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=blob:raise ValueError('Modified pinned source '+key)
  result[key]=sha(data)
 return result

def function_body(text,name):
 start=text.index(name);left=text.index('{',start);depth=1;right=left+1
 while depth:
  if text[right]=='{':depth+=1
  if text[right]=='}':depth-=1
  right+=1
 return left,right

def transform(text):
 if sha(text.encode()) not in {a for _,a in PINS.values()}:raise ValueError('Unreviewed allocator source')
 helper='''
#include "allocation_metrics_hook.h"
#include <cstdint>
namespace {
template<class T> std::uint64_t ernie_metric_id(T value) noexcept { return (std::uint64_t)value; }
std::uint64_t ernie_metric_device(const ncnn::VkAllocator* owner) noexcept { return owner->vkdev ? ernie_metric_id(owner->vkdev->vkdevice()) : 0; }
void ernie_metric_alloc(const ncnn::VkAllocator* owner, VkDeviceMemory memory, std::uint64_t bytes, std::uint32_t type, bool imported) noexcept {
    const auto& properties=owner->vkdev->info.physical_device_memory_properties();
    // Vulkan validated this memoryTypeIndex before allocation succeeded.
    const auto& t=properties.memoryTypes[type];
    ernie::allocation_memory_created(ernie_metric_id(owner->vkdev->vkdevice()),ernie_metric_id(owner),ernie_metric_id(memory),bytes,{type,t.heapIndex,t.propertyFlags,imported,properties.memoryHeaps[t.heapIndex].flags});
}
}
'''
 # Limit helpers to Vulkan build, including the public Vulkan types.
 text=text.replace('namespace ncnn {','#if NCNN_VULKAN\n'+helper+'#endif\n\nnamespace ncnn {',1)
 for name,imported in [('VkAllocator::allocate_memory(',False),('VkAllocator::allocate_dedicated_memory(',False),('VkAllocator::allocate_import_host_memory(',True)]:
  a,b=function_body(text,name);body=text[a:b];assert body.count('return memory;')==1
  body=body.replace('return memory;',f'ernie_metric_alloc(this,memory,size,memory_type_index,{str(imported).lower()});\n    return memory;');text=text[:a]+body+text[b:]
 pattern=r'vkFreeMemory\(vkdev->vkdevice\(\), (ptr->memory|memory), 0\);'
 count=len(re.findall(pattern,text))
 if count!=10:raise ValueError('Unexpected free call count '+str(count))
 text=re.sub(pattern,lambda m:m[0]+'\n        ernie::allocation_memory_destroyed(ernie_metric_device(this),ernie_metric_id(this),ernie_metric_id('+m[1]+'));',text)
 for name,code in [
  ('VkAllocator::VkAllocator(const VulkanDevice* _vkdev)','ernie::allocation_allocator_created(ernie_metric_device(this),ernie_metric_id(this));'),
  ('VkAllocator::~VkAllocator()','ernie::allocation_allocator_destroyed(ernie_metric_device(this),ernie_metric_id(this));')]:
  a,b=function_body(text,name);text=text[:b-1]+'    '+code+'\n'+text[b-1:]
 for cls,role in [('VkBlobAllocator','Blob'),('VkWeightAllocator','Weight'),('VkStagingAllocator','Staging'),('VkWeightStagingAllocator','Staging')]:
  a,b=function_body(text,cls+'::'+cls+'(const VulkanDevice*');text=text[:a+1]+f'\n    ernie::allocation_allocator_role(ernie_metric_device(this),ernie_metric_id(this),ernie::AllocationRole::{role});'+text[a+1:]
 return text

def derive(source,output):
 source=Path(source).resolve();output=Path(output).resolve()
 if source==output or source in output.parents:raise ValueError('Derived output must be outside original source')
 revision=git(source,'rev-parse','HEAD').decode().strip()
 if revision not in PINS:raise ValueError('Wrong ncnn pin')
 files=inventory(source);digest=sha(json.dumps(files,sort_keys=True,separators=(',',':')).encode())
 if digest!=PINS[revision][0]:raise ValueError('Full source identity mismatch '+digest)
 original=(source/'src/allocator.cpp').read_text();modified=transform(original)
 import difflib
 patch=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/src/allocator.cpp',tofile='b/src/allocator.cpp'))
 if output.exists():
  provenance=json.loads((output/'allocation-metrics-provenance.json').read_text())
  if provenance['full_source_sha256']!=digest or provenance['patcher_sha256']!=sha(Path(__file__).read_bytes()):raise ValueError('Stale derived source; use a new build directory')
  expected={name:(sha(modified.encode()) if name=='src/allocator.cpp' else d) for name,d in files.items() if not name.endswith('/GITLINK')}
  actual={str(p.relative_to(output)):sha(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name not in ['allocation-metrics-provenance.json','allocation-metrics.patch']}
  if actual!=expected:raise ValueError('Modified derived source')
  if (output/'allocation-metrics.patch').read_text()!=patch:raise ValueError('Modified patch record')
  return digest
 output.mkdir(parents=True)
 for name in files:
  if name.endswith('/GITLINK'):continue
  target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
 (output/'src/allocator.cpp').write_text(modified)
 (output/'allocation-metrics.patch').write_text(patch)
 (output/'allocation-metrics-provenance.json').write_text(json.dumps({'revision':revision,'full_source_sha256':digest,'files':files,'derived_allocator_sha256':sha(modified.encode()),'scope':'Linux ncnn allocator Vulkan memory events; Python and Android excluded','patcher_sha256':sha(Path(__file__).read_bytes())},indent=2))
 return digest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(derive(a.source,a.output))
