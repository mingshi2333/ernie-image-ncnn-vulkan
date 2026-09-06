#!/usr/bin/env python3
"""CPU-only conservative Linux loader inventory. Unresolved entries never authorize mapped libraries.

Search roots follow Khronos Vulkan-Loader LoaderDriverInterface/LoaderLayerInterface.
This is a discovery superset, not a claim to emulate loader selection/order. Actual
mapped-file authentication remains mandatory. Never invokes Vulkan or loads a model.
"""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def elf_bits(p):
 with Path(p).open('rb') as f:b=f.read(20)
 if b[:4]!=b'\x7fELF' or len(b)<20:raise ValueError('Not ELF: '+str(p))
 if b[4] not in (1,2):raise ValueError('Unknown ELF class')
 return 32 if b[4]==1 else 64

def search_roots(env, sysconf=('/etc','/usr/local/etc')):
 home=Path(env['HOME'])
 bases=[env.get('XDG_CONFIG_HOME') or str(home/'.config'),*(env.get('XDG_CONFIG_DIRS') or '/etc/xdg').split(':'),*sysconf,env.get('XDG_DATA_HOME') or str(home/'.local/share'),*(env.get('XDG_DATA_DIRS') or '/usr/local/share:/usr/share').split(':')]
 roots=[str(Path(b)/'vulkan'/kind) for b in bases if b for kind in ('icd.d','implicit_layer.d','explicit_layer.d')]
 # Include overridden and additional roots as a conservative superset. Do not
 # mutate process environment to select a different driver or layer.
 for key in ('VK_DRIVER_FILES','VK_ICD_FILENAMES','VK_ADD_DRIVER_FILES','VK_LAYER_PATH','VK_ADD_LAYER_PATH','VK_IMPLICIT_LAYER_PATH','VK_ADD_IMPLICIT_LAYER_PATH'):
  roots.extend(x for x in env.get(key,'').split(':') if x)
 roots.extend(str(Path(b)/'vulkan/loader_settings.d') for b in [str(home/'.local/share'),env.get('XDG_DATA_HOME',''),'/etc'] if b)
 return list(dict.fromkeys(roots))

def resolve_library(manifest,name,cache,env):
 p=Path(name)
 if p.is_absolute():candidates=[p]
 elif '/' in name:candidates=[manifest.parent/p]
 else:
  candidates=[Path(d)/name for d in env.get('LD_LIBRARY_PATH','').split(':') if d]
  candidates.extend(Path(x) for x in cache.get(name,[]))
 found=[]
 for c in candidates:
  if c.is_file() and c.resolve() not in found:found.append(c.resolve())
 if not found:raise ValueError('Unresolved library '+name)
 valid=[p for p in found if elf_bits(p)==64]
 if not valid:return None
 # Do not silently pick between competing ELF64 basename resolutions.
 if len(valid)>1:raise ValueError('Ambiguous ELF64 library '+name+': '+str(valid))
 return valid[0]

def manifests(roots):
 membership={};files=set()
 for value in roots:
  p=Path(value)
  if p.is_dir():members=sorted(str(x.absolute()) for x in p.glob('*.json'))
  elif p.is_file():members=[str(p.absolute())]
  else:members=[]
  membership[value]={'exists':p.exists(),'members':members};files.update(members)
 return membership,sorted(files)

def inspect_manifest(path,cache,env):
 data=json.loads(path.read_text());entries=[]
 if 'ICD' in data:entries.append(('ICD',data['ICD']))
 if 'layer' in data:entries.append(('layer',data['layer']))
 entries.extend(('layer',x) for x in data.get('layers',[]))
 if not entries:raise ValueError('No ICD or layer entry')
 rows=[]
 for kind,item in entries:
  if 'library_path' not in item:
   if item.get('component_layers'):rows.append({'kind':kind,'status':'meta_layer','components':item['component_layers']});continue
   raise ValueError('Missing library_path')
  lib=resolve_library(path,item['library_path'],cache,env)
  rows.append({'kind':kind,'name':item.get('name'),'declared':item['library_path'],'path':str(lib) if lib else None,'status':'ELF64' if lib else 'excluded_ELF32'})
 return rows

def build(env):
 ld=subprocess.run(['ldconfig','-p'],capture_output=True,text=True,check=True).stdout;cache={}
 for line in ld.splitlines():
  if '=>' in line:cache.setdefault(line.split()[0],[]).append(line.split('=>')[1].strip())
 roots=search_roots(env);membership,files=manifests(roots);bound={};records={};failures=[];libs=set();layer_env=set()
 for name in files:
  p=Path(name);bound[name]=sha(p)
  try:
   if p.parent.name=='loader_settings.d':raise ValueError('Loader settings require separate policy review')
   data=json.loads(p.read_text());items=[data.get('layer',{})]+data.get('layers',[])
   for item in items:layer_env.update(item.get('enable_environment',{}));layer_env.update(item.get('disable_environment',{}))
   rows=inspect_manifest(p,cache,env);records[name]=rows
   libs.update(Path(r['path']) for r in rows if r.get('path'))
  except (ValueError,OSError,KeyError) as e:failures.append({'manifest':name,'error':str(e)})
 closure={};queue=list(libs)
 while queue:
  p=queue.pop()
  if str(p) in closure:continue
  r=subprocess.run(['ldd',str(p)],capture_output=True,text=True)
  closure[str(p)]={'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode};bound[str(p)]=sha(p)
  if r.returncode or 'not found' in r.stdout:failures.append({'library':str(p),'error':'ldd unresolved/error'});continue
  for name in re.findall(r'(?:=>\s*)?(/[^\s()]+)',r.stdout):
   d=Path(name).resolve()
   if d.is_file() and elf_bits(d)==64 and str(d) not in closure:queue.append(d)
 return {'status':'cpu_catalogued_with_unresolved' if failures else 'cpu_catalogued_not_execution_validated','scope':'documented Linux search-root superset; no driver filtering or model execution','environment':{k:v for k,v in env.items() if k.startswith(('VK_','XDG_','LD_')) or k=='HOME'},'layer_environment':{k:env.get(k) for k in sorted(layer_env)},'roots':roots,'directory_membership':membership,'manifests':records,'unresolved':failures,'dependency_closure':closure,'ldconfig_stdout':ld,'bound':bound}

def verify_inventory(catalog,env,require_resolved=False):
 observed={k:v for k,v in env.items() if k.startswith(('VK_','XDG_','LD_')) or k=='HOME'}
 if observed!=catalog['environment']:raise ValueError('Loader environment changed')
 if any(env.get(k)!=v for k,v in catalog.get('layer_environment',{}).items()):raise ValueError('Layer activation environment changed')
 if manifests(catalog['roots'])[0]!=catalog['directory_membership']:raise ValueError('Loader directory membership changed')
 if require_resolved and catalog['unresolved']:raise ValueError('Unresolved loader inventory')
 for name,digest in catalog['bound'].items():
  if sha(name)!=digest:raise ValueError('Loader file changed: '+name)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists():p.error('Use a new output')
 result=build(dict(os.environ));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'manifests':len(result['manifests']),'bound':len(result['bound']),'unresolved':result['unresolved']}))
