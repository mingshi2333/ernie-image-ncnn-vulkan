from pathlib import Path
import hashlib,json,subprocess,sys
b=Path(__file__).resolve().parent
side=sys.argv[1] if len(sys.argv)==2 else ''
if side != 'on':raise SystemExit('usage: worker.py on')
p=json.loads((b/'plan.json').read_text())
def sha(x):
 h=hashlib.sha256()
 with x.open('rb') as f:
  for d in iter(lambda:f.read(1<<20),b''):h.update(d)
 return h.hexdigest()
def check(x,item):
 if not x.is_file() or x.stat().st_size!=item['size'] or sha(x)!=item['sha256']:raise SystemExit('identity mismatch: '+str(x))
for n,item in p['inputs'].items():check(b/n,item)
check(b/f'ernie-image-{side}',p['builds'][side])
cache=p['builds'][side]['cmake_cache'];check(b/cache['file'],cache)
if p['builds'][side]['required_cache_value'] not in (b/cache['file']).read_text().splitlines():raise SystemExit('build option mismatch')
for n,item in p['source_files'].items():check(b/p['source_directory']/n,item)
for n,item in p['ncnn_sources'].items():check(b/n,item)
root=Path(p['model_root'])
for n,item in p['model_files'].items():check(root/n,item)
out=b/side
for n in ('native.png','metrics.json','driver.log'):
 if (out/n).exists():raise SystemExit('refuse overwrite: '+str(out/n))
if (out/'trace').exists():raise SystemExit('refuse overwrite trace')
with (out/'driver.log').open('wb') as log:
 r=subprocess.run(p['commands'][side],stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(r.returncode)
