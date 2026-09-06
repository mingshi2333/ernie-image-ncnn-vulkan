import hashlib,json,os,subprocess,sys
from pathlib import Path
kind,cpus=sys.argv[1:]
assert kind in ('cpu','vulkan')
out=Path('outputs/d1-install-build-'+kind+'-v2');out.mkdir(exist_ok=False)
source=Path('outputs/d1-install-source-v2');metadata=source/'source-identity.json'
f=json.loads(metadata.read_text())
for p,sha in f['files'].items(): assert hashlib.sha256((source/p).read_bytes()).hexdigest()==sha,p
command=['systemd-run','--user','--scope','--unit=ernie-d1-install-'+kind+'-v2','-p','MemoryMax=4G','-p','MemorySwapMax=0','-p','CPUQuota=200%','/usr/bin/time','-v','taskset','-c',cpus,'cmake','--build','build-install-'+kind+'-v2','--parallel','2']
env=os.environ.copy();env.update(XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
(out/'identity.json').write_text(json.dumps(dict(command=command,source_identity_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(Path('build-install-'+kind+'-v2/CMakeCache.txt').read_bytes()).hexdigest()),indent=2)+'\n')
with (out/'build.log').open('w') as log:
 completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
changes=[p for p,sha in f['files'].items() if hashlib.sha256((source/p).read_bytes()).hexdigest()!=sha]
(out/'result.json').write_text(json.dumps(dict(exit_code=completed.returncode,source_changes_after_build=changes))+'\n')
print(kind,'build_exit_code',completed.returncode,'source_changes',changes)
raise SystemExit(completed.returncode or bool(changes))
