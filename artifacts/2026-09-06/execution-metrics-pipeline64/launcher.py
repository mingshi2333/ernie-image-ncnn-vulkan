from pathlib import Path
import hashlib,subprocess,sys
b=Path(__file__).resolve().parent
expected={'plan_sha256': 'd20b6b2ec68a286dc551eec730604cbf98b5be5a9678bc2036cc5c1efe1aec39', 'worker_sha256': 'aee653c47c0ec3cab3404d12e6ec7a70c28be627d02efec9cd2b9c1915e3a1b7', 'supervisor_sha256': 'b1e660e6fc8391bad9ab394a1c47c49aa24c8be93b9d52305f5fba1ee49a822f'}
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,k in (("plan.json","plan_sha256"),("worker.py","worker_sha256"),("supervisor.py","supervisor_sha256")):
 if h(b/n)!=expected[k]:raise SystemExit("authorization mismatch: "+n)
raise SystemExit(subprocess.run([sys.executable,str(b/"supervisor.py"),*sys.argv[1:]]).returncode)
