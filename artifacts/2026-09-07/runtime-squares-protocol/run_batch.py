import json
from pathlib import Path
import subprocess
import time
base=Path(__file__).resolve().parent
results=[]
for step in json.loads((base/'commands.json').read_text()):
 print(json.dumps({'starting':step['case']+' '+step['phase']}),flush=True)
 start=time.monotonic()
 with (base/(step['case']+'-'+step['phase']+'.log')).open('xb') as log:
  process=subprocess.run(step['argv'],stdout=log,stderr=subprocess.STDOUT)
 results.append({**step,'return_code':process.returncode,'wall_seconds':time.monotonic()-start})
 (base/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results[-1]),flush=True)
 if process.returncode:raise SystemExit(process.returncode)
