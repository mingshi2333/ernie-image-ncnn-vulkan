import json,hashlib,math
from pathlib import Path
import numpy as np
root=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=[]
names=['pipeline1024-chinese-s64-fp32-v1','pipeline1024-chinese-s64-fp32-kahan-v1','pipeline1024-chinese-s64-fp32-chunked-v1','pipeline1024-chinese-s64-fp32-reference-text-v1','pipeline1024-chinese-vectordown-fp32-v1']
def metrics(a,b):
 x=np.fromfile(a,'<f4').astype('f8');y=np.fromfile(b,'<f4').astype('f8');assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all();d=x-y
 return {'nrmse':float(np.sqrt(np.sum(d*d)/np.sum(y*y))),'max_abs_error':float(np.abs(d).max()),'error_l2':float(np.sqrt(np.sum(d*d))),'reference_l2':float(np.sqrt(np.sum(y*y)))}
for name in names:
 p=root/'outputs'/name;r=json.loads((p/'result.json').read_text());f=json.loads((p/'reference/fixture.json').read_text());tf=json.loads((p/'reference/text/fixture.json').read_text());cmd=r['command']
 prompt=Path(cmd[cmd.index('--prompt-file')+1]).read_bytes() if '--prompt-file' in cmd else cmd[cmd.index('--prompt')+1].encode()
 ids=list(map(int,(p/'trace/ids.txt').read_text().split()))
 assert prompt==f['prompt'].encode()==tf['prompt'].encode() and ids==f['ids']==tf['ids']
 assert f['config']=={'packed_width':64,'packed_height':64,'text_bucket':64,'dit_text_tokens':64,'text_layers':25,'dit_layers':36}
 assert tf['tokens']==64 and tf['valid_tokens']==32 and tf['blocks']==25
 assert sha(p/'ernie-image.snapshot')==r['runner_sha256'];assert sha(p/'scripts/validate_pipeline.py')==r['validator_sha256']
 for file,digest in r['source_snapshot'].items():assert sha(p/'scripts'/file)==digest
 assert sha(p/'reference/fixture.json')==r['reference_fixture_sha256']=='81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7'
 assert r['device']=='vulkan' and r['dit_precision']=='fp32' and r['text_scheduler_vae_precision']=='fp32'
 rows={x['tensor']:x for x in r['comparisons']}
 for key in ['initial','text','padded-text','constant-0','constant-1','constant-2']:
  assert sha(p/'trace'/(key+'.f32'))==rows[key]['sha256'];assert sha(p/'reference'/f['inputs'][key]['file'])==f['inputs'][key]['sha256']
 assert sha(p/'trace/initial.f32')==f['inputs']['initial']['sha256']
 embedding=None
 if '--embeddings' in cmd:
  ep=Path(cmd[cmd.index('--embeddings')+1]);embedding={'path':str(ep),'sha256':sha(ep)};assert ep.read_bytes()==(p/'trace/text.f32').read_bytes()
 stages=[]
 for i in range(8):
  for key in ['prediction','step']:
   item=f['outputs'][i][key];assert sha(p/'reference'/item['file'])==item['sha256'];assert sha(p/'trace'/f'{key}-{i}.f32')==rows[f'{key}-{i}']['sha256']
  prediction=metrics(p/'trace'/f'prediction-{i}.f32',p/'reference'/f'prediction-{i}.f32');sample=metrics(p/'trace'/f'step-{i}.f32',p/'reference'/f'step-{i}.f32');stages.append({'step':i,'prediction':prediction,'sample':sample})
 results.append({'run':name,'prompt_repr':repr(prompt.decode()),'prompt_sha256':hashlib.sha256(prompt).hexdigest(),'prompt_bytes':len(prompt),'trailing_lf':prompt.endswith(b'\n'),'ids':ids,'ids_file_sha256':sha(p/'trace/ids.txt'),'bucket':64,'valid_tokens':32,'runner_sha256':r['runner_sha256'],'validator_sha256':r['validator_sha256'],'python_snapshot_files_verified':len(r['source_snapshot']),'native_build_source_manifest_complete':False,'embedding':embedding,'reference_padded_text_exact':(p/'trace/padded-text.f32').read_bytes()==(p/'reference/padded-text.f32').read_bytes(),'precision':r['text_scheduler_vae_precision'],'steps':stages})
# Independently inspect historical head/block traces with bounded sequential arrays.
p=root/'outputs/diagnostic-chinese-step0-stages-v1';r=json.loads((p/'result.json').read_text());f=json.loads((p/'oracle/fixture/fixture.json').read_text());assert sha(p/'runner.snapshot')==r['prediction']['runner_sha256'];assert sha(p/'oracle/fixture/fixture.json')==r['prediction']['fixture_sha256']
for key,item in f['inputs'].items():assert sha(p/'oracle/fixture'/item['file'])==item['sha256']
stages=[]
for index in [0,1,13,15,19,23,28,29,34,35]:
 name=f'block-{index}';item=f['stages'][name];record=next(x for x in r['stages'] if x['stage']==name)
 assert sha(p/'oracle/fixture'/item['file'])==item['sha256'];assert sha(p/'trace'/f'{name}.f32')==record['actual_sha256']
 stages.append({'block':index,**metrics(p/'trace'/f'{name}.f32',p/'oracle/fixture'/item['file'])})
e=root/'outputs/diagnostic-chinese-exact-heads-v1';er=json.loads((e/'native/result.json').read_text());ef=json.loads((e/'fixture/fixture.json').read_text());assert sha(e/'runner.snapshot')==er['runner_sha256'];assert sha(e/'fixture/fixture.json')==er['fixture_sha256']
for name,item in ef['inputs'].items():assert sha(e/'fixture'/item['file'])==item['sha256']
assert sha(e/'fixture'/ef['expected']['file'])==ef['expected']['sha256'];assert sha(e/'native/actual.f32')==er['actual_sha256']
output={'scope':'Existing CPU tensor analysis; not new runtime evidence','trajectories':results,'historical_stage_runner':r['prediction']['runner_sha256'],'historical_selected_block_errors':stages,'historical_exact_head_hidden_state':metrics(e/'native/actual.f32',e/'fixture'/ef['expected']['file']),'historical_exact_head_fixture_sha256':sha(e/'fixture/fixture.json'),'historical_exact_head_inputs':ef['inputs'],'source_script_sha256':sha(__file__)}
Path('.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-next-evidence.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'prompts':[{'run':r['run'],'sha':r['prompt_sha256'],'LF':r['trailing_lf'],'tokens':len(r['ids'])} for r in results],'first_step':[{'run':r['run'],'prediction':r['steps'][0]['prediction'],'sample':r['steps'][0]['sample']} for r in results],'stages':stages,'exact_heads':output['historical_exact_head_hidden_state']},indent=2))
