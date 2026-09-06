#!/usr/bin/env python3
"""Read-only PE evidence verification; writes only a new requested artifact folder.
Run from the reviewed worktree: python review.py SOURCE NEW_OUTPUT
No model imports or inference. Stream hashes, one pair of vocabulary logits at a time.
"""
import hashlib,json,shutil,sys,sysconfig
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while block:=f.read(1<<20):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify(p,d):
 actual=sha(p)
 if actual!=d:raise ValueError(f'Checksum mismatch {p}: {actual} != {d}')
 return actual

def main(source,out):
 source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 snapshot=read(source/'snapshot.json');contract=read(source/'contract.json');journal=read(source/'journal.json')
 assert journal['status']=='passed' and len(journal['cases'])==len(contract['cases'])==12
 verify(source/'snapshot.json',journal['source_snapshot_sha256']);verify(source/'contract.json',snapshot['contract_sha256'])
 verify(source/'ernie-pe-runner.snapshot',snapshot['runner_sha256'])
 assert len(snapshot['tools'])==58
 for p,d in snapshot['tools'].items():verify(source/'snapshots'/p,d)
 for p in ['snapshot.json','contract.json','journal.json']:shutil.copy2(source/p,out/p)
 model=Path(snapshot['model_directory']);verify(model/'manifest.json',snapshot['model_manifest_sha256']);manifest=read(model/'manifest.json')
 # Hash original packaged files without loading any model or allocating tensors.
 for p,d in manifest['files'].items():verify(model/p,d)
 official=Path('models/official');names=[f'pe-block-{i:02d}' for i in range(26)]+['pe-embed','pe-norm']
 weights=[]
 for name in names:
  m=read(official/(name+'.manifest.json'));assert m['revision']=='bc68c81e2a1730a394d5fc9fae70713dee940140'
  verify(official/(name+'.safetensors'),m['sha256']);weights.append(m['sha256'])
 model_source=Path(sysconfig.get_paths()['purelib'])/'transformers/models/ministral3/modeling_ministral3.py'
 for name,info in contract['tokenizer_sources'].items():verify(Path('models/pe-tokenizer')/name,info['sha256'])
 summaries=[]
 for case,j in zip(contract['cases'],journal['cases']):
  cid=case['id'];assert j['id']==cid and j['status']=='passed';folder=source/cid;refdir=folder/'reference';natdir=folder/'native-validation';native=natdir/'native'
  ref=read(refdir/'reference.json');result=read(natdir/'result.json');verify(natdir/'result.json',j['result_sha256']);verify(refdir/'reference.json',result['reference_manifest_sha256'])
  assert result['passed'] and result['return_code']==0 and result['gates']==dict(nrmse=.0002,atol=.0002,rtol=.0002)
  assert result['runner_sha256']==snapshot['runner_sha256'];verify(natdir/'ernie-pe-runner',snapshot['runner_sha256'])
  assert result['validator_sha256']==snapshot['tools']['validate_pe.py'] and ref['script_sha256']==snapshot['tools']['reference_pe.py']
  assert result['model_manifest_sha256']==snapshot['model_manifest_sha256'] and ref['source_weight_sha256']==weights
  verify(model_source,ref['model_source_sha256']);assert ref['official_model_revision']==manifest['official_model_revision']
  assert ref['transformers_revision']=='7d9754a05193eb79b1d86aa744b622b8068008cd'
  assert ref['batch_case_id']==result['batch_case_id']==cid
  for key in ['width','height','max_tokens','input_tokens']:assert ref[key]==case[key]
  assert ref['input_prompt']==case['prompt'] and case['mode']=='greedy' and case['temperature']==0 and case['top_p']==1
  assert (natdir/'prompt.txt').read_bytes()==case['prompt'].encode()
  for p,d in ref['files'].items():verify(refdir/p,d)
  for p,d in result['native_files'].items():verify(native/p,d)
  for p in ['input-ids.txt','generated-ids.txt','enhanced.txt']:assert (refdir/p).read_bytes()==(native/p).read_bytes() and result['exact'][p]
  ids=list(map(int,(refdir/'input-ids.txt').read_text().split()));generated=list(map(int,(refdir/'generated-ids.txt').read_text().split()))
  digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
  assert len(ids)==case['input_tokens'] and digest==case['input_ids_sha256']==ref['batch_input_ids_sha256']
  count=len(generated);assert count==ref['generated_tokens']==j['generated_tokens']==j['logits_passed']==len(result['logits'])
  assert count<=case['max_tokens'] and ref['eos']==(generated[-1]==2)
  status=dict(line.split() for line in (native/'status.txt').read_text().splitlines());assert int(status['eos'])==int(ref['eos'])
  if not ref['eos']:assert count==case['max_tokens']
  for directory in [refdir,native]:assert {p.name for p in directory.glob('logits-*.f32')}=={f'logits-{i}.f32' for i in range(count)}
  steps=[]
  for i in range(count):
   a=np.fromfile(refdir/f'logits-{i}.f32','<f4').astype('f8');b=np.fromfile(native/f'logits-{i}.f32','<f4').astype('f8')
   assert a.shape==b.shape==(131072,) and np.isfinite(a).all() and np.isfinite(b).all()
   assert int(np.argmax(a))==int(np.argmax(b))==generated[i]
   error=b-a;nrmse=float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(a*a)),1e-12));maximum=float(np.max(np.abs(error)));limit=.0002+.0002*float(np.max(np.abs(a)))
   recorded=result['logits'][i];assert recorded['token']==i and recorded['passed']
   for k,v in [('nrmse',nrmse),('max_abs_error',maximum),('max_abs_limit',limit)]:assert v==recorded[k]
   assert nrmse<=.0002 and maximum<=limit
   steps.append({'token':i,'nrmse':nrmse,'max_abs_error':maximum,'max_abs_limit':limit})
  for stage in ['official','native']:
   process=j[stage];assert process['return_code']==0 and process['status']=='completed';verify(process['log'],process['log_sha256'])
   assert abs(process['wall_seconds']-(process['finished_monotonic_ns']-process['started_monotonic_ns'])/1e9)<1e-8
  files={str(p.relative_to(folder)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(folder.rglob('*')) if p.is_file()}
  summary={'id':cid,'status':'independently_reverified','input_tokens':len(ids),'generated_tokens':count,'eos':ref['eos'],'all_steps_recomputed':count,'max_nrmse':max(s['nrmse'] for s in steps),'max_abs_error':max(s['max_abs_error'] for s in steps),'max_error_to_limit_ratio':max(s['max_abs_error']/s['max_abs_limit'] for s in steps),'files':files,'reference_identity':{k:ref[k] for k in ['official_model_revision','transformers_revision','source_weight_sha256','model_source_sha256','script_sha256']}}
  (out/(cid+'.json')).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');summaries.append({k:v for k,v in summary.items() if k not in ['files','reference_identity']});print(cid,count,flush=True)
 audit={'status':'passed','cases':summaries,'source_directory':str(source.resolve()),'review_script_sha256':sha(__file__),'verified_source_snapshots':len(snapshot['tools']),'model_file_count_rehashed':len(manifest['files']),'official_components_rehashed':len(weights),'full_output_capacity_executed':False,'full_cache_capacity_executed':False,'sampling_same_seed_is_parity_oracle':False}
 (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 shutil.copy2(__file__,out/'review.py')
if __name__=='__main__':main(*sys.argv[1:])
