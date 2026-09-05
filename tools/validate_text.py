#!/usr/bin/env python3
"""Validate converted text block fixtures or saved real-prompt hidden states."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import numpy as np
import torch
from prepare_block import ROOT, sha256
from export_dit_block import metrics, save_tensor
from export_text_block import config, load_block
from transformers import AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
from safetensors import safe_open

def real_reference(models, prompt, output):
    if [json.loads((m/'model.json').read_text())['block'] for m in models] != list(range(25)):
        raise ValueError('Full text path must contain blocks 0..24')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/tokenizer',local_files_only=True)
    ids=tokenizer(prompt,add_special_tokens=True,truncation=True,padding=False)['input_ids']
    if not 1<=len(ids)<=32:raise ValueError('Prompt exceeds reviewed 32-token bucket')
    output.mkdir(parents=True)
    (output/'ids.txt').write_text('\n'.join(map(str,ids))+'\n')
    path=ROOT/'models/official/text-embed.safetensors'
    source=json.loads(path.with_suffix('.manifest.json').read_text())
    if sha256(path)!=source['sha256']:raise ValueError('Embedding source checksum differs')
    with safe_open(path,framework='pt',device='cpu') as file:
        table=file.get_slice('language_model.model.embed_tokens.weight')
        x=torch.stack([table[token].float() for token in ids])[None]
    # Reference runs exactly the valid length, testing that native right padding
    # cannot leak through causal attention into the retained prefix.
    positions=torch.arange(len(ids))[None]
    cos,sin=MistralRotaryEmbedding(config())(x,positions)
    mask=torch.full((len(ids),len(ids)),torch.finfo(torch.float32).min).triu(1)[None,None]
    for index, model in enumerate(models):
        block,source=load_block(index)
        if source['sha256']!=json.loads((model/'model.json').read_text())['weights_sha256']:
            raise ValueError('Converted text block source differs')
        x=block(x,attention_mask=mask,position_ids=positions,position_embeddings=(cos,sin),use_cache=False)
        del block
    fixture={'prompt':prompt,'ids':ids,'tokens':32,'valid_tokens':len(ids),'blocks':25,
             'scope':'Real prompt through official embedding and Mistral blocks 0..24; no final norm',
             'source_sha256':sha256(__file__),'expected':save_tensor(output/'expected.f32',x),
             'gates':{'fp32':{'nrmse':.0002,'atol':.0002,'rtol':.0002},'fp16':{'nrmse':.03,'atol':.03,'rtol':.03},
                      'bf16':{'nrmse':.15,'atol':.2,'rtol':.2}}}
    (output/'fixture.json').write_text(json.dumps(fixture,indent=2,ensure_ascii=False)+'\n')
    return fixture

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',type=Path,required=True)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--prompt')
    p.add_argument('--embedding-package',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'build/ernie-text-runner')
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--bf16',action='store_true')
    args=p.parse_args()
    if args.output.exists() or (args.fixture is None)==(args.prompt is None):p.error('Use new output and exactly one fixture/prompt')
    args.output.mkdir(parents=True)
    runner=args.output/'runner.snapshot';shutil.copy2(args.runner,runner)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    manifests=[]
    for model in args.model:
        m=json.loads((model/'model.json').read_text())
        if any(sha256(model/name)!=digest for name,digest in m['files'].items()):raise ValueError('Model checksum differs')
        manifests.append(sha256(model/'model.json'))
    reference_path=args.fixture or args.output/'reference'
    fixture=json.loads((reference_path/'fixture.json').read_text()) if args.fixture else real_reference(args.model,args.prompt,reference_path)
    ref=reference_path/fixture['expected']['file']
    if sha256(ref)!=fixture['expected']['sha256']:raise ValueError('Reference checksum differs')
    if args.fixture:
        for item in fixture['inputs'].values():
            if sha256(reference_path/item['file'])!=item['sha256']:raise ValueError('Input checksum differs')
    elif args.embedding_package is None:p.error('Real prompt requires embedding package')
    results=[]
    variants=[('cpu','fp32')] if args.cpu_only else [('cpu','fp32'),('vulkan','fp32'),('vulkan','fp16')]
    if args.bf16 and not args.cpu_only:variants.append(('vulkan','bf16'))
    for backend,precision in variants:
        out=args.output/(backend+'-'+precision);out.mkdir()
        cmd=[str(runner.resolve()),'--output',str((out/'output.f32').resolve()),'--tokens',str(fixture['tokens']),
             '--backend',backend,'--precision',precision]
        for m in args.model:cmd+=['--model',str(m.resolve())]
        if args.fixture:cmd+=['--fixture',str(reference_path.resolve())]
        else:
            package=json.loads((args.embedding_package/'model.json').read_text())
            for name in ('embedding','rope'):
                entry=package[name]
                if sha256(args.embedding_package/entry['file'])!=entry['sha256']:raise ValueError('Embedding package checksum differs')
            cmd+=['--ids',str((reference_path/'ids.txt').resolve()),'--embeddings',str((args.embedding_package/'embeddings.bf16').resolve()),
                  '--frequencies',str((args.embedding_package/'rope-inv-freq.f32').resolve())]
        result={'backend':backend,'precision':precision,'passed':False,'command':cmd,'scope':fixture['scope'],
                'runner_sha256':sha256(runner),'validator_sha256':sha256(__file__),'model_manifests':manifests,
                'fixture_sha256':sha256(reference_path/'fixture.json')}
        try:
            with (out/'run.log').open('w') as log:
                run=subprocess.Popen(['/usr/bin/time','-v',*cmd],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:run.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid,signal.SIGKILL);run.wait();raise
            result['return_code']=run.returncode
            if run.returncode:raise RuntimeError('Text runner failed')
            a=np.fromfile(out/'output.f32','<f4');b=np.fromfile(ref,'<f4')
            if a.shape!=b.shape or not np.isfinite(a).all():raise RuntimeError('Invalid text output')
            error=metrics(torch.from_numpy(b),torch.from_numpy(a));gate=fixture['gates'][precision]
            result.update(error)
            result['passed']=bool(error['nrmse']<=gate['nrmse'] and error['max_abs_error']<=gate['atol']+gate['rtol']*error['reference_max_abs'])
            result['sha256']=sha256(out/'output.f32')
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:result['failure']=str(error)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append(result);(args.output/'matrix.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:result.get(k) for k in ('backend','precision','passed','nrmse','failure')}),flush=True)
    return 0 if all(r['passed'] for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
