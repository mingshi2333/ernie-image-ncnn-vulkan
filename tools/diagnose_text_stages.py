#!/usr/bin/env python3
"""Bounded CPU text-stage diagnostics on a saved real prompt; never quality closure.

Snapshots this self-contained worker and runner, pins actual inputs, source and
weights. Official/native free-running blocks are compared separately from a
selected native block teacher-forced with an official input and constants.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

# Exact producers in the reviewed 2048-token text graph. CHW means head/token/
# head-dimension for RoPE/SDPA, and [1,token,width] for unprojected linear outputs.
BLOBS={'norm':'6','q':'10','k':'13','v':'16','rope_q':'45','rope_k':'58',
       'attention_heads':'59','attention':'61','attention_projection':'62',
       'attention_residual':'63','post_norm':'66','gate':'69','silu':'70',
       'up':'71','gated':'72','down':'73','output':'out0'}
TYPES={'6':'RMSNorm','10':'Gemm','13':'Gemm','16':'Gemm','45':'Concat','58':'Concat',
       '59':'SDPA','61':'Reshape','62':'Gemm','63':'BinaryOp','66':'RMSNorm',
       '69':'Gemm','70':'Swish','71':'Gemm','72':'BinaryOp','73':'Gemm','out0':'BinaryOp'}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def compare(reference,actual):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(actual,dtype=np.float64)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Stage tensors must have same nonempty finite shape')
    delta=b-a
    return dict(nrmse=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(a.ravel()),1e-30)),
                max_abs_error=float(np.max(np.abs(delta))),
                max_index=list(map(int,np.unravel_index(abs(delta).argmax(),delta.shape))),
                bitwise_equal=a.astype('<f4').tobytes()==b.astype('<f4').tobytes())

def summarize_layers(rows):
    if [r['layer'] for r in rows]!=list(range(len(rows))):raise ValueError('Layer sequence must be contiguous from 0')
    # Report observations, not a data-selected new numerical gate or cause.
    increases=[dict(layer=r['layer'],nrmse_increase=r['free_running']['nrmse']-(rows[i-1]['free_running']['nrmse'] if i else 0)) for i,r in enumerate(rows)]
    return dict(layers=rows,largest_nrmse_increase=max(increases,key=lambda x:x['nrmse_increase']) if increases else None,
                scope='free-running observations; teacher-forced metrics are separate and cannot close trajectory quality')

def verify_blob_map(param):
    producers={}
    for line in Path(param).read_text().splitlines()[2:]:
        t=line.split();n,m=map(int,t[2:4]);producers.update({x:t[0] for x in t[4+n:4+n+m]})
    for blob,kind in TYPES.items():
        if producers.get(blob)!=kind:raise ValueError('Unexpected graph producer for '+blob)
    return dict(BLOBS)

def save(path,value):
    a=np.ascontiguousarray(value,dtype='<f4')
    if not np.isfinite(a).all():raise ValueError('Nonfinite output')
    with Path(path).open('xb') as f:a.tofile(f)
    return dict(shape=list(a.shape),sha256=sha(path))

def execute(command,folder,timeout=1800):
    folder.mkdir(parents=True,exist_ok=False)
    record=dict(command=command,status='running',started_monotonic_ns=time.monotonic_ns())
    (folder/'run.json').write_text(json.dumps(record,indent=2))
    with (folder/'log.txt').open('wb') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                              env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'})
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=proc.returncode;record['status']='timeout'
    record.update(return_code=code,finished_monotonic_ns=time.monotonic_ns())
    if record['status']!='timeout':record['status']='completed' if code==0 else 'failed'
    (folder/'run.json').write_text(json.dumps(record,indent=2))
    if code:raise RuntimeError('Text subprocess failed: '+str(folder))
    return record

def official_worker(args):
    import inspect
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Mistral3Config
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer,MistralRotaryEmbedding,apply_rotary_pos_emb
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    root=args.project;out=args.output;out.mkdir(parents=True,exist_ok=False)
    config_path=root/'models/official/text_encoder-config.json'
    cfg=Mistral3Config.from_dict(json.loads(config_path.read_text())).text_config
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    if (cfg.hidden_size,cfg.head_dim,cfg.num_attention_heads,cfg.num_key_value_heads)!=(3072,128,32,8):raise ValueError('Wrong text configuration')
    ids=list(map(int,args.ids.read_text().split()));count=len(ids)
    lock=json.loads((root/'sources.lock.json').read_text());weight_hashes={}
    def weight_file(label):
        path=root/'models/official'/label;manifest=json.loads(path.with_suffix('.manifest.json').read_text())
        if manifest['revision']!=lock['official_model']['revision'] or sha(path)!=manifest['sha256']:raise ValueError('Weight identity changed')
        weight_hashes[label]=manifest['sha256'];return path
    path=weight_file('text-embed.safetensors')
    with safe_open(path,framework='pt',device='cpu') as f:
        table=f.get_slice('language_model.model.embed_tokens.weight');x=torch.stack([table[i].float() for i in ids])[None]
    rotary=MistralRotaryEmbedding(cfg);positions=torch.arange(count)[None];cos,sin=rotary(x,positions)
    mask=torch.full((count,count),torch.finfo(torch.float32).min).triu(1)[None,None]
    tensors={'embedding':save(out/'embedding.f32',x.numpy()),'cos':save(out/'cos.f32',cos.numpy()),'sin':save(out/'sin.f32',sin.numpy())}
    padded=torch.zeros((1,args.tokens,3072));padded[:,:count]=x
    fullcos,fullsin=rotary(padded,torch.arange(args.tokens)[None])
    teacher=out/'teacher';teacher.mkdir()
    save(teacher/'in1.f32',fullcos.numpy());save(teacher/'in2.f32',fullsin.numpy())
    save(teacher/'in3.f32',torch.full((args.tokens,args.tokens),torch.finfo(torch.float32).min).triu(1).numpy())
    stages={}
    for index in range(args.layers):
        path=weight_file(f'text-block-{index:02d}.safetensors')
        with torch.device('meta'):block=MistralDecoderLayer(cfg,index)
        state={k.removeprefix(f'language_model.model.layers.{index}.'):v.float() for k,v in load_file(path).items()}
        block.load_state_dict(state,strict=True,assign=True);del state
        block.eval().requires_grad_(False)
        hooks=[];captured={}
        if index==args.stage_layer:
            padded.zero_();padded[:,:count]=x;save(teacher/'in0.f32',padded.numpy())
            modules={'norm':block.input_layernorm,'q':block.self_attn.q_proj,'k':block.self_attn.k_proj,'v':block.self_attn.v_proj,
                     'attention_projection':block.self_attn.o_proj,'post_norm':block.post_attention_layernorm,
                     'gate':block.mlp.gate_proj,'up':block.mlp.up_proj,'down':block.mlp.down_proj}
            for name,module in modules.items():
                hooks.append(module.register_forward_hook(lambda mod,inp,result,key=name:captured.__setitem__(key,result.detach().clone())))
            for name,module in [('attention',block.self_attn.o_proj),('attention_residual',block.post_attention_layernorm),('gated',block.mlp.down_proj)]:
                hooks.append(module.register_forward_pre_hook(lambda mod,inp,key=name:captured.__setitem__(key,inp[0].detach().clone())))
        x=block(x,attention_mask=mask,position_ids=positions,position_embeddings=(cos,sin),use_cache=False)
        tensors[f'layer-{index}']=save(out/f'layer-{index}.f32',x.numpy())
        if index==args.stage_layer:
            captured['output']=x
            captured['silu']=torch.nn.functional.silu(captured['gate'])
            q=captured['q'].reshape(1,count,32,128).transpose(1,2);k=captured['k'].reshape(1,count,8,128).transpose(1,2)
            captured['rope_q'],captured['rope_k']=apply_rotary_pos_emb(q,k,cos,sin)
            captured['attention_heads']=captured['attention'].reshape(1,count,32,128).transpose(1,2)
            for name,value in captured.items():
                value=value.squeeze(0) if value.ndim==4 else value
                stages[name]=save(out/(name+'.f32'),value.numpy())
        for hook in hooks:hook.remove()
        del block,captured
    report=dict(status='completed',scope='official free-running valid-prefix only',tokens=count,bucket=args.tokens,
                tensors=tensors,stages=stages,weights=weight_hashes,
                source_sha256=sha(__file__),official_source_sha256=sha(inspect.getfile(MistralDecoderLayer)),
                torch_version=torch.__version__,config_sha256=sha(config_path),ids_sha256=sha(args.ids),
                attention_scaling=rotary.attention_scaling)
    (out/'result.json').write_text(json.dumps(report,indent=2))

def load(path,shape):
    value=np.fromfile(path,dtype='<f4')
    if value.size!=int(np.prod(shape)):raise ValueError('Tensor byte size mismatch: '+str(path))
    return value.reshape(shape)

def orchestrate(args):
    root=args.project;out=args.output;out.mkdir(parents=True,exist_ok=False)
    snapshots=out/'snapshots';snapshots.mkdir();script=snapshots/'diagnose_text_stages.py';shutil.copy2(__file__,script)
    runner=snapshots/'ernie-text-runner';shutil.copy2(args.runner,runner)
    ids=out/'ids.txt';shutil.copy2(args.ids,ids);count=len(ids.read_text().split())
    if not 0<count<=args.tokens<=2048 or not 1<=args.layers<=25:raise ValueError('Invalid dimensions')
    package=args.package
    # Verify native model files against the sealed package manifest without importing live tools.
    manifest=json.loads((package/'manifest.json').read_text())
    required=['text/embeddings.bf16','text/rope-inv-freq.f32']+[f'text/block-{i:02d}/text.ncnn.{suffix}' for i in range(args.layers) for suffix in ('param','bin')]
    hashes={}
    for name in required:
        path=package/name
        if sha(path)!=manifest['files'][name]:raise ValueError('Native package tensor checksum mismatch: '+name)
        hashes[name]=manifest['files'][name]
    report=dict(status='running',scope='text-stage diagnostic only; not full image quality',runner_sha256=sha(runner),
                script_sha256=sha(script),ids_sha256=sha(ids),package_manifest_sha256=sha(package/'manifest.json'),native_files=hashes,
                model_revision=manifest['official_model_revision'],processes={})
    (out/'result.json').write_text(json.dumps(report,indent=2))
    try:
        command=[sys.executable,str(script),'--worker','--project',str(root),'--ids',str(ids),'--output',str(out/'official'),
                 '--tokens',str(args.tokens),'--layers',str(args.layers),'--stage-layer',str(args.stage_layer)]
        report['processes']['official']=execute(command,out/'official-process')
        native=[str(runner),'--output',str(out/'native-output.f32'),'--ids',str(ids),'--tokens',str(args.tokens),
                '--embeddings',str(package/'text/embeddings.bf16'),'--frequencies',str(package/'text/rope-inv-freq.f32'),
                '--backend','cpu','--trace-dir',str(out/'native')]
        for i in range(args.layers):native+=['--model',str(package/f'text/block-{i:02d}')]
        report['processes']['native']=execute(native,out/'native-process')
        mapping=verify_blob_map(package/f'text/block-{args.stage_layer:02d}/text.ncnn.param')
        teacher=[str(runner),'--model',str(package/f'text/block-{args.stage_layer:02d}'),'--fixture',str(out/'official/teacher'),
                 '--tokens',str(args.tokens),'--valid-tokens',str(count),'--output',str(out/'teacher-output.f32'),
                 '--backend','cpu','--trace-dir',str(out/'teacher')]
        for blob in mapping.values():teacher+=['--trace-blob',blob]
        report['processes']['teacher']=execute(teacher,out/'teacher-process')
        official=json.loads((out/'official/result.json').read_text());shape=(1,count,3072)
        rows=[]
        for i in range(args.layers):rows.append(dict(layer=i,free_running=compare(load(out/f'official/layer-{i}.f32',shape),load(out/f'native/layer-{i}.f32',shape))))
        stages={}
        for name,blob in mapping.items():
            meta=json.loads((out/f'teacher/blob-{blob}.json').read_text());ref=load(out/f'official/{name}.f32',official['stages'][name]['shape'])
            actual=load(out/f'teacher/blob-{blob}.f32',meta['shape'])
            if actual.shape!=ref.shape:raise ValueError('Explicit stage layout mismatch: '+name)
            stages[name]=dict(blob=blob,teacher_forced=compare(ref,actual),shape=list(ref.shape))
        report.update(status='diagnostic_completed',free_running=summarize_layers(rows),teacher_forced=dict(layer=args.stage_layer,stages=stages),
                      embedding=compare(load(out/'official/embedding.f32',shape),load(out/'native/embedding.f32',shape)),
                      constants={name:compare(load(out/f'official/{name}.f32',(1,count,128)),load(out/f'native/constant-{i}.f32',(1,count,128))) for i,name in enumerate(('cos','sin'))},
                      official_provenance=official,blob_map=mapping)
        if args.historical_text:
            historical=load(args.historical_text,shape);native_final=load(out/f'native/layer-{args.layers-1}.f32',shape)
            report['historical_native_reproduction']=dict(historical_sha256=sha(args.historical_text),actual_sha256=sha(out/f'native/layer-{args.layers-1}.f32'),**compare(historical,native_final))
        report['files']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='result.json'}
    except Exception as exc:
        report.update(status='failed',failure=str(exc));raise
    finally:(out/'result.json').write_text(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ids',type=Path,required=True)
    p.add_argument('--package',type=Path);p.add_argument('--runner',type=Path);p.add_argument('--tokens',type=int,default=2048);p.add_argument('--layers',type=int,default=25);p.add_argument('--stage-layer',type=int,default=0);p.add_argument('--historical-text',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    for name in ('project','output','ids','package','runner','historical_text'):
        if getattr(a,name) is not None:setattr(a,name,getattr(a,name).resolve())
    if a.output.exists() or not 0<=a.stage_layer<a.layers<=25:p.error('Use new output and stage-layer in [0,layers)')
    if a.worker:official_worker(a)
    else:
        if not a.runner or not a.package:p.error('Require runner/package')
        orchestrate(a)
if __name__=='__main__':main()
