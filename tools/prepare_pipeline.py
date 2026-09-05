#!/usr/bin/env python3
"""Assemble a verified local model package by linking converted components."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from prepare_block import ROOT,sha256
from rebucket_dit import verify_runtime
from validate_dit_heads import verify as verify_head

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dit',action='append',type=Path,required=True)
    p.add_argument('--text',action='append',type=Path,required=True)
    p.add_argument('--input-head',type=Path,required=True)
    p.add_argument('--output-head',type=Path,required=True)
    p.add_argument('--vae',type=Path,required=True)
    p.add_argument('--embedding-package',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=ROOT/'models/tokenizer')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or len(args.dit)!=36 or len(args.text)!=25:p.error('Use new output, 36 DiT and 25 text blocks')
    revision=json.loads((ROOT/'sources.lock.json').read_text())['official_model']['revision']
    tokenizer=json.loads((args.tokenizer/'manifest.json').read_text())
    if (tokenizer['revision']!=revision or set(tokenizer['files'])!={'tokenizer.json','tokenizer_config.json'}
        or any(sha256(args.tokenizer/name)!=digest for name,digest in tokenizer['files'].items())):
        raise ValueError('Tokenizer source revision or checksum differs')
    pre=verify_head(args.input_head);post=verify_head(args.output_head);vae=verify_head(args.vae)
    if (pre['component']!='input' or post['component']!='output' or vae['component']!='vae'
        or any(pre[k]!=post[k] for k in ('height','width','text_tokens','tokens','weights'))
        or (vae['height'],vae['width'])!=(2*pre['height'],2*pre['width'])):raise ValueError('Component dimensions differ')
    for i,path in enumerate(args.dit):
        m=verify_runtime(path)
        if m['block']!=i or m['tokens']!=pre['tokens']:raise ValueError('DiT block order or shape mismatch')
    for i,path in enumerate(args.text):
        m=json.loads((path/'model.json').read_text())
        if m['block']!=i or m['tokens']!=32 or any(sha256(path/name)!=value for name,value in m['files'].items()):
            raise ValueError('Text block order, shape or checksum mismatch')
    package=json.loads((args.embedding_package/'model.json').read_text())
    for name in ('embedding','rope'):
        item=package[name]
        if sha256(args.embedding_package/item['file'])!=item['sha256']:raise ValueError('Invalid embedding package')
    output=args.output.resolve();output.mkdir(parents=True)
    for path in ('text','dit','vae'):(output/path).mkdir()
    source_hashes={}
    def link(source,target):
        (output/target).symlink_to(source.resolve(),target_is_directory=source.is_dir())
        if source.is_dir() and (source/'model.json').is_file():source_hashes[target]=sha256(source/'model.json')
        elif source.is_file():source_hashes[target]=sha256(source)
    for i,path in enumerate(args.dit):link(path,f'dit/block-{i:02d}')
    for i,path in enumerate(args.text):link(path,f'text/block-{i:02d}')
    link(args.input_head,'dit/input');link(args.output_head,'dit/output')
    link(args.embedding_package/'embeddings.bf16','text/embeddings.bf16')
    link(args.embedding_package/'rope-inv-freq.f32','text/rope-inv-freq.f32')
    link(args.tokenizer,'tokenizer')
    source_hashes['tokenizer/manifest.json']=sha256(args.tokenizer/'manifest.json')
    for name,digest in tokenizer['files'].items():source_hashes['tokenizer/'+name]=digest
    for name in ('head.ncnn.param','head.ncnn.bin'):link(args.vae/name,'vae/'+name)
    table=torch.cat([1./(256**(torch.arange(0,dim,2,dtype=torch.float32)/dim)) for dim in (32,48,48)])
    table.numpy().astype('<f4').tofile(output/'dit/rope-inv-freq.f32')
    bn_path=ROOT/'models/official/vae-bn.safetensors'
    bn_manifest=json.loads(bn_path.with_suffix('.manifest.json').read_text())
    revision=json.loads((ROOT/'sources.lock.json').read_text())['official_model']['revision']
    if bn_manifest['revision']!=revision or bn_manifest['prefix']!='bn.' or sha256(bn_path)!=bn_manifest['sha256']:
        raise ValueError('VAE BN source differs')
    bn=load_file(bn_path)
    for key,name in [('bn.running_mean','bn-mean.f32'),('bn.running_var','bn-variance.f32')]:
        value=bn[key].float()
        if value.shape!=(128,) or not torch.isfinite(value).all() or (key.endswith('var') and (value<0).any()):
            raise ValueError('Invalid BN statistics')
        value.numpy().astype('<f4').tofile(output/'vae'/name)
    cfg={'packed_width':pre['width'],'packed_height':pre['height'],'text_bucket':32,'dit_text_tokens':pre['text_tokens'],
         'text_layers':25,'dit_layers':36}
    (output/'model.cfg').write_text(''.join(f'{k} {v}\n' for k,v in cfg.items()))
    manifest={'schema_version':1,'scope':'Local symlink package; requires source directories to remain present',
        'config':cfg,'source_manifests':source_hashes,'vae_manifest_sha256':sha256(args.vae/'model.json'),
        'bn_source_sha256':bn_manifest['sha256'],'builder_sha256':sha256(__file__),'official_model_revision':revision,
        'files':{name:sha256(output/name) for name in ['model.cfg','dit/rope-inv-freq.f32','vae/bn-mean.f32','vae/bn-variance.f32']}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'package':str(output),'resolution':[pre['width']*16,pre['height']*16],'text_bucket':32}),flush=True)

if __name__=='__main__':main()
