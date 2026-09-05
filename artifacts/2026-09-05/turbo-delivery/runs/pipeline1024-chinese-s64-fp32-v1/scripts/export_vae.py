#!/usr/bin/env python3
"""Export only the pinned AutoencoderKLFlux2 decoder and post-quant convolution."""
import argparse
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from diffusers import AutoencoderKLFlux2
from safetensors.torch import load_file
from export_dit_heads import export_component
from prepare_block import ROOT, sha256

def load_vae():
    cfg=json.loads((ROOT/'models/official/vae-config.json').read_text())
    with torch.device('meta'):
        model=AutoencoderKLFlux2.from_config(cfg)
    manifests={}
    for label,prefix,module in [('decoder','decoder.',model.decoder),('post-quant','post_quant_conv.',model.post_quant_conv)]:
        path=ROOT/f'models/official/vae-{label}.safetensors'
        manifest=json.loads(path.with_suffix('.manifest.json').read_text())
        if (manifest['prefix']!=prefix or sha256(path)!=manifest['sha256']
            or manifest['revision']!=json.loads((ROOT/'sources.lock.json').read_text())['official_model']['revision']):
            raise ValueError('VAE component checksum, prefix or revision differs')
        state={name.removeprefix(prefix):value.float() for name,value in load_file(path).items()}
        module.load_state_dict(state,strict=True,assign=True)
        manifests[label]=manifest
    return model.eval().requires_grad_(False),manifests

class Decode(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.post_quant_conv=model.post_quant_conv
        self.decoder=model.decoder
    def forward(self,x):
        return self.decoder(self.post_quant_conv(x))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--height',type=int,default=8)
    p.add_argument('--width',type=int,default=8)
    p.add_argument('--reference-only',action='store_true',help='Save the official fixture for specialize_vae.py without large pnnx evaluation')
    args=p.parse_args()
    if args.output.exists() or min(args.height,args.width)<1 or max(args.height,args.width)>128:
        p.error('Use new output and latent dimensions in [1,128]')
    if args.height*args.width>4096 and not args.reference_only:
        p.error('Large whole-VAE pnnx evaluation exceeds the 32GB host budget; use --reference-only then specialize_vae.py')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    model,manifests=load_vae()
    x=torch.randn(1,32,args.height,args.width,generator=torch.Generator().manual_seed(20260905))
    expected=model._decode(x,return_dict=False)[0]
    metadata={'component':'vae','height':args.height,'width':args.width,'text_tokens':0,'tokens':0,
        'weights':{k:v['sha256'] for k,v in manifests.items()},
        'official_revision':next(iter(manifests.values()))['revision'],
        'scope':'Official VAE decoder with synthetic unpacked latent; BN/unpack happens before this component',
        'official_source_sha256':sha256(inspect.getfile(AutoencoderKLFlux2)),
        'vae_config_sha256':sha256(ROOT/'models/official/vae-config.json')}
    export_component(Decode(model),(x,),(expected,),args.output.resolve(),metadata,args.reference_only)

if __name__=='__main__':main()
