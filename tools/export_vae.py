#!/usr/bin/env python3
"""Export only the pinned AutoencoderKLFlux2 decoder and post-quant convolution."""
import argparse
import inspect
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from diffusers import AutoencoderKLFlux2
from safetensors.torch import load_file
from export_dit_heads import export_component
from prepare_block import ROOT, sha256
from vae_shape_contract import validate_decoder_shape

def load_vae(official_root=None):
    official_root=Path(official_root) if official_root is not None else ROOT/'models/official'
    cfg=json.loads((official_root/'vae-config.json').read_text())
    with torch.device('meta'):
        model=AutoencoderKLFlux2.from_config(cfg)
    manifests={}
    for label,prefix,module in [('decoder','decoder.',model.decoder),('post-quant','post_quant_conv.',model.post_quant_conv)]:
        path=official_root/f'vae-{label}.safetensors'
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
    p.add_argument('--fixed-1376x768',action='store_true',help='Permit only latent 96x172 reference-only; never whole-VAE pnnx')
    p.add_argument('--official-root',type=Path,default=ROOT/'models/official')
    p.add_argument('--input-f32',type=Path,help='Saved finite FP32 little-endian unpacked latent, exact selected shape')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--runtime-identity',type=Path,help='Frozen runtime allowlist for guarded reference execution')
    p.add_argument('--runtime-report',type=Path,help='New actual worker runtime inventory directory')
    p.add_argument('--runtime-probe',type=Path,help='Import/input-only worker inventory; return before model construction')
    args=p.parse_args()
    if bool(args.runtime_identity)!=bool(args.runtime_report):p.error('Both runtime identity and report are required')
    if args.runtime_identity and not args.reference_only:p.error('Runtime inventory requires reference-only')
    if args.output.exists():p.error('Use a new output directory')
    if not 1<=args.threads<=4:p.error('Use 1..4 CPU threads')
    try:
        validate_decoder_shape(args.height,args.width,fixed=args.fixed_1376x768,reference_only=args.reference_only)
    except ValueError as error:p.error(str(error))
    torch.set_num_threads(args.threads);torch.set_grad_enabled(False)
    official_root=args.official_root.resolve()
    if args.input_f32:
        raw=args.input_f32.read_bytes()
        if len(raw)!=32*args.height*args.width*4:p.error('Saved input has wrong exact FP32 size')
        import numpy as np
        array=np.frombuffer(raw,dtype='<f4').copy().reshape(1,32,args.height,args.width)
        if not np.isfinite(array).all():p.error('Saved input must be finite')
        x=torch.from_numpy(array)
    else:
        x=torch.randn(1,32,args.height,args.width,generator=torch.Generator().manual_seed(20260905))
    if args.runtime_probe:
        if not args.reference_only:p.error('Runtime probe requires reference-only')
        from vae_reference_scope import capture
        capture(args.runtime_probe,import_reference=False)
        return
    collector=None
    if args.runtime_identity:
        from vae_reference_scope import WorkerRuntime
        collector=WorkerRuntime(args.runtime_identity,args.runtime_report)
    success=False
    try:
        if collector:collector.checkpoint('before_model')
        model,manifests=load_vae(official_root)
        if collector:collector.checkpoint('after_model')
        expected=model._decode(x,return_dict=False)[0]
        if collector:collector.checkpoint('after_first_forward')
        metadata={'component':'vae','height':args.height,'width':args.width,'text_tokens':0,'tokens':0,
            'weights':{k:v['sha256'] for k,v in manifests.items()},
            'official_revision':next(iter(manifests.values()))['revision'],
            'saved_input_sha256':hashlib.sha256(raw).hexdigest() if args.input_f32 else None,
            'input_generation':'saved_FP32_bytes' if args.input_f32 else 'torch_seed_20260905',
            'fixed_1376x768_candidate':args.fixed_1376x768,
            'threads':args.threads,'native_acceptance_eligible':False,
            'scope':'Official VAE decoder with synthetic unpacked latent; BN/unpack happens before this component',
            'official_source_sha256':sha256(inspect.getfile(AutoencoderKLFlux2)),
            'vae_config_sha256':sha256(official_root/'vae-config.json')}
        export_component(Decode(model),(x,),(expected,),args.output.resolve(),metadata,args.reference_only)
        if collector:collector.checkpoint('after_second_forward')
        success=True
    finally:
        if collector:collector.finish(success)

if __name__=='__main__':main()
