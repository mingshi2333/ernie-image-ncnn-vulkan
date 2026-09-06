#!/usr/bin/env python3
"""Small CPU encoder export, with separate reference/trace and pnnx processes.
Run via validate_img2img_encoder.py to enforce the process-group resource budget.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import numpy as np
try:
    from package_model import ROOT,sha256
    from audit_port_weights import official_inventory,OFFICIAL_REVISION,OFFICIAL_REPOSITORY
except ImportError:
    from tools.package_model import ROOT,sha256
    from tools.audit_port_weights import official_inventory,OFFICIAL_REVISION,OFFICIAL_REPOSITORY


def dimensions(width,height):
    if any(type(x) is not int or x<16 or x>64 or x%16 for x in (width,height)):
        raise ValueError('Only small multiples of16 in [16,64] are allowed for whole encoder export')


def rgb_fixture(width,height):
    dimensions(width,height)
    y,x,c=np.indices((height,width,3),dtype=np.int32)
    return ((13*x+29*y+71*c+(x*y)%19)%256).astype(np.uint8)


def normalize_rgb(rgb):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB uint8 HWC')
    # Match the requested FP32 operation order, not division or a fused alternative.
    return ((rgb.astype(np.float32)-np.float32(127.5))*np.float32(1/127.5)).transpose(2,0,1)[None].copy()


def tensor(path,value):
    a=value.detach().cpu().contiguous().numpy().astype('<f4',copy=False)
    if not np.isfinite(a).all():raise ValueError('Nonfinite official output')
    path.write_bytes(a.tobytes());return dict(file=path.name,shape=list(a.shape),dtype='F32',layout='NCHW',sha256=sha256(path))


def reference(output,width,height):
    dimensions(width,height)
    if output.exists():raise ValueError('Use a new reference directory')
    output.mkdir(parents=True)
    import inspect
    import torch
    from torch import nn
    from torch.nn import functional as F
    from diffusers import AutoencoderKLFlux2
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from safetensors.torch import load_file
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    official=ROOT/'models/official';config_path=official/'vae-config.json'
    source=json.loads((official/'vae-config.source.json').read_text());config=json.loads(config_path.read_text())
    if source.get('revision')!=OFFICIAL_REVISION or source.get('sha256')!=sha256(config_path) or source.get('url')!=f'{OFFICIAL_REPOSITORY}/resolve/{OFFICIAL_REVISION}/vae/config.json':raise ValueError('Invalid VAE config provenance')
    if config.get('latent_channels')!=32 or config.get('patch_size')!=[2,2] or config.get('batch_norm_eps')!=1e-4:raise ValueError('Unreviewed VAE configuration')
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    dist=importlib.metadata.distribution('diffusers');origin=json.loads(dist.read_text('direct_url.json'))
    if origin['url']!='https://github.com/huggingface/diffusers/archive/'+lock['diffusers']['revision']+'.zip':raise ValueError('Unpinned diffusers installation')
    selected=['vae-encoder.safetensors','vae-quant.safetensors','vae-bn.safetensors']
    official_inventory(official,selected) # strict provenance, SHA, full safetensors ranges
    with torch.device('meta'):model=AutoencoderKLFlux2.from_config(config)
    manifests={}
    for label,prefix,module in [('encoder','encoder.',model.encoder),('quant','quant_conv.',model.quant_conv),('bn','bn.',model.bn)]:
        p=official/f'vae-{label}.safetensors';m=json.loads(p.with_suffix('.manifest.json').read_text())
        if m['prefix']!=prefix:raise ValueError('Wrong component prefix')
        state=load_file(p)
        if any(not k.startswith(prefix) for k in state):raise ValueError('Foreign component tensor')
        state={k[len(prefix):]:v.float() if v.is_floating_point() else v for k,v in state.items()}
        module.load_state_dict(state,strict=True,assign=True);manifests[label]=m
    model.eval().requires_grad_(False)
    if model.bn.affine or model.bn.eps!=1e-4:raise ValueError('Unexpected BN contract')
    class Encode(nn.Module):
        def __init__(self,m):
            super().__init__();self.encoder=m.encoder;self.quant_conv=m.quant_conv;self.bn=m.bn
        def forward(self,x):
            mean=self.quant_conv(self.encoder(x))[:,:32]
            packed=F.pixel_unshuffle(mean,2)
            return mean,packed,self.bn(packed)
    rgb=rgb_fixture(width,height);(output/'input.rgb').write_bytes(rgb.tobytes())
    x=torch.from_numpy(normalize_rgb(rgb));mean=model.encode(x).latent_dist.mode();packed=F.pixel_unshuffle(mean,2);normalized=model.bn(packed)
    expected=(mean,packed,normalized);wrapper=Encode(model).eval();actual=wrapper(x)
    bitwise=[np.array_equal(a.detach().numpy().view(np.uint32),b.detach().numpy().view(np.uint32)) for a,b in zip(actual,expected)]
    if not all(bitwise):raise ValueError('Wrapper differs bitwise from official encode().latent_dist.mode() pipeline')
    fixture=dict(component='vae-encoder',width=width,height=height,text_tokens=0,
        scope='Small deterministic RGB official mode/packed/normalized boundaries; no img2img denoising',
        official_revision=OFFICIAL_REVISION,weights={k:v['sha256'] for k,v in manifests.items()},
        source_manifests={k:sha256(official/f'vae-{k}.manifest.json') for k in manifests},
        official_source_sha256=sha256(inspect.getfile(AutoencoderKLFlux2)),distribution_source_sha256=sha256(inspect.getfile(DiagonalGaussianDistribution)),
        diffusers_revision=lock['diffusers']['revision'],vae_config_sha256=sha256(config_path),
        posterior='mode_first_32_channels_no_sampling',packing='pixel_unshuffle_2',encoder_bn=dict(eps=1e-4,affine=False),decoder_inverse_bn_eps=1e-5,
        rgb=dict(file='input.rgb',shape=[height,width,3],layout='HWC_RGB',sha256=sha256(output/'input.rgb'),normalization='FP32 (v-127.5)*(1/127.5)'),
        wrapper_bitwise_equal=bitwise,boundaries=['mean','packed','normalized'],
        inputs={'in0':tensor(output/'in0.f32',x)},expected={f'out{i}':tensor(output/f'out{i}.f32',v) for i,v in enumerate(expected)},
        gates={'fp32':{'atol':0.0002,'rtol':0.0002,'nrmse':0.00002}})
    (output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    traced=torch.jit.trace(wrapper,(x,),check_trace=False);traced.save(str(output/'head.pt'))
    (output/'trace.json').write_text(json.dumps({'head_pt_sha256':sha256(output/'head.pt'),'exporter_sha256':sha256(__file__),'torch':torch.__version__,'threads':2},indent=2)+'\n')
    print(json.dumps({'stage':'reference_trace','wrapper_bitwise_equal':bitwise,'boundaries':[list(t.shape) for t in expected]}),flush=True)


def convert(output):
    fixture=json.loads((output/'fixture.json').read_text());dimensions(fixture['width'],fixture['height'])
    trace=json.loads((output/'trace.json').read_text())
    if sha256(output/'head.pt')!=trace['head_pt_sha256']:raise ValueError('Trace changed before conversion')
    if (output/'conversion.json').exists():raise ValueError('Do not overwrite prior conversion')
    # Resolve binary without importing pnnx's torch-dependent Python wrappers.
    dist=importlib.metadata.distribution('pnnx');binary=Path(dist.locate_file('pnnx/pnnx'))
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    if sha256(binary)!=lock['pnnx']['binary_sha256']:raise ValueError('Converter binary checksum differs')
    command=[str(binary),'head.pt',f'inputshape=[1,3,{fixture["height"]},{fixture["width"]}]','fp16=0','device=cpu']
    with (output/'conversion.log').open('w') as log:r=subprocess.run(command,cwd=output,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    diagnostics=[s for s in (output/'conversion.log').read_text().splitlines() if 'not supported' in s or 'unsupported' in s.lower()]
    param=output/'head.ncnn.param';lines=param.read_text().splitlines()[2:] if param.exists() else []
    result=dict(command=command,pnnx_sha256=sha256(binary),return_code=r.returncode,unsupported_diagnostics=diagnostics,operators=sorted({x.split()[0] for x in lines if x.strip()}),unconverted=[x for x in lines if x.startswith(('pnnx.','aten::','Tensor.'))])
    (output/'conversion.json').write_text(json.dumps(result,indent=2)+'\n')
    if r.returncode or diagnostics or result['unconverted'] or not param.exists():raise ValueError('Encoder conversion failed; retained conversion.log')
    names=['head.ncnn.param','head.ncnn.bin','fixture.json','conversion.json','trace.json','input.rgb','in0.f32','out0.f32','out1.f32','out2.f32']
    manifest=dict(schema_version=1,component='vae-encoder',source_sha256=sha256(__file__),weights=fixture['weights'],ncnn_revision=lock['ncnn']['revision'],files={n:sha256(output/n) for n in names})
    (output/'model.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=32);p.add_argument('--height',type=int,default=32);p.add_argument('--stage',choices=['reference','convert'],required=True);a=p.parse_args()
    if a.stage=='reference':reference(a.output.resolve(),a.width,a.height)
    else:convert(a.output.resolve())
if __name__=='__main__':main()
