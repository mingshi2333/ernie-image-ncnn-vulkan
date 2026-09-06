#!/usr/bin/env python3
"""Freeze official 512x384 VAE encoder boundaries without tracing or pnnx."""
import argparse,inspect,json
from pathlib import Path
import numpy as np
from reference_img2img import load_encoder
from export_vae_encoder import normalize_rgb,tensor
from package_model import ROOT,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use a new output directory')
 a.output.mkdir(parents=True)
 import torch
 from diffusers import AutoencoderKLFlux2
 from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
 model,manifests,config_path=load_encoder();w,h=512,384
 y,x,c=np.indices((h,w,3),dtype=np.int32);rgb=((13*x+29*y+71*c+(x*y)%19)%256).astype(np.uint8)
 (a.output/'input.rgb').write_bytes(rgb.tobytes());inp=torch.from_numpy(normalize_rgb(rgb))
 mean=model.encode(inp).latent_dist.mode();packed=torch.nn.functional.pixel_unshuffle(mean,2);normalized=model.bn(packed)
 fixture={'component':'vae-encoder','width':w,'height':h,'text_tokens':0,'scope':'Official 512x384 encoder fixture only; no trace, pnnx, denoising, or acceptance claim',
  'official_revision':next(iter(manifests.values()))['revision'],'weights':{k:v['sha256'] for k,v in manifests.items()},
  'source_manifests':{k:sha256(ROOT/'models/official'/f'vae-{k}.manifest.json') for k in manifests},
  'official_source_sha256':sha256(inspect.getfile(AutoencoderKLFlux2)),'distribution_source_sha256':sha256(inspect.getfile(DiagonalGaussianDistribution)),
  'diffusers_revision':json.loads((ROOT/'sources.lock.json').read_text())['diffusers']['revision'],'vae_config_sha256':sha256(config_path),
  'posterior':'mode_first_32_channels_no_sampling','packing':'pixel_unshuffle_2','encoder_bn':{'eps':1e-4,'affine':False},'decoder_inverse_bn_eps':1e-5,
  'rgb':{'file':'input.rgb','shape':[h,w,3],'layout':'HWC_RGB','sha256':sha256(a.output/'input.rgb'),'normalization':'FP32 (v-127.5)*(1/127.5)'},
  'wrapper_bitwise_equal':[True,True,True],'boundaries':['mean','packed','normalized'],'inputs':{'in0':tensor(a.output/'in0.f32',inp)},
  'expected':{f'out{i}':tensor(a.output/f'out{i}.f32',v) for i,v in enumerate([mean,packed,normalized])},'gates':{'fp32':{'atol':.0002,'rtol':.0002,'nrmse':.00002}}}
 (a.output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n');print(json.dumps({'output':str(a.output),'boundaries':[list(x.shape) for x in [mean,packed,normalized]]}))
if __name__=='__main__':main()
