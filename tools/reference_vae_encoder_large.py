#!/usr/bin/env python3
"""Freeze official 512x384 VAE encoder boundaries without tracing or pnnx."""
import argparse,inspect,json
from pathlib import Path
import numpy as np
try:
 from reference_img2img import load_encoder
 from export_vae_encoder import normalize_rgb,tensor
 from package_model import ROOT,sha256
except ImportError:
 from tools.reference_img2img import load_encoder
 from tools.export_vae_encoder import normalize_rgb,tensor
 from tools.package_model import ROOT,sha256

def run_reference(output,w,h,rgb,scope):
 output=Path(output)
 if output.exists():raise ValueError('Use a new output directory')
 output.mkdir(parents=True)
 import torch
 from diffusers import AutoencoderKLFlux2
 from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
 model,manifests,config_path=load_encoder()
 if rgb.shape!=(h,w,3) or rgb.dtype!=np.uint8:raise ValueError('Invalid fixed RGB input')
 (output/'input.rgb').write_bytes(rgb.tobytes());inp=torch.from_numpy(normalize_rgb(rgb))
 mean=model.encode(inp).latent_dist.mode();packed=torch.nn.functional.pixel_unshuffle(mean,2);normalized=model.bn(packed)
 fixture={'component':'vae-encoder','width':w,'height':h,'text_tokens':0,'scope':scope,
  'official_revision':next(iter(manifests.values()))['revision'],'weights':{k:v['sha256'] for k,v in manifests.items()},
  'source_manifests':{k:sha256(ROOT/'models/official'/f'vae-{k}.manifest.json') for k in manifests},
  'official_source_sha256':sha256(inspect.getfile(AutoencoderKLFlux2)),'distribution_source_sha256':sha256(inspect.getfile(DiagonalGaussianDistribution)),
  'diffusers_revision':json.loads((ROOT/'sources.lock.json').read_text())['diffusers']['revision'],'vae_config_sha256':sha256(config_path),
  'posterior':'mode_first_32_channels_no_sampling','packing':'pixel_unshuffle_2','encoder_bn':{'eps':1e-4,'affine':False},'decoder_inverse_bn_eps':1e-5,
  'rgb':{'file':'input.rgb','shape':[h,w,3],'layout':'HWC_RGB','sha256':sha256(output/'input.rgb'),'normalization':'FP32 (v-127.5)*(1/127.5)'},
  'wrapper_bitwise_equal':[True,True,True],'boundaries':['mean','packed','normalized'],'inputs':{'in0':tensor(output/'in0.f32',inp)},
  'expected':{f'out{i}':tensor(output/f'out{i}.f32',v) for i,v in enumerate([mean,packed,normalized])},'gates':{'fp32':{'atol':.0002,'rtol':.0002,'nrmse':.00002}}}
 (output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
 return fixture

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 w,h=512,384;y,x,c=np.indices((h,w,3),dtype=np.int32);rgb=((13*x+29*y+71*c+(x*y)%19)%256).astype(np.uint8)
 fixture=run_reference(a.output,w,h,rgb,'Official 512x384 encoder fixture only; no trace, pnnx, denoising, or acceptance claim')
 print(json.dumps({'output':str(a.output),'boundaries':[v['shape'] for v in fixture['expected'].values()]}))
if __name__=='__main__':main()
