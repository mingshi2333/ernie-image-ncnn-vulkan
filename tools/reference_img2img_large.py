#!/usr/bin/env python3
"""Official-module 512x384 strength-zero decoder continuation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from reference_img2img import load_encoder,save_tensor
from export_vae import load_vae
from package_model import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--encoder-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use a new output directory')
 f=json.loads((a.encoder_reference/'fixture.json').read_text())
 if (f.get('width'),f.get('height'))!=(512,384) or f.get('posterior')!='mode_first_32_channels_no_sampling' or f.get('encoder_bn')!={'eps':1e-4,'affine':False} or f.get('decoder_inverse_bn_eps')!=1e-5:raise ValueError('Wrong official encoder fixture contract')
 entry=f['expected']['out2'];path=a.encoder_reference/entry['file']
 if sha256(path)!=entry['sha256'] or entry['shape']!=[1,128,24,32]:raise ValueError('Wrong normalized encoder boundary')
 a.output.mkdir();import torch;from PIL import Image
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
 encoder,_,_=load_encoder();normalized=torch.from_numpy(np.fromfile(path,'<f4').copy().reshape(entry['shape']))
 unpacked=torch.nn.functional.pixel_shuffle(normalized*torch.sqrt(encoder.bn.running_var.reshape(1,128,1,1)+1e-5)+encoder.bn.running_mean.reshape(1,128,1,1),2)
 decoder,manifests=load_vae();decoded=decoder._decode(unpacked,return_dict=False)[0]
 pixels=((decoded[0]/2+.5).clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8');Image.fromarray(pixels).save(a.output/'reference.png')
 result={'schema_version':1,'scope':'official 512x384 strength-zero encoder boundary continuation; no text or DiT','encoder_fixture_sha256':sha256(a.encoder_reference/'fixture.json'),'strength':0.,'decoder_inverse_bn_eps':1e-5,
  'decoder_weights':{k:v['sha256'] for k,v in manifests.items()},'unpacked':save_tensor(a.output/'unpacked.f32',unpacked.numpy()),'decoded':save_tensor(a.output/'decoded.f32',decoded.numpy()),'png_sha256':sha256(a.output/'reference.png')}
 (a.output/'reference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(a.output),'png_sha256':result['png_sha256']}))
if __name__=='__main__':main()
