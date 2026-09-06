#!/usr/bin/env python3
"""Official-module strength-zero continuation for fixed reviewed encoder fixtures."""
import argparse,importlib.metadata,inspect,json
from pathlib import Path
import numpy as np
try:
 from reference_img2img import load_encoder,save_tensor
 from export_vae import load_vae
 from package_model import sha256
except ImportError:
 from tools.reference_img2img import load_encoder,save_tensor
 from tools.export_vae import load_vae
 from tools.package_model import sha256

REVIEWED_FIXTURES = {
 (512,384): 'ee5ce5db3cb9912ff5e824bf062d1376bcf9b2e91e942d06924ecd187a93e1a9',
 (1024,1024): '88f2e8b7ad63a47fd993b069282dcb8bca9042cf56a470efa84237b711d9f7d9',
}

def reviewed_fixture(directory):
 directory=Path(directory);f=json.loads((directory/'fixture.json').read_text());shape=(f.get('width'),f.get('height'))
 if REVIEWED_FIXTURES.get(shape)!=sha256(directory/'fixture.json'):raise ValueError('Unreviewed official encoder fixture')
 if f.get('posterior')!='mode_first_32_channels_no_sampling' or f.get('encoder_bn')!={'eps':1e-4,'affine':False} or f.get('decoder_inverse_bn_eps')!=1e-5:raise ValueError('Wrong official encoder fixture contract')
 entry=f['expected']['out2'];path=directory/entry['file']
 if sha256(path)!=entry['sha256'] or entry['shape']!=[1,128,shape[1]//16,shape[0]//16]:raise ValueError('Wrong normalized encoder boundary')
 return f,entry,path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--encoder-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use a new output directory')
 f,entry,path=reviewed_fixture(a.encoder_reference)
 a.output.mkdir();import torch;from PIL import Image
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
 encoder,_,_=load_encoder();normalized=torch.from_numpy(np.fromfile(path,'<f4').copy().reshape(entry['shape']))
 unpacked=torch.nn.functional.pixel_shuffle(normalized*torch.sqrt(encoder.bn.running_var.reshape(1,128,1,1)+1e-5)+encoder.bn.running_mean.reshape(1,128,1,1),2)
 decoder,manifests=load_vae();decoded=decoder._decode(unpacked,return_dict=False)[0]
 pixels=((decoded[0]/2+.5).clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8');Image.fromarray(pixels).save(a.output/'reference.png')
 from diffusers import AutoencoderKLFlux2
 dist=importlib.metadata.distribution('diffusers');direct_url=json.loads(dist.read_text('direct_url.json'))
 result={'schema_version':1,'scope':f"official {f['width']}x{f['height']} strength-zero encoder boundary continuation; no text or DiT",'encoder_fixture_sha256':sha256(a.encoder_reference/'fixture.json'),'strength':0.,'decoder_inverse_bn_eps':1e-5,
  'decoder_weights':{k:v['sha256'] for k,v in manifests.items()},
  'decoder_source_manifests':{k:sha256(Path('models/official')/f'vae-{k}.manifest.json') for k in manifests},
  'vae_config_sha256':sha256(Path('models/official/vae-config.json')),
  'official_source_sha256':sha256(inspect.getfile(AutoencoderKLFlux2)),
  'diffusers_installed':{'version':dist.version,'direct_url':direct_url,'class_file':inspect.getfile(AutoencoderKLFlux2)},
  'unpacked':save_tensor(a.output/'unpacked.f32',unpacked.numpy()),'decoded':save_tensor(a.output/'decoded.f32',decoded.numpy()),'png_sha256':sha256(a.output/'reference.png')}
 (a.output/'reference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(a.output),'png_sha256':result['png_sha256']}))
if __name__=='__main__':main()
