#!/usr/bin/env python3
"""Fetch the pinned text and VAE components needed by the native PE-off pipeline."""
import argparse
import json
from pathlib import Path
import urllib.request
from fetch_component import ROOT,fetch_component,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-only',action='store_true')
    p.add_argument('--vae-only',action='store_true')
    args=p.parse_args()
    if args.text_only and args.vae_only:p.error('Choose at most one component family')
    source=json.loads((ROOT/'sources.lock.json').read_text())['official_model']
    directory=ROOT/'models/official';directory.mkdir(parents=True,exist_ok=True)
    for folder in (['text_encoder'] if args.text_only else ['vae'] if args.vae_only else ['text_encoder','vae']):
        url=f"{source['url']}/resolve/{source['revision']}/{folder}/config.json"
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(1024*1024+1)
        if len(data)>1024*1024:raise ValueError('Config unexpectedly large')
        json.loads(data)
        path=directory/(folder+'-config.json')
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing config differs from pinned source')
        if not path.exists():path.write_bytes(data)
        metadata={'url':url,'revision':source['revision'],'sha256':sha256(path)}
        path.with_suffix('.source.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if not args.vae_only:
        fetch_component('language_model.model.embed_tokens.',directory/'text-embed.safetensors',
                        subfolder='text_encoder',single_file='model.safetensors')
        for i in range(25):
            fetch_component(f'language_model.model.layers.{i}.',directory/f'text-block-{i:02d}.safetensors',
                            subfolder='text_encoder',single_file='model.safetensors')
    if not args.text_only:
        for label,prefix in [('decoder','decoder.'),('post-quant','post_quant_conv.'),('bn','bn.')]:
            fetch_component(prefix,directory/f'vae-{label}.safetensors',subfolder='vae',single_file='diffusion_pytorch_model.safetensors')

if __name__=='__main__':main()
