#!/usr/bin/env python3
"""Specialize two audited spatial reshapes and verify against an independent VAE fixture.

The pinned VAE consists of spatial convolutions, GroupNorm, nearest 2x upsampling,
and one attention block. Only its flatten/unflatten pair has absolute H/W values.
This is a reviewed graph specialization, not a successful large pnnx export.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from prepare_block import ROOT, sha256
from validate_dit_heads import verify

GRAPH_SHA256 = '6ecb591c473c56f3d9e923c845be2125e6e42b33ee2cd8d4d1783db68e32c4c5'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Sealed independently executed official VAE fixture')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    base=verify(args.template)
    ref=json.loads((args.reference/'fixture.json').read_text())
    if sha256(args.template/'head.ncnn.param')!=GRAPH_SHA256:
        raise ValueError('VAE graph has not been reviewed')
    if (base['component']!='vae' or ref['component']!='vae'
        or any(base[k]!=ref[k] for k in ('weights','official_revision','vae_config_sha256','official_source_sha256'))):
        raise ValueError('Reference and template VAE sources differ')
    h,w=ref['height'],ref['width']
    if not 1<=h<=128 or not 1<=w<=128 or ref['inputs']['in0']['shape']!=[1,32,h,w] or ref['expected']['out0']['shape']!=[1,3,8*h,8*w]:
        raise ValueError('Unsupported fixture shape')
    for entry in [*ref['inputs'].values(),*ref['expected'].values()]:
        path=args.reference/entry['file']
        if (Path(entry['file']).name!=entry['file'] or sha256(path)!=entry['sha256']
            or path.stat().st_size!=int(np.prod(entry['shape']))*4):
            raise ValueError('Reference tensor checksum or shape differs')
    lines=(args.template/'head.ncnn.param').read_text().splitlines()
    changed=[]
    for i,line in enumerate(lines):
        parts=line.split()
        if parts and parts[0]=='Reshape':
            expected={'reshape_99':['0=64','1=512'],'reshape_100':['0=8','1=8','2=512']}
            if parts[1] not in expected or parts[6:]!=expected[parts[1]]:
                raise ValueError('Unreviewed spatial reshape')
            params=[f'0={h*w}','1=512'] if parts[1]=='reshape_99' else [f'0={w}',f'1={h}','2=512']
            lines[i]=' '.join(parts[:6]+params)
            changed.append({'before':line,'after':lines[i]})
    if len(changed)!=2:raise ValueError('Expected precisely two spatial reshapes')
    out=args.output.resolve();out.mkdir(parents=True)
    (out/'head.ncnn.param').write_text('\n'.join(lines)+'\n')
    (out/'head.ncnn.bin').symlink_to((args.template/'head.ncnn.bin').resolve())
    shutil.copy2(args.reference/'fixture.json',out/'fixture.json')
    for entry in [*ref['inputs'].values(),*ref['expected'].values()]:
        (out/entry['file']).symlink_to((args.reference/entry['file']).resolve())
    conversion={'method':'reviewed_spatial_reshape_specialization','changes':changed,
        'template_manifest_sha256':sha256(args.template/'model.json'),
        'template_graph_sha256':GRAPH_SHA256,'independent_reference_sha256':sha256(args.reference/'fixture.json'),
        'scope':'Conversion candidate; must pass full-resolution official-reference validation before use'}
    (out/'conversion.json').write_text(json.dumps(conversion,indent=2)+'\n')
    names=['head.ncnn.param','head.ncnn.bin','fixture.json','conversion.json','in0.f32','out0.f32']
    manifest={'schema_version':1,'component':'vae','weights':ref['weights'],'source_sha256':sha256(__file__),
        'ncnn_revision':json.loads((ROOT/'sources.lock.json').read_text())['ncnn']['revision'],
        'files':{name:sha256(out/name) for name in names}}
    (out/'model.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),'resolution':[8*w,8*h],'changed_reshapes':len(changed)}))

if __name__=='__main__':main()
