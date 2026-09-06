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
from vae_shape_contract import validate_decoder_shape, specialize_decoder_graph

GRAPH_SHA256 = '6ecb591c473c56f3d9e923c845be2125e6e42b33ee2cd8d4d1783db68e32c4c5'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Sealed independently executed official VAE fixture')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-1376x768', action='store_true', help='Candidate-only latent 96x172; requires independent official reference')
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    base=verify(args.template)
    template_manifest=json.loads((args.template/'model.json').read_text())
    template_weights=template_manifest['files'].get('head.ncnn.bin')
    if template_weights is None or sha256(args.template/'head.ncnn.bin')!=template_weights:
        raise ValueError('Template must bind the complete VAE weight stream')
    ref=json.loads((args.reference/'fixture.json').read_text())
    if sha256(args.template/'head.ncnn.param')!=GRAPH_SHA256:
        raise ValueError('VAE graph has not been reviewed')
    if (base['component']!='vae' or ref['component']!='vae'
        or any(base[k]!=ref[k] for k in ('weights','official_revision','vae_config_sha256','official_source_sha256'))):
        raise ValueError('Reference and template VAE sources differ')
    h,w=ref['height'],ref['width']
    validate_decoder_shape(h,w,fixed=args.fixed_1376x768)
    if (set(ref['inputs'])!={'in0'} or set(ref['expected'])!={'out0'} or
        ref['inputs']['in0']['shape']!=[1,32,h,w] or ref['expected']['out0']['shape']!=[1,3,8*h,8*w]):
        raise ValueError('Unsupported fixture shape')
    for entry in [*ref['inputs'].values(),*ref['expected'].values()]:
        path=args.reference/entry['file']
        if (Path(entry['file']).name!=entry['file'] or sha256(path)!=entry['sha256']
            or path.stat().st_size!=int(np.prod(entry['shape']))*4):
            raise ValueError('Reference tensor checksum or shape differs')
    graph,changed=specialize_decoder_graph((args.template/'head.ncnn.param').read_text(),h,w,
                                           fixed=args.fixed_1376x768)
    out=args.output.resolve();out.mkdir(parents=True)
    (out/'head.ncnn.param').write_text(graph)
    (out/'head.ncnn.bin').symlink_to((args.template/'head.ncnn.bin').resolve())
    shutil.copy2(args.reference/'fixture.json',out/'fixture.json')
    for entry in [*ref['inputs'].values(),*ref['expected'].values()]:
        (out/entry['file']).symlink_to((args.reference/entry['file']).resolve())
    if sha256(out/'head.ncnn.bin')!=template_weights:
        raise ValueError('Template weights changed after validation')
    conversion={'method':'reviewed_spatial_reshape_specialization','changes':changed,
        'template_manifest_sha256':sha256(args.template/'model.json'),
        'weights_sha256':template_weights,
        'fixed_1376x768_candidate':args.fixed_1376x768,'native_acceptance_eligible':False,
        'template_graph_sha256':GRAPH_SHA256,'independent_reference_sha256':sha256(args.reference/'fixture.json'),
        'scope':'Conversion candidate; must pass full-resolution official-reference validation before use'}
    (out/'conversion.json').write_text(json.dumps(conversion,indent=2)+'\n')
    names=['head.ncnn.param','head.ncnn.bin','fixture.json','conversion.json','in0.f32','out0.f32']
    manifest={'schema_version':1,'component':'vae','weights':ref['weights'],'source_sha256':sha256(__file__),
        'ncnn_revision':json.loads((ROOT/'sources.lock.json').read_text())['ncnn']['revision'],
        'files':{name:sha256(out/name) for name in names}}
    if manifest['files']['head.ncnn.bin']!=conversion['weights_sha256']:
        raise ValueError('VAE weights changed while creating candidate')
    (out/'model.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),'resolution':[8*w,8*h],'changed_reshapes':len(changed)}))

if __name__=='__main__':main()
