#!/usr/bin/env python3
"""Reuse verified shape-independent weights with an independently exported static graph."""
import argparse
import json
from pathlib import Path
from build_dit_weights import GRAPH_SHA256,graph_hash
from prepare_block import ROOT,sha256
from validate_dit_block import verify

def residual_graph(graph, restore=False):
    lines=graph.splitlines();changed=0
    for i,line in enumerate(lines):
        parts=line.split()
        if len(parts)>1 and parts[1] in ('add_16','add_22'):
            expected='ErnieResidualAdd' if restore else 'BinaryOp'
            if parts[0]!=expected or parts[2:4]!=['2','1'] or parts[7:]!=['0=0']:
                raise ValueError('Unreviewed residual sum')
            parts[0]='BinaryOp' if restore else 'ErnieResidualAdd'
            lines[i]=' '.join(parts);changed+=1
    if changed!=2:raise ValueError('Expected two residual sums')
    return '\n'.join(lines)+'\n'

def verify_runtime(model):
    manifest=json.loads((model/'model.json').read_text())
    if manifest.get('kind')!='inference_static_bucket':
        return verify(model,model)[0]
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    graph=(model/'block.ncnn.param').read_text()
    if manifest.get('fp32_residual'):graph=residual_graph(graph,restore=True)
    if (manifest['official_model_revision']!=lock['official_model']['revision']
        or manifest['ncnn_revision']!=lock['ncnn']['revision'] or not 0<=manifest['block']<36
        or graph_hash(graph,manifest['tokens'])!=GRAPH_SHA256
        or any(Path(name).name!=name or sha256(model/name)!=digest for name,digest in manifest['files'].items())):
        raise ValueError('Invalid inference graph, source revision or file checksum')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fp32-residual',action='store_true',help='Keep both residual sums and skip paths in FP32')
    args=p.parse_args()
    if args.output.exists() or len(args.model)!=36:p.error('Use new output and all 36 source blocks')
    template=verify(args.template,args.template)[0]
    graph=(args.template/'block.ncnn.param').read_text()
    if graph_hash(graph,template['tokens'])!=GRAPH_SHA256:raise ValueError('Template topology differs')
    if args.fp32_residual:graph=residual_graph(graph)
    args.output.mkdir(parents=True)
    entries=[]
    for i,path in enumerate(args.model):
        base=verify_runtime(path)
        source_graph=(path/'block.ncnn.param').read_text()
        if base.get('fp32_residual'):source_graph=residual_graph(source_graph,restore=True)
        if base['block']!=i or graph_hash(source_graph,base['tokens'])!=GRAPH_SHA256:
            raise ValueError('Source block topology or order differs')
        storage=base if base.get('kind')=='inference_static_bucket' else base.get('weight_storage',{})
        if (storage.get('graph_contract_sha256')!=GRAPH_SHA256
            or len(storage.get('reconstructed_fp32_sha256',''))!=64):
            raise ValueError('Require lossless BF16 source with reviewed weight bindings')
        if i==0 and sha256(args.template/'block.ncnn.bin')!=storage['reconstructed_fp32_sha256']:
            raise ValueError('Independent target export does not reconstruct the same FP32 weights')
        out=args.output/f'block-{i:02d}';out.mkdir()
        (out/'block.ncnn.param').write_text(graph)
        (out/'block.ncnn.bin').symlink_to((path/'block.ncnn.bin').resolve())
        m={'schema_version':1,'kind':'inference_static_bucket','block':i,'tokens':template['tokens'],
           'fp32_residual':args.fp32_residual,
           'scope':'Static graph exported independently; reused lossless weights; no standalone numerical fixture in this package',
           'official_model_revision':base['official_model_revision'],'ncnn_revision':base['ncnn_revision'],
           'weights_sha256':base['weights_sha256'],'base_manifest_sha256':sha256(path/'model.json'),
           'template_manifest_sha256':sha256(args.template/'model.json'),'graph_contract_sha256':GRAPH_SHA256,
           'builder_sha256':sha256(__file__),'reconstructed_fp32_sha256':storage['reconstructed_fp32_sha256'],
           'files':{name:sha256(out/name) for name in ('block.ncnn.param','block.ncnn.bin')}}
        (out/'model.json').write_text(json.dumps(m,indent=2)+'\n')
        entries.append(m);print(json.dumps({'block':i,'tokens':template['tokens'],'linked':True}),flush=True)
    (args.output/'model.json').write_text(json.dumps({'blocks':entries,'kind':'inference_static_bucket'},indent=2)+'\n')

if __name__=='__main__':main()
