#!/usr/bin/env python3
"""Read-only audit of pinned static source packages; never instantiates a graph.

Only enumerated fields are normalized, then the entire graph is hash checked.
Matching these static instances is not permission to generate arbitrary shapes.
"""
import argparse
import hashlib
import json
from pathlib import Path

TEXT = {**{f'gemm_{i}': {'7':'text_bucket'} for i in range(7)},
        **{f'reshape_{i}': {'2':'text_bucket'} for i in (10,11,12)},
        'reshape_13':{'1':'text_bucket'}, 'unsqueeze_18':{'1':'text_bucket'}, 'unsqueeze_19':{'1':'text_bucket'}}
DIT = {**{f'gemm_{i}': {'7':'total_tokens'} for i in range(7)},
       **{f'reshape_{i}': {'2':'total_tokens'} for i in (12,13,14)},
       'reshape_15':{'1':'total_tokens'}, 'unsqueeze_20':{'1':'total_tokens'}, 'unsqueeze_21':{'1':'total_tokens'}}
RULES = {'text':TEXT, 'dit':DIT,
         'input':{'reshape_7':{'0':'image_tokens'},'gemm_0':{'7':'dit_text_tokens'}},
         'output':{'gemm_0':{'7':'total_tokens'},'slice_0':{'-23310':'image_tokens_array'},'reshape_5':{'0':'packed_width','1':'packed_height'}},
         'vae':{'reshape_99':{'0':'vae_pixels'},'reshape_100':{'0':'vae_width','1':'vae_height'}}}
# These complete normalized graph hashes are tied to the reviewed static source instances.
CONTRACT_HASHES = {'text': 'dda8db13b6e20a00c1133485c3be2ef37db56b1db058e42a405b51d24fc64b0e', 'dit': '51db4065837c28e235784fd0dd196f109ca86822143c025a44ee9372240e4132', 'input': '3710a502802138e8605ca54a9132fb9a9a1888841586c1d7067f7043769bbecb', 'output': '98bf1afc154dd05cf419aafad4aa383af311e44658e644d0c322ffb50431533c', 'vae': '6d68a0f10423b6ca243e229c8a9e8fe98c442b0d061ad5d38667f752cd9eb48f'}
PINNED_MANIFESTS = {'9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154': 'portable-turbo1024-s32-v1', '9c14feef90fe8e3d189466ce2cef3fce78b9fdf772ffdacee5afe02b40d90752': 'pipeline64-residual-v1', 'ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59': 'turbo512x384-s2048-portable', '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1': 'turbo1024-s64-portable'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dimensions(config):
    w,h,t,d=(config[k] for k in ('packed_width','packed_height','text_bucket','dit_text_tokens'))
    if any(type(v) is not int for v in (w,h,t,d)) or not (1<=w<=128 and 1<=h<=128 and t in (32,64,2048) and t<=d<=2048):
        raise ValueError('Invalid static shape metadata')
    return dict(config,image_tokens=w*h,total_tokens=w*h+d,image_tokens_array='1,'+str(w*h),vae_width=w*2,vae_height=h*2,vae_pixels=w*h*4)


def normalize_graph(graph, kind, config):
    """Audit exact parameter locations; no replacement is ever written to model files."""
    if kind not in RULES:raise ValueError('Unknown component kind')
    values=dimensions(config);rules=RULES[kind];seen=set();nodes=set();lines=[];changes=[]
    for line in graph.splitlines():
        f=line.split()
        if len(f)>1:
            name=f[1]
            if name in nodes:raise ValueError('Duplicate graph node')
            nodes.add(name)
            if name in rules:
                start=4+int(f[2])+int(f[3])
                for key,formula in rules[name].items():
                    target=key+'='+str(values[formula])
                    hits=[i for i in range(start,len(f)) if f[i].split('=',1)[0]==key]
                    if len(hits)!=1 or f[hits[0]]!=target:raise ValueError('Unreviewed shape field '+name+':'+key)
                    changes.append(dict(node=name,operator=f[0],parameter=key,value=str(values[formula]),formula=formula))
                    f[hits[0]]=key+'='+formula.upper();seen.add((name,key))
        lines.append(' '.join(f))
    if seen!={(n,k) for n,fields in rules.items() for k in fields}:raise ValueError('Missing shape node')
    return hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),changes


def graph_files():
    return [(f'text/block-{i:02d}/text.ncnn.param','text') for i in range(25)]+[(f'dit/block-{i:02d}/block.ncnn.param','dit') for i in range(36)]+[('dit/input/head.ncnn.param','input'),('dit/output/head.ncnn.param','output'),('vae/head.ncnn.param','vae')]


def audit_package(root):
    root=Path(root);manifest_path=root/'manifest.json';manifest_sha=sha(manifest_path)
    if manifest_sha not in PINNED_MANIFESTS:raise ValueError('Unknown static package manifest hash')
    m=json.loads(manifest_path.read_text());config=m['config'];dimensions(config)
    if sha(root/'model.cfg')!=m['files']['model.cfg']:raise ValueError('Config checksum mismatch')
    raw=(root/'model.cfg').read_text().split()
    if len(raw)%2 or {raw[i]:int(raw[i+1]) for i in range(0,len(raw),2)}!=config:raise ValueError('Config metadata mismatch')
    graphs=[]
    for relative,kind in graph_files():
        path=root/relative;digest=sha(path)
        if m['schema_version']==2:expected=m['files'][relative]
        elif relative in m['source_manifests']:expected=m['source_manifests'][relative]
        else:
            component=str(Path(relative).parent);meta_path=root/component/'model.json'
            if sha(meta_path)!=m['source_manifests'][component]:raise ValueError('Source manifest checksum mismatch')
            expected=json.loads(meta_path.read_text())['files'][path.name]
        if digest!=expected:raise ValueError('Static graph checksum mismatch: '+relative)
        canonical,fields=normalize_graph(path.read_text(),kind,config)
        if canonical!=CONTRACT_HASHES[kind]:raise ValueError('Unknown graph topology: '+relative)
        shape_nodes=[]
        for line in path.read_text().splitlines()[2:]:
            f=line.split()
            if f and f[0] in {'Reshape','Permute','Crop','Slice','Gemm','Concat','ExpandDims','Flatten','Interp','MemoryData','ErnieImageRoPE'}:
                start=4+int(f[2])+int(f[3])
                shape_nodes.append(dict(operator=f[0],node=f[1],parameters=f[start:]))
        graphs.append(dict(path=relative,sha256=digest,contract_sha256=canonical,kind=kind,allowed_shape_fields=fields,shape_sensitive_nodes=shape_nodes))
    return dict(package=str(root),manifest_sha256=manifest_sha,config=config,graphs=graphs,status='static_graph_text_verified',runtime_generation_validated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'schema_version':1,'scope':'Read-only static graph shape audit; mathematical planning and runtime graph generation are separate','packages':[audit_package(x) for x in a.package],'rules':RULES,'contract_hashes':CONTRACT_HASHES,'dynamic_instantiation_supported':False}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
