#!/usr/bin/env python3
"""Bounded-memory weight value audit; content matches do not prove graph mapping.

Canonical bytes are little-endian FP32 in official tensor order. The scanner
supports the pinned graphs' common ncnn weights and fails closed at unknown
weighted layers. mmap creates views only; conversion buffers are <= 4 MiB.
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
import numpy as np
try:
    from package_model import sha256, safe_name
    from port_graph_contract import graph_weight_mappings
except ImportError:
    from tools.package_model import sha256, safe_name
    from tools.port_graph_contract import graph_weight_mappings

CHUNK=1<<20
OFFICIAL_REVISION='bc68c81e2a1730a394d5fc9fae70713dee940140'
OFFICIAL_REPOSITORY='https://huggingface.co/baidu/ERNIE-Image-Turbo'
DTYPE_SIZES={'F32':4,'F16':2,'BF16':2,'I64':8}
MHA_EVIDENCE={
    'ncnn_revision':'f6f734f44d66f469fefee9ee401fd1cb5e3d573e',
    'loader_source':'https://github.com/Tencent/ncnn/blob/f6f734f44d66f469fefee9ee401fd1cb5e3d573e/src/layer/multiheadattention.cpp',
    'loader_source_sha256':'f5aa50405bda6b1d62b234ce10f455eb68f704913f3faffc6ab59343403e5237',
    'load_model_lines':[29,62],
    'forward_weight_order_lines':[125,138,166,179,207,220,350,362],
    'linear_storage':'row-major [output_features,input_features], no transpose',
    'scope':'projection weight/bias roles and bytes, not full attention graph equivalence',
}
CROP_EVIDENCE={
    'ncnn_revision':'f6f734f44d66f469fefee9ee401fd1cb5e3d573e',
    'crop_cpp_sha256':'e879147fcca9304d9d8be4a6c32c700a26d2624cc7d9e1bef6081c67aabc141d',
    'crop_h_sha256':'febe8f78104f83f1afa61df87dd83a223247e4c9a7b40a329a0827d2b599c89a',
    'layer_cpp_sha256':'2bd7ccf956b74032cd80a5d446898b10e0a8721a1561b4717cf9ca03ad831fa1',
    'basis':'Crop::load_param reads scalar/array/expression ParamDict fields; Crop has no load_model override; Layer::load_model consumes zero bytes',
    'source':'https://github.com/Tencent/ncnn/blob/f6f734f44d66f469fefee9ee401fd1cb5e3d573e/src/layer/crop.cpp',
}


def crop_weights(params):
    # Explicit pinned load_param IDs; array IDs serialize as -23300-id.
    scalars={0,1,2,3,4,5,6,7,8,13,14,15}
    arrays={-23309,-23310,-23311};expressions={19,20,21}
    for key,value in params.items():
        index=int(key)
        if index in scalars:
            if not re.fullmatch(r'-?\d+',value):raise ValueError('Invalid Crop scalar')
        elif index in arrays:
            fields=value.split(',')
            if any(not re.fullmatch(r'-?\d+',v) for v in fields):raise ValueError('Invalid Crop array')
            if int(fields[0])<0 or int(fields[0])!=len(fields)-1:raise ValueError('Invalid Crop array length')
        elif index in expressions:
            if not value:raise ValueError('Empty Crop expression')
        else:raise ValueError('Unsupported Crop parameter '+key)
    return []

VAE_TAIL_EVIDENCE={'ncnn_revision': 'f6f734f44d66f469fefee9ee401fd1cb5e3d573e', 'source_sha256': {'reorg.cpp': '57defcdab6d4251b3fad234ef302af93d648e1849923636f60c1301e1ec2d95d', 'reorg.h': 'cf24e4375dd4a1003235b7cd7ccf9ab437cdae09919302f7f3d26e65e01fc00c', 'batchnorm.cpp': '4303d94e4508efc87d0b8261696bfaab4a1a3dfba8a74c212ce232fe459af153', 'batchnorm.h': '1e0380d0703761d865bad5087682d5da6bccfdec0a6295a145154d90d7bf584c'}, 'reorg': 'load_param IDs0(stride),1(mode); no load_model override, zero bytes', 'batchnorm': 'load_param IDs0(channels),1(eps); load_model lines26-40: raw FP32 slope, mean, variance, bias; all channels elements'}

PINNED_VAE_ATTENTION={
    'f4469da7c4cdf42375cf83adec075aa02b4d289c68e2fbd0f76142d4f3571bf9':('attention_66','decoder'),
    '7abee618d294ea7120f755e4b245b274c54ed3264b84b8cea366083b968e913d':('attention_51','encoder'),
}


def normalized_hash(path, offset, count, dtype, transpose_shape=None):
    sizes=DTYPE_SIZES
    if dtype not in sizes or count < 0 or offset < 0 or offset+count*sizes[dtype]>Path(path).stat().st_size:
        raise ValueError('Invalid tensor bounds or dtype')
    storage_dtype={'F32':'<f4','F16':'<f2','BF16':'<u2','I64':'<i8'}[dtype]
    digest=hashlib.sha256()
    def add(a):
        if dtype=='BF16':a=(a.astype('<u4')<<16).view('<f4')
        values=np.asarray(a,dtype='<f4')
        if not np.isfinite(values).all():raise ValueError('Nonfinite canonical weight values')
        digest.update(values.tobytes())
    if transpose_shape:
        rows,cols=transpose_shape
        if rows*cols!=count:raise ValueError('Invalid transpose shape')
        # One source column at a time, bounded even for multi-GB matrices.
        view=np.memmap(path,mode='r',offset=offset,dtype=storage_dtype,shape=(count,))
        matrix=view.reshape(rows,cols)
        for col in range(cols):
            for row in range(0,rows,CHUNK):add(matrix[row:row+CHUNK,col])
        del matrix,view
    else:
        with Path(path).open('rb') as stream:
            stream.seek(offset)
            for start in range(0,count,CHUNK):
                data=stream.read(min(CHUNK,count-start)*sizes[dtype])
                add(np.frombuffer(data,dtype=storage_dtype))
    return digest.hexdigest()


def _unique_json(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key: '+key)
        result[key]=value
    return result


def official_inventory(root, filenames=None):
    """Only authenticated pinned official components enter content matching."""
    rows=[];root=Path(root)
    paths=sorted(root.glob('*.safetensors')) if filenames is None else [root/safe_name(n) for n in filenames]
    if not paths:raise ValueError('No official safetensors components selected')
    for path in paths:
        manifest_path=path.with_suffix('.manifest.json')
        if not manifest_path.is_file():raise ValueError('Missing official provenance: '+str(path))
        provenance=json.loads(manifest_path.read_text(),object_pairs_hook=_unique_json)
        if provenance.get('revision')!=OFFICIAL_REVISION or provenance.get('repository')!=OFFICIAL_REPOSITORY:
            raise ValueError('Official repository/revision mismatch: '+str(path))
        expected=provenance.get('sha256')
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Missing/invalid official file SHA256: '+str(path))
        if sha256(path)!=expected:raise ValueError('Official file checksum mismatch: '+str(path))
        with path.open('rb') as f:
            prefix=f.read(8)
            if len(prefix)!=8:raise ValueError('Truncated safetensors header length')
            n=struct.unpack('<Q',prefix)[0]
            if n<2 or n>32*1024*1024 or 8+n>path.stat().st_size:raise ValueError('Invalid safetensors header size')
            header=json.loads(f.read(n),object_pairs_hook=_unique_json)
        if not isinstance(header,dict):raise ValueError('Invalid safetensors header object')
        tensors=[];payload_size=path.stat().st_size-8-n
        for name,t in header.items():
            if name=='__metadata__':continue
            if not isinstance(t,dict) or t.get('dtype') not in DTYPE_SIZES:raise ValueError('Unsupported official tensor metadata')
            shape=t.get('shape');offsets=t.get('data_offsets')
            if not isinstance(shape,list) or any(type(x)!=int or x<0 for x in shape):raise ValueError('Invalid tensor dimensions')
            if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x)!=int or x<0 for x in offsets):raise ValueError('Invalid tensor offsets')
            a,b=offsets;count=math.prod(shape)
            if b<a or b>payload_size or b-a!=count*DTYPE_SIZES[t['dtype']]:raise ValueError('Tensor byte range/shape mismatch')
            tensors.append((a,b,name,t,count))
        position=0
        for a,b,name,t,count in sorted(tensors):
            if a!=position:raise ValueError('Overlapping or noncontiguous tensor ranges')
            position=b
        if position!=payload_size:raise ValueError('Unreferenced safetensors payload bytes')
        for a,b,name,t,count in tensors:
            rows.append(dict(file=path.name,name=name,shape=t['shape'],dtype=t['dtype'],
                             revision=provenance['revision'],source_sha256=expected,
                             canonical_sha256=normalized_hash(path,8+n+a,count,t['dtype'])))
    return rows


def mha_shapes(p):
    """Pinned ncnn load_param/load_model and forward row loops define these shapes."""
    embed=int(p.get('0',0));size=int(p.get('2',0));heads=int(p.get('1',1))
    kdim=int(p.get('3',embed));vdim=int(p.get('4',embed))
    if int(p.get('18',0)):raise ValueError('Unsupported MultiHeadAttention int8 scales')
    if embed<=0 or size<=0 or size%embed or heads<=0 or embed%heads or kdim<=0 or vdim<=0:
        raise ValueError('Invalid MultiHeadAttention dimensions')
    qdim=size//embed
    return {'q.weight':[embed,qdim],'q.bias':[embed],
            'k.weight':[embed,kdim],'k.bias':[embed],
            'v.weight':[embed,vdim],'v.bias':[embed],
            'out.weight':[qdim,embed],'out.bias':[qdim]}


WEIGHTLESS=set('Input Split BinaryOp UnaryOp ErnieImageRoPE GELU Permute Reshape SDPA ExpandDims RotaryEmbed Swish Tile Interp PixelShuffle Concat Slice Sigmoid Packing Flatten Softmax'.split())


def layer_weights(kind,p):
    g=lambda k,d=0:int(p.get(str(k),d))
    if kind=='Crop':return crop_weights(p)
    if kind=='Reorg':
        if set(p)-{'0','1'} or g(0,1)<=0 or g(1) not in (0,1):raise ValueError('Unsupported Reorg parameters')
        return []
    if kind=='BatchNorm':
        if set(p)-{'0','1'} or g(0)<=0 or not math.isfinite(float(p.get('1',0))) or float(p.get('1',0))<0:raise ValueError('Unsupported BatchNorm parameters')
        return [(role,g(0),1,None) for role in ('slope','mean','variance','bias')]
    if kind in WEIGHTLESS:return []
    if kind=='MultiHeadAttention':
        return [(role,math.prod(shape),0 if role.endswith('weight') else 1,None) for role,shape in mha_shapes(p).items()]
    if kind=='Gemm':
        if g(18) or g(4):raise ValueError('Unsupported Gemm A/int8')
        rows=[]
        if g(5):rows.append(('B',g(8)*g(9),0,None if g(3) else (g(9),g(8))))
        if g(6) and g(10)!=-1:
            counts={0:1,1:g(7),2:g(7),3:g(7)*g(8),4:g(8)}
            rows.append(('C',counts[g(10)],0,None))
        return rows
    if kind in ('RMSNorm','LayerNorm'):
        return [(role,g(0),1,None) for role in (('gamma',) if kind=='RMSNorm' else ('gamma','beta'))] if g(2,1) else []
    if kind=='GroupNorm':return [(role,g(1),1,None) for role in ('gamma','beta')] if g(3,1) else []
    if kind in ('Convolution','ConvolutionDepthWise','InnerProduct','Embed'):
        if g(18 if kind=='Embed' else 8):raise ValueError('Unsupported int8 weights')
        count=g(6) if kind.startswith('Convolution') else g(2) if kind=='InnerProduct' else g(3)
        bias=g(5) if kind.startswith('Convolution') else g(1) if kind=='InnerProduct' else g(2)
        return [('weight',count,0,None)]+([('bias',g(0),1,None)] if bias else [])
    if kind=='MemoryData':return [('constant',g(0)*max(g(1),1)*max(g(2),1)*max(g(11),1),g(21,1),None)]
    raise ValueError('Unsupported layer: '+kind)


def reference_inventory(root):
    rows=[];gaps=[]
    for param in sorted(Path(root).rglob('*.ncnn.param')):
        binary=param.with_suffix('.bin')
        if not binary.exists():gaps.append(dict(file=str(binary),reason='missing binary'));continue
        offset=0
        param_sha256=sha256(param)
        try:
            with binary.open('rb') as f:
                for line in param.read_text().splitlines()[2:]:
                    t=line.split();kind,name=t[:2];i=4+int(t[2])+int(t[3]);p=dict(x.split('=',1) for x in t[i:])
                    for role,count,load_type,transpose in layer_weights(kind,p):
                        dtype='F32'
                        if load_type==0:
                            f.seek(offset);tag=struct.unpack('<I',f.read(4))[0];offset+=4
                            if tag==0x01306b47:dtype='F16'
                            elif tag not in (0,0x0002c056):raise ValueError('Unsupported ncnn tag '+hex(tag))
                        start=offset;size=count*(2 if dtype=='F16' else 4);offset+=(size+3)//4*4
                        row=dict(file=str(binary.relative_to(root)),layer=name,kind=kind,role=role,
                                 offset=start,count=count,storage_dtype=dtype,transpose_shape=transpose,
                                 param_sha256=param_sha256,canonical_sha256=normalized_hash(binary,start,count,dtype,transpose))
                        if kind=='MultiHeadAttention':
                            row.update(logical_shape=mha_shapes(p)[role],layout_evidence=MHA_EVIDENCE)
                            pinned=PINNED_VAE_ATTENTION.get(param_sha256)
                            if pinned and name==pinned[0]:
                                projection,attribute=role.split('.')
                                target={'q':'to_q','k':'to_k','v':'to_v','out':'to_out.0'}[projection]
                                row['logical_target']=f'{pinned[1]}.mid_block.attentions.0.{target}.{attribute}'
                                row['logical_mapping_basis']='pinned VAE graph instance + ncnn loader/forward role + named official tensor equality'
                            else:row['logical_mapping_status']='unverified_graph_instance'
                        rows.append(row)
            if offset!=binary.stat().st_size:raise ValueError(f'Unconsumed bytes {binary.stat().st_size-offset}')
        except (ValueError,KeyError,struct.error) as exc:gaps.append(dict(file=str(param.relative_to(root)),reason=str(exc),offset=offset))
    return rows,gaps


def audit_port_weights(source: Path, official: Path, output: Path, official_files=None) -> dict:
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    official_rows=official_inventory(official,official_files)
    (output/'official-tensors.json').write_text(json.dumps(official_rows,indent=2))
    peer,gaps=reference_inventory(source);lookup={}
    for row in official_rows:lookup.setdefault(row['canonical_sha256'],[]).append(row['file']+':'+row['name'])
    named={r['name']:r for r in official_rows}
    logical=[]
    for row in peer:
        row['official_content_matches']=lookup.get(row['canonical_sha256'],[])
        row['status']='value_match' if row['official_content_matches'] else 'unmatched'
        if row['kind']=='MultiHeadAttention':
            target=named.get(row.get('logical_target'))
            if not row.get('logical_target'):status='unverified_graph_instance'
            elif target is None:status='official_component_missing'
            elif target['shape']!=row['logical_shape']:status='logical_shape_mismatch'
            elif target['canonical_sha256']!=row['canonical_sha256']:status='logical_value_mismatch'
            else:status='logical_projection_value_match'
            row['logical_mapping_status']=status
            logical.append({k:row.get(k) for k in ('file','layer','role','logical_target','logical_shape','canonical_sha256','logical_mapping_status')})
            if status!='logical_projection_value_match':gaps.append(dict(file=row['file'],layer=row['layer'],role=row['role'],reason=status))
    graph_mappings=graph_weight_mappings(source,peer,official_rows)
    report=dict(schema_version=1,status='unproven',comparison_scope='product_comparison_only',
                reason='Content equality does not establish full logical graph correspondence; derived constants and unsupported paths remain explicit',
                buffer_elements=CHUNK,official_tensor_count=len(official_rows),reference_tensor_count=len(peer),
                matched_tensor_count=sum(bool(r['official_content_matches']) for r in peer),
                unmatched_tensor_count=sum(not r['official_content_matches'] for r in peer),gaps=gaps,tensors=peer,
                logical_projection_mappings=logical,graph_weight_mappings=graph_mappings,official_components=[p.name for p in sorted(Path(official).glob('*.safetensors'))] if official_files is None else list(official_files),
                mha_layout_evidence=MHA_EVIDENCE,crop_serialization_evidence=CROP_EVIDENCE,vae_tail_serialization_evidence=VAE_TAIL_EVIDENCE,allowed_to_close_S=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--official',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--official-files',nargs='+',help='Explicit authenticated component subset; missing components remain gaps');a=p.parse_args()
    r=audit_port_weights(a.source,a.official,a.output,a.official_files);print(json.dumps({k:v for k,v in r.items() if k!='tensors'},indent=2))
