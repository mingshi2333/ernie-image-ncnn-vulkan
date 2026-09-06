#!/usr/bin/env python3
"""Eight narrowly bound peer weight relations; never a whole-graph proof."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import numpy as np
try:
    from audit_port_weights import reference_inventory, official_inventory, normalized_hash
except ImportError:
    from tools.audit_port_weights import reference_inventory, official_inventory, normalized_hash

PINS = {
  "outputs/reference-port-v1/assets-manifest.json": "147e7710a43acc1bcf0b1d6ebd44dc114173720343a8e0891f6e9ce6e765780d",
  "models/official/dit-final_norm.safetensors": "12fa6e315102b34387b8790eb4f59adcfbe83187bcc8a734f5146b4646ee2067",
  "models/official/dit-final_norm.manifest.json": "91e4f57fd238792b4f0c8c96bb98c80257919016e9819bae050a88f8d5c3829b",
  "models/official/vae-bn.safetensors": "994f11f035c926c3be5a59802b401ab5304bbab7d53053995d4737b76b8411a6",
  "models/official/vae-bn.manifest.json": "ed837a63df954f47c17853675bcdf25579bedf2f373eac88ea2f66d21559f243",
  "models/official/vae-config.json": "4d5ba5e01de06d589dd46e2955ab97e2b0968703dce31b9ebe1d2a38c141836d",
  "models/official/vae-config.source.json": "eb12aa896984caded27202c6620f22f5fa6a752ac9f4d8dacfe919430a331814",
  "tools/audit_port_weights.py": "1e1f583f96bad7613cfbc1f47da1d48be8adb5f4f7bf8fda97f78d8a0feab15d",
  "outputs/reference-port-v1/assets/dit/finalizer.ncnn.param": "3175e42943b968f90047d2993ab7d4e538566a4cf247b8f7c76c1d439d72a620",
  "outputs/reference-port-v1/assets/dit/finalizer.ncnn.bin": "6cbbc765bfc00be61cc45bb4deb816b67d784473a5438625f73fdd2232c7979b",
  "outputs/reference-port-v1/assets/dit/preprocessor.ncnn.param": "611af908f0d31148a1be312ce54b548f676bcfca18013bde637d6f51c87370c5",
  "outputs/reference-port-v1/assets/dit/preprocessor.ncnn.bin": "7a5471f6416834e7389ef7efe1fa5f72ac3632213894df4712037e9199f13869",
  "outputs/reference-port-v1/assets/vae/encoder_full.ncnn.param": "7abee618d294ea7120f755e4b245b274c54ed3264b84b8cea366083b968e913d",
  "outputs/reference-port-v1/assets/vae/encoder_full.ncnn.bin": "7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce",
  "outputs/reference-port-v1/assets/vae/decoder_full.ncnn.param": "f4469da7c4cdf42375cf83adec075aa02b4d289c68e2fbd0f76142d4f3571bf9",
  "outputs/reference-port-v1/assets/vae/decoder_full.ncnn.bin": "44fe82334caff6ff8a636d52dd9dad017c86443582b34cda49f6816a2a0734e8",
  "outputs/reference-port-v1/source/src/ncnn/src/layer/gemm.cpp": "49081dc97b3b49e0af3f38e23fde0d56ac2e1c5426f863776c538ef91000fe39",
  "outputs/reference-port-v1/source/src/ncnn/src/layer/binaryop.h": "960ec8762a4382908a17c517c11788390fe7e027896813426ca28360a0dfb011",
  "outputs/reference-port-v1/source/src/ncnn/src/layer/batchnorm.cpp": "4303d94e4508efc87d0b8261696bfaab4a1a3dfba8a74c212ce232fe459af153",
  "outputs/reference-port-v1/source/src/ncnn/src/layer/convolutiondepthwise.cpp": "3af6164e7011a7fd0cf7d91845029db45bcadbe05190135b9086305b4646215d",
  "outputs/reference-port-v1/source/src/ncnn/src/layer/memorydata.cpp": "74984437b782561e7a0e4b03926b225e95a12d207da376803191a843cc1ef5b0",
  ".venv/lib/python3.14/site-packages/diffusers/models/transformers/transformer_ernie_image.py": "0f1814f63008298707afea5d7bd22d0a16073f0a8e858037c198faa28027e926",
  ".venv/lib/python3.14/site-packages/diffusers/models/embeddings.py": "4eb810f715786eb1f24f2a4641e529817f7c951b9caf2f7925811329a03ec796",
  ".venv/lib/python3.14/site-packages/diffusers/models/autoencoders/autoencoder_kl_flux2.py": "7d9a976c1e4f42615e8c422f1643d86b49c4339221bd04b67f518b718ebd6c2d"
}
ASSETS='outputs/reference-port-v1/assets'
STEMS=['dit/finalizer','dit/preprocessor','vae/encoder_full','vae/decoder_full']


def digest(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def require_pin(path, expected):
    if digest(path)!=expected:raise ValueError('Pinned file changed: '+str(path))


def nodes(text):
    result={}
    for line in text.splitlines()[2:]:
        words=line.split();kind,name=words[:2];ni,no=map(int,words[2:4]);end=4+ni+no
        if name in result:raise ValueError('Duplicate node')
        result[name]=(kind,words[4:4+ni],words[4+ni:end],dict(x.split('=',1) for x in words[end:]))
    return result


def finalizer_contract(text):
    n=nodes(text)
    expected={
      'gemm_0':('Gemm',['3'],['4'],{'10':'4','2':'0','3':'1','4':'0','5':'1','6':'1','7':'0','8':'4096','9':'4096'}),
      'gemm_1':('Gemm',['2'],['5'],{'10':'4','2':'0','3':'1','4':'0','5':'1','6':'1','7':'0','8':'4096','9':'4096'}),
      'splitncnn_0':('Split',['in1'],['2','3'],{}),
      'ln_3':('LayerNorm',['in0'],['6'],{'0':'4096','1':'1.000000e-6','2':'0'}),
      'add_0':('BinaryOp',['4'],['7'],{'0':'0','1':'1','2':'1.0'}),
      'mul_1':('BinaryOp',['6','7'],['8'],{'0':'2'}),
      'add_2':('BinaryOp',['8','5'],['9'],{'0':'0'})}
    for name,value in expected.items():
        if n.get(name)!=value:raise ValueError('Unreviewed scale/shift graph: '+name)
    return {'gemm_0':'scale_rows_0_4096','gemm_1':'shift_rows_4096_8192'}


def tensor_range(path, name, shape):
    with path.open('rb') as file:
        n=struct.unpack('<Q',file.read(8))[0];h=json.loads(file.read(n))
    t=h[name]
    if t['dtype']!='BF16' or t['shape']!=shape:raise ValueError('Official tensor shape/dtype changed')
    a,b=t['data_offsets']
    if b-a!=math.prod(shape)*2:raise ValueError('Tensor size changed')
    return 8+n+a


def derived_metrics(actual, expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid derived tensor')
    diff=actual.astype('f8')-expected.astype('f8')
    return {'bitwise_equal':actual.tobytes()==expected.tobytes(),'different_values':int(np.count_nonzero(actual!=expected)),
            'max_abs_error':float(np.abs(diff).max()),'nrmse':float(np.linalg.norm(diff)/max(np.linalg.norm(expected.astype('f8')),1e-30)),
            'actual_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'expected_sha256':hashlib.sha256(expected.tobytes()).hexdigest()}


def official_frequency(source):
    # Execute the exact reviewed function prefix through torch.exp, before it
    # multiplies by timesteps. No model import or formula algebra substitution.
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_timestep_embedding')
    end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.exp')
    function.body=function.body[:end+1]+[ast.Return(value=ast.Name(id='emb',ctx=ast.Load()))]
    tree=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));scope={'torch':torch,'math':math}
    exec(compile(tree,str(source),'exec'),scope)
    value=scope['get_timestep_embedding'](torch.tensor([1.],dtype=torch.float32),4096,False,0).numpy()
    return value,{'torch':torch.__version__,'device':'cpu','threads':2,'executed_prefix':ast.unparse(tree)}


def audit(root, output):
    root=root.resolve();output=output.resolve()
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True)
    for name,expected in PINS.items():require_pin(root/name,expected)
    manifest=json.loads((root/'outputs/reference-port-v1/assets-manifest.json').read_text())
    if manifest['revision']!='140a052f7919f279de7f697fa54f33bd1c0cac2b':raise ValueError('Peer revision differs')
    inventory={x['path']:x for x in manifest['files']}
    for stem in STEMS:
        for ext in ['param','bin']:
            name=stem+'.ncnn.'+ext;item=inventory[name];path=root/ASSETS/name
            if item['sha256']!=PINS[ASSETS+'/'+name] or item['size']!=path.stat().st_size or item['url']!='https://huggingface.co/wuyex/ernie-image-ncnn/resolve/'+manifest['revision']+'/'+name:
                raise ValueError('Asset manifest identity differs')
    # Reparse actual binary streams to EOF; audit-v6 offsets/hashes are not inputs.
    with tempfile.TemporaryDirectory() as directory:
        view=Path(directory)
        for stem in STEMS:
            for ext in ['param','bin']:
                path=view/(stem+'.ncnn.'+ext);path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(root/ASSETS/path.relative_to(view))
        rows,gaps=reference_inventory(view)
    if gaps:raise ValueError('Actual binary parser gaps: '+str(gaps))
    selected={(r['file'],r['layer'],r['role']):r for r in rows}
    official_inventory(root/'models/official',['dit-final_norm.safetensors','vae-bn.safetensors'])
    finalizer_contract((root/ASSETS/'dit/finalizer.ncnn.param').read_text())
    result=[];official=root/'models/official/dit-final_norm.safetensors'
    for layer,half in [('gemm_0',0),('gemm_1',1)]:
        for role,attribute,shape in [('B','weight',[8192,4096]),('C','bias',[8192])]:
            r=selected[('dit/finalizer.ncnn.bin',layer,role)];count=math.prod(shape)//2
            if r['count']!=count or r['transpose_shape'] is not None or r['storage_dtype']!='F32':raise ValueError('Peer logical layout differs')
            offset=tensor_range(official,'final_norm.linear.'+attribute,shape)+half*count*2
            expected=normalized_hash(official,offset,count,'BF16')
            result.append({'relation':'official_final_norm_row_slice','peer':r,'official_tensor':'final_norm.linear.'+attribute,
                           'official_shape':shape,'row_range':[half*4096,(half+1)*4096],
                           'role':'scale' if half==0 else 'shift','expected_sha256':expected,
                           'bitwise_equal':r['canonical_sha256']==expected})
    freq,environment=official_frequency(root/'.venv/lib/python3.14/site-packages/diffusers/models/embeddings.py')
    r=selected[('dit/preprocessor.ncnn.bin','pnnx_fold_109','constant')]
    if r['count']!=2048 or r['storage_dtype']!='F32':raise ValueError('Frequency layout differs')
    actual=np.fromfile(root/ASSETS/r['file'],dtype='<f4',count=r['count'],offset=r['offset'])
    result.append({'relation':'Timesteps_frequency_4096_downshift0_maxperiod10000','peer':r,**derived_metrics(actual,freq)})
    bn=root/'models/official/vae-bn.safetensors';offset=tensor_range(bn,'bn.running_var',[128])
    var=(np.fromfile(bn,'<u2',count=128,offset=offset).astype('<u4')<<16).view('<f4')
    for file,layer,role,value,basis in [
      ('vae/decoder_full.ncnn.bin','convdw_103','weight',np.sqrt(var+np.float32(1e-4)),'sqrt(BF16 running_var expanded FP32 + 1e-4)'),
      ('vae/encoder_full.ncnn.bin','bn_0','slope',np.ones(128,dtype='<f4'),'nonaffine BN slope=1'),
      ('vae/encoder_full.ncnn.bin','bn_0','bias',np.zeros(128,dtype='<f4'),'nonaffine BN bias=0')]:
        r=selected[(file,layer,role)]
        if r['count']!=128 or r['storage_dtype']!='F32':raise ValueError('VAE constant layout differs')
        actual=np.fromfile(root/ASSETS/file,dtype='<f4',count=128,offset=r['offset'])
        result.append({'relation':basis,'peer':r,**derived_metrics(actual,value)})
    report={'status':'bounded_relations_checked_not_full_graph_proof','allowed_to_close_S':False,
            'source_pins':PINS,'tool_sha256':digest(__file__),'actual_selected_stream_rows':len(rows),'parser_gaps':gaps,
            'relation_count':len(result),'exact_relation_count':sum(r['bitwise_equal'] for r in result),
            'rows':result,'frequency_execution':environment,
            'vae_epsilon_boundary':'Peer inverse BN uses 1e-4; official ERNIE pipeline inverse uses 1e-5. Do not equate these graphs.'}
    (output/'relations.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,output/'audit_port_relations.py.snapshot')
    (output/'official-frequency.f32').write_bytes(freq.tobytes())
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=audit(a.root,a.output);print(json.dumps({k:v for k,v in r.items() if k not in ['rows','source_pins','frequency_execution']}))
