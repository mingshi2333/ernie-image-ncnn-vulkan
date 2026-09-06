#!/usr/bin/env python3
"""Write lossless BF16 text blocks for the reviewed 32-token GQA graph."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
from safetensors import safe_open
from export_text_block import config
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
from prepare_block import ROOT, sha256

GRAPH_SHA = '912dc8a3f8839cca841004f8c5afb4df5c9d5015d4fedbb4deb77d67a62e2523'
BINDINGS = {
    'rmsn_8': ('input_layernorm.weight',[3072]),
    'gemm_0': ('self_attn.q_proj.weight',[4096,3072]),
    'gemm_1': ('self_attn.k_proj.weight',[1024,3072]),
    'gemm_2': ('self_attn.v_proj.weight',[1024,3072]),
    'gemm_3': ('self_attn.o_proj.weight',[3072,4096]),
    'rmsn_9': ('post_attention_layernorm.weight',[3072]),
    'gemm_4': ('mlp.gate_proj.weight',[9216,3072]),
    'gemm_5': ('mlp.up_proj.weight',[9216,3072]),
    'gemm_6': ('mlp.down_proj.weight',[3072,9216]),
}

def component(path, prefix):
    m = json.loads(path.with_suffix('.manifest.json').read_text())
    revision = json.loads((ROOT/'sources.lock.json').read_text())['official_model']['revision']
    if m['revision'] != revision or m['prefix'] != prefix or sha256(path) != m['sha256']:
        raise ValueError('Component revision, prefix or checksum differs')
    with path.open('rb') as file:
        length = struct.unpack('<Q', file.read(8))[0]
        if not 2 <= length <= 16*1024*1024:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(file.read(length))
    return m, header, length+8

def build(template, output, start, end, include_embedding):
    graph = (template/'text.ncnn.param').read_bytes()
    source_manifest = json.loads((template/'model.json').read_text())
    if (hashlib.sha256(graph).hexdigest() != GRAPH_SHA or source_manifest['tokens'] != 32
        or any(sha256(template/name) != value for name,value in source_manifest['files'].items())):
        raise ValueError('Template differs from reviewed 32-token text graph')
    output.mkdir(parents=True)
    manifest = {'schema_version':1,'tokens':32,'layers_required':25,'hidden_state':'hidden_states[-2] = output of block 24; no final norm',
                'scope':'Text path only; no vision branch or LM head', 'source_sha256':sha256(__file__),
                'config_sha256':sha256(ROOT/'models/official/text_encoder-config.json'),'blocks':[]}
    for index in range(start,end):
        prefix = f'language_model.model.layers.{index}.'
        path = ROOT/f'models/official/text-block-{index:02d}.safetensors'
        weights, header, data_start = component(path,prefix)
        if set(header)-{'__metadata__'} != {prefix+name for name,_ in BINDINGS.values()}:
            raise ValueError('Unexpected text block tensors')
        directory = output/f'block-{index:02d}'
        directory.mkdir()
        restored_hash = hashlib.sha256()
        sections = []
        with path.open('rb') as inp, (directory/'text.ncnn.bin.partial').open('xb') as out:
            for line in graph.decode().splitlines()[2:]:
                fields = line.split()
                if fields[0] not in ('Gemm','RMSNorm'):
                    continue
                name,shape = BINDINGS[fields[1]]
                item = header[prefix+name]
                begin,stop = item['data_offsets']
                if item['dtype'] != 'BF16' or item['shape'] != shape or stop-begin != math.prod(shape)*2:
                    raise ValueError('Unexpected text weight layout')
                affine = fields[0] == 'RMSNorm'
                inp.seek(data_start+begin)
                if not affine:
                    out.write(struct.pack('<I',0x01348B83)); restored_hash.update(b'\0'*4)
                remaining = stop-begin
                while remaining:
                    data = inp.read(min(remaining,1024*1024))
                    if not data or len(data)%2: raise ValueError('Truncated text component')
                    restored = (np.frombuffer(data,'<u2').astype('<u4')<<16).tobytes()
                    if not np.isfinite(np.frombuffer(restored,'<f4')).all(): raise ValueError('Non-finite text weight')
                    out.write(restored if affine else data);restored_hash.update(restored)
                    remaining -= len(data)
                if not affine: out.write(b'\0'*((-(stop-begin))%4))
                sections.append({'layer':fields[1],'tensor':prefix+name,'shape':shape,'storage':'fp32' if affine else 'bf16'})
        (directory/'text.ncnn.bin.partial').rename(directory/'text.ncnn.bin')
        (directory/'text.ncnn.param').write_bytes(graph)
        entry = {'block':index,'tokens':32,'weights_sha256':weights['sha256'],'graph_sha256':GRAPH_SHA,
            'reconstructed_fp32_sha256':restored_hash.hexdigest(),'sections':sections,
            'files':{name:sha256(directory/name) for name in ('text.ncnn.param','text.ncnn.bin')}}
        if index == 0 and entry['reconstructed_fp32_sha256'] != sha256(template/'text.ncnn.bin'):
            raise ValueError('Reconstructed first block differs from pnnx FP32 weights')
        (directory/'model.json').write_text(json.dumps(entry,indent=2)+'\n')
        manifest['blocks'].append(entry)
        print(json.dumps({'block':index,'converted':True}),flush=True)
    if include_embedding:
        path = ROOT/'models/official/text-embed.safetensors'
        weights,header,data_start = component(path,'language_model.model.embed_tokens.')
        name = 'language_model.model.embed_tokens.weight'
        item = header[name]
        if set(header)-{'__metadata__'} != {name} or item['dtype'] != 'BF16' or item['shape'] != [131072,3072]:
            raise ValueError('Invalid embedding table')
        begin,stop = item['data_offsets']
        if stop-begin != 131072*3072*2: raise ValueError('Invalid embedding byte count')
        with path.open('rb') as inp, (output/'embeddings.bf16').open('xb') as out:
            inp.seek(data_start+begin)
            remaining = stop-begin
            while remaining:
                data = inp.read(min(remaining,1024*1024))
                if not data: raise ValueError('Truncated embedding table')
                out.write(data); remaining -= len(data)
        manifest['embedding'] = {'file':'embeddings.bf16','sha256':sha256(output/'embeddings.bf16'),'source_sha256':weights['sha256']}
    rope = MistralRotaryEmbedding(config())
    if rope.attention_scaling != 1.:
        raise ValueError('Native text path requires configured unit YaRN attention factor')
    rope.inv_freq.detach().cpu().numpy().astype('<f4').tofile(output/'rope-inv-freq.f32')
    manifest['rope'] = {'file':'rope-inv-freq.f32','sha256':sha256(output/'rope-inv-freq.f32'),'attention_scaling':1.}
    (output/'model.json').write_text(json.dumps(manifest,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=25)
    parser.add_argument('--embedding',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or not 0 <= args.start < args.end <= 25:
        parser.error('Use new output and 0 <= start < end <= 25')
    build(args.template,args.output,args.start,args.end,args.embedding)

if __name__=='__main__':main()
