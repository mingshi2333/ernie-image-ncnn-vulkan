#!/usr/bin/env python3
"""Audit the official output contract with a tiny 26-layer Mistral3 model."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import torch
from transformers import Mistral3Config,Mistral3Model
from transformers.models.mistral.modeling_mistral import MistralModel,MistralAttention
from prepare_block import ROOT,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    args.output.mkdir(parents=True)
    official=ROOT/'models/official/text_encoder-config.json'
    cfg=json.loads(official.read_text())
    value=copy.deepcopy(cfg)
    value['text_config'].update(hidden_size=32,intermediate_size=64,num_attention_heads=2,
        num_key_value_heads=1,head_dim=16,vocab_size=32,use_cache=False)
    value['vision_config'].update(hidden_size=32,intermediate_size=64,num_attention_heads=2,head_dim=16,num_hidden_layers=1)
    config=Mistral3Config.from_dict(value);config._attn_implementation='eager'
    torch.manual_seed(17);torch.set_num_threads(2);torch.set_grad_enabled(False)
    model=Mistral3Model(config).eval()
    captured={}
    hooks=[layer.register_forward_hook(lambda module,inputs,out,i=i:captured.update({i:out.detach()}))
           for i,layer in enumerate(model.language_model.layers)]
    result={'scope':'Synthetic tiny weights; official 26-layer output recording and config dispatch contract, not real-weight model parity',
        'config_sha256':sha256(official),'source_sha256':sha256(__file__),
        'official_sources':{name:sha256(inspect.getfile(cls)) for name,cls in [('Mistral3Model',Mistral3Model),('MistralModel',MistralModel)]},
        'effective_model':type(model.language_model).__name__,'effective_attention':type(model.language_model.layers[0].self_attn).__name__,
        'llama4_scaling_called': '_get_llama_4_attn_scale' in inspect.getsource(MistralAttention.forward),'cases':[]}
    for ids in ([1],[1,3,5,7],[1,31,0,6,14,12,2,5,9]):
        captured.clear();out=model(torch.tensor([ids]),output_hidden_states=True,use_cache=False)
        matches=[i for i,x in captured.items() if torch.equal(x,out.hidden_states[-2])]
        result['cases'].append({'ids':ids,'hidden_state_count':len(out.hidden_states),'second_last_matches_decoder_outputs':matches,
            'last_is_normalized_final':bool(torch.equal(out.last_hidden_state,out.hidden_states[-1])),
            'passed':matches==[24] and len(out.hidden_states)==27 and torch.equal(out.last_hidden_state,out.hidden_states[-1])})
    for hook in hooks:hook.remove()
    result['passed']=result['effective_model']=='MistralModel' and not result['llama4_scaling_called'] and all(c['passed'] for c in result['cases'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
