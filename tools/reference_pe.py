#!/usr/bin/env python3
"""Run the pinned official FP32 PE model and save a greedy decoding oracle."""
import argparse
import inspect
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Ministral3ForCausalLM
from build_text_weights import component
from export_pe_block import config
from prepare_block import ROOT, sha256
from prompt_io import read_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.max_tokens <= 2048:
        p.error('Use a new directory and 1..2048 output tokens')
    if min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
        p.error('Dimensions must be positive multiples of 16')
    out = args.output.resolve(); out.mkdir(parents=True)
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/pe-tokenizer', local_files_only=True)
    prompt = read_prompt(args.prompt_file)
    content = json.dumps(dict(prompt=prompt, width=args.width, height=args.height), ensure_ascii=False)
    formatted = tokenizer.apply_chat_template([dict(role='user', content=content)],
                                             tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(formatted, return_tensors='pt')
    if inputs.input_ids.shape[1] > 2048:
        raise ValueError('PE input exceeds the reviewed 2048-token input limit')
    with torch.device('meta'):
        model = Ministral3ForCausalLM(config())
    sources = []
    for index in range(26):
        path = ROOT/f'models/official/pe-block-{index:02d}.safetensors'
        prefix = f'model.layers.{index}.'
        source, _, _ = component(path, prefix)
        state = {k.removeprefix(prefix): v.float() for k, v in load_file(path).items()}
        model.model.layers[index].load_state_dict(state, strict=True, assign=True)
        del state
        sources.append(source['sha256'])
        print(json.dumps({'reference_load_block': index}), flush=True)
    for name, prefix, layer in [('pe-embed', 'model.embed_tokens.', model.model.embed_tokens),
                                ('pe-norm', 'model.norm.', model.model.norm)]:
        path = ROOT/f'models/official/{name}.safetensors'
        source, _, _ = component(path, prefix); sources.append(source['sha256'])
        layer.load_state_dict({k.removeprefix(prefix): v.float() for k, v in load_file(path).items()},
                              strict=True, assign=True)
    # The package builder independently checks equality of both stored tensors.
    embed = json.loads((ROOT/'models/official/pe-embed.manifest.json').read_text())
    head = json.loads((ROOT/'models/official/pe-lm-head.manifest.json').read_text())
    if embed['tensors'][0]['sha256'] != head['tensors'][0]['sha256']:
        raise ValueError('Official PE tied weights differ')
    model.tie_weights()
    # Meta construction leaves the non-persistent YaRN buffer on meta.
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding
    model.model.rotary_emb = Ministral3RotaryEmbedding(config())
    if any(x.is_meta for x in model.parameters()):
        raise ValueError('Reference contains unloaded parameters')
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False,
                                pad_token_id=11, eos_token_id=2,
                                return_dict_in_generate=True, output_logits=True)
    ids = result.sequences[0, inputs.input_ids.shape[1]:].tolist()
    for index, logits in enumerate(result.logits):
        logits[0].float().numpy().astype('<f4').tofile(out/f'logits-{index}.f32')
    (out/'input-ids.txt').write_text(''.join(f'{x}\n' for x in inputs.input_ids[0].tolist()))
    (out/'generated-ids.txt').write_text(''.join(f'{x}\n' for x in ids))
    (out/'enhanced.txt').write_text(tokenizer.decode(ids, skip_special_tokens=True).strip())
    (out/'formatted.txt').write_text(formatted)
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    metadata = dict(scope='Official Ministral3ForCausalLM.generate, CPU FP32 SDPA, greedy, real prompt',
                    input_prompt=prompt, width=args.width, height=args.height, max_tokens=args.max_tokens,
                    input_tokens=inputs.input_ids.shape[1], generated_tokens=len(ids), eos=bool(ids and ids[-1] == 2),
                    official_model_revision=lock['official_model']['revision'],
                    transformers_revision=lock['transformers']['revision'], source_weight_sha256=sources,
                    model_source_sha256=sha256(inspect.getfile(Ministral3ForCausalLM)),
                    script_sha256=sha256(__file__),
                    files={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'reference.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'reference_complete': True, 'generated_tokens': len(ids), 'eos': metadata['eos']}), flush=True)


if __name__ == '__main__': main()
