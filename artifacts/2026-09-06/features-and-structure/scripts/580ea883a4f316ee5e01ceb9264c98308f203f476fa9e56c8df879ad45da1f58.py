#!/usr/bin/env python3
"""Fetch the optional pinned Ministral3 prompt enhancer in bounded components."""
import argparse
import json
import urllib.request
from fetch_component import ROOT, fetch_component, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=26)
    p.add_argument('--metadata-only', action='store_true')
    args = p.parse_args()
    if not 0 <= args.start <= args.end <= 26:
        p.error('Require 0 <= start <= end <= 26')
    source = json.loads((ROOT/'sources.lock.json').read_text())['official_model']
    targets = {'pe/config.json': ROOT/'models/official/pe-config.json',
               **{f'pe_tokenizer/{name}': ROOT/'models/pe-tokenizer'/name
                  for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}}
    for name, target in targets.items():
        url = f"{source['url']}/resolve/{source['revision']}/{name}"
        metadata = target.with_suffix(target.suffix+'.source.json')
        if target.is_file() and metadata.is_file():
            entry = json.loads(metadata.read_text())
            if entry['url'] != url or sha256(target) != entry['sha256']:
                raise ValueError('Existing PE metadata differs from its pinned source')
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(32*1024*1024 + 1)
        if len(data) > 32*1024*1024:
            raise ValueError('Unexpected PE metadata size')
        if name.endswith('.json'):
            json.loads(data)
        else:
            data.decode('utf-8')
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Existing PE metadata differs')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        metadata.write_text(json.dumps({'url': url, 'revision': source['revision'],
                                       'sha256': sha256(target)}, indent=2)+'\n')
    if args.metadata_only:
        return
    for index in range(args.start, args.end):
        fetch_component(f'model.layers.{index}.', ROOT/f'models/official/pe-block-{index:02d}.safetensors',
                        subfolder='pe', single_file='model.safetensors')
    if args.start == 0:
        for name, prefix in (('embed', 'model.embed_tokens.'), ('norm', 'model.norm.'), ('lm-head', 'lm_head.')):
            fetch_component(prefix, ROOT/f'models/official/pe-{name}.safetensors',
                            subfolder='pe', single_file='model.safetensors')


if __name__ == '__main__':
    main()
