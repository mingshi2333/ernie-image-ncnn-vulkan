#!/usr/bin/env python3
"""Check native PE formatting, exact token IDs and decoding against official assets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from transformers import AutoTokenizer
from prepare_block import ROOT, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, default=ROOT/'models/pe-tokenizer')
    p.add_argument('--runner', type=Path, default=ROOT/'build/ernie-pe-runner')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Use a new output directory')
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    sources = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        meta = json.loads((args.tokenizer/(name+'.source.json')).read_text())
        if meta['revision'] != lock['official_model']['revision'] or sha256(args.tokenizer/name) != meta['sha256']:
            raise ValueError('PE tokenizer asset differs from the pinned source')
        sources[name] = meta
    out = args.output.resolve(); out.mkdir(parents=True)
    runner = out/'runner.snapshot'; shutil.copy2(args.runner, runner)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = ['A red apple on a wooden table.', ' 一只猫\r\n☃️ "quoted" \\ backslash ',
               '\u2028\u2029\t\b\f\r\n', '', '\x01\x1f café 中文', 'cat ' * 2100]
    rows = []
    for index, prompt in enumerate(prompts):
        source = out/f'prompt-{index}.txt'; source.write_bytes(prompt.encode())
        native = out/f'case-{index}'
        run = subprocess.run([str(runner), str(args.tokenizer.resolve()), str(source), '512', '384', '1',
                              str(native), 'tokenize'], capture_output=True, text=True, timeout=20)
        content = json.dumps(dict(prompt=prompt, width=512, height=384), ensure_ascii=False)
        formatted = tokenizer.apply_chat_template([dict(role='user', content=content)],
                                                 tokenize=False, add_generation_prompt=False)
        ids = tokenizer(formatted)['input_ids']
        if len(ids) > 2048:
            row = dict(case=index, tokens=len(ids), over_capacity=True, return_code=run.returncode,
                       passed=run.returncode != 0 and 'capacity' in run.stderr)
        else:
            exact = dict(format=(native/'formatted.txt').is_file() and (native/'formatted.txt').read_bytes()==formatted.encode(),
                         ids=(native/'input-ids.txt').is_file() and list(map(int, (native/'input-ids.txt').read_text().split()))==ids,
                         decode=(native/'decoded.txt').is_file() and (native/'decoded.txt').read_bytes()==tokenizer.decode(ids, skip_special_tokens=True).encode())
            row = dict(case=index, tokens=len(ids), return_code=run.returncode, exact=exact,
                       passed=run.returncode==0 and all(exact.values()))
        rows.append(row)
        (out/f'case-{index}.log').write_text(run.stdout+run.stderr)
    result = dict(passed=all(x['passed'] for x in rows), cases=rows, sources=sources,
                  validator_sha256=sha256(__file__), runner_sha256=sha256(runner))
    (out/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result['cases']), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__': main()
