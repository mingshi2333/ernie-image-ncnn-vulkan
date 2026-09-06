#!/usr/bin/env python3
"""Check native PE formatting, exact token IDs and decoding against official assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from transformers import AutoTokenizer
from prepare_block import ROOT, sha256


DEVELOPMENT_CASES = [
    dict(id='pe-dev-00-en', prompt='A red apple on a wooden table.', width=512, height=384, max_tokens=64),
    dict(id='pe-dev-01-zh', prompt='一只白猫坐在写有“春天”的蓝色门牌旁。', width=512, height=512, max_tokens=96),
    dict(id='pe-dev-02-ja', prompt='駅の案内板に「北口」と書かれ、赤い傘が二本ある。', width=512, height=512, max_tokens=96),
    dict(id='pe-dev-03-quotes', prompt='Render exactly: "quoted", \'single\', and a \\ backslash.', width=640, height=384, max_tokens=128),
    dict(id='pe-dev-04-newline', prompt='first line\nsecond line\r\n第三行', width=512, height=384, max_tokens=128),
    dict(id='pe-dev-05-whitespace', prompt=' \t  leading and trailing ideographic space\u3000 ', width=512, height=384, max_tokens=64),
    dict(id='pe-dev-06-empty', prompt='', width=512, height=384, max_tokens=32),
    dict(id='pe-dev-07-control', prompt='\x01\x1f café 中文 ☃️', width=512, height=384, max_tokens=64),
    dict(id='pe-dev-08-wide', prompt='A bilingual information poster with four aligned panels.', width=1376, height=768, max_tokens=256),
    dict(id='pe-dev-09-portrait', prompt='A low contrast portrait under soft window light.', width=768, height=1376, max_tokens=256),
    dict(id='pe-dev-10-long-output', prompt='Describe a complex overhead room layout with exact object relations.', width=1024, height=1024, max_tokens=2048),
]


def _formatted_ids(tokenizer, prompt, width, height):
    content = json.dumps(dict(prompt=prompt, width=width, height=height), ensure_ascii=False)
    formatted = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                              add_generation_prompt=False)
    return formatted, tokenizer(formatted)['input_ids']


def _near_capacity_prompt(tokenizer):
    """Return deterministic complete text whose formatted input is as close as possible below 2048."""
    low, high = 1, 4096
    old_limit = getattr(tokenizer, 'model_max_length', 2048)
    tokenizer.model_max_length = max(old_limit, 10000)
    try:
        while low < high:
            middle = (low + high + 1) // 2
            if len(_formatted_ids(tokenizer, 'cat ' * middle, 512, 384)[1]) <= 2048:
                low = middle
            else:
                high = middle - 1
    finally:
        tokenizer.model_max_length = old_limit
    return 'cat ' * low


def input_ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def build_development_batch(tokenizer, tokenizer_sources=None):
    specs = DEVELOPMENT_CASES + [dict(id='pe-dev-11-near-capacity', prompt=_near_capacity_prompt(tokenizer),
                                      width=512, height=384, max_tokens=32)]
    cases = []
    for spec in specs:
        _, ids = _formatted_ids(tokenizer, spec['prompt'], spec['width'], spec['height'])
        if not ids or len(ids) > 2048:
            raise ValueError(f"{spec['id']} does not satisfy the PE input capacity contract")
        cases.append(dict(spec, mode='greedy', temperature=0.0, top_p=1.0,
                          input_tokens=len(ids), input_ids_sha256=input_ids_sha256(ids),
                          acceptance_status='pending_real_official_and_native_pe'))
    manifest = dict(schema_version=1, suite='pe-development-v1', case_count=12,
                    status='inputs_frozen_model_acceptance_pending',
                    tokenizer_sources=tokenizer_sources or {'status':'unit_fixture'}, cases=cases)
    manifest['manifest_sha256'] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    validate_development_batch(manifest, tokenizer)
    return manifest


def validate_development_batch(manifest, tokenizer):
    unsigned = dict(manifest)
    expected_hash = unsigned.pop('manifest_sha256', None)
    actual_hash = hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError('PE development batch manifest checksum differs')
    if manifest.get('schema_version') != 1 or manifest.get('suite') != 'pe-development-v1':
        raise ValueError('Unsupported PE development batch schema')
    if not isinstance(manifest.get('tokenizer_sources'), dict) or not manifest['tokenizer_sources']:
        raise ValueError('PE tokenizer source identity is required')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or manifest.get('case_count') != 12 or len(cases) != 12:
        raise ValueError('PE development batch must contain exactly 12 cases')
    required = {'id','prompt','width','height','max_tokens','mode','temperature','top_p',
                'input_tokens','input_ids_sha256','acceptance_status'}
    if len({case.get('id') for case in cases}) != 12:
        raise ValueError('PE development case IDs must be unique')
    for case in cases:
        if set(case) != required:
            raise ValueError('PE development case fields differ')
        if type(case['width']) is not int or type(case['height']) is not int or min(case['width'],case['height']) < 16 or case['width'] % 16 or case['height'] % 16:
            raise ValueError('Invalid PE development dimensions')
        if type(case['max_tokens']) is not int or not 1 <= case['max_tokens'] <= 2048:
            raise ValueError('Invalid PE development output limit')
        if case['mode'] != 'greedy' or type(case['temperature']) not in (int,float) or case['temperature'] != 0 or type(case['top_p']) not in (int,float) or case['top_p'] != 1:
            raise ValueError('PE development batch must use explicit greedy settings')
        _, ids = _formatted_ids(tokenizer, case['prompt'], case['width'], case['height'])
        if not ids or len(ids) > 2048 or case['input_tokens'] != len(ids) or case['input_ids_sha256'] != input_ids_sha256(ids):
            raise ValueError('PE development input identity differs')
        if case['acceptance_status'] != 'pending_real_official_and_native_pe':
            raise ValueError('PE model acceptance must remain explicitly pending')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, default=ROOT/'models/pe-tokenizer')
    p.add_argument('--runner', type=Path, default=ROOT/'build/ernie-pe-runner')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-contract', action='store_true', help='Freeze the 12-case input contract only')
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.batch_contract:
        manifest = build_development_batch(tokenizer, sources)
        (out/'pe-development-batch.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({'batch_contract_complete': True, 'case_count': 12,
                          'model_acceptance': 'pending'}, ensure_ascii=False), flush=True)
        return
    runner = out/'runner.snapshot'; shutil.copy2(args.runner, runner)
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
