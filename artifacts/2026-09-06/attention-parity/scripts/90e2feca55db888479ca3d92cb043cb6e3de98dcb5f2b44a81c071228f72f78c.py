#!/usr/bin/env python3
"""Require exact native token IDs against the official Transformers tokenizer call."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import tokenizers
from transformers import AutoTokenizer
from prepare_block import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/tokenizer')
    parser.add_argument('--runner', type=Path, default=ROOT / 'build-tokenizer/ernie-tokenize')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    manifest = json.loads((args.model / 'manifest.json').read_text())
    if manifest['revision'] != json.loads((ROOT / 'sources.lock.json').read_text())['official_model']['revision']:
        raise ValueError('Tokenizer revision differs')
    for name, value in manifest['files'].items():
        if sha256(args.model / name) != value:
            raise ValueError('Tokenizer checksum mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = [('', 'empty'), (' ', 'one-space'), (' \t\n\r\n ', 'whitespace'),
             ('一只橘猫坐在窗边，柔和的晨光。', 'chinese'),
             ('A red fox under the northern lights.', 'english'),
             ('白い猫が桜の木の下で眠っている。', 'japanese'),
             ('Красная площадь зимой, мягкий вечерний свет.', 'russian'),
             ('مدينة مضاءة تحت السماء الزرقاء', 'arabic'),
             ('👩🏽‍🚀 🌌 🐈‍⬛ ❤️', 'emoji'), ('café cafe\u0301 ＡＢＣ　abc', 'unicode-forms'),
             ('<s>hello</s><pad><unk>', 'special-tokens'), ('A\x00B', 'nul'),
             ('0 12345 -0.25 1e-6 // \\ # [] {}', 'symbols'),
             ('猫 cat кошка 猫 cat кошка ' * 500, 'long-truncation'),
             ('word ' * 2046, 'near-limit'), ('word ' * 2048, 'over-limit')]
    # Search for actual counterexamples, without claiming flag-specific
    # differential coverage if this vocabulary sample has none.
    raw = json.loads((args.model / 'tokenizer.json').read_text())
    if raw['model'].get('ignore_merges') is not True:
        raise ValueError('Expected the reviewed ignore_merges tokenizer')
    candidates = [word for word in raw['model']['vocab'] if word.isascii() and word.isalpha() and 4 <= len(word) <= 20]
    random.Random(20260905).shuffle(candidates)
    cases.extend((word, f'vocabulary-{i}') for i, word in enumerate(candidates[:32]))
    raw['model']['ignore_merges'] = False
    incorrect = tokenizers.Tokenizer.from_str(json.dumps(raw))
    counterexamples = []
    for word in candidates[:10000]:
        actual = tokenizer(word, add_special_tokens=True, truncation=True, padding=False)['input_ids']
        if actual != incorrect.encode(word, add_special_tokens=True).ids:
            counterexamples.append(word)
            cases.append((word, f'ignore-merges-{len(counterexamples)}'))
            if len(counterexamples) == 8:
                break
    args.output.mkdir(parents=True)
    fixtures = []
    for index, (prompt, name) in enumerate(cases):
        file = args.output / f'prompt-{index:02d}.txt'
        file.write_bytes(prompt.encode('utf-8'))
        ids = tokenizer(prompt, add_special_tokens=True, truncation=True, padding=False)['input_ids']
        if not ids:
            ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0]
        fixtures.append({'name': name, 'file': file.name, 'sha256': sha256(file), 'ids': ids})
    fixture = {'scope': 'Exact tokenizer IDs only; no text encoder execution', 'model_manifest_sha256': sha256(args.model / 'manifest.json'),
               'python_tokenizers_version': tokenizers.__version__, 'generator_sha256': sha256(__file__),
               'ignore_merges': {'preserved_in_json': True, 'scanned_candidates': min(len(candidates), 10000),
                                 'counterexamples': counterexamples,
                                 'scope': 'No flag-specific differential coverage when counterexamples is empty'}, 'cases': fixtures}
    (args.output / 'fixture.json').write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + '\n')
    command = [str(args.runner.resolve()), str(args.model.resolve()), *(str((args.output / entry['file']).resolve()) for entry in fixtures)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (args.output / 'runner.log').write_text(run.stdout + run.stderr)
    result = {'command': command, 'runner_sha256': sha256(args.runner), 'fixture_sha256': sha256(args.output / 'fixture.json'),
              'return_code': run.returncode, 'passed': False, 'cases': []}
    if run.returncode == 0:
        actual = [json.loads(line) for line in run.stdout.splitlines()]
        for index, entry in enumerate(fixtures):
            ids = actual[index] if index < len(actual) else None
            result['cases'].append({'name': entry['name'], 'tokens': len(entry['ids']), 'passed': ids == entry['ids']})
        result['passed'] = len(actual) == len(fixtures) and all(row['passed'] for row in result['cases'])
    invalid = args.output / 'invalid-utf8.txt'
    invalid.write_bytes(b'\xff\xfe')
    rejected = subprocess.run([str(args.runner.resolve()), str(args.model.resolve()), str(invalid.resolve())],
                              capture_output=True, text=True, timeout=60)
    (args.output / 'invalid-utf8.log').write_text(rejected.stdout + rejected.stderr)
    result['rejects_invalid_utf8'] = rejected.returncode != 0
    result['passed'] &= result['rejects_invalid_utf8']
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
