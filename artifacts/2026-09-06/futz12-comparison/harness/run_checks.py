"""Reproduce source-layout and tokenizer comparisons without loading neural weights."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = OUT / 'source'
META = OUT / 'model-metadata'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = OUT / 'local-checks-v2.json'
    if output.exists():
        raise ValueError('Preserve the previous check result')
    for filename, base in [('source-manifest.json', SOURCE), ('model-metadata-manifest.json', META)]:
        manifest = json.loads((OUT / filename).read_text())
        for entry in manifest['files']:
            assert sha(base / entry['path']) == entry['sha256'], entry['path']

    configure_command = ['cmake', '-S', str(SOURCE), '-B', str(OUT / 'readme-build-v2'), '-DCMAKE_BUILD_TYPE=Release']
    configure = subprocess.run(configure_command, capture_output=True, text=True, timeout=30)
    (OUT / 'readme-build-v2.log').write_text(configure.stdout + configure.stderr)
    assert configure.returncode != 0 and 'does not appear to contain CMakeLists.txt' in configure.stderr

    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/tokenizer', local_files_only=True)
    fixture_root = ROOT / 'outputs/tokenizer-v2'
    fixture = json.loads((fixture_root / 'fixture.json').read_text())
    cases = []
    for entry in fixture['cases']:
        path = fixture_root / entry['file']
        assert sha(path) == entry['sha256']
        cases.append((entry['name'], path, 'existing_48_cases'))
    cases += [(path.name, path, 'published_prompt') for path in sorted((SOURCE / 'assets').glob('prompt*.txt'))]
    paths = [str(path) for _, path, _ in cases]
    commands = {
        'ours': [str(ROOT / 'build-tokenizer/ernie-tokenize'), str(ROOT / 'models/tokenizer'), *paths],
        'futz12': [str(OUT / 'futz12-tokenize'), str(META / 'text_encoder/tokenizer/vocab.txt'),
                   str(META / 'text_encoder/tokenizer/merges.txt'), *paths],
    }
    actual = {}
    for label, command in commands.items():
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (OUT / (label + '-tokenizer-v2.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, label
        actual[label] = [json.loads(line) for line in run.stdout.splitlines()]
        assert len(actual[label]) == len(cases)
    rows = []
    for index, (name, path, group) in enumerate(cases):
        text = path.read_bytes().decode('utf-8')
        expected = tokenizer(text, add_special_tokens=True, truncation=True, padding=False)['input_ids']
        matches = {label: values[index] == expected for label, values in actual.items()}
        differences = {}
        for label in actual:
            if not matches[label]:
                ids = actual[label][index]
                first = next((i for i, (a, b) in enumerate(zip(ids, expected)) if a != b), min(len(ids), len(expected)))
                differences[label] = {'first_different_index': first, 'actual_tokens': len(ids),
                                      'actual_window': ids[max(first-2, 0):first+6],
                                      'official_window': expected[max(first-2, 0):first+6]}
        rows.append({'name': name, 'group': group, 'file': str(path.relative_to(ROOT)),
                     'sha256': sha(path), 'official_tokens_including_bos': len(expected),
                     'fits_current_64_token_package': len(expected) <= 64,
                     'ids_exact': matches, 'differences': differences})

    official = json.loads((ROOT / 'models/tokenizer/tokenizer.json').read_text())
    vocab = [line.rstrip('\r') for line in (META / 'text_encoder/tokenizer/vocab.txt').read_text().split('\n') if line]
    missing = [(token, token_id) for token, token_id in official['model']['vocab'].items()
               if token_id >= len(vocab) or vocab[token_id] != token]
    graph_layers = {}
    for path in sorted(META.glob('*/*.param')):
        lines = path.read_text().splitlines()
        graph_layers[str(path.relative_to(META))] = dict(Counter(line.split()[0] for line in lines[2:] if line.split()))
    result = {
        'scope': 'Source layout and isolated tokenizer execution only; no full reference build, neural weights or image generation',
        'source_revision': '8dcd6e4411137d8abe92c9d78581c4c96d5182c6',
        'readme_configure': {'command': configure_command, 'return_code': configure.returncode,
                             'root_cmake_missing_in_complete_git_tree': True},
        'model_vocabulary': {'official_base_entries': len(official['model']['vocab']), 'futz12_entries': len(vocab),
                             'different_base_entries': len(missing), 'first_differences': missing[:3]},
        'tokenizer': {'cases': len(rows), 'exact': {label: sum(row['ids_exact'][label] for row in rows) for label in commands},
                      'results': rows, 'commands': commands, 'binary_sha256': {label: sha(Path(command[0])) for label, command in commands.items()}},
        'graph_layers': graph_layers,
        'evidence_sha256': {name: sha(OUT / name) for name in ['run_checks.py', 'tokenizer_comparison.cpp',
            'source-manifest.json', 'model-metadata-manifest.json', 'readme-build-v2.log',
            'ours-tokenizer-v2.log', 'futz12-tokenizer-v2.log']},
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'configure': configure.returncode, 'vocabulary': result['model_vocabulary'],
                      'tokenizer_exact': result['tokenizer']['exact'], 'cases': len(rows),
                      'different_cases': [row for row in rows if not all(row['ids_exact'].values())]}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
