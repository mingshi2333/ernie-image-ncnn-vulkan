#!/usr/bin/env python3
"""Download the two pinned official tokenizer JSON files and record full checksums."""
import argparse
import json
from pathlib import Path
import urllib.request
from prepare_block import ROOT, sha256


def fetch(output):
    output = Path(output)
    model = json.loads((ROOT / 'sources.lock.json').read_text())['official_model']
    manifest = output / 'manifest.json'
    if output.exists():
        if manifest.is_file():
            previous = json.loads(manifest.read_text())
            if previous['revision'] == model['revision'] and all(sha256(output / name) == value for name, value in previous['files'].items()):
                return previous
        raise ValueError('Existing tokenizer directory has no matching verified manifest')
    output.mkdir(parents=True)
    files = {}
    for name in ['tokenizer_config.json', 'tokenizer.json']:
        url = f"{model['url']}/resolve/{model['revision']}/tokenizer/{name}"
        temporary = output / (name + '.partial')
        total = 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as target:
            while data := response.read(1024 * 1024):
                total += len(data)
                if total > 64 * 1024 * 1024:
                    raise ValueError('Unexpectedly large tokenizer JSON')
                target.write(data)
        json.loads(temporary.read_text())
        temporary.rename(output / name)
        files[name] = sha256(output / name)
    result = {'repository': model['url'], 'revision': model['revision'], 'files': files,
              'downloader_sha256': sha256(__file__)}
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/tokenizer')
    args = parser.parse_args()
    print(json.dumps(fetch(args.output)))
