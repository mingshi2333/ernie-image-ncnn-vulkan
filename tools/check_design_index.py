#!/usr/bin/env python3
"""Check local links and reviewed source identities in the small design index."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root).resolve()
    index = root / 'docs/DESIGN-INDEX.md'
    source = json.loads((root / 'docs/design-sources.json').read_text())
    errors = []
    for name, expected in source['local_files'].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            errors.append('Missing source: ' + name)
            continue
        # Canonical text hashes also work with Git CRLF checkouts on Windows.
        actual = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        if actual != expected:
            errors.append('Source changed; reconcile the design and evidence: ' + name)
    links = 0
    for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', index.read_text(encoding='utf-8')):
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        links += 1
        path = (index.parent / unquote(url.path)).resolve()
        if root not in path.parents or not path.is_file():
            errors.append('Broken local design link: ' + target)
    return {'passed': not errors, 'sources_checked': len(source['local_files']),
            'local_links_checked': links, 'errors': errors}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
