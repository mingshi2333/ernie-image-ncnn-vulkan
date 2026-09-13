import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
stage = Path('/var/tmp/ernie-hf-release-20260913-v1')
sys.path.insert(0, str(root))
from tools.download_model import acquire, download, SafeRedirect

report = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'anonymous': True, 'checks': []}
for package in ('turbo', 'pe'):
    manifest = json.loads((root/f'docs/models/{package}-v1.json').read_text())
    small = copy.deepcopy(manifest)
    small['files'] = [f for f in manifest['files'] if f['size'] < 100000][:2]
    if package == 'pe':
        small['files'].append(next(f for f in manifest['files'] if f['path'] == 'tokenizer/tokenizer.json'))
    result = download(small, stage/f'public-download-sample/{package}', timeout=60, emit=lambda _: None)
    report['checks'].append({'kind': 'fresh_anonymous_download', 'package': package, **result})
    print(f'{package}: anonymous sample download and SHA256 passed', flush=True)
    result = acquire(manifest, stage/f'repo/{package}',
                     verifier=stage/'evidence/ernie-image' if package == 'turbo' else None,
                     emit=lambda _: None)
    report['checks'].append({'kind': 'complete_manifest_existing_local_files', 'package': package, **result})
    print(f'{package}: complete immutable manifest verified against existing local bytes', flush=True)

largest = max(manifest['files'], key=lambda x: x['size'])
opener = urllib.request.build_opener(SafeRedirect())
for start in (0, largest['size'] - 1024*1024):
    end = start + 1024*1024 - 1
    req = urllib.request.Request(largest['url'], headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with opener.open(req, timeout=60) as response:
        assert response.status == 206
        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{largest["size"]}'
        remote = response.read(1024*1024 + 1)
    with (stage/'repo/pe'/largest['path']).open('rb') as stream:
        stream.seek(start)
        local = stream.read(1024*1024)
    assert remote == local
    report['checks'].append({'kind': 'anonymous_range_download', 'package': 'pe',
        'path': largest['path'], 'offset': start, 'bytes': len(remote), 'http_status': 206,
        'sha256': hashlib.sha256(remote).hexdigest(), 'matches_local': True})
report['all_passed'] = True
report['scope'] = 'Fresh anonymous sample downloads, first/last 1 MiB of the largest weight, and complete immutable manifests against already verified local packages. The entire 31 GB release was not downloaded a second time.'
(stage/'evidence/download-checks.json').write_text(json.dumps(report, indent=2)+'\n')
print('All publication download checks passed.', flush=True)
