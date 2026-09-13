#!/usr/bin/env python3
"""Check the exact uploaded inventory and emit immutable project download lists."""
import hashlib
import json
from pathlib import Path
import sys
import time
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
STAGE = Path('/var/tmp/ernie-hf-release-20260913-v1')
sys.path.insert(0, str(ROOT))
from tools.release_manifest import validate_manifest

local = json.loads((STAGE/'evidence/local-inventory.json').read_text())
uploaded = json.loads((STAGE/'evidence/storage-finalized.json').read_text())
provenance = json.loads((STAGE/'repo/provenance.json').read_text())
repo_id, revision = uploaded['repo_id'], uploaded['revision']
assert repo_id == local['repository'] == 'akashimio/ERNIE-Image-Turbo-ncnn'
api = HfApi()
info = api.model_info(repo_id, revision=revision, files_metadata=True)
assert info.sha == revision
remote = {f.rfilename: f for f in info.siblings}
expected = {f['path']: f for f in local['all_files']}
assert set(remote) == set(expected), {'missing': sorted(set(expected)-set(remote)),
                                      'extra':sorted(set(remote)-set(expected))}
checks = []
for path, item in sorted(expected.items()):
    actual = remote[path]
    assert actual.size == item['size'], (path, actual.size, item['size'])
    if actual.lfs is not None:
        checksum = actual.lfs.sha256
        assert actual.lfs.size == item['size']
        method = 'Hub LFS SHA256 and byte size'
    else:
        downloaded = Path(hf_hub_download(repo_id, path, revision=revision,
                          cache_dir=STAGE/'remote-metadata-cache'))
        with downloaded.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        method = 'Downloaded Git file SHA256 and byte size'
    assert checksum == item['sha256'], path
    checks.append(dict(path=path, size=actual.size, sha256=checksum, verification=method))
report = {'repository':repo_id,'revision':revision,'private_at_verification':info.private,
          'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'files':len(checks),'bytes':sum(x['size'] for x in checks),'all_passed':True,
          'verification_scope':'All remote file identities and sizes; LFS data is checked against the server content digest, not fully downloaded again.',
          'checks':checks}
(STAGE/'evidence/remote-inventory-check.json').write_text(json.dumps(report,indent=2)+'\n')
for name, schema, capabilities, script in (
    ('turbo','ernie-shared-weight-package-v1',
     ['ernie_schema3','ernie_custom_layers','text_buckets_32_64_2048'], 'tools/package_dynamic_model.py'),
    ('pe','ernie-prompt-enhancer-schema1',
     ['ernie_pe_schema1','cpu_fp32','native_kv_cache'], 'tools/pe_package.py')):
    manifest = {
        'schema_version':1,'model_revision':revision,'graph_schema':schema,
        'required_capabilities':capabilities,
        'conversion':{'source':f'https://huggingface.co/{repo_id}/blob/{revision}/provenance.json',
                      'script_sha256':provenance['tools_at_release_commit'][script]},
        'license':{'identifier':'Apache-2.0',
                   'source':f'https://huggingface.co/{repo_id}/blob/{revision}/LICENSE',
                   'notice':'Original model: Baidu ERNIE-Image-Turbo. Converted distribution: mingshi2333 / akashimio. See docs/models/LICENSE-ERNIE-Image and docs/models/NOTICE. Original conversion identities remain in the package; script_sha256 identifies the verifier used to prepare this release.'},
        'files':[{'path':f['path'].removeprefix(name+'/'),
                  'url':f'https://huggingface.co/{repo_id}/resolve/{revision}/{f["path"]}',
                  'size':f['size'],'sha256':f['sha256']} for f in local['packages'][name]['files']]}
    validate_manifest(manifest)
    target=ROOT/f'docs/models/{name}-v1.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'repository':repo_id,'revision':revision,'files_verified':len(checks),
                  'bytes':report['bytes'],'manifests_written':['turbo-v1.json','pe-v1.json'],
                  'private':info.private}))
