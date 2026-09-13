"""Publish verified CI runtime archives alongside the existing immutable models."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

STAGE = Path(__file__).resolve().parent
ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
REPO = 'akashimio/ERNIE-Image-Turbo-ncnn'
BASE = '9924de97ebe85c540ce09e207142a6efee614be7'
SOURCE = 'd2fd973f7658de5162e8edfee13171bb2e6c2e15'
RUN = '34772511807'
payload, evidence = STAGE/'repo', STAGE/'evidence'
api = HfApi()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def entry(path):
    return {'path': path.relative_to(payload).as_posix(), 'size': path.stat().st_size, 'sha256': digest(path)}

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

assert not (evidence/'publication.json').exists(), 'This publication already completed; inspect the receipt before retrying.'
run = json.loads(subprocess.check_output(['gh','run','view',RUN,'--repo','mingshi2333/ernie-image-ncnn-vulkan',
                                          '--json','databaseId,status,conclusion,headSha,jobs,url'], env={**os.environ,'GOMAXPROCS':'2'}))
assert run['headSha'] == SOURCE and run['status'] == 'completed' and run['conclusion'] == 'success', run
assert len(run['jobs']) == 5 and all(j['conclusion'] == 'success' for j in run['jobs'])
save(evidence/'ci-run.json',run)
platforms = []
for platform in ('linux-x86_64', 'windows-x86_64', 'macos-arm64'):
    name = 'ernie-image-' + platform
    archive = payload/'runtime'/(name+'.zip')
    sha = payload/'runtime'/(name+'.sha256')
    assert sha.read_text().split()[0] == digest(archive)
    with zipfile.ZipFile(archive) as zipped:
        assert not zipped.testzip()
        manifest = json.loads(zipped.read(name+'/build-info/runtime.json'))
        assert manifest['platform'] == platform and manifest['source_revision'] == SOURCE
        assert manifest['ci_run'] == RUN
        inventory = json.loads(zipped.read(name+'/files.json'))
        assert set(zipped.namelist()) == {name+'/'+p for p in inventory} | {name+'/files.json'}
        for p, identity in inventory.items():
            content = zipped.read(name+'/'+p)
            assert len(content)==identity['size'] and hashlib.sha256(content).hexdigest()==identity['sha256'], p
        assert 'Download' in zipped.read(name+'/build-info/help.txt').decode() or 'Launcher options' in zipped.read(name+'/build-info/help.txt').decode()
        platforms.append({**entry(archive), 'platform':platform, 'source_revision':SOURCE,
                          'ncnn_revision':manifest['ncnn_revision'], 'native_tests':manifest['tests'],
                          'files':len(inventory)+1,'dependencies':manifest['dependencies'],
                          'archive_file_checksums_verified':True})

before = api.model_info(REPO, files_metadata=True)
assert not before.private and before.sha==BASE, (before.private, before.sha)
old_payload = Path('/var/tmp/ernie-hf-release-20260913-v1/repo')
old_inventory = json.loads((old_payload/'files.json').read_text())
for name in ('files.json','provenance.json','.gitattributes','README.md'):
    downloaded = Path(hf_hub_download(REPO,name,revision=BASE,token=False,cache_dir=STAGE/'metadata-cache'))
    assert digest(downloaded)==digest(old_payload/name), name

runtime_index = {'schema_version':1,'source_revision':SOURCE,'ci_run':run['url'],
                 'models_revision':BASE,'platforms':platforms,
                 'usage':'Extract one runtime ZIP. python3 run.py --prompt TEXT (Windows: python). Model weights download separately on first use.'}
save(payload/'runtime/manifest.json', runtime_index)
provenance = json.loads((old_payload/'provenance.json').read_text())
provenance['runtime_release'] = {'source_revision':SOURCE,'ci_run':run['url'],
    'manifest':'runtime/manifest.json','weights_and_package_manifests_changed':False,
    'verification':'Native framework CI, relocated launcher checks, ZIP file checksums; Linux real-model smoke recorded separately.'}
old_lfs = [f.rfilename for f in before.siblings if f.lfs]
attributes = '* -text\n' + ''.join(f'{p} filter=lfs diff=lfs merge=lfs -text\n' for p in sorted(old_lfs + [p['path'] for p in platforms]))
(payload/'.gitattributes').write_text(attributes)
provenance['storage']['lfs_file_count'] += len(platforms)
save(payload/'provenance.json',provenance)
# The model card is maintained in Git, then copied byte-for-byte to HF.
assert (payload/'README.md').read_bytes() == (ROOT/'docs/models/MODEL-CARD.md').read_bytes()
changed = {p.relative_to(payload).as_posix(): entry(p) for p in payload.rglob('*') if p.is_file() and p.name!='files.json'}
merged = {item['path']:item for item in old_inventory['files']}
merged.update(changed)
save(payload/'files.json',{'schema_version':1,'scope':old_inventory['scope'],'files':[merged[k] for k in sorted(merged)]})
paths = sorted([*changed,'files.json'])
save(evidence/'publication-plan.json',{'repository':REPO,'parent_revision':BASE,'source_revision':SOURCE,'ci_run':RUN,
    'files':[entry(payload/p) for p in paths],'unchanged_model_files':150,'runtime_index':runtime_index})
commit = api.create_commit(repo_id=REPO, parent_commit=BASE,
    operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=payload/p) for p in paths],
    commit_message='Publish native runtime downloads and first-run model launcher')
save(evidence/'upload-commit.json',{'repository':REPO,'revision':commit.oid,'files':paths})
# Match .gitattributes to the Hub's actual storage classification.
info = api.model_info(REPO, revision=commit.oid, files_metadata=True)
lfs_paths = sorted(f.rfilename for f in info.siblings if f.lfs)
actual_attributes = '* -text\n' + ''.join(f'{p} filter=lfs diff=lfs merge=lfs -text\n' for p in lfs_paths)
if actual_attributes != attributes:
    (payload/'.gitattributes').write_text(actual_attributes)
    provenance['storage']['lfs_file_count'] = len(lfs_paths)
    save(payload/'provenance.json',provenance)
    for p in ('.gitattributes','provenance.json'):
        merged[p] = entry(payload/p)
    save(payload/'files.json',{'schema_version':1,'scope':old_inventory['scope'],'files':[merged[k] for k in sorted(merged)]})
    commit = api.create_commit(repo_id=REPO,parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=payload/p) for p in ('.gitattributes','provenance.json','files.json')],
        commit_message='Align runtime archive storage inventory')
    info = api.model_info(REPO, revision=commit.oid, files_metadata=True)
assert not info.private
expected = {**merged,'files.json':entry(payload/'files.json')}
actual = {f.rfilename:f for f in info.siblings}
assert set(expected)==set(actual)

def check(path):
    item,remote = expected[path],actual[path]
    assert remote.size == item['size'],path
    # Actually download every new runtime and small metadata file anonymously.
    if remote.lfs and not path.startswith('runtime/'):
        sha=remote.lfs.sha256
        method='Hub LFS SHA256 and size'
    else:
        local=Path(hf_hub_download(REPO,path,revision=info.sha,token=False,cache_dir=STAGE/'public-cache'))
        sha=digest(local)
        method='Anonymous download and SHA256'
    assert sha == item['sha256'], path
    return {**item,'verification':method}
with ThreadPoolExecutor(max_workers=6) as pool:
    checked=list(pool.map(check,sorted(expected)))
receipt={'repository':REPO,'url':'https://huggingface.co/'+REPO,'revision':info.sha,'previous_revision':BASE,
         'published_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'private':info.private,
         'source_revision':SOURCE,'ci_run':run['url'],'files':len(checked),
         'model_files_unchanged':True,'all_file_identities_verified':True,
         'runtime_archives_anonymously_downloaded_and_hashed':True,'platforms':platforms}
save(evidence/'remote-inventory.json',{'revision':info.sha,'checks':checked})
save(evidence/'publication.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
