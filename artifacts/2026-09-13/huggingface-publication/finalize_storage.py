#!/usr/bin/env python3
"""Align Git LFS attributes with the Hub's actual storage modes before publication."""
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, CommitOperationAdd

stage=Path('/var/tmp/ernie-hf-release-20260913-v1')
complete=json.loads((stage/'evidence/upload-complete.json').read_text())
api=HfApi()
info=api.model_info(complete['repo_id'],revision=complete['revision'],files_metadata=True)
assert info.private and info.sha==complete['revision']
assert api.model_info(complete['repo_id']).sha==info.sha
lfs_paths=sorted(f.rfilename for f in info.siblings if f.lfs is not None)
attributes='* -text\n'+''.join(f'{p} filter=lfs diff=lfs merge=lfs -text\n' for p in lfs_paths)
payload=stage/'repo'
for name in ['.gitattributes','provenance.json','files.json']:
    shutil.copyfile(payload/name,stage/'evidence'/('initial-'+name.lstrip('.')))
shutil.copyfile(stage/'evidence/local-inventory.json',stage/'evidence/initial-local-inventory.json')
(payload/'.gitattributes').write_text(attributes)
provenance=json.loads((payload/'provenance.json').read_text())
provenance['storage']={'backend':'Hugging Face Hub / Xet and Git',
    'lfs_file_count':len(lfs_paths),'attributes':'Exact paths mirror the uploaded storage modes, including shared objects and optional PE data.',
    'weights_and_package_manifests_changed':False}
(payload/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')

def entry(name):
    p=payload/name
    with p.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':name,'size':p.stat().st_size,'sha256':checksum}

changed={name:entry(name) for name in ['.gitattributes','provenance.json']}
inventory=json.loads((payload/'files.json').read_text())
inventory['files']=[changed.get(f['path'],f) for f in inventory['files']]
(payload/'files.json').write_text(json.dumps(inventory,indent=2)+'\n')
changed['files.json']=entry('files.json')
local=json.loads((stage/'evidence/local-inventory.json').read_text())
local['all_files']=[changed.get(f['path'],f) for f in local['all_files']]
(stage/'evidence/local-inventory.json').write_text(json.dumps(local,indent=2)+'\n')
commit=api.create_commit(repo_id=complete['repo_id'],parent_commit=info.sha,
    operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=payload/name) for name in changed],
    commit_message='Record exact model file storage and release provenance')
result={**complete,'previous_revision':complete['revision'],'revision':commit.oid,
        'changed_files':list(changed),'weights_and_package_manifests_changed':False,'lfs_file_count':len(lfs_paths)}
(stage/'evidence/storage-finalized.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
