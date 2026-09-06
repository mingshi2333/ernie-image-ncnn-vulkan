#!/usr/bin/env python3
"""Build/verify an OFFLINE shared-weight candidate, not a runnable schema-3 package.

contract.json deliberately replaces manifest.json. Native runtime validation still
rejects schema 3. Only two pinned portable static instances may be represented;
schema-1 migration and new dynamic shapes remain pending independent evidence.
Encoder components are intentionally absent. Adding encoder inventory, posterior/packing,
and separate encoder/decoder BN epsilon identities requires a new format revision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
try:
    from audit_shape_contract import audit_package, normalize_graph, CONTRACT_HASHES, PINNED_MANIFESTS, graph_files
    from package_model import verify_package, runtime_files, sha256
except ImportError:
    from tools.audit_shape_contract import audit_package, normalize_graph, CONTRACT_HASHES, PINNED_MANIFESTS, graph_files
    from tools.package_model import verify_package, runtime_files, sha256

FORMAT='ernie-shared-weights-offline-candidate-v1'
POLICY={'candidate_schema':3,'runtime_supported':False,'quality_status':'pending',
        'scope':'pinned static instances only; no dynamic quality or runtime acceptance'}
KINDS=dict(graph_files())


def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('Duplicate JSON key')
        out[key]=value
    return out


def read_json(path):
    if path.stat().st_size>4*1024*1024:raise ValueError('Metadata exceeds bounded size')
    return json.loads(path.read_text(),object_pairs_hook=unique)


def digest_name(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Invalid object SHA256')
    return value


def verify_object(root, digest, expected_size=None):
    digest=digest_name(digest);objects=root/'objects';path=objects/digest
    if objects.is_symlink() or path.is_symlink() or not path.is_file():raise ValueError('Missing/nonportable object')
    if expected_size is not None and (type(expected_size) is not int or expected_size<0 or path.stat().st_size!=expected_size):raise ValueError('Object size mismatch')
    if sha256(path)!=digest:raise ValueError('Object checksum mismatch')
    return path


def verify_graph(path,kind,config):
    if path.stat().st_size>1024*1024:raise ValueError('Graph exceeds bounded size')
    canonical,_=normalize_graph(path.read_text(),kind,config)
    if canonical!=CONTRACT_HASHES[kind]:raise ValueError('Unknown complete graph hash')


def verify_candidate(root):
    root=Path(root)
    if (root/'manifest.json').exists():raise ValueError('Offline candidate must not masquerade as a runtime package')
    if (root/'contract.json').is_symlink():raise ValueError('Nonportable contract')
    c=read_json(root/'contract.json')
    if (set(c)!={'format','policy','instances','objects'} or c['format']!=FORMAT or c['policy']!=POLICY
            or not isinstance(c['policy'],dict) or any(type(c['policy'][k]) is not type(v) for k,v in POLICY.items())):
        raise ValueError('Invalid offline candidate policy/schema')
    if not isinstance(c['instances'],list) or not 1<=len(c['instances'])<=2 or not isinstance(c['objects'],dict):raise ValueError('Invalid instance/object inventory')
    seen=set();used=set();verified=set()
    def obj(digest):
        digest_name(digest)
        if digest not in c['objects']:raise ValueError('Missing declared object')
        size=c['objects'][digest]
        if type(size) is not int or size<0:raise ValueError('Invalid object size')
        if digest not in verified:verify_object(root,digest,size);verified.add(digest)
        used.add(digest);return root/'objects'/digest
    for instance in c['instances']:
        if not isinstance(instance,dict) or set(instance)!={'source_manifest_sha256','config','runtime_bindings'}:raise ValueError('Invalid instance fields')
        digest=digest_name(instance['source_manifest_sha256'])
        if digest not in PINNED_MANIFESTS or digest in seen:raise ValueError('Unknown/duplicate pinned instance')
        seen.add(digest);m=read_json(obj(digest))
        if m.get('schema_version')!=2 or m.get('portable') is not True:raise ValueError('Only existing portable schema-2 inputs are supported')
        # The full source manifest hash is pinned; its config, revisions, inventory and sizes cannot be redefined.
        if (not isinstance(instance['config'],dict) or any(type(v) is not int for v in instance['config'].values())
                or instance['config']!=m['config'] or set(m['files'])!=set(runtime_files())):raise ValueError('Static metadata mismatch')
        bindings=instance['runtime_bindings']
        if not isinstance(bindings,dict) or bindings!=m['files']:raise ValueError('Missing/changed complete runtime bindings')
        for name,d in bindings.items():
            path=obj(d)
            if type(m['file_sizes'][name]) is not int or path.stat().st_size!=m['file_sizes'][name]:raise ValueError('Pinned runtime size mismatch')
            if name in KINDS:verify_graph(path,KINDS[name],m['config'])
            if name=='model.cfg':
                words=path.read_text().split()
                if len(words)!=12 or len(set(words[::2]))!=6 or dict(zip(words[::2],map(int,words[1::2])))!=m['config']:raise ValueError('Static model.cfg mismatch')
    if used!=set(c['objects']):raise ValueError('Unbound objects in candidate inventory')
    # Object store is entirely sealed. Reports belong beside, not inside, this package candidate.
    if {p.name for p in (root/'objects').iterdir()}!=used:raise ValueError('Unlisted object files')
    if {p.name for p in root.iterdir()}!={'contract.json','objects'}:raise ValueError('Unlisted candidate files')
    return c


def build_candidate(sources,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new candidate directory')
    if not 1<=len(sources)<=2:raise ValueError('Use one or two pinned portable static instances')
    inputs=[];seen=set()
    for source in map(Path,sources):
        m=read_json(source/'manifest.json');digest=sha256(source/'manifest.json')
        if digest not in PINNED_MANIFESTS:raise ValueError('Unknown static source manifest')
        if m.get('schema_version')!=2 or m.get('portable') is not True:raise ValueError('Schema-1 migration remains pending')
        audit_package(source)
        m,files=verify_package(source)
        if digest in seen:raise ValueError('Duplicate source instance')
        seen.add(digest);inputs.append((source,m,files,digest))
    output.mkdir();(output/'objects').mkdir();objects={};instances=[]
    def copy(path,digest):
        digest_name(digest)
        if digest not in objects:
            target=output/'objects'/digest;shutil.copyfile(path,target)
            if sha256(target)!=digest:raise ValueError('Source changed during copy')
            objects[digest]=target.stat().st_size
        return digest
    for source,m,files,digest in inputs:
        copy(source/'manifest.json',digest)
        bindings={name:copy(source/name,d) for name,d in files.items()}
        instances.append(dict(source_manifest_sha256=digest,config=m['config'],runtime_bindings=bindings))
    c=dict(format=FORMAT,policy=POLICY,instances=instances,objects=objects)
    (output/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
    return verify_candidate(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,action='append');p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify and not a.source and not a.output:c=verify_candidate(a.verify)
    elif a.source and a.output and not a.verify:c=build_candidate(a.source,a.output)
    else:p.error('Use --source/--output or --verify')
    print(json.dumps({'offline_candidate_verified':True,'runtime_supported':False,'quality_status':'pending','instances':len(c['instances']),'shared_objects':len(c['objects'])}))
if __name__=='__main__':main()
