#!/usr/bin/env python3
"""Build and verify immutable shared-weight packages from pinned static exports.

--schema3 enables the native shared-object format, with up to three independent
text templates. Runtime shape planning belongs to the native program; package
verification alone is not evidence that every shape has passed image comparison.
Optional encoder components require their own reviewed target-size evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
try:
    from audit_shape_contract import audit_package, normalize_graph, CONTRACT_HASHES, PINNED_MANIFESTS, graph_files
    from package_model import ROOT, verify_package, runtime_files, sha256
except ImportError:
    from tools.audit_shape_contract import audit_package, normalize_graph, CONTRACT_HASHES, PINNED_MANIFESTS, graph_files
    from tools.package_model import ROOT, verify_package, runtime_files, sha256

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


def verify_candidate(root, _runtime=None, _extra_bindings=()):
    root=Path(root)
    if _runtime is None and (root/'manifest.json').exists():raise ValueError('Offline candidate must not masquerade as a runtime package')
    if (root/'contract.json').is_symlink():raise ValueError('Nonportable contract')
    c=read_json(root/'contract.json') if _runtime is None else _runtime
    if (set(c)!={'format','policy','instances','objects'} or c['format']!=FORMAT or c['policy']!=POLICY
            or not isinstance(c['policy'],dict) or any(type(c['policy'][k]) is not type(v) for k,v in POLICY.items())):
        raise ValueError('Invalid offline candidate policy/schema')
    if not isinstance(c['instances'],list) or not 1<=len(c['instances'])<=3 or not isinstance(c['objects'],dict):raise ValueError('Invalid instance/object inventory')
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
    for digest in _extra_bindings:obj(digest)
    if used!=set(c['objects']):raise ValueError('Unbound objects in candidate inventory')
    # Object store is entirely sealed. Reports belong beside, not inside, this package candidate.
    if {p.name for p in (root/'objects').iterdir()}!=used:raise ValueError('Unlisted object files')
    if {p.name for p in root.iterdir()}!={('contract.json' if _runtime is None else 'manifest.json'),'objects'}:raise ValueError('Unlisted candidate files')
    return c


def build_candidate(sources,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new candidate directory')
    if not 1<=len(sources)<=3:raise ValueError('Use one to three pinned portable static instances')
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


def shared_contract():
    return read_json(ROOT/'tokenizer/schema3_contract.json')

def encoder_manifest(trusted):
    """Drop evidence and sizes from the on-disk runtime declaration."""
    out={k:v for k,v in trusted.items() if k not in ('files','evidence')}
    out['files']={name:item['sha256'] for name,item in trusted['files'].items()}
    return out

def reviewed_encoder(contract,source_digest,evidence):
    """Authenticate the sole reviewed encoder evidence before copying bytes."""
    trusted=contract.get('reviewed_encoders',{}).get(source_digest)
    if trusted is None:raise ValueError('No reviewed encoder for this static source')
    evidence=Path(evidence);fixture=read_json(evidence/'fixture.json');conversion=read_json(evidence/'conversion.json')
    identity=trusted['evidence']
    if sha256(evidence/'fixture.json')!=identity['official_fixture_sha256'] or sha256(evidence/'conversion.json')!=identity['conversion_sha256']:
        raise ValueError('Encoder evidence identity differs')
    if (fixture.get('width'),fixture.get('height'),fixture.get('official_revision'))!=(trusted['width'],trusted['height'],identity['official_revision']):raise ValueError('Encoder fixture scope differs')
    if fixture.get('source_manifests')!={'encoder':identity['official_encoder_manifest_sha256'],'quant':identity['official_quant_manifest_sha256'],'bn':identity['official_bn_manifest_sha256']}:raise ValueError('Encoder official source identity differs')
    if (fixture.get('posterior'),fixture.get('packing'),fixture.get('encoder_bn'),fixture.get('decoder_inverse_bn_eps'))!=(trusted['posterior'],trusted['packing'],{'eps':trusted['encoder_bn_eps'],'affine':trusted['encoder_bn_affine']},trusted['decoder_inverse_bn_eps']):raise ValueError('Encoder mathematical identity differs')
    fixed={'method':'reviewed_encoder_spatial_reshape_specialization','template_param_sha256':'75d493995616b451e51ddecc0dc352a3f200557baab3f98d742cc374ed6d0977','template_bin_sha256':trusted['files']['vae/encoder.ncnn.bin']['sha256'],'reference_fixture_sha256':identity['official_fixture_sha256']}
    if any(conversion.get(k)!=v for k,v in fixed.items()) or len(conversion.get('changes',[]))!=2:raise ValueError('Encoder specialization evidence differs')
    sources={'vae/encoder.ncnn.param':evidence/'head.ncnn.param','vae/encoder.ncnn.bin':evidence/'head.ncnn.bin'}
    for logical,path in sources.items():
        item=trusted['files'][logical]
        if not path.is_file() or path.stat().st_size!=item['size'] or sha256(path)!=item['sha256']:raise ValueError('Reviewed encoder runtime file differs')
    return trusted,sources


def verify_shared_package(root):
    """Runnable package protocol for pinned static instances; no new quality claim."""
    root=Path(root);m=read_json(root/'manifest.json');contract=shared_contract()
    if (root/'manifest.json').is_symlink() or set(m)!={'schema_version','format','instances','objects','math','encoder','generation_quality_status'}:
        raise ValueError('Invalid schema-3 metadata')
    for key in ['schema_version','format','math','generation_quality_status']:
        # JSON canonical bytes also distinguish bool from integer/float.
        if json.dumps(m[key],sort_keys=True)!=json.dumps(contract[key],sort_keys=True):raise ValueError('Unreviewed schema-3 '+key)
    if m['encoder']!=contract['encoder']:
        matches=[source for source,value in contract.get('reviewed_encoders',{}).items() if m['encoder']==encoder_manifest(value)]
        selected={instance.get('source_manifest_sha256') for instance in m.get('instances',[])}
        if len(matches)!=1 or matches[0] not in selected:raise ValueError('Unreviewed schema-3 encoder')
    for instance in m['instances']:
        pinned=contract['source_manifests'].get(instance.get('source_manifest_sha256'))
        if pinned is None or instance.get('config')!=pinned:raise ValueError('Unknown schema-3 source/config')
    offline=dict(format=FORMAT,policy=POLICY,instances=m['instances'],objects=m['objects'])
    extra=[] if m['encoder']==contract['encoder'] else list(m['encoder']['files'].values())
    verify_candidate(root,_runtime=offline,_extra_bindings=extra)
    return m


def build_shared_package(sources,output,encoder=None):
    """Reuse validated static inputs, copy each unique weight once, emit native schema3."""
    contract=shared_contract()
    for source in sources:
        if sha256(Path(source)/'manifest.json') not in contract['source_manifests']:raise ValueError('Unknown schema-3 source')
    encoder_verified=None
    if encoder is not None:
        if len(sources)!=1:raise ValueError('Reviewed encoder packages contain exactly one static instance')
        encoder_verified=reviewed_encoder(contract,sha256(Path(sources[0])/'manifest.json'),encoder)
    candidate=build_candidate(sources,output)
    m={k:contract[k] for k in ['schema_version','format','math','encoder','generation_quality_status']}
    m.update(instances=candidate['instances'],objects=candidate['objects'])
    output=Path(output)
    if encoder is not None:
        trusted,encoder_sources=encoder_verified
        for logical,path in encoder_sources.items():
            digest=trusted['files'][logical]['sha256'];target=output/'objects'/digest
            if not target.exists():shutil.copyfile(path,target)
            if target.stat().st_size!=trusted['files'][logical]['size'] or sha256(target)!=digest:raise ValueError('Encoder changed during CAS copy')
            m['objects'][digest]=target.stat().st_size
        for logical in ['vae/bn-mean.f32','vae/bn-variance.f32']:
            if candidate['instances'][0]['runtime_bindings'][logical]!=trusted['files'][logical]['sha256']:raise ValueError('Source package has different encoder BN identity')
        m['encoder']=encoder_manifest(trusted)
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (output/'contract.json').unlink()
    return verify_shared_package(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--schema3',action='store_true');p.add_argument('--source',type=Path,action='append');p.add_argument('--encoder',type=Path);p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify and not a.source and not a.output:c=(verify_shared_package if a.schema3 else verify_candidate)(a.verify)
    elif a.source and a.output and not a.verify:
        if a.encoder and not a.schema3:p.error('--encoder requires --schema3')
        c=build_shared_package(a.source,a.output,a.encoder) if a.schema3 else build_candidate(a.source,a.output)
    else:p.error('Use --source/--output or --verify')
    print(json.dumps({'offline_candidate_verified':not a.schema3,'native_package_protocol':a.schema3,'quality_status':'pending','instances':len(c['instances']),'shared_objects':len(c['objects'])}))
if __name__=='__main__':main()
