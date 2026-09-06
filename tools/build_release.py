#!/usr/bin/env python3
"""Build a local runtime/SDK review draft, never a publishing authorization.

Python 3.11+. ELF inspection uses readelf/ldconfig; it does not execute the input.
System shared libraries are requirements, not silently copied into the archive.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tomllib

MAX_NOTICE = 2 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def record(path):
    return {'size':path.stat().st_size, 'sha256':sha256(path)}


def regular(path):
    if path.is_symlink() or not path.is_file(): raise ValueError('Expected regular file: '+str(path))
    return path


def inspect_elf(executable):
    """Read DT_NEEDED; unresolved/transitive closure remains explicitly unverified."""
    with executable.open('rb') as stream:
        magic=stream.read(4)
    if magic != b'\x7fELF':
        return {'status':'unavailable', 'reason':'Only ELF inspection implemented', 'needed':[]}
    if not shutil.which('readelf'):
        return {'status':'unavailable', 'reason':'readelf missing', 'needed':[]}
    result = subprocess.run(['readelf','-d',str(executable)], capture_output=True,text=True,
                            timeout=30,env={**os.environ,'LC_ALL':'C'})
    if result.returncode:
        return {'status':'unavailable','reason':'readelf failed','needed':[]}
    needed = re.findall(r'\(NEEDED\).*\[(.*?)\]',result.stdout)
    paths = re.findall(r'\((?:RPATH|RUNPATH)\).*\[(.*?)\]',result.stdout)
    return {'status':'direct_dependencies_observed','needed':needed,'search_paths':paths,
            'transitive_resolution':'not_verified','bundled':False,
            'inspection_host':{'system':platform.system(),'machine':platform.machine()}}


def system_notices(dependencies, destination):
    """Record host library candidates and RPM notices without bundling libraries."""
    result={'scope':'host ldconfig candidates, not observed runtime loading', 'libraries':[], 'gaps':[]}
    if not shutil.which('ldconfig') or not shutil.which('rpm'):
        result['gaps'].append('ldconfig/RPM provenance inspection unavailable');return result
    cache=subprocess.run(['ldconfig','-p'],capture_output=True,text=True,timeout=30).stdout
    copied=set()
    for name in dependencies.get('needed',[]):
        paths=[Path(line.rsplit(' => ',1)[1]) for line in cache.splitlines()
               if line.strip().startswith(name+' ') and ' => ' in line]
        entry={'soname':name,'candidates':[]};result['libraries'].append(entry)
        if not paths:result['gaps'].append(name+': no host ldconfig candidate')
        for candidate in paths:
            if not candidate.is_file():continue
            actual=candidate.resolve()
            q=subprocess.run(['rpm','-qf','--qf','%{NEVRA}\n%{LICENSE}\n',str(actual)],capture_output=True,text=True,timeout=30)
            if q.returncode:
                result['gaps'].append(name+': library origin not identified');continue
            lines=q.stdout.splitlines()
            item={'path':str(actual),**record(actual),'rpm':lines[0],'declared_license':lines[1] if len(lines)>1 else None,'notices':[]}
            entry['candidates'].append(item)
            listing=subprocess.run(['rpm','-ql',lines[0]],capture_output=True,text=True,timeout=30)
            for value in listing.stdout.splitlines():
                path=Path(value)
                if not value.startswith('/usr/share/licenses/') or not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_NOTICE:continue
                relative=Path('system')/path.relative_to('/usr/share/licenses')
                target=destination/relative
                if relative not in copied:
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);copied.add(relative)
                item['notices'].append({'path':relative.as_posix(),'source':str(path),**record(target)})
            if not item['notices']:result['gaps'].append(lines[0]+': installed license text absent')
    return result


def select_files(prefix, include_sdk):
    """An allowlist, not an unrestricted recursive copy of the install root."""
    selected = []
    for name in ('bin/ernie-image','bin/ernie-image.exe'):
        if (prefix/name).exists(): selected.append(name)
    if len(selected) != 1: raise ValueError('Expected exactly one installed CLI')
    for name in ('LICENSE','RUNNING.md','sources.lock.json'):
        regular(prefix/'share/ernie-image'/name)
        selected.append('share/ernie-image/'+name)
    if include_sdk:
        for sub in ('include','lib','lib64'):
            if (prefix/sub).is_symlink(): raise ValueError('Symlink SDK directory')
            if not (prefix/sub).exists(): continue
            for p in sorted((prefix/sub).rglob('*')):
                if p.is_symlink(): raise ValueError('Symlink SDK entry')
                if p.is_file():
                    if p.suffix not in ('.h','.hpp','.a','.lib','.cmake','.pc'):
                        raise ValueError('Unexpected SDK file: '+str(p))
                    selected.append(p.relative_to(prefix).as_posix())
        if not any(x.endswith('/ErnieConfig.cmake') for x in selected):
            raise ValueError('Requested SDK missing ErnieConfig.cmake')
    for name in selected:
        p=prefix/name
        for ancestor in (p,*p.parents):
            if ancestor==prefix.parent: break
            if ancestor.is_symlink(): raise ValueError('Symlink selected path')
        regular(p)
    return selected


def collect_notices(source, destination, cargo_registry=None, ncnn_source=None):
    """Conservative lockfile inventory, not a claim about the linked crate graph."""
    destination.mkdir()
    entries=[]; gaps=[]
    def add(component,path,relative,declared=None,expected=None):
        if not path or not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_NOTICE:
            gaps.append(component+': notice unavailable');return
        if expected and sha256(path)!=expected:
            gaps.append(component+': notice differs from cargo checksum');return
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        entries.append({'component':component,'source':str(path),'path':relative,
                        'declared_license':declared,**record(target)})
    if source is None:
        return {'entries':entries,'gaps':['No fixed source supplied'],'licenses_complete':False}
    add('project',source/'LICENSE','project/LICENSE','MIT')
    ncnn_source=ncnn_source or source/'third_party/ncnn'
    pinned=json.loads((source/'sources.lock.json').read_text()).get('ncnn',{}).get('revision') if (source/'sources.lock.json').is_file() else None
    def git_text(directory,*args):
        r=subprocess.run(['git','-C',str(directory),*args],capture_output=True,timeout=30)
        return r.stdout if r.returncode==0 else None
    actual=git_text(ncnn_source,'rev-parse','HEAD') if ncnn_source.is_dir() else None
    if pinned and actual and actual.decode().strip()==pinned:
        for component,directory,revision in [('ncnn',ncnn_source,pinned)]:
            content=git_text(directory,'show',revision+':LICENSE.txt')
            if content:
                target=destination/component/'LICENSE.txt';target.parent.mkdir();target.write_bytes(content)
                entries.append({'component':component,'source':str(directory),'revision':revision,'path':component+'/LICENSE.txt',**record(target)})
            else:gaps.append(component+': pinned notice unavailable')
        tree=git_text(ncnn_source,'ls-tree',pinned,'glslang')
        revision=tree.decode().split()[2] if tree and tree.decode().startswith('160000 commit ') else None
        content=git_text(ncnn_source/'glslang','show',revision+':LICENSE.txt') if revision else None
        if content:
            target=destination/'glslang/LICENSE.txt';target.parent.mkdir();target.write_bytes(content)
            entries.append({'component':'glslang','source':str(ncnn_source/'glslang'),'revision':revision,'path':'glslang/LICENSE.txt',**record(target)})
        else:gaps.append('glslang: pinned notice unavailable')
    else:gaps.append('ncnn/glslang: exact pinned source unavailable')
    stb=source/'third_party/stb/stb_image.h'
    if stb.is_file():
        text=stb.read_text(); marker='This software is available under 2 licenses -- choose whichever you prefer.'
        if text.count(marker)==1:
            notice=text.split(marker,1)[1].rsplit('*/',1)[0]
            target=destination/'stb-image.txt';target.write_text(marker+'\n'+notice+'\n')
            entries.append({'component':'stb_image','source':str(stb),'source_sha256':sha256(stb),
                            'path':'stb-image.txt','declared_license':'MIT OR Unlicense',**record(target)})
        else:gaps.append('stb_image: license section not recognized')
    else:gaps.append('stb_image: source absent')
    lock=source/'tokenizer/Cargo.lock'; crates=[]
    if not lock.is_file():gaps.append('Rust lockfile missing')
    else:
        packages=tomllib.loads(lock.read_text())['package']
        for package in packages:
            if 'source' not in package:
                crates.append({'name':package['name'],'version':package['version'],'scope':'local bridge; project license'})
                continue
            name=package['name'];version=package['version'];item={'name':name,'version':version,'source':package['source'],'checksum':package.get('checksum')}
            crates.append(item)
            archives=list((cargo_registry.parent/'cache').glob('*/'+name+'-'+version+'.crate')) if cargo_registry else []
            if len(archives)==1 and package.get('checksum') and sha256(archives[0])==package['checksum']:
                with tarfile.open(archives[0]) as archive:
                    base=name+'-'+version+'/'
                    members={m.name:m for m in archive.getmembers() if m.isfile() and m.size<=MAX_NOTICE}
                    cargo=members.get(base+'Cargo.toml')
                    if cargo is None:gaps.append(name+': package Cargo.toml missing');continue
                    metadata=tomllib.loads(archive.extractfile(cargo).read().decode())['package']
                    item['declared_license']=metadata.get('license');item['authenticated_archive']=str(archives[0])
                    selected=[]
                    for member in members.values():
                        relative=member.name.removeprefix(base)
                        if not member.name.startswith(base) or Path(relative).is_absolute() or '..' in Path(relative).parts:continue
                        if ('/' not in relative and re.match(r'(?i)^(LICENSE|COPYING|NOTICE)([._-].*|$)',relative)) or relative==metadata.get('license-file'):
                            selected.append((relative,member))
                    if not selected:gaps.append(name+'-'+version+': authenticated archive has no notice text')
                    for relative,member in selected:
                        target=destination/'rust'/(name+'-'+version)/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(archive.extractfile(member).read())
                        entries.append({'component':name+'-'+version,'source':str(archives[0])+'::'+member.name,'archive_sha256':package['checksum'],
                            'path':target.relative_to(destination).as_posix(),'declared_license':item.get('declared_license'),**record(target)})
                continue
            matches=list(cargo_registry.glob('*/'+name+'-'+version)) if cargo_registry else []
            if len(matches)!=1:
                gaps.append(name+'-'+version+': exact cached source absent/ambiguous');continue
            folder=matches[0];checksum=folder/'.cargo-checksum.json'
            if not checksum.is_file():gaps.append(name+': cargo checksum missing');continue
            checks=json.loads(checksum.read_text())
            if checks.get('package')!=package.get('checksum') or not package.get('checksum'):
                gaps.append(name+': cached package identity mismatch');continue
            cargo=folder/'Cargo.toml'
            if checks['files'].get('Cargo.toml')!=sha256(cargo):
                gaps.append(name+': Cargo.toml checksum mismatch');continue
            metadata=tomllib.loads(cargo.read_text())['package'];item['declared_license']=metadata.get('license')
            notices=[p for p in folder.iterdir() if p.is_file() and re.match(r'(?i)^(LICENSE|COPYING|NOTICE)([._-].*|$)',p.name)]
            if metadata.get('license-file'):
                rel=Path(metadata['license-file'])
                if not rel.is_absolute() and '..' not in rel.parts and (folder/rel).is_file():notices.append(folder/rel)
            if not notices:gaps.append(name+'-'+version+': license text absent')
            for path in sorted(set(notices)):
                rel=path.relative_to(folder).as_posix();expected=checks['files'].get(rel)
                if not expected:gaps.append(name+': unbound notice '+rel);continue
                add(name+'-'+version,path,'rust/'+name+'-'+version+'/'+rel,item.get('declared_license'),expected)
    return {'entries':entries,'gaps':gaps,'rust_lock_packages':crates,
            'inventory_scope':'conservative Cargo.lock packages, includes build/optional crates; actual link closure not proven',
            'licenses_complete':False,
            'remaining_review':['Confirm actual static linked dependency/notice closure',
                'System dynamic dependency notices and redistribution basis not established',
                'License expressions/texts recorded, no owner risk acceptance inferred']}


def build_release(install, output, *, include_sdk=False, platform_label='unverified',
                  source=None, build_evidence=None, install_evidence=None, cargo_registry=None, ncnn_source=None):
    prefix=Path(install).absolute();out=Path(output).absolute()
    for path in (out,*out.parents):
        if path.is_symlink(): raise ValueError('Symlink output path')
    if out==prefix or prefix in out.parents: raise ValueError('Output must be outside installation')
    selected=select_files(prefix,include_sdk)
    out.mkdir(parents=True,exist_ok=False)
    payload=out/'ernie-runtime';payload.mkdir()
    try:
        files=[]
        for name in selected:
            src=prefix/name;dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True)
            before=record(src);shutil.copy2(src,dest)
            if record(dest)!=before or record(src)!=before:raise ValueError('Installation changed while copying')
            files.append({'path':name,**before})
        binary=payload/selected[0]
        dependencies=inspect_elf(binary)
        license_report=collect_notices(Path(source) if source else None,payload/'licenses',Path(cargo_registry) if cargo_registry else None, Path(ncnn_source) if ncnn_source else None)
        system=system_notices(dependencies,payload/'licenses')
        license_report['system_dependencies']=system
        license_report['gaps'].extend(system['gaps'])
        evidence={};gaps=[]
        if source:
            identity=Path(source)/'source-identity.json'
            if identity.is_file():
                value=json.loads(identity.read_text());changed=[]
                for name,expected in value['files'].items():
                    # Frozen identity is evidence input, never a copy instruction.
                    path=Path(source)/name
                    if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or sha256(path)!=expected:changed.append(name)
                evidence['source']={'identity_sha256':sha256(identity),'changed_files':changed,'base_commit':value.get('base_commit')}
                if changed:gaps.append('Frozen source identity mismatch')
            else:gaps.append('Fixed source identity absent')
        else:gaps.append('Source identity absent')
        if build_evidence:
            directory=Path(build_evidence);identity=json.loads((directory/'identity.json').read_text());result=json.loads((directory/'result.json').read_text())
            evidence['build']={'identity':identity,'result':result,'identity_sha256':sha256(directory/'identity.json'),'result_sha256':sha256(directory/'result.json')}
            if result.get('exit_code')!=0 or result.get('source_changes_after_build')!=[]:gaps.append('Build evidence did not pass')
            if identity.get('source_identity_sha256')!=evidence.get('source',{}).get('identity_sha256'):gaps.append('Build/source identity not linked')
        else:gaps.append('Build evidence absent')
        if install_evidence:
            p=Path(install_evidence);value=json.loads(p.read_text());known={f['path']:f for f in value.get('installed_files',[])}
            matched=all(known.get(f['path'],{}).get('sha256')==f['sha256'] and known.get(f['path'],{}).get('bytes')==f['size'] for f in files)
            evidence['installation']={'sha256':sha256(p),'selected_files_match':matched,'status':value.get('status'),
                'source_isolated':value.get('source_isolated'),'network_disabled':value.get('network_disabled'),'scope':value.get('scope')}
            if not matched or value.get('status')!='passed':gaps.append('Installation evidence not bound/passed')
        else:gaps.append('Installation evidence absent')
        result={'schema_version':1,'status':'local_review_draft','distributable':False,'published':False,
            'public_url':None,'include_sdk':include_sdk,'platform':{'requested_label':platform_label,'verified':False,
                'reason':'No independently verified release-platform contract; installation API evidence is scoped separately'},
            'selected_install_files':files,'dependencies':dependencies,'licenses':license_report,'evidence':evidence,
            'blockers':gaps+license_report['gaps']+license_report.get('remaining_review',[]),
            'model_assets_included':False,'full_model_validation':'not_performed'}
        result['blockers'].append('Release platform verification pending')
        (payload/'release.json').write_text(json.dumps(result,indent=2)+'\n')
        (payload/'DRAFT-README.txt').write_text('LOCAL REVIEW DRAFT — NOT A VERIFIED DISTRIBUTABLE RELEASE\n'
            'No model weights are included. System dynamic libraries listed in release.json are required and are not bundled.\n'
            'License and platform review remain pending. See share/ernie-image/RUNNING.md for the installed API scope.\n')
        members=[p for p in sorted(payload.rglob('*')) if p.is_file()]
        (out/'files.json').write_text(json.dumps([{'path':p.relative_to(payload).as_posix(),**record(p)} for p in members],indent=2)+'\n')
        archive=out/'ernie-runtime-draft.tar.gz'
        with archive.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed, tarfile.open(fileobj=compressed,mode='w|') as tar:
            for p in members:
                info=tar.gettarinfo(str(p),arcname='ernie-runtime/'+p.relative_to(payload).as_posix());info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                with p.open('rb') as stream:tar.addfile(info,stream)
        (out/'SHA256SUMS').write_text(sha256(archive)+'  '+archive.name+'\n'+sha256(out/'files.json')+'  files.json\n')
        return result
    except BaseException as error:
        (out/'FAILED.txt').write_text(str(error)+'\n');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--include-sdk',action='store_true');p.add_argument('--platform',default='unverified')
    p.add_argument('--source',type=Path);p.add_argument('--build-evidence',type=Path);p.add_argument('--install-evidence',type=Path)
    p.add_argument('--cargo-registry',type=Path);p.add_argument('--ncnn-source',type=Path)
    a=p.parse_args()
    try:
        r=build_release(a.install,a.output,include_sdk=a.include_sdk,platform_label=a.platform,source=a.source,
            build_evidence=a.build_evidence,install_evidence=a.install_evidence,cargo_registry=a.cargo_registry,ncnn_source=a.ncnn_source)
        print(json.dumps({'output':str(a.output),'status':r['status'],'distributable':r['distributable'],'blockers':r['blockers']},indent=2))
    except (OSError,ValueError) as error:p.exit(1,f'Release draft failed: {error}\n')
if __name__=='__main__':main()
