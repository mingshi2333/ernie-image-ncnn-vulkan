#!/usr/bin/env python3
"""Acquire authenticated bytes only; run the native model verifier separately."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.request
from urllib.parse import urlsplit
try:
    from .release_manifest import load_manifest, validate_manifest
except ImportError:
    from release_manifest import load_manifest, validate_manifest

CHUNK = 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''): h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    p = root
    for part in Path(relative).parts:
        p = p / part
        if p.is_symlink(): raise ValueError('Refusing symlink: '+str(p))
        if p.exists() and not (p.is_dir() or p.is_file()):
            raise ValueError('Refusing special filesystem object: '+str(p))
    return p


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if new.scheme != 'https' and not (old.scheme == new.scheme == 'http' and new.hostname in ('localhost','127.0.0.1','::1')):
            raise ValueError('Unsafe download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(f, root, opener, timeout):
    dest = checked_path(root, f['path'])
    if dest.exists():
        if not dest.is_file() or dest.stat().st_size != f['size'] or sha256(dest) != f['sha256']:
            raise ValueError('Existing target is not authenticated: '+str(dest))
        return 'verified_existing'
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = checked_path(root, f['path']+'.part')
    meta = checked_path(root, f['path']+'.part.json')
    identity = dict(f)
    offset = 0; old = None
    if part.exists() or meta.exists():
        if not part.is_file() or not meta.is_file(): raise ValueError('Incomplete resume metadata')
        if part.stat().st_nlink != 1 or meta.stat().st_nlink != 1:
            raise ValueError('Refusing shared hardlinked resume files')
        if meta.stat().st_size > 65536: raise ValueError('Oversized resume metadata')
        old = json.loads(meta.read_text())
        if not isinstance(old, dict) or set(old) != {'file','etag','last_modified'}:
            raise ValueError('Invalid resume metadata')
        for key in ('etag','last_modified'):
            if old[key] is not None and (not isinstance(old[key], str) or any(ord(c)<32 for c in old[key])):
                raise ValueError('Invalid resume HTTP validator')
        if old.get('file') != identity: raise ValueError('Resume manifest identity changed')
        offset = part.stat().st_size
        if offset > f['size']: raise ValueError('Oversized partial file')
        if offset == f['size']:
            if sha256(part) != f['sha256']: raise ValueError('Complete partial checksum mismatch')
            os.link(part, dest); part.unlink(); meta.unlink(); return 'verified_partial'
    headers = {'Accept-Encoding':'identity','User-Agent':'ernie-model-download/1'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
        validator = old.get('etag') or old.get('last_modified')
        if validator and not validator.startswith('W/'):
            headers['If-Range'] = validator
    try:
        response = opener.open(urllib.request.Request(f['url'], headers=headers), timeout=timeout)
    except urllib.error.HTTPError as error:
        error.close()
        raise
    with response:
        status = response.status
        etag = response.headers.get('ETag'); modified = response.headers.get('Last-Modified')
        if response.headers.get('Content-Encoding', 'identity') != 'identity': raise ValueError('Encoded response')
        if offset and old and ((old.get('etag') and etag != old['etag']) or (not old.get('etag') and old.get('last_modified') and modified != old['last_modified'])):
            raise ValueError('Remote object version changed')
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
            if not match or tuple(map(int,match.groups())) != (offset,f['size']-1,f['size']):
                raise ValueError('Invalid Content-Range')
        elif status == 200:
            offset = 0  # Server ignored Range: restart, never append a full response.
        else: raise ValueError('Unexpected HTTP status')
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != f['size']-offset: raise ValueError('Wrong Content-Length')
        if shutil.disk_usage(root).free < f['size']-offset: raise OSError('Insufficient disk capacity')
        metadata = {'file':identity,'etag':etag,'last_modified':modified}
        # Metadata is local recovery state, never authority for a final file.
        meta.write_text(json.dumps(metadata, sort_keys=True)+'\n')
        total = offset
        with part.open('ab' if offset else 'wb') as output:
            while True:
                data = response.read(min(CHUNK, f['size']-total+1))
                if not data: break
                if total+len(data) > f['size']: raise ValueError('Oversized HTTP body')
                output.write(data); total += len(data)
            output.flush(); os.fsync(output.fileno())
        if total != f['size']: raise ValueError('Truncated HTTP body; partial retained')
    if sha256(part) != f['sha256']: raise ValueError('Downloaded checksum mismatch; partial retained')
    # Atomic publication without overwriting any target created during download.
    os.link(part, dest)
    part.unlink(); meta.unlink()
    return 'downloaded'


def download(manifest, output, *, timeout=60, emit=print):
    m = validate_manifest(manifest)
    root = Path(output).absolute()
    for p in (root, *root.parents):
        if p.is_symlink(): raise ValueError('Symlink output directory')
    root.mkdir(parents=True, exist_ok=True)
    emit(f"Total: {sum(f['size'] for f in m['files'])} bytes; destination: {root}")
    lock = root/'.download.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    try:
        # Validate all existing paths before any network request or data mutation.
        remaining = 0
        for f in m['files']:
            dest = checked_path(root, f['path'])
            for suffix in ('.part','.part.json'): checked_path(root, f['path']+suffix)
            if dest.exists():
                if not dest.is_file() or dest.stat().st_size != f['size'] or sha256(dest) != f['sha256']:
                    raise ValueError('Existing target is not authenticated: '+str(dest))
            else: remaining += f['size']  # Conservative: permits server-200 full restart.
        if shutil.disk_usage(root).free < remaining: raise OSError('Insufficient disk capacity')
        opener = urllib.request.build_opener(SafeRedirect())
        result = []
        for f in m['files']:
            status = download_file(f,root,opener,timeout)
            emit(f"{f['path']}: {status}")
            result.append({'path':f['path'],'status':status,'sha256':f['sha256']})
        return {'status':'download_integrity_verified','model_semantics':'not_verified','files':result}
    finally:
        os.close(fd); lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try: download(load_manifest(args.manifest), args.output)
    except (OSError,ValueError,http.client.HTTPException) as error: parser.exit(1, f'Download failed: {error}\n')

if __name__ == '__main__': main()
