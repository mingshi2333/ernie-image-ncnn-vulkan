#!/usr/bin/env python3
"""Strict download-manifest schema; hashes prove bytes, not model semantics/licensing."""
import json
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit, unquote


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('Unexpected manifest fields')


def _text(value):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError('Expected nonempty text')


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256')


def safe_path(value):
    _text(value)
    p = PurePosixPath(value)
    if not p.parts or p.is_absolute() or str(p) != value or any(x in ('.', '..') for x in p.parts):
        raise ValueError('Unsafe relative path')
    if any('\\' in x or ':' in x or x.endswith(('.', ' ')) for x in p.parts):
        raise ValueError('Unsafe path component')
    if any(x.lower().split('.')[0] in {'con', 'prn', 'aux', 'nul', *('com'+str(i) for i in range(1,10)), *('lpt'+str(i) for i in range(1,10))} for x in p.parts):
        raise ValueError('Reserved path component')
    if any(x.endswith(('.part', '.part.json')) or x == '.download.lock' for x in p.parts):
        raise ValueError('Reserved downloader path')
    return value


def validate_manifest(m):
    _keys(m, ['schema_version','model_revision','graph_schema','required_capabilities','conversion','license','files'])
    if type(m['schema_version']) is not int or m['schema_version'] != 1:
        raise ValueError('Unsupported download schema')
    if not isinstance(m['model_revision'], str) or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', m['model_revision']):
        raise ValueError('Expected immutable model revision')
    _text(m['graph_schema'])
    if not isinstance(m['required_capabilities'], list) or not m['required_capabilities']:
        raise ValueError('Missing capabilities')
    for cap in m['required_capabilities']: _text(cap)
    _keys(m['conversion'], ['source','script_sha256'])
    _text(m['conversion']['source']); digest(m['conversion']['script_sha256'])
    _keys(m['license'], ['identifier','source','notice'])
    for v in m['license'].values(): _text(v)
    if not isinstance(m['files'], list) or not m['files']: raise ValueError('Empty download')
    names = set()
    for f in m['files']:
        _keys(f, ['path','url','size','sha256'])
        safe_path(f['path']); digest(f['sha256'])
        name = f['path'].casefold()
        if name in names or any(name.startswith(n+'/') or n.startswith(name+'/') for n in names):
            raise ValueError('Conflicting download paths')
        names.add(name)
        if type(f['size']) is not int or f['size'] < 0: raise ValueError('Invalid file size')
        _text(f['url']); u = urlsplit(f['url'])
        if u.username or u.password or u.fragment or not u.hostname:
            raise ValueError('Unsafe URL')
        if u.scheme != 'https' and not (u.scheme == 'http' and u.hostname in ('localhost','127.0.0.1','::1')):
            raise ValueError('Downloads require HTTPS (HTTP loopback fixtures only)')
        if m['model_revision'] not in unquote(u.path).split('/'):
            raise ValueError('URL path must bind the immutable model revision')
    return m


def load_manifest(path):
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result: raise ValueError('Duplicate JSON key')
            result[k] = v
        return result
    with open(path, encoding='utf-8') as stream:
        return validate_manifest(json.load(stream, object_pairs_hook=unique))
