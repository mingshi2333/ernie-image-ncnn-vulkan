#!/usr/bin/env python3
"""Verify and materialize a runtime-only model package without external symlinks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_files():
    names = ['model.cfg', 'text/embeddings.bf16', 'text/rope-inv-freq.f32',
             'dit/rope-inv-freq.f32', 'vae/bn-mean.f32', 'vae/bn-variance.f32',
             'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json',
             'vae/head.ncnn.param', 'vae/head.ncnn.bin']
    for kind, count, stem in [('text', 25, 'text'), ('dit', 36, 'block')]:
        for index in range(count):
            names += [f'{kind}/block-{index:02d}/{stem}.ncnn.{suffix}' for suffix in ('param', 'bin')]
    for head in ('input', 'output'):
        names += [f'dit/{head}/head.ncnn.{suffix}' for suffix in ('param', 'bin')]
    return sorted(names)


def safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or any(p in ('', '.', '..') for p in name.split('/')) or Path(name).is_absolute():
        raise ValueError(f'Unsafe package path: {name!r}')
    return name


def verify_package(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    if manifest.get('official_model_revision') != lock['official_model']['revision']:
        raise ValueError('Official model revision differs')
    expected = dict(manifest['files'])
    if manifest['schema_version'] == 1:
        for name, digest in manifest['source_manifests'].items():
            path = root/safe_name(name)
            if path.is_dir():
                path /= 'model.json'
                component = json.loads(path.read_text())
                for filename, checksum in component['files'].items():
                    expected[name+'/'+safe_name(filename)] = checksum
            else:
                expected[name] = digest
            if sha256(path) != digest:
                raise ValueError(f'Source manifest checksum differs: {name}')
        # Compatibility with early local packages that did not index tokenizer
        # files at the root. The tokenizer manifest still pins both digests.
        tokenizer = json.loads((root/'tokenizer/manifest.json').read_text())
        if tokenizer['revision'] != manifest['official_model_revision']:
            raise ValueError('Tokenizer revision differs')
        for name, digest in tokenizer['files'].items():
            key = 'tokenizer/'+safe_name(name)
            if key in expected and expected[key] != digest:
                raise ValueError('Tokenizer manifests disagree')
            expected[key] = digest
    elif manifest['schema_version'] == 2:
        if manifest.get('ncnn_revision') != lock['ncnn']['revision']:
            raise ValueError('ncnn revision differs')
        if set(expected) != set(runtime_files()) or set(manifest['file_sizes']) != set(expected):
            raise ValueError('Runtime file inventory differs')
    else:
        raise ValueError('Unsupported package schema')
    for name in runtime_files():
        path = root/name
        if name not in expected or not path.is_file():
            raise ValueError(f'Missing runtime file or checksum: {name}')
        if manifest.get('portable'):
            relative = Path(name)
            if any((root/part).is_symlink() for part in (relative, *relative.parents)):
                raise ValueError(f'Portable package contains a symlink: {name}')
        if manifest['schema_version'] == 2 and path.stat().st_size != manifest['file_sizes'][name]:
            raise ValueError(f'File size differs: {name}')
        if sha256(path) != expected[name]:
            raise ValueError(f'File checksum differs: {name}')
    tokens = (root/'model.cfg').read_text().split()
    if len(tokens) != 12 or len(set(tokens[::2])) != 6:
        raise ValueError('Malformed model.cfg')
    cfg = dict(zip(tokens[::2], map(int, tokens[1::2])))
    if cfg != manifest['config']:
        raise ValueError('Configuration and manifest disagree')
    if (set(cfg) != {'packed_width','packed_height','text_bucket','dit_text_tokens','text_layers','dit_layers'}
        or not 1 <= cfg['packed_width'] <= 128 or not 1 <= cfg['packed_height'] <= 128
        or not 1 <= cfg['text_bucket'] <= cfg['dit_text_tokens'] <= 2048
        or cfg['packed_width']*cfg['packed_height']+cfg['dit_text_tokens'] > 6144
        or cfg['text_layers'] != 25 or cfg['dit_layers'] != 36):
        raise ValueError('Unsupported model configuration')
    return manifest, {name: expected[name] for name in runtime_files()}


def package_model(source, output, link=False):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new package output directory')
    manifest, files = verify_package(source)
    output.mkdir(parents=True)
    for name in files:
        destination = output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link:
            destination.symlink_to((source/name).resolve())
        else:
            shutil.copyfile(source/name, destination)
        if sha256(destination) != files[name]:
            raise ValueError(f'Copied file differs: {name}')
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    packed = {'schema_version': 2, 'portable': not link,
              'scope': 'Self-contained runtime model files' if not link else 'Local development links; source files must remain present',
              'config': manifest['config'], 'official_model_revision': manifest['official_model_revision'],
              'ncnn_revision': lock['ncnn']['revision'], 'files': files,
              'file_sizes': {name: (output/name).stat().st_size for name in files},
              'provenance': {'source_manifest_sha256': sha256(source/'manifest.json'),
                             'packager_sha256': sha256(__file__)}}
    if manifest['schema_version'] == 1:
        packed['source_weights'] = {
            kind: [json.loads((source/f'{kind}/block-{index:02d}/model.json').read_text())['weights_sha256']
                   for index in range(count)] for kind, count in [('text',25),('dit',36)]}
    elif 'source_weights' in manifest:
        packed['source_weights'] = manifest['source_weights']
    (output/'manifest.json').write_text(json.dumps(packed, indent=2)+'\n')
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New portable directory; omit to verify the existing package')
    parser.add_argument('--link', action='store_true', help='Development links instead of a portable copy')
    args = parser.parse_args()
    if args.link and not args.output:
        parser.error('--link requires --output')
    if args.output:
        manifest = package_model(args.model, args.output, args.link)
    else:
        manifest, _ = verify_package(args.model)
    print(json.dumps({'verified': True, 'model': str(args.output or args.model),
                      'portable': manifest.get('portable', False), 'files': len(runtime_files())}))


if __name__ == '__main__':
    main()
