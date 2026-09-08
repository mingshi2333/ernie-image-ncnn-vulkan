#!/usr/bin/env python3
"""Verify and materialize a runtime-only model package without external symlinks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .ncnn_compat import compatible_model_revision
else:
    from ncnn_compat import compatible_model_revision

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
        if not compatible_model_revision(manifest.get('ncnn_revision'), lock):
            raise ValueError('ncnn revision differs')
        if set(expected) != set(runtime_files()) or set(manifest['file_sizes']) != set(expected):
            raise ValueError('Runtime file inventory differs')
        if type(manifest.get('portable')) is not bool:
            raise ValueError('Missing portable package flag')
    else:
        raise ValueError('Unsupported package schema')
    for name in runtime_files():
        path = root/name
        if name not in expected or not path.is_file():
            raise ValueError(f'Missing runtime file or checksum: {name}')
        if manifest.get('portable'):
            relative = Path(name)
            if any((root/part).is_symlink() for part in (relative, *relative.parents) if part != Path('.')):
                raise ValueError(f'Portable package contains a symlink: {name}')
        if manifest['schema_version'] == 2 and (type(manifest['file_sizes'][name]) is not int
                or path.stat().st_size != manifest['file_sizes'][name]):
            raise ValueError(f'File size differs: {name}')
        if sha256(path) != expected[name]:
            raise ValueError(f'File checksum differs: {name}')
    tokens = (root/'model.cfg').read_text().split()
    if len(tokens) != 12 or len(set(tokens[::2])) != 6:
        raise ValueError('Malformed model.cfg')
    cfg = dict(zip(tokens[::2], map(int, tokens[1::2])))
    if cfg != manifest['config'] or any(type(v) is not int for v in manifest['config'].values()):
        raise ValueError('Configuration and manifest disagree')
    if (set(cfg) != {'packed_width','packed_height','text_bucket','dit_text_tokens','text_layers','dit_layers'}
        or not 1 <= cfg['packed_width'] <= 128 or not 1 <= cfg['packed_height'] <= 128
        or not 1 <= cfg['text_bucket'] <= cfg['dit_text_tokens'] <= 2048
        or cfg['packed_width']*cfg['packed_height']+cfg['dit_text_tokens'] > 6144
        or cfg['text_layers'] != 25 or cfg['dit_layers'] != 36):
        raise ValueError('Unsupported model configuration')
    return manifest, {name: expected[name] for name in runtime_files()}


def package_model(source, output, link=False, fixed1376=False):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new package output directory')
    replacements = {}
    target_config = None
    if fixed1376:
        # Reuse the exact audited shape fields. This explicit fixed package does
        # not add a shape/source to the native schema-3 shared-weight registry.
        if __package__:
            from .prepare_shape_1376 import SOURCE_SHA, TARGET, candidate_graph
            from .audit_shape_contract import graph_files
        else:
            from prepare_shape_1376 import SOURCE_SHA, TARGET, candidate_graph
            from audit_shape_contract import graph_files
        if sha256(source/'manifest.json') != SOURCE_SHA:
            raise ValueError('Fixed 1376x768 requires the reviewed 1024x1024/s64 source package')
    manifest, files = verify_package(source)
    if fixed1376:
        target_config = dict(TARGET)
        replacements = {
            name: candidate_graph((source/name).read_text(), kind, manifest['config']).encode('utf-8')
            for name, kind in graph_files()}
        replacements['model.cfg'] = ''.join(f'{k} {v}\n' for k, v in target_config.items()).encode('utf-8')
    output.mkdir(parents=True)
    packed_files = {}
    for name in files:
        destination = output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in replacements:
            destination.write_bytes(replacements[name])
        elif link:
            destination.symlink_to((source/name).resolve())
        else:
            shutil.copyfile(source/name, destination)
        expected = hashlib.sha256(replacements[name]).hexdigest() if name in replacements else files[name]
        if sha256(destination) != expected:
            raise ValueError(f'Copied file differs: {name}')
        packed_files[name] = expected
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    packed = {'schema_version': 2, 'portable': not link,
              'scope': 'Self-contained runtime model files' if not link else 'Local development links; source files must remain present',
              'config': target_config or manifest['config'], 'official_model_revision': manifest['official_model_revision'],
              # Copying an existing package does not change its recorded revision.
              'ncnn_revision': manifest.get('ncnn_revision', lock['ncnn']['revision']), 'files': packed_files,
              'file_sizes': {name: (output/name).stat().st_size for name in files},
              'provenance': {'source_manifest_sha256': sha256(source/'manifest.json'),
                             'packager_sha256': sha256(__file__)}}
    if fixed1376:
        packed['provenance']['fixed_shape'] = {
            'width': 1376, 'height': 768, 'text_bucket': 64,
            'method': 'Complete graph hashes and enumerated static shape fields',
            'shape_tool_sha256': sha256(ROOT/'tools/prepare_shape_1376.py'),
            'shape_contract_sha256': sha256(ROOT/'tools/audit_shape_contract.py'),
            'status': 'Experimental fixed shape; no shared registry or broad quality acceptance'}
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
    parser.add_argument('--fixed-1376x768', action='store_true',
                        help='Experimental fixed 1376x768/s64 package from the pinned 1024x1024/s64 source')
    args = parser.parse_args()
    if (args.link or args.fixed_1376x768) and not args.output:
        parser.error('--link and --fixed-1376x768 require --output')
    if args.output:
        manifest = package_model(args.model, args.output, args.link, args.fixed_1376x768)
    else:
        manifest, _ = verify_package(args.model)
    print(json.dumps({'verified': True, 'model': str(args.output or args.model),
                      'portable': manifest.get('portable', False), 'files': len(runtime_files())}))


if __name__ == '__main__':
    main()
