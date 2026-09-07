"""Select a verified package instance for an existing pipeline oracle.

Shared packages retain their CAS layout. This adapter supplies configuration and
provenance to the validator; the native runtime resolves every actual component.
"""
from pathlib import Path

from package_dynamic_model import read_json, shared_contract, verify_shared_package
from package_model import sha256


def select_shared_instance(instances, width=None, height=None, valid_text_tokens=None):
    """Mirror native source/geometry selection using authenticated metadata only."""
    if not instances:
        raise ValueError('Missing shared sources')
    if width is None:
        if height is not None or len(instances) != 1:
            raise ValueError('Select a shared resolution with --width and --height')
        width = instances[0]['config']['packed_width'] * 16
        height = instances[0]['config']['packed_height'] * 16
    if (type(width) is not int or type(height) is not int or
            not 16 <= width <= 2048 or not 16 <= height <= 2048 or
            width % 16 or height % 16 or width * height > 2097152):
        raise ValueError('Invalid shared runtime dimensions')
    contract = shared_contract()
    seen = set()
    for item in instances:
        digest = item['source_manifest_sha256']
        if item['config'] != contract['source_manifests'].get(digest) or digest in seen:
            raise ValueError('Unknown or duplicate shared source configuration')
        seen.add(digest)
    if valid_text_tokens is None:
        source = min(instances, key=lambda item: (
            (item['config']['packed_width'] * 16, item['config']['packed_height'] * 16) != (width, height),
            item['config']['text_bucket']))
    else:
        if type(valid_text_tokens) is not int or not 1 <= valid_text_tokens <= 2048:
            raise ValueError('Reference must contain 1..2048 token IDs')
        candidates = [item for item in instances if item['config']['text_bucket'] >= valid_text_tokens]
        if not candidates:
            raise ValueError('Reference exceeds available text buckets')
        source = min(candidates, key=lambda item: item['config']['text_bucket'])
    target = {**source['config'], 'packed_width': width // 16, 'packed_height': height // 16}
    if target == source['config']:
        return source, None
    return {**source, 'config': target}, {
        'source_config': source['config'], 'target_config': target,
        'scope': 'Reviewed spatial graph instantiation; existing weights; target encoder unavailable',
    }


def validation_package(model, width=None, height=None, *, reference=None, reference_only=False):
    model = Path(model)
    if (width is None) != (height is None):
        raise ValueError('Specify validation width and height together')
    if width is not None and (type(width) is not int or type(height) is not int
                              or width < 16 or height < 16 or width % 16 or height % 16):
        raise ValueError('Validation dimensions must be positive multiples of 16')
    manifest = read_json(model / 'manifest.json')
    if manifest.get('schema_version') != 3:
        config = manifest['config']
        if width is not None and (width != config['packed_width'] * 16
                                  or height != config['packed_height'] * 16):
            raise ValueError('Validation dimensions differ from the fixed package')
        return config, None
    if reference is None or reference_only:
        raise ValueError('Shared-package validation requires an existing verified --reference')
    # Authenticate the entire object inventory before selecting an instance.
    manifest = verify_shared_package(model)
    # Selection uses the saved oracle IDs; full fixture authentication follows in
    # pipeline_reference before any native execution. A mismatched oracle fails.
    fixture = read_json(Path(reference) / 'fixture.json')
    ids = fixture.get('ids')
    if not isinstance(ids, list) or any(type(v) is not int or not 0 <= v < 131072 for v in ids):
        raise ValueError('Invalid reference token IDs')
    selected, runtime_target = select_shared_instance(manifest['instances'], width, height, len(ids))
    binding = {
        'schema_version': 3,
        'source_manifest_sha256': selected['source_manifest_sha256'],
        'shared_manifest_sha256': sha256(model / 'manifest.json'),
        'runtime_bindings': selected['runtime_bindings'],
        'scope': 'Verified pinned static instance; native CAS resolution; existing official oracle',
    }
    if runtime_target is not None:
        binding['runtime_target'] = runtime_target
        binding['scope'] = 'Verified pinned source with reviewed runtime spatial target; existing official oracle'
    return selected['config'], binding
