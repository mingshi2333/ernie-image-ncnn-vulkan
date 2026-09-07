"""Select a verified package instance for an existing pipeline oracle.

Shared packages retain their CAS layout. This adapter supplies configuration and
provenance to the validator; the native runtime resolves every actual component.
"""
from pathlib import Path

from package_dynamic_model import read_json, shared_contract, verify_shared_package
from package_model import sha256


def select_shared_instance(instances, width=None, height=None):
    """Select a source and an explicitly reviewed spatial target without I/O."""
    exact = [item for item in instances if width is None or
             (item['config']['packed_width'] * 16 == width
              and item['config']['packed_height'] * 16 == height)]
    if len(exact) == 1:
        return exact[0], None
    if exact or width is None:
        raise ValueError('Select exactly one shared instance with --width and --height')
    contract = shared_contract()
    candidates = []
    for item in instances:
        digest = item['source_manifest_sha256']
        if item['config'] != contract['source_manifests'].get(digest):
            continue
        for target in contract.get('reviewed_runtime_targets', {}).get(digest, []):
            if (target['packed_width'] * 16, target['packed_height'] * 16) == (width, height):
                if any(target[k] != item['config'][k]
                       for k in ('text_bucket', 'dit_text_tokens', 'text_layers', 'dit_layers')):
                    raise ValueError('Runtime target changes the reviewed model contract')
                candidates.append(({**item, 'config': target}, {
                    'source_config': item['config'], 'target_config': target,
                    'scope': 'Reviewed spatial graph instantiation; existing weights; target encoder unavailable',
                }))
    if len(candidates) != 1:
        raise ValueError('Select exactly one available shared runtime target')
    return candidates[0]


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
    selected, runtime_target = select_shared_instance(manifest['instances'], width, height)
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
