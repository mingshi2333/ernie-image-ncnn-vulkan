"""Select a verified package instance for an existing pipeline oracle.

Shared packages retain their CAS layout. This adapter supplies configuration and
provenance to the validator; the native runtime resolves every actual component.
"""
from pathlib import Path

from package_dynamic_model import read_json, verify_shared_package
from package_model import sha256


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
    candidates = manifest['instances']
    if width is not None:
        candidates = [item for item in candidates
                      if item['config']['packed_width'] * 16 == width
                      and item['config']['packed_height'] * 16 == height]
    if len(candidates) != 1:
        raise ValueError('Select exactly one shared instance with --width and --height')
    selected = candidates[0]
    return selected['config'], {
        'schema_version': 3,
        'source_manifest_sha256': selected['source_manifest_sha256'],
        'shared_manifest_sha256': sha256(model / 'manifest.json'),
        'runtime_bindings': selected['runtime_bindings'],
        'scope': 'Verified pinned static instance; native CAS resolution; existing official oracle',
    }
