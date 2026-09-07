"""Fail-closed structure and reviewed provenance of full pipeline references."""
import hashlib
import json
from pathlib import Path
import re

# Shared-package validation currently reuses reviewed historical official runs.
# Add entries only with independently reviewed execution and source evidence.
# This binds the whole fixture, including official environment and tensor hashes;
# a caller cannot establish provenance by editing fields inside that fixture.
REVIEWED_SHARED_REFERENCES = {
    'dbfd637a06f43812307c5745a3df86cc7daa501f5be40afa5c50d0e7af36ebc1': {
        'source_manifest_sha256': '9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154',
        'evidence': 'artifacts/2026-09-07/runtime-squares/README.md',
        'scope': '1024x1024 official FP32 development reference; common saved initial bytes; 32 text bucket and 64 DiT slots; eight complete steps; native comparison remains 24/25',
    },
    '731ebbab21841d98635d65e72960804c21cb42007a25928483cf7b069ebffd7f': {
        'source_manifest_sha256': '9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154',
        'evidence': 'artifacts/2026-09-07/runtime-squares/README.md',
        'scope': '768x768 official FP32 development reference; common saved initial bytes; 32 text bucket and 64 DiT slots; eight complete steps; native comparison remains 18/25',
    },
    '307dbd99c6beb731fafccf86822873bb9f961737ce8f224da34f8c78909e84d0': {
        'source_manifest_sha256': '9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154',
        'evidence': 'artifacts/2026-09-07/runtime-rectangles-and-mapped512/README.md',
        'scope': '512x384 official FP32 development reference; common saved initial bytes; 32 text bucket and 64 DiT slots; eight complete steps',
    },
    '20688b0323061d965cdf43ed4dcfb8f6898926c5221832279f6113b0fd9d7d43': {
        'source_manifest_sha256': '9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154',
        'evidence': 'artifacts/2026-09-07/runtime-rectangles-and-mapped512/README.md',
        'scope': '384x512 official FP32 development reference; common saved initial bytes; 32 text bucket and 64 DiT slots; eight complete steps',
    },
    'e68f3d8c112ef7a988161b539845eff6c356fe00fdc0d3e7444d625e99bc50d3': {
        'source_manifest_sha256': '9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154',
        'evidence': 'artifacts/2026-09-07/runtime-images-and-sdk/README.md',
        'scope': '512x512 official FP32 development reference; common saved initial bytes; CUDA blocks with TF32 disabled; eight complete steps',
    },
    '03ceedfd835cadc1a4450e31b43195fdc217dc59743b5a6c76948c8566802e63': {
        'source_manifest_sha256': '9cc1dc0e605256e405a049b6a8d58f98a9f138b2a16c6d40be6bb35595743154',
        'evidence': 'artifacts/2026-09-07/runtime-images-and-sdk/README.md',
        'scope': '64x64 official FP32 development reference; native saved initial bytes; eight complete steps; 32 text bucket and 64 DiT slots',
    },
    '930d1593ea6e4246e03edd990fbdf3a805de1b378ea2a5bbc8b15695a762f128': {
        'source_manifest_sha256': '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1',
        'evidence': 'artifacts/2026-09-07/fixed1376-native-pipeline/README.md',
        'scope': '1376x768 apple official FP32 development reference; eight complete steps; unchanged 1024 source weights',
    },
    '8d7638b1808b4b435565d4045ffc39e3a06fb1de35f8a52f23de75aec2feae11': {
        'source_manifest_sha256': 'ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59',
        'evidence': 'artifacts/2026-09-06/native-vector-pipeline/README.md',
        'scope': '512x384 PE/apple official FP32 development reference; eight complete steps',
    },
}


def full_reference_contract(fixture, config, prompt, steps):
    """Require every conditioning, prediction, Euler and decoder boundary."""
    if (not isinstance(fixture, dict) or fixture.get('complete') is not True
            or type(steps) is not int or not 1 <= steps <= 1000
            or type(fixture.get('steps')) is not int or fixture['steps'] != steps
            or type(fixture.get('start_step', 0)) is not int or fixture.get('start_step', 0) != 0
            or fixture.get('prompt') != prompt or fixture.get('config') != config
            or not isinstance(fixture.get('config'), dict)
            or any(type(v) is not int for v in fixture['config'].values())):
        raise ValueError('Saved reference is not the requested complete pipeline')
    h, w, tokens = config['packed_height'], config['packed_width'], config['dit_text_tokens']
    ids = fixture.get('ids')
    if (not isinstance(ids, list) or not 1 <= len(ids) <= min(tokens, config['text_bucket'])
            or any(type(v) is not int or not 0 <= v < 131072 for v in ids)):
        raise ValueError('Invalid complete reference token IDs')
    total = h * w + tokens
    latent = [1, 128, h, w]
    required_inputs = {'initial': latent, 'text': [1, len(ids), 3072],
                       'padded-text': [1, tokens, 3072],
                       'constant-0': [1, total, 128], 'constant-1': [1, total, 128],
                       'constant-2': [total, total]}
    required_final = {'final': latent, 'unpacked': [1, 32, h * 2, w * 2],
                      'decoded': [1, 3, h * 16, w * 16]}
    seen = set()

    def group(actual, required, suffix=''):
        if not isinstance(actual, dict) or set(actual) != set(required):
            raise ValueError('Missing or unexpected complete reference boundaries')
        for name, shape in required.items():
            item = actual[name]
            filenames = {name + suffix + '.f32'}
            if name == 'initial':
                filenames.add('initial-input.f32')
            if (not isinstance(item, dict) or item.get('file') not in filenames
                    or item['file'] in seen or item.get('shape') != shape
                    or not isinstance(item.get('shape'), list)
                    or any(type(v) is not int for v in item['shape'])
                    or item.get('dtype') != 'float32_le'
                    or not isinstance(item.get('sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
                raise ValueError('Invalid complete reference tensor contract')
            seen.add(item['file'])

    group(fixture.get('inputs'), required_inputs)
    outputs = fixture.get('outputs')
    if not isinstance(outputs, list) or len(outputs) != steps:
        raise ValueError('Reference must contain every denoising step')
    for index, output in enumerate(outputs):
        group(output, {'prediction': latent, 'step': latent}, '-' + str(index))
    group(fixture.get('final'), required_final)
    return 9 + 2 * steps


def reviewed_shared_reference(path, binding, package_manifest=None):
    """Authenticate a whole reference and its selected source-package binding."""
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    reviewed = REVIEWED_SHARED_REFERENCES.get(digest)
    if reviewed is None or reviewed['source_manifest_sha256'] != binding.get('source_manifest_sha256'):
        raise ValueError('Shared-package oracle has no matching reviewed source binding')
    if package_manifest is not None:
        package_manifest = Path(package_manifest)
        if package_manifest.is_symlink() or package_manifest.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Unbounded or indirect shared manifest')
        data = package_manifest.read_bytes()
        if hashlib.sha256(data).hexdigest() != binding.get('shared_manifest_sha256'):
            raise ValueError('Shared result manifest binding differs')
        manifest = json.loads(data)
        matches = [i for i in manifest['instances']
                   if i['source_manifest_sha256'] == binding['source_manifest_sha256']]
        if (manifest.get('schema_version') != 3 or len(matches) != 1
                or matches[0]['runtime_bindings'] != binding.get('runtime_bindings')):
            raise ValueError('Shared result instance binding differs')
        from pipeline_package import select_shared_instance
        fixture = json.loads(path.read_text())
        target_config = fixture['config']
        selected, runtime_target = select_shared_instance(
            manifest['instances'], target_config['packed_width'] * 16, target_config['packed_height'] * 16,
            len(fixture['ids']))
        if (selected['config'] != target_config or binding.get('runtime_target') != runtime_target or
                selected['source_manifest_sha256'] != binding['source_manifest_sha256']):
            raise ValueError('Shared result runtime target binding differs')
        # A new self-consistent shared manifest is not a trusted source manifest.
        # The selected source object must retain the registry's immutable digest.
        objects = package_manifest.parent / 'objects'
        source = objects / binding['source_manifest_sha256']
        if (objects.is_symlink() or source.is_symlink() or not source.is_file()
                or source.stat().st_size > 4 * 1024 * 1024):
            raise ValueError('Missing bounded source manifest object')
        source_bytes = source.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != binding['source_manifest_sha256']:
            raise ValueError('Pinned source manifest object differs')
        pinned = json.loads(source_bytes)
        if (pinned.get('schema_version') != 2 or pinned.get('portable') is not True
                or matches[0]['config'] != pinned.get('config')
                or matches[0]['runtime_bindings'] != pinned.get('files')):
            raise ValueError('Shared instance differs from the pinned full source inventory')
    return {'fixture_sha256': digest, **reviewed}
