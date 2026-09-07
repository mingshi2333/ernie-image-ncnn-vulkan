"""Verified model selection and native completion records for a single timing run."""
import json
import math
import struct
from pathlib import Path

from package_model import verify_package
from package_dynamic_model import verify_shared_package
from pipeline_package import select_shared_instance


SETTING_NAMES = ('device', 'precision', 'vae_device', 'vae_convolution', 'text_device',
                 'text_down_vector', 'threads', 'steps', 'seed', 'dit_weights',
                 'gpu_reserve_mib', 'dit_cache_mib', 'ram_reserve_mib', 'model_loading')


def requested_settings(args):
    return {**{name: getattr(args, name) for name in SETTING_NAMES},
            'img2img': args.strength is not None, 'strength': args.strength,
            'resize': args.resize if args.strength is not None else 'none'}


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def denoising_steps(steps, strength):
    if strength is None:
        return steps
    return min(steps, max(1, math.floor(float32(float32(steps * float32(strength)) + .5))))


def verify_noise(path, width, height):
    raw = Path(path).read_bytes()
    if len(raw) != (width // 16) * (height // 16) * 128 * 4:
        raise ValueError('Saved FP32 noise byte count differs from runtime dimensions')
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)):
        raise ValueError('Saved FP32 noise contains non-finite values')


def verify_benchmark_package(model, width=None, height=None):
    """Authenticate all files/objects, retaining schema-1/2 fixed geometry rules."""
    if (width is None) != (height is None):
        raise ValueError('Specify width and height together')
    raw = json.loads((Path(model) / 'manifest.json').read_text())
    if raw.get('schema_version') == 3:
        manifest = verify_shared_package(model)
        instance, _ = select_shared_instance(manifest['instances'], width, height)
        return manifest, instance['config']
    manifest, _ = verify_package(model)
    cfg = manifest['config']
    if width is not None and (width, height) != (cfg['packed_width'] * 16, cfg['packed_height'] * 16):
        raise ValueError('Requested dimensions differ from the fixed package')
    return manifest, cfg


def validate_native_report(report, args, manifest, width, height):
    """Bind observed source/geometry/settings; logs never establish identity."""
    if type(report) is not dict or type(report.get('schema_version')) is not int or report['schema_version'] != 1 or report.get('status') != 'success':
        raise ValueError('Missing successful native generation report')
    if report.get('shape_order') != 'WH' or report.get('shape') != [width, height]:
        raise ValueError('Native report dimensions differ from the request')
    if type(report.get('trace_enabled')) is not bool or report['trace_enabled'] != args.trace:
        raise ValueError('Native report tracing differs from the request')
    if type(report.get('allocation_instrumentation')) is not bool:
        raise ValueError('Native report must identify allocation instrumentation')
    settings = requested_settings(args)
    observed = report.get('request')
    if observed != settings or any(type(observed[k]) is not type(v) for k, v in settings.items() if k != 'strength'):
        raise ValueError('Native report settings differ from the request')
    if args.strength is not None and type(observed['strength']) not in (int, float):
        raise ValueError('Native report strength is not numeric')
    uses_vulkan = args.device == 'vulkan' or args.vae_device == 'vulkan'
    index = report.get('vulkan_gpu_index')
    if (type(index) is not int or (uses_vulkan and not 0 <= index <= 63)
            or (not uses_vulkan and index != -1) or (args.gpu is not None and index != args.gpu)):
        raise ValueError('Native report Vulkan device differs from the request')
    loading = report.get('model_loading_requested')
    if loading not in ('stdio', 'mapped') or (args.model_loading != 'default' and loading != args.model_loading):
        raise ValueError('Native report loading policy differs from the request')
    ids = report.get('token_ids')
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 2048 or
            any(type(v) is not int or not 0 <= v < 131072 for v in ids)):
        raise ValueError('Native report has invalid consumed token IDs')
    if not isinstance(report.get('prompt'), str):
        raise ValueError('Native report is missing consumed prompt')
    pe = report.get('pe', {})
    if type(pe) is not dict or pe.get('enabled') is not (args.pe_model is not None):
        raise ValueError('Native report prompt enhancement differs from the request')
    if args.pe_model and (pe.get('greedy') is not True or pe.get('max_tokens') != args.pe_max_tokens):
        raise ValueError('Native report PE settings differ from the request')
    schema = manifest['schema_version']
    binding = None
    if schema == 3:
        selected, target = select_shared_instance(manifest['instances'], width, height, len(ids))
        cfg = selected['config']
        source = target['source_config'] if target else cfg
        binding = {'source_manifest_sha256': selected['source_manifest_sha256'],
                   'runtime_bindings': selected['runtime_bindings'], 'runtime_target': target}
    else:
        cfg = source = manifest['config']
        if len(ids) > cfg['text_bucket']:
            raise ValueError('Native tokens exceed fixed package capacity')
    expected = {'schema_version': schema, 'source_width': source['packed_width'] * 16,
                'source_height': source['packed_height'] * 16, 'text_bucket': cfg['text_bucket'],
                'dit_text_tokens': cfg['dit_text_tokens']}
    if report.get('model') != expected or any(type(v) is not int for v in report['model'].values()):
        raise ValueError('Native report source does not match verified package selection')
    for name in ('generation_seconds', 'image_write_seconds', 'total_seconds', 'vae_and_image_encode_seconds'):
        value = report.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Native report has invalid timing')
    for name, keys in (
        ('placement_requests', ('gpu', 'ram', 'budget_unavailable')),
        ('weight_cache', ('hits', 'loads', 'peak_charged_bytes', 'peak_nets', 'evictions', 'budget_unavailable')),
    ):
        values = report.get(name, {})
        if type(values) is not dict or set(values) != set(keys) or any(type(v) is not int or not 0 <= v <= 2**64-1 for v in values.values()):
            raise ValueError('Native report has invalid memory counters')
    progress = report.get('progress')
    if not isinstance(progress, list):
        raise ValueError('Native report is missing stage progress')
    for item in progress:
        if (not isinstance(item, dict) or item.get('stage') not in ('verify', 'text', 'denoise') or
                type(item.get('current')) is not int or type(item.get('total')) is not int or
                type(item.get('seconds')) not in (int, float) or
                not math.isfinite(item['seconds']) or item['seconds'] < 0):
            raise ValueError('Native report has invalid stage progress')
    text = [item for item in progress if item['stage'] == 'text']
    if len(text) != 1 or text[0]['current'] != len(ids) or text[0]['total'] != cfg['text_bucket']:
        raise ValueError('Native report text progress differs from consumed tokens')
    steps = [item for item in progress if item['stage'] == 'denoise']
    expected_steps = denoising_steps(args.steps, args.strength)
    if ([item['current'] for item in steps] != list(range(1, expected_steps + 1))
            or any(item['total'] != expected_steps for item in steps)):
        raise ValueError('Native report does not contain the requested complete denoising schedule')
    return cfg, binding
