"""Synthetic native producer for benchmark orchestration tests, not model evidence."""
import json
from pathlib import Path
from PIL import Image

FIXED_CONFIG = {'packed_width': 32, 'packed_height': 24, 'text_bucket': 64,
                'dit_text_tokens': 64, 'text_layers': 25, 'dit_layers': 36}
FIXED_MANIFEST = {'schema_version': 2, 'config': FIXED_CONFIG}
NOISE = b'\0' * (32 * 24 * 128 * 4)


def fake_timing(command, timeout, log):
    def value(flag, default=None):
        return command[command.index(flag) + 1] if flag in command else default
    width, height = int(value('--width')), int(value('--height'))
    Image.new('RGB', (width, height)).save(Path(value('--output')))
    request = {k: value('--' + k.replace('_', '-')) for k in (
        'device', 'precision', 'vae_device', 'vae_convolution', 'text_device', 'dit_weights', 'model_loading')}
    request.update({k: int(value('--' + k.replace('_', '-'))) for k in (
        'threads', 'steps', 'seed', 'gpu_reserve_mib', 'dit_cache_mib', 'ram_reserve_mib')})
    strength = float(value('--strength')) if '--strength' in command else None
    request.update(text_down_vector='--text-down-vector' in command, img2img=strength is not None,
                   strength=strength, resize=value('--resize', 'none'))
    steps = request['steps'] if strength is None else max(1, int(request['steps'] * strength + .5))
    report = {'schema_version': 1, 'status': 'success', 'shape': [width, height], 'shape_order': 'WH',
        'model': {'schema_version': 2, 'source_width': 512, 'source_height': 384, 'text_bucket': 64, 'dit_text_tokens': 64},
        'request': request, 'trace_enabled': '--trace-dir' in command, 'allocation_instrumentation': False,
        'vulkan_gpu_index': int(value('--gpu', 0)) if request['device'] == 'vulkan' or request['vae_device'] == 'vulkan' else -1,
        'model_loading_requested': 'mapped' if request['model_loading'] == 'mapped' else 'stdio',
        'placement_requests': {'gpu': 0, 'ram': 0, 'budget_unavailable': 0},
        'weight_cache': dict.fromkeys(('hits', 'loads', 'peak_charged_bytes', 'peak_nets', 'evictions', 'budget_unavailable'), 0),
        'prompt': Path(value('--prompt-file')).read_bytes().decode('utf-8-sig'), 'token_ids': [1, 2, 3],
        'pe': {'enabled': False}, 'generation_seconds': 1., 'image_write_seconds': .25,
        'total_seconds': 1.25, 'vae_and_image_encode_seconds': .5,
        'progress': [{'stage': 'text', 'current': 3, 'total': 64, 'seconds': .5}] +
                    [{'stage': 'denoise', 'current': i, 'total': steps, 'seconds': .01} for i in range(1, steps + 1)]}
    Path(value('--report-json')).write_text(json.dumps(report))
    return {'return_code': 0, 'failure_category': None, 'wall_started_monotonic_ns': 1,
            'wall_finished_monotonic_ns': 2, 'wall_seconds': 1e-9}
