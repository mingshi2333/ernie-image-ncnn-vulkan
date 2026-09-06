"""Explicit CLI/input contracts for the pinned candidate and reference ports.

Both pinned native patchified dumps are CHW. HWC is supported only when
explicitly declared; shape equality never establishes channel identity.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re
import numpy as np


class Unavailable(ValueError):
    """The executable cannot express this configuration faithfully."""


def canonical_latent(raw: np.ndarray, layout: str, shape: tuple[int, int, int]) -> np.ndarray:
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Expected positive CHW shape')
    raw = np.asarray(raw)
    if raw.size != int(np.prod(shape)) or raw.dtype.kind != 'f':
        raise ValueError('Wrong latent size or non-floating dtype')
    c, h, w = shape
    if layout == 'CHW':
        result = raw.reshape(shape)
    elif layout == 'HWC':
        result = raw.reshape(h, w, c).transpose(2, 0, 1)
    else:
        raise ValueError('Unknown latent layout: ' + layout)
    result = np.ascontiguousarray(result, dtype='<f4')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite latent')
    return result


def canonical_sha256(raw, layout, shape):
    return hashlib.sha256(canonical_latent(raw, layout, shape).tobytes()).hexdigest()


def verify_pair(left: dict, right: dict) -> None:
    """Require observed initial hashes and proven weight identity, fail closed."""
    for key in ('prompt_sha256', 'noise_sha256', 'initial_canonical_sha256', 'shape',
                'steps', 'cfg', 'pe', 'dtype_by_stage', 'weights_canonical_sha256'):
        if key not in left or key not in right or left[key] is None or right[key] is None:
            raise ValueError('Missing pair identity: ' + key)
        if left[key] != right[key]:
            raise ValueError('Pair mismatch: ' + key)
    for side in (left, right):
        if side['noise_sha256'] != side['initial_canonical_sha256']:
            raise ValueError('Observed initial dump differs from frozen noise')
        if side.get('weight_identity_status') != 'proven':
            raise ValueError('Unproven weight identity: product comparison only')
        if not side['weights_canonical_sha256']:
            raise ValueError('Empty weight identity')
        if not isinstance(side.get('device_by_stage'),dict):
            raise ValueError('Missing stage devices')
        for stage in ('text', 'dit', 'vae', 'scheduler'):
            if stage not in side['dtype_by_stage'] or stage not in side['device_by_stage'] or not side['device_by_stage'][stage]:
                raise ValueError('Missing stage mode: ' + stage)


@dataclass
class PortAdapter:
    kind: str
    binary: Path
    model: Path
    input_root: Path
    config: dict = field(default_factory=dict)

    def command(self, case: dict, output: Path, trace: bool) -> list[str]:
        if self.kind not in ('candidate', 'reference'):
            raise ValueError('Unknown port kind')
        output = Path(output)
        width, height = case['shape']
        if case.get('cfg') != 1 or case.get('steps') != 8:
            raise Unavailable('Only Turbo 8-step CFG=1 contract is calibrated')
        if case.get('input_image_path') or case.get('split') == 'img2img':
            raise Unavailable('Image-to-image paired saved-noise contract not implemented')
        noise = self.input_root / case['noise_path']
        if case.get('noise_layout') != 'CHW' or case.get('noise_dtype') != '<f4':
            raise ValueError('Frozen noise must be FP32 CHW')
        if noise.stat().st_size != 128 * (height // 16) * (width // 16) * 4:
            raise ValueError('Frozen noise shape mismatch')
        if hashlib.sha256(noise.read_bytes()).hexdigest() != case['noise_sha256']:
            raise ValueError('Frozen noise hash mismatch')
        prompt_bytes = (self.input_root / case['prompt_path']).read_bytes()
        if hashlib.sha256(prompt_bytes).hexdigest() != case['prompt_sha256']:
            raise ValueError('Frozen prompt hash mismatch')
        # The peer strips CR/LF in --prompt-file; literal argv preserves the bytes.
        prompt = prompt_bytes.decode('utf-8')
        if '\0' in prompt:
            raise ValueError('Prompt contains NUL')
        precision = self.config.get('precision', case['dtype_by_stage']['dit'])
        pe = case['pe']
        command = [str(self.binary), '--model', str(self.model), '--prompt', prompt,
                   '--output', str(output/'image.png'), '--width', str(width),
                   '--height', str(height), '--steps', str(case['steps']), '--seed', str(case.get('seed',42))]
        if self.kind == 'reference':
            if precision not in ('fp32', 'bf16'):
                raise Unavailable('Pinned reference explicitly disables FP16 storage and arithmetic')
            command += ['--input-latents', str(noise), '--threads', str(self.config.get('threads',4))]
            command += ['--cpu'] if self.config.get('device','vulkan') == 'cpu' else ['--gpu', str(self.config.get('gpu',0))]
            if precision == 'fp32': command += ['--fp32-storage']
            if not self.config.get('low_vram',True):
                if not self.config.get('device_capacity_preflight_passed',False):
                    raise Unavailable('Device-resident weights require recorded capacity preflight')
                command += ['--no-low-vram']
            command += ['--use-pe' if pe['enabled'] else '--no-pe']
            if trace:
                command += ['--dump-initial-latents',str(output/'initial.f32'),
                            '--dump-latents',str(output/'final.f32'), '--dump-step-prefix',str(output/'step')]
        else:
            if self.config.get('threads',4) != 4:
                raise Unavailable('Pinned candidate fixes CPU threads to 4')
            command += ['--latent',str(noise),'--precision',precision,
                        '--device',self.config.get('device','vulkan'), '--vae-device',self.config.get('vae_device','cpu')]
            if trace: command += ['--trace-dir',str(output/'trace')]
            if pe['enabled']:
                if not self.config.get('pe_model'): raise Unavailable('Candidate PE package missing')
                command += ['--pe-model', str(self.config['pe_model'])]
        if pe['enabled']:
            greedy = pe.get('greedy', pe.get('sampling') == 'greedy')
            if greedy and pe['temperature'] != 0:
                raise ValueError('Greedy PE requires temperature=0 on the pinned reference')
            if self.kind == 'candidate' and greedy: command += ['--pe-greedy']
            for field_name, flag in [('max_new_tokens','--pe-tokens' if self.kind=='reference' else '--pe-max-tokens'),
                                     ('temperature','--pe-temperature'),('top_p','--pe-top-p'),('seed','--pe-seed')]:
                if field_name == 'seed' and greedy:
                    value = pe.get('seed', case.get('seed', 42))
                else:
                    if field_name not in pe: raise ValueError('Missing explicit PE parameter: '+field_name)
                    value = pe[field_name]
                command += [flag,str(value)]
        return command

    def modes(self, case):
        precision=self.config.get('precision',case['dtype_by_stage']['dit'])
        device=self.config.get('device','vulkan')
        if self.kind=='reference':
            dtype={s:precision for s in ('text','dit','vae')}
            devices={s:device for s in ('text','dit','vae')}
        else:
            dtype={'text':'fp32','dit':precision,'vae':'fp32'}
            devices={'text':'cpu','dit':device,'vae':self.config.get('vae_device','cpu')}
        dtype['scheduler']='fp32';devices['scheduler']='cpu' if self.kind=='reference' else device
        if case['pe']['enabled']: dtype['pe']='fp32';devices['pe']='cpu'
        return {'dtype_by_stage':dtype,'device_by_stage':devices}

    def parse_log(self, log: str) -> dict:
        """Keep optional internal observations distinct from external wall time."""
        result={'internal_timing_scope':'diagnostic_only','reported_seconds':None}
        match=re.search(r'(?:total_seconds|elapsed_seconds)\s*[=:]\s*([\d.]+)',log)
        if match: result['reported_seconds']=float(match[1])
        match=re.search(r'DiT loaded: gpu-id=(-?\d+) vulkan=(\d+) bf16_storage=(\d+) low_vram=(\d+)',log)
        if match: result['reference_runtime']=dict(zip(('gpu_id','vulkan','bf16_storage','low_vram'),map(int,match.groups())))
        result['allocation_error']=bool(re.search(r'(out of memory|allocate.*failed|allocation.*failed)',log,re.I))
        return result


def calibration_grid():
    rows=[]
    for kind in ('candidate','reference'):
        for threads in (4,8):
            for precision in ('fp32','fp16','bf16'):
                for low in ((True,False) if kind=='reference' else (None,)):
                    reason=('candidate fixes CPU threads to 4' if kind=='candidate' and threads!=4 else
                            'reference FP16 unsupported' if precision=='fp16' else
                            'device-resident capacity preflight required' if not low else None)
                    rows.append(dict(port=kind,threads=threads,precision=precision,low_vram=low,
                                     status='unavailable' if reason else 'pending',reason=reason))
    return rows
