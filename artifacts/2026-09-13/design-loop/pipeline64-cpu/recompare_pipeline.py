"""Recheck immutable saved outputs after the row-mask trace reader fix; no model execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
BASE = Path('/var/tmp/ernie-evidence-loop-20260913-v1')
RUN = BASE/'pipeline64-cpu'
OUT = BASE/'pipeline64-cpu-reanalysis'
sys.path.insert(0, str(ROOT/'tools'))
from validate_pipeline import compare_trace_tensor, sha256
from pipeline_reference import full_reference_contract

torch.set_num_threads(2)
old = json.loads((RUN/'result.json').read_text())
assert old['return_code'] == 0 and old['conditioning_source'] == 'native_text_encoder'
assert sha256(RUN/'ernie-image.snapshot') == old['runner_sha256']
ref = RUN/'reference'
assert sha256(ref/'fixture.json') == old['reference_fixture_sha256']
fixture = json.loads((ref/'fixture.json').read_text())
full_reference_contract(fixture, fixture['config'], fixture['prompt'], fixture['steps'])
assert sha256(ref/'reference.png') == fixture['reference_png_sha256']
assert [int(x) for x in (RUN/'trace/ids.txt').read_text().split()] == fixture['ids']
for entry in old['comparisons']:
    assert sha256(RUN/'trace'/(entry['tensor']+'.f32')) == entry['sha256']
gates = json.loads((RUN/'gates.json').read_text())
paths = [RUN/'result.json', RUN/'gates.json', RUN/'native.png', RUN/'native.log', RUN/'ernie-image.snapshot',
         ref/'fixture.json', ref/'reference.png', *sorted((RUN/'trace').glob('*'))]
identities = {str(p.resolve()): sha256(p) for p in paths if p.is_file()}
OUT.mkdir()
shutil.copy2(__file__, OUT/'recompare_pipeline.py')
shutil.copy2(ROOT/'tools/validate_pipeline.py', OUT/'validate_pipeline.py')
result = dict(old)
result.pop('failure', None)
result['passed'] = False
result['comparisons'] = []
result['reanalysis'] = {'scope': 'Read the same completed native output with explicit row-mask semantics; no native rerun',
    'original_result_sha256': sha256(RUN/'result.json'), 'original_gate_sha256': sha256(RUN/'gates.json'),
    'comparison_source_sha256': sha256(ROOT/'tools/validate_pipeline.py'),
    'reanalysis_script_sha256': sha256(__file__), 'input_identities': identities}
entries = [(name, item, True) for name, item in fixture['inputs'].items()]
for i, outputs in enumerate(fixture['outputs']):
    entries.extend((f'{name}-{i}', item, False) for name, item in outputs.items())
entries.extend((name, item, False) for name, item in fixture['final'].items())
for name, item, conditioning in entries:
    result['comparisons'].append(compare_trace_tensor(name, item, ref, RUN/'trace',
        gates['conditioning' if conditioning else old['dit_precision']]))
native = np.array(Image.open(RUN/'native.png').convert('RGB')).astype('f8')
expected = np.array(Image.open(ref/'reference.png').convert('RGB')).astype('f8')
assert native.shape == expected.shape
error = abs(native-expected); gate = gates[old['dit_precision']]
result['png'] = {'shape': list(native.shape), 'mae': float(error.mean()), 'max_abs': float(error.max()),
    'sha256': sha256(RUN/'native.png'), 'passed': bool(error.mean() <= gate['pixel_mae'] and error.max() <= gate['pixel_max'])}
decoded = np.fromfile(RUN/'trace/decoded.f32', '<f4').reshape(3, native.shape[0], native.shape[1])
quantized = (np.clip(decoded/2+.5, 0, 1).transpose(1, 2, 0)*255).round().astype('uint8')
result['native_png_quantization_exact'] = bool(np.array_equal(native, quantized))
result['passed'] = result['native_png_quantization_exact'] and result['png']['passed'] and all(c['passed'] for c in result['comparisons'])
assert all(sha256(Path(p)) == digest for p, digest in identities.items())
(OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'passed': result['passed'], 'tensors_passed': sum(c['passed'] for c in result['comparisons']),
    'tensor_count': len(result['comparisons']), 'png': result['png'],
    'failures': [c for c in result['comparisons'] if not c['passed']]}))
