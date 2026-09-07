import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# Reproduce the completed read-only audit without changing its original record.
base = Path(__file__).resolve().parent
case = Path('/var/tmp/ernie-runtime-squares-v1/768')
rows = json.loads((case / 'comparison.json').read_text())['comparisons']
byname = {r['name']: r for r in rows}
fixture = json.loads((case / 'official/reference/fixture.json').read_text())
schedule = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.)
schedule.set_timesteps(sigmas=torch.linspace(1., 0., 9)[:-1], device='cpu')
sigmas = schedule.sigmas.numpy()
results = []
def read(origin, name):
    directory = case / ('native/trace' if origin == 'native' else 'official/reference')
    filename = fixture['inputs']['initial']['file'] if origin == 'official' and name == 'initial' else name + '.f32'
    raw = (directory / filename).read_bytes()
    digest = byname[name]['native_sha256' if origin == 'native' else 'reference_sha256']
    assert hashlib.sha256(raw).hexdigest() == digest
    values = np.frombuffer(raw, '<f4')
    assert values.size == 294912 and np.isfinite(values).all()
    return values
for origin in ('native', 'official'):
    for step in range(8):
        sample = read(origin, 'initial' if step == 0 else f'step-{step-1}')
        prediction = read(origin, f'prediction-{step}')
        actual = read(origin, f'step-{step}')
        delta = np.float32(sigmas[step+1] - sigmas[step])
        expected = np.add(sample, np.multiply(delta, prediction, dtype='f4'), dtype='f4')
        results.append(dict(origin=origin, step=step, delta=float(delta), elements=actual.size,
            exact_bytes=actual.tobytes() == expected.tobytes(),
            max_abs=float(np.abs(actual.astype('f8') - expected.astype('f8')).max())))
saved = json.loads((base / 'euler-recurrence.json').read_text())
assert saved['rows'] == results
assert saved['all_exact'] == all(row['exact_bytes'] for row in results)
assert saved['total_elements'] == sum(row['elements'] for row in results)
print(json.dumps({'reproduced_saved_rows':len(results),'all_exact':saved['all_exact']}))
