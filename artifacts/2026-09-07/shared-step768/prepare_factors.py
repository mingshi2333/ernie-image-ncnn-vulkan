import hashlib
import json
from pathlib import Path
import shutil

base = Path(__file__).resolve().parent
out = base / 'factors'
out.mkdir()
original = json.loads((base / 'plan.json').read_text())
replay = json.loads((base / 'native-comparison.json').read_text())
teacher = json.loads((base / 'official-comparison.json').read_text())
assert replay['exact_bytes'] and replay['criterion_passed'] and teacher['criterion_passed']
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert replay['plan_sha256'] == teacher['plan_sha256'] == sha(base / 'plan.json')
for name, expected in original['bindings'].items():
    path = Path(name)
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
fixtures, commands = {}, {}
for phase, changed in (('latent', 'in0'), ('text', 'in1')):
    target = out / (phase + '-fixture')
    target.mkdir()
    entries = {}
    for name, item in original['fixtures']['native']['inputs'].items():
        origin = 'official' if name == changed else 'native'
        expected = original['fixtures'][origin]['inputs'][name]
        source = base / (origin + '-fixture') / item['file']
        assert sha(source) == expected['sha256']
        dest = target / item['file']
        shutil.copyfile(source, dest); dest.chmod(0o444)
        entries[name] = {**expected, 'source': str(source)}
    assert {name for name, item in entries.items() if item['sha256'] != original['fixtures']['native']['inputs'][name]['sha256']} == {changed}
    fixtures[phase] = dict(changed_input=changed, inputs=entries,
        scope='Only this named input changes from the exact native replay; descriptive prediction distance, no official quality gate.')
    (target / 'fixture.json').write_text(json.dumps(fixtures[phase], indent=2) + '\n')
    command = list(original['commands']['native'][0])
    command[command.index('--fixture') + 1] = str(target)
    command[command.index('--output') + 1] = str(out / phase / 'actual.f32')
    commands[phase] = [command]
worker = (base / 'run.py').read_text().replace("('official', 'native')", "('latent', 'text')")
supervisor = (base / 'supervisor.py').read_text().replace("('official', 'native')", "('latent', 'text')").replace('ernie-step768-v1-', 'ernie-step768-factors-v1-')
(out / 'run.py').write_text(worker)
(out / 'supervisor.py').write_text(supervisor)
bindings = dict(original['bindings'])
bound = [base / 'plan.json', base / 'native-comparison.json', base / 'official-comparison.json',
         base / 'native/actual.f32', base / 'official/actual.f32', base / 'prepare_factors.py',
         out / 'run.py', out / 'supervisor.py']
for phase in fixtures:
    bound.extend((out / (phase + '-fixture')).iterdir())
bindings.update({str(p):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in bound})
plan = dict(protocol='shared-768-step6-single-factor-v1', step=6, steps=8,
            native_acceptance_eligible=False, official_gate_applied=False,
            scope='Adaptive development diagnosis after exact native replay and passing teacher forcing. Latent-only and text-only replacements are hybrid inputs, not complete official inputs; their output distances are descriptive and effects are not additive.',
            base_plan_sha256=sha(base / 'plan.json'), source_head=original['source_head'],
            resource_limits=original['resource_limits'], commands=commands, fixtures=fixtures,
            references=dict(native=str(base / 'native-fixture/expected.f32'), official=str(base / 'official-fixture/expected.f32')),
            bindings=bindings)
(out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'plan':sha(out/'plan.json'), 'bindings':len(bindings), 'factors':list(fixtures)}))
