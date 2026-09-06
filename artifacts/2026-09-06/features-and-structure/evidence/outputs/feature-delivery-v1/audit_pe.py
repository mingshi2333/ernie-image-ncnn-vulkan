"""Recheck saved PE bytes and logits without executing inference again."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATES = dict(nrmse=.0002, atol=.0002, rtol=.0002)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def audit(version):
    run = ROOT/f'outputs/pe-apple-validation-{version}'
    reference = ROOT/f'outputs/pe-apple-reference-{version}'
    result = json.loads((run/'result.json').read_text())
    meta = json.loads((reference/'reference.json').read_text())
    assert result['gates'] == GATES and result['return_code'] == 0
    check(run/'ernie-pe-runner', result['runner_sha256'])
    check(ROOT/'tools/validate_pe.py', result['validator_sha256'])
    check(ROOT/'tools/reference_pe.py', meta['script_sha256'])
    check(ROOT/'models/pe-cpu-v1/manifest.json', result['model_manifest_sha256'])
    check(reference/'reference.json', result['reference_manifest_sha256'])
    for directory, files in ((reference, meta['files']), (run/'native', result['native_files'])):
        for name, expected in files.items():
            assert Path(name).name == name
            check(directory/name, expected)
    for name in ('input-ids.txt', 'generated-ids.txt', 'enhanced.txt'):
        assert result['exact'][name]
        assert (run/'native'/name).read_bytes() == (reference/name).read_bytes()
    assert len(result['logits']) == meta['generated_tokens']
    verdicts = []
    for index, row in enumerate(result['logits']):
        assert row['token'] == index
        expected = np.fromfile(reference/f'logits-{index}.f32', '<f4').astype('f8')
        actual = np.fromfile(run/f'native/logits-{index}.f32', '<f4').astype('f8')
        assert actual.shape == expected.shape == (131072,)
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        error = actual - expected
        nrmse = float(np.sqrt(np.sum(error*error)/np.sum(expected*expected)))
        maximum = float(np.max(np.abs(error)))
        limit = GATES['atol'] + GATES['rtol']*float(np.max(np.abs(expected)))
        for key, value in (('nrmse', nrmse), ('max_abs_error', maximum), ('max_abs_limit', limit)):
            assert np.isclose(row[key], value, rtol=1e-12, atol=1e-15), (index, key)
        passed = nrmse <= GATES['nrmse'] and maximum <= limit
        assert passed == row['passed']
        verdicts.append(passed)
    assert all(verdicts) == result['passed']
    status = dict(line.split(' ', 1) for line in (run/'native/status.txt').read_text().splitlines())
    assert bool(int(status['eos'])) == meta['eos']
    return dict(run=str(run.relative_to(ROOT)), tokens=meta['generated_tokens'], eos=meta['eos'],
                passed=result['passed'], exact_ids_and_text=True, logits_recomputed=len(verdicts),
                worst_nrmse=max(row['nrmse'] for row in result['logits']),
                reference_and_native_hashes_checked=len(meta['files'])+len(result['native_files']),
                result_sha256=digest(run/'result.json'), gates=GATES,
                cache_buffer_changes=int(status['cache_buffer_changes']))


if __name__ == '__main__':
    rows = [audit(version) for version in ('v1', 'v2')]
    output = Path(__file__).with_name('pe-independent-audit.json')
    output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))
