import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
out = root / 'outputs/cache-headroom-v1'
results = json.loads((out / 'checks.json').read_text()) if (out / 'checks.json').exists() else []
targets = ['ernie-image', 'ernie-host-memory-contract', 'ernie-weight-session-contract',
           'ernie-pipeline-api-contract', 'ernie-request-validation-contract']
for build in ('build-dev', 'build-install-cpu'):
    for kind, command in (
        ('configure', ['cmake', '-S', str(root), '-B', build]),
        ('build', ['cmake', '--build', build, '--parallel', '2', '--target', *targets]),
        ('ctest', ['ctest', '--test-dir', build, '--output-on-failure', '-V', '-R',
                   '^(host_memory_cpu|weight_session_.*|installed_cpp_consumer)$' if '--policy-only' in sys.argv else
                   '^(pipeline_api_contract|request_validation_contract|cli_contract|host_memory_cpu|weight_session_cpu|weight_session_vulkan|weight_session_mapped_vulkan|weight_session_bf16_file_vulkan|weight_session_mapped_bf16_file_vulkan|installed_cpp_consumer)$'])
    ):
        log = out / f'{build}-{kind}.log'
        if log.exists():
            archived = out / f'{build}-{kind}-attempt-{len(results)}.log'
            log.rename(archived)
            for item in results:
                if item['log'] == str(log):
                    item['log'] = str(archived)
        start = time.monotonic()
        with log.open('w') as stream:
            run = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        item = {'command': command, 'returncode': run.returncode,
                'seconds': time.monotonic() - start, 'log': str(log)}
        results.append(item)
        (out / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(item), flush=True)
        if run.returncode:
            print(log.read_text()[-5000:], flush=True)
            raise SystemExit(run.returncode)
