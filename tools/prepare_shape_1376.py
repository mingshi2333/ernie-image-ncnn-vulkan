#!/usr/bin/env python3
"""Freeze a non-runnable 1376x768/s64 component plan; never run models or register shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import random
import struct
import subprocess
import os
from audit_shape_contract import audit_package, normalize_graph, CONTRACT_HASHES, RULES, dimensions
from prepare_block import ROOT, sha256
from source_inventory import source_files

SOURCE_SHA = '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1'
TARGET = dict(packed_width=86, packed_height=48, text_bucket=64,
              dit_text_tokens=64, text_layers=25, dit_layers=36)


def candidate_graph(graph, kind, source):
    before, _ = normalize_graph(graph, kind, source)
    if before != CONTRACT_HASHES[kind]:
        raise ValueError('Unreviewed complete source graph')
    values = dimensions(TARGET); lines = []
    for line in graph.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in RULES[kind]:
            start = 4 + int(fields[2]) + int(fields[3])
            for key, formula in RULES[kind][fields[1]].items():
                index = next(i for i in range(start, len(fields)) if fields[i].split('=', 1)[0] == key)
                fields[index] = key + '=' + str(values[formula])
        lines.append(' '.join(fields))
    result = '\n'.join(lines) + '\n'
    after, _ = normalize_graph(result, kind, TARGET)
    if after != before:
        raise ValueError('Candidate changed more than enumerated shape fields')
    return result


def prepare(package, output, official_root, runner, python):
    package=Path(package).resolve();output=Path(output).resolve()
    if output.exists(): raise ValueError('Use a new plan directory')
    if sha256(package/'manifest.json') != SOURCE_SHA:
        raise ValueError('Only the reviewed 1024/s64 source is eligible for this plan')
    audit=audit_package(package)
    manifest=json.loads((package/'manifest.json').read_text())
    # This is only a graph/metadata operation. Weight content must be streamed and
    # verified against these exact pins before any later component execution.
    weights={name: digest for name,digest in manifest['files'].items() if name.endswith('.bin')}
    output.mkdir(parents=True)
    input_path=output/'vae-input.f32';generator=random.Random(20260905)
    with input_path.open('xb') as stream:
        for _ in range(32*96*172):stream.write(struct.pack('<f',generator.gauss(0,1)))
    snapshot=output/'source';records={}
    for path in source_files(ROOT):
        relative=path.relative_to(ROOT);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);records[str(relative)]=sha256(dest)
    frozen_runner=output/'head-runner.snapshot';shutil.copy2(runner,frozen_runner)
    graphs=[]
    for row in audit['graphs']:
        path=package/row['path'];text=candidate_graph(path.read_text(),row['kind'],audit['config'])
        dest=output/'graphs'/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        graphs.append({'path':row['path'],'kind':row['kind'],'source_sha256':row['sha256'],
                       'candidate_sha256':sha256(dest),'contract_sha256':row['contract_sha256']})
    lock=json.loads((snapshot/'sources.lock.json').read_text())
    python_invocation=os.path.abspath(python)
    environment=json.loads(subprocess.check_output([python_invocation,'-c',
        "import sys,json,importlib.util,importlib.metadata as m; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'executable':sys.executable,'packages':{n:{'origin':importlib.util.find_spec(n).origin,'version':m.version(n),'direct_url':m.distribution(n).read_text('direct_url.json')} for n in ('torch','diffusers','safetensors')}}))"],text=True))
    official_metadata={name:sha256(Path(official_root)/name) for name in ('vae-config.json','vae-decoder.manifest.json','vae-post-quant.manifest.json')}
    sources={'files':records,'head_runner_sha256':sha256(frozen_runner),
             'python':python_invocation,'python_resolved_executable':str(Path(python).resolve()),
             'python_sha256':sha256(python),'python_environment':environment,
             'official_root':str(Path(official_root).resolve()),'official_metadata_sha256':official_metadata,
             'official_revision':lock['official_model']['revision'],'ncnn_revision':lock['ncnn']['revision']}
    (output/'source-identity.json').write_text(json.dumps(sources,indent=2)+'\n')
    reference=output/'official-vae';candidate=output/'vae-candidate'
    plan={'schema_version':1,'status':'prepared_not_executed','native_acceptance_eligible':False,
          'production_registry_modified':False,'scope':'fixed component candidates; no formal corpus or quality result',
          'source_manifest_sha256':SOURCE_SHA,'source_package':str(package),'source_config':audit['config'],
          'target_config':TARGET,'shape_order':'WH','output_shape':[1376,768],'vae_latent_shape':[1,32,96,172],
          'input':{'path':str(input_path),'sha256':sha256(input_path),'size_bytes':input_path.stat().st_size,
                   'shape':[1,32,96,172],'dtype':'<f4','generator':'Python random.Random(20260905).gauss then little-endian FP32; saved bytes define input'},
          'image_tokens':4128,'sequence_tokens':4192,'source_identity_sha256':sha256(output/'source-identity.json'),
          'graphs':graphs,'weights_expected_sha256':weights,'weights_content_verified':False,
          'weights_policy':'No bin rewritten; stream every selected component bin against source manifest before execution',
          'resources':{'official_torch_threads':2,'native_ncnn_threads':4,'scope_cpu_budget':2,'cpu_affinity':[12,14],'memory_max_bytes':16*1024**3,
                       'swap_max_bytes':0,'host_available_min_bytes':3*1024**3,'gpu_authorized':False,
                       'guard_status':'required_external_supervisor_not_started','timeout_seconds':1800},
          'steps':[
              {'id':'official_vae','status':'requires_resource_slot_and_installed_runtime_freeze',
               'argv':[python_invocation,str(snapshot/'tools/export_vae.py'),'--output',str(reference),
                       '--height','96','--width','172','--reference-only','--fixed-1376x768','--threads','2',
                       '--official-root',str(Path(official_root).resolve()),'--input-f32',str(input_path),
                       '--runtime-identity',str(output/'runtime/identity.json'),
                       '--runtime-report',str(output/'worker-runtime')]},
              {'id':'native_vae','status':'requires_official_fixture_then_specialize_vae_full_template_and_bin_hash_audit',
               'candidate_directory':str(candidate),'runner':str(frozen_runner),
               'validation':'validate_dit_heads.py --cpu-only --vae-convolution direct; unchanged fixture gates'},
              {'id':'heads','status':'requires_frozen_official_and_native_input_output_fixtures',
               'shape':{'height':48,'width':86,'text_tokens':64},
               'tool':'export_dit_heads.py; export one component at a time under guard; verify all input head 8 outputs and output head 1 output'},
              {'id':'dit','status':'requires_single_block_then_full_36_block_same_input_official_validation',
               'sequence_tokens':4192,'bins':'same source bytes; no numerical changes'},
              {'id':'pipeline','status':'blocked_until_components_pass_and_new_source_registry_independently_reviewed'}]}
    runtime_environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                         'MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    runtime_environment.pop('PYTHONPATH',None)
    subprocess.run([python_invocation,str(snapshot/'tools/vae_reference_scope.py'),'--capture-runtime',str(output/'runtime')],
                   env=runtime_environment,check=True)
    # Observe the actual exporter/import/input path without constructing a model.
    from vae_reference_scope import process_identity
    probe_process=subprocess.Popen(plan['steps'][0]['argv']+['--runtime-probe',str(output/'worker-import-probe')],
                                   env=runtime_environment)
    plan['worker_probe_process']=process_identity(probe_process.pid)
    if probe_process.wait()!=0:raise ValueError('Worker import probe failed')
    runtime_path=output/'runtime/identity.json'
    runtime=json.loads(runtime_path.read_text())
    probe=json.loads((output/'worker-import-probe/identity.json').read_text())
    if (probe['process']!=plan['worker_probe_process'] or probe['prefix']!=runtime['prefix']
        or probe['executable']!=runtime['executable']):
        raise ValueError('Worker import probe interpreter differs')
    for name,row in probe['files'].items():
        if name in runtime['files'] and runtime['files'][name]!=row:
            raise ValueError('Runtime changed between parent and worker probes: '+name)
    runtime['parent_only_files']=sorted(set(runtime['files'])-set(probe['files']))
    runtime['worker_only_files']=sorted(set(probe['files'])-set(runtime['files']))
    runtime['files'].update(probe['files'])
    runtime['required_files']=sorted(probe['files'])
    runtime['required_mapped_files']=probe['mapped_files']
    runtime['required_coverage_scope']='all files and mappings observed in actual exporter import/input probe before model construction'
    runtime['worker_probe_sha256']=sha256(output/'worker-import-probe/identity.json')
    runtime_path.write_text(json.dumps(runtime,indent=2)+'\n')
    plan['runtime_identity_sha256']=sha256(runtime_path)
    launcher=['systemd-run','--user','--scope','--quiet',
              '--unit=ernie-vae1376-'+hashlib.sha256(str(output).encode()).hexdigest()[:12],
              '-p','MemoryMax=17179869184','-p','MemorySwapMax=0','-p','CPUQuota=200%',
              'taskset','-c','12,14','env','-u','PYTHONPATH','CUDA_VISIBLE_DEVICES=',
              'OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','HF_HUB_OFFLINE=1',
              'TRANSFORMERS_OFFLINE=1',python_invocation,str(snapshot/'tools/vae_reference_scope.py'),
              '--run-plan',str(output/'plan.json')]
    plan['launcher_argv']=launcher
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','output','official-root','runner','python'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();plan=prepare(a.package,a.output,a.official_root,a.runner,a.python)
    print(json.dumps({'status':plan['status'],'graphs':len(plan['graphs']),'weights_content_verified':False}))

if __name__=='__main__':main()
