#!/usr/bin/env python3
"""Run immutable per-case evidence; speed aggregation belongs to port_metrics.

Manifest contains frozen B1 cases. Optional --ports JSON supplies adapter
settings without modifying the frozen manifest. Development permits missing functionality but never declares a win.
"""
import argparse
import hashlib
import json
import subprocess
import os
import signal
import shutil
import time
from pathlib import Path
import numpy as np
from port_adapters import PortAdapter, Unavailable, canonical_sha256, verify_pair, calibration_grid
from package_model import sha256, verify_package, safe_name
from source_inventory import source_files
from port_metrics import summarize_pairs
from acceptance_manifest import verify_inputs


def verify_reference_assets(model):
    manifest_path=Path(model).parent/'assets-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('revision')!='140a052f7919f279de7f697fa54f33bd1c0cac2b':
        raise ValueError('Reference model revision mismatch')
    api=json.loads((Path(model).parent/'model-api.json').read_text())
    expected={e['rfilename']:e for e in api['siblings']}
    if api.get('sha')!=manifest['revision'] or {e['path'] for e in manifest['files']}!=set(expected):
        raise ValueError('Incomplete reference asset inventory')
    for entry in manifest['files']:
        path=Path(model)/safe_name(entry['path']);remote=expected[entry['path']]
        if path.stat().st_size!=entry['size'] or entry['size']!=remote['size'] or sha256(path)!=entry['sha256']:
            raise ValueError('Reference asset checksum mismatch: '+str(path))
        if remote.get('lfs') and entry['sha256']!=remote['lfs']['sha256']:
            raise ValueError('Reference LFS identity mismatch')
    return sha256(manifest_path)


def run_side(adapter,case,out,trace,timeout):
    out.mkdir(parents=True,exist_ok=False)
    record={'status':'incomplete','quality_status':'incomplete','scope':'end_to_end','trace':trace,
            **{k:case[k] for k in ('prompt_sha256','noise_sha256','shape','steps','cfg','pe')},
            **adapter.modes(case),'weight_identity_status':'unproven','weights_canonical_sha256':None}
    record.update(input_id=case['prompt_sha256'],noise_dtype=case['noise_dtype'],
                  model_id=json.dumps(case['model_identity'],sort_keys=True),
                  pe_identity=json.dumps(case['pe'],sort_keys=True),
                  precision_by_stage=record['dtype_by_stage'])
    try:
        command=adapter.command(case,out,trace)
        if adapter.kind=='candidate':verify_package(adapter.model)
        else: record['asset_manifest_sha256']=verify_reference_assets(adapter.model)
        snapshot=out/'runner.snapshot'
        shutil.copy2(adapter.binary,snapshot);command[0]=str(snapshot.resolve())
        record.update(command=command,binary_sha256=sha256(snapshot))
        actual_command=['/usr/bin/time','-v','-o',str(out/'resource.txt'),*command]
        record['actual_command']=actual_command
        (out/'launch.json').write_text(json.dumps(record,indent=2))
        start=time.monotonic_ns()
        with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
            try:
                proc=subprocess.Popen(actual_command,stdout=stdout,stderr=stderr,start_new_session=True)
                code=proc.wait(timeout=timeout);record['status']='ok' if code==0 else 'crashed' if code<0 or code>=128 else 'failed'
                record['exit_code']=code
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                record['status']='timeout';record['exit_code']=None
        end=time.monotonic_ns()
        record.update(started_monotonic_ns=start,finished_monotonic_ns=end,wall_seconds=(end-start)/1e9)
        record['log_observations']=adapter.parse_log((out/'stderr.log').read_text(errors='replace'))
        if record['status']=='ok' and not (out/'image.png').is_file():record['status']='incomplete'
        record['files']={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}
        initial=out/('trace/initial.f32' if adapter.kind=='candidate' else 'initial.f32')
        if initial.exists():
            record['initial_canonical_sha256']=canonical_sha256(np.fromfile(initial,dtype='<f4'),'CHW',tuple(case['noise_shape']))
            if record['initial_canonical_sha256']!=case['noise_sha256']:record['status']='input_mismatch'
    except Unavailable as exc:record.update(status='unavailable',reason=str(exc))
    except (ValueError,OSError) as exc:record.update(status='incomplete',reason=str(exc))
    (out/'result.json').write_text(json.dumps(record,indent=2));return record


def compare(manifest,suite,output,development=False,ports_file=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=Path(manifest).resolve();data=json.loads(manifest.read_text());root=manifest.parent
    (output/'manifest.json').write_bytes(manifest.read_bytes())
    verify_inputs(data,root)
    config=json.loads(Path(ports_file).read_text()) if ports_file else {}
    if ports_file: (output/'ports.json').write_bytes(Path(ports_file).read_bytes())
    splits={'calibration':{'development'},'quality':{'formal'},'performance':{'performance'}}[suite]
    cases=[c for c in data['cases'] if c['split'] in splits]
    pairs=[]
    for case in cases:
        pair={'case_id':case['id'],'input_id':case['prompt_sha256'],'noise_id':case['noise_sha256'],
              'model_id':json.dumps(case['model_identity'],sort_keys=True),'precision':case['dtype_by_stage']['dit'],
              'pe_enabled':case['pe']['enabled'],'trace':suite!='performance'}
        for kind in ('reference','candidate'):
            cfg=config.get(kind)
            if cfg is None:pair[kind]={'status':'incomplete','reason':'No executable/model configuration'};continue
            adapter=PortAdapter(kind,Path(cfg['binary']).resolve(),Path(cfg['model']).resolve(),root,cfg.get('config',{}))
            pair[kind]=run_side(adapter,case,output/case['id']/kind,suite!='performance',data.get('timeout_seconds',1800))
        try:verify_pair(pair['candidate'],pair['reference']);pair['pair_contract']='matched'
        except ValueError as exc:pair.update(pair_contract='unproven_or_mismatched',pair_contract_reason=str(exc))
        pairs.append(pair)
    report={'schema_version':1,'suite':suite,'development':development,'status':'incomplete',
            'superiority_claim':False,'pairs':pairs,'metrics':summarize_pairs(pairs),
            'calibration_grid':calibration_grid(),'manifest_sha256':sha256(manifest),
            'sources':{str(p.relative_to(Path(__file__).resolve().parents[1])):sha256(p) for p in source_files(Path(__file__).resolve().parents[1])},
            'limitations':['No automated full trajectory quality verdict yet; successful process is not quality acceptance',
                           'Weight graph correspondence unproven; cannot close S',
                           'Performance suite observations are not the frozen five-pair AB/BA protocol']}
    (output/'result.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--suite',choices=['calibration','quality','performance'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--development',action='store_true');p.add_argument('--ports',type=Path);a=p.parse_args()
    r=compare(a.manifest,a.suite,a.output,a.development,a.ports)
    print(json.dumps({'status':r['status'],'superiority_claim':r['superiority_claim'],'cases':len(r['pairs'])}))
    if not a.development and r['status']!='complete':raise SystemExit(2)
