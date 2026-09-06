#!/usr/bin/env python3
"""Small official/native encoder boundary validation with a 2GiB cgroup ceiling."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import numpy as np
try:
    from package_model import ROOT,sha256
except ImportError:
    from tools.package_model import ROOT,sha256

LIMIT=2*1024**3
MIN_AVAILABLE=3*1024**3
FP32_GATES={"atol":0.0002,"rtol":0.0002,"nrmse":0.00002}


def verify(model):
    manifest=json.loads((model/'model.json').read_text());fixture=json.loads((model/'fixture.json').read_text())
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    names={'head.ncnn.param','head.ncnn.bin','fixture.json','conversion.json','trace.json','input.rgb','in0.f32','out0.f32','out1.f32','out2.f32'}
    if manifest.get('component')!='vae-encoder' or manifest.get('ncnn_revision')!=lock['ncnn']['revision'] or fixture.get('official_revision')!=lock['official_model']['revision'] or manifest.get('weights')!=fixture.get('weights'):
        raise ValueError('Encoder source identity mismatch')
    if set(manifest['files'])!=names:raise ValueError('Incomplete encoder graph/fixture inventory')
    for name,digest in manifest['files'].items():
        if sha256(model/name)!=digest:raise ValueError('Encoder checksum mismatch: '+name)
    for entry in [*fixture['inputs'].values(),*fixture['expected'].values()]:
        if Path(entry['file']).name!=entry['file'] or sha256(model/entry['file'])!=entry['sha256'] or (model/entry['file']).stat().st_size!=int(np.prod(entry['shape']))*4:raise ValueError('Encoder tensor checksum/size mismatch')
    return fixture


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Cannot read MemAvailable')


def rss(pids):
    total=0
    for pid in pids:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024;break
        except (OSError,ProcessLookupError):pass
    return total


def bounded_run(command,output,timeout=900):
    """No fallback without a hard cgroup cap. Low memory stops the whole scope."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(command=command,limit_bytes=LIMIT,min_available_bytes=MIN_AVAILABLE,passed=False,
                rss_guard_bytes=1900*1024**2,peak_group_rss_bytes=0,peak_cgroup_memory_bytes=0)
    uid=os.getuid();unit='ernie-f2-'+uuid.uuid4().hex
    env={**os.environ,'XDG_RUNTIME_DIR':f'/run/user/{uid}','DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{uid}/bus',
         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
    wrapped=['systemd-run','--user','--scope','--quiet','--unit='+unit,'--property=MemoryMax='+str(LIMIT),'--property=MemorySwapMax=0',*command]
    result['supervised_command']=wrapped
    cg=Path(f'/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope')
    start=time.monotonic();process=None
    try:
        if available()<MIN_AVAILABLE:raise RuntimeError('Host available memory below3GiB before launch')
        with (output/'runner.log').open('w') as log:
            process=subprocess.Popen(wrapped,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while process.poll() is None:
                if available()<MIN_AVAILABLE:raise RuntimeError('Host available memory below3GiB')
                pids=[]
                if cg.exists():
                    result['cgroup_seen']=True
                    pids=[int(p) for p in (cg/'cgroup.procs').read_text().split()]
                    current=int((cg/'memory.current').read_text());result['peak_cgroup_memory_bytes']=max(result['peak_cgroup_memory_bytes'],current)
                    result['memory_events']=(cg/'memory.events').read_text()
                else:
                    # Include launcher/children before the transient scope exists.
                    for p in Path('/proc').iterdir():
                        if p.name.isdigit():
                            try:
                                if os.getpgid(int(p.name))==process.pid:pids.append(int(p.name))
                            except ProcessLookupError:pass
                current=rss(pids);result['peak_group_rss_bytes']=max(result['peak_group_rss_bytes'],current)
                if current>result['rss_guard_bytes']:raise RuntimeError('Process-group RSS crossed1900MiB guard')
                if time.monotonic()-start>timeout:raise RuntimeError('Stage timeout')
                time.sleep(.02)
            result['return_code']=process.wait()
            result['passed']=process.returncode==0 and result.get('cgroup_seen',False)
            if not result['passed']:result['failure']='Stage failed or hard cgroup could not be observed'
    except (OSError,RuntimeError) as error:
        result['failure']=str(error)
        if process is not None and process.poll() is None:
            subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=KILL',unit+'.scope'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            result['return_code']=process.wait(timeout=10)
    result['seconds']=time.monotonic()-start
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare_arrays(actual,reference,limits):
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():raise ValueError('Invalid encoder boundary')
    a=actual.astype(np.float64);r=reference.astype(np.float64);d=a-r
    nrmse=float(np.linalg.norm(d)/max(np.linalg.norm(r),1e-30));maximum=float(np.abs(d).max())
    cap=limits['atol']+limits['rtol']*float(np.abs(r).max())
    return dict(nrmse=nrmse,max_abs_error=maximum,max_abs_limit=cap,nrmse_limit=limits['nrmse'],passed=nrmse<=limits['nrmse'] and maximum<=cap)


def validate(model,runner,output):
    fixture=verify(model)
    if fixture['component']!='vae-encoder' or fixture['boundaries']!=['mean','packed','normalized'] or set(fixture['expected'])!={'out0','out1','out2'}:raise ValueError('Wrong encoder boundaries')
    if fixture['encoder_bn']!={'eps':1e-4,'affine':False} or fixture['decoder_inverse_bn_eps']!=1e-5:raise ValueError('Unexpected asymmetric BN identity')
    if fixture.get('gates',{}).get('fp32')!=FP32_GATES or fixture.get('wrapper_bitwise_equal')!=[True,True,True] or any(type(v) is not bool for v in fixture.get('wrapper_bitwise_equal',[])):raise ValueError('Unreviewed quality gates or wrapper mismatch')
    try:
        from export_vae_encoder import dimensions
    except ImportError:
        from tools.export_vae_encoder import dimensions
    dimensions(fixture['width'],fixture['height'])
    if type(fixture['encoder_bn']['affine']) is not bool:raise ValueError('Wrong BN affine type')
    w,h=fixture['width'],fixture['height']
    expected_shapes=[[1,32,h//8,w//8],[1,128,h//16,w//16],[1,128,h//16,w//16]]
    if fixture['inputs']['in0']['shape']!=[1,3,h,w] or [fixture['expected'][f'out{i}']['shape'] for i in range(3)]!=expected_shapes:raise ValueError('Wrong encoder tensor shapes')
    try:
        from export_vae_encoder import normalize_rgb
    except ImportError:
        from tools.export_vae_encoder import normalize_rgb
    rgb_meta=fixture['rgb'];rgb_path=model/rgb_meta['file']
    if rgb_meta['file']!='input.rgb' or rgb_meta['shape']!=[h,w,3] or sha256(rgb_path)!=rgb_meta['sha256']:raise ValueError('RGB identity mismatch')
    rgb=np.fromfile(rgb_path,np.uint8).reshape(h,w,3)
    if normalize_rgb(rgb).tobytes()!= (model/'in0.f32').read_bytes():raise ValueError('RGB FP32 normalization bytes mismatch')
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'ernie-head-runner.snapshot';shutil.copy2(runner,snapshot)
    command=[str(snapshot.resolve()),'--model',str(model.resolve()),'--fixture',str(model.resolve()),'--output',str((output/'actual').resolve()),'--component','vae-encoder','--width',str(fixture['width']),'--height',str(fixture['height']),'--text-tokens','0','--backend','cpu','--precision','fp32','--vae-convolution','direct']
    run=bounded_run(command,output/'process');result=dict(passed=False,process=run,runner_sha256=sha256(snapshot),model_sha256=sha256(model/'model.json'),boundaries={})
    try:
        if run['passed']:
            for name,entry in fixture['expected'].items():
                actual=output/'actual'/f'{name}.f32'
                values=compare_arrays(np.fromfile(actual,'<f4'),np.fromfile(model/entry['file'],'<f4'),FP32_GATES)
                result['boundaries'][fixture['boundaries'][int(name[-1])]]={**values,'actual_sha256':sha256(actual)}
            result['passed']=all(v['passed'] for v in result['boundaries'].values())
    except (ValueError,OSError) as error:result['failure']=str(error)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--runner',type=Path);p.add_argument('--width',type=int,default=32);p.add_argument('--height',type=int,default=32);p.add_argument('--export-only',action='store_true');a=p.parse_args()
    if a.output.exists() or (a.model and a.export_only) or (not a.export_only and not a.runner):p.error('Use new output and --runner, or --export-only with no --model')
    a.output.mkdir(parents=True);result={'scope':'small encoder boundary only, not15case img2img','passed':False}
    model=a.model or a.output/'model'
    if not a.model:
        for stage in ['reference','convert']:
            command=[sys.executable,str(ROOT/'tools/export_vae_encoder.py'),'--output',str(model.resolve()),'--width',str(a.width),'--height',str(a.height),'--stage',stage]
            r=bounded_run(command,a.output/stage)
            result[stage]=r
            if not r['passed']:
                (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return 1
    if a.export_only:result.update(export_complete=True,validation_pending=True)
    else:
        result['validation']=validate(model,a.runner,a.output/'validation');result['passed']=result['validation']['passed']
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if a.export_only or result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
