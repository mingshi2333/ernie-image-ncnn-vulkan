#!/usr/bin/env python3
"""Small official/native encoder boundary validation with a 2GiB cgroup ceiling."""
import argparse
import importlib.metadata
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
# Only these independently executed static graphs have reviewed boundary evidence.
# New shapes/exports remain candidates until separately reviewed and explicitly pinned.
REVIEWED_GRAPHS={'32x32': {'head.ncnn.param': '75d493995616b451e51ddecc0dc352a3f200557baab3f98d742cc374ed6d0977', 'head.ncnn.bin': '7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce'}, '64x32': {'head.ncnn.param': '40c2b384c05e9f816aa1ff1703e3bb03bdec73d0c9d38a128556591131a95a95', 'head.ncnn.bin': '7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce'}}


def verify(model):
    """Validate this diagnostic candidate against local pinned official sources.

    This authenticates the reviewed export contract, not arbitrary graph semantics or
    production img2img quality. The native boundary comparison remains mandatory.
    """
    try:
        return _verify(Path(model))
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Malformed encoder contract') from error


def _verify(model):
    try:
        from export_vae_encoder import dimensions, normalize_rgb, rgb_fixture
        from audit_port_weights import official_inventory, OFFICIAL_REVISION, OFFICIAL_REPOSITORY
    except ImportError:
        from tools.export_vae_encoder import dimensions, normalize_rgb, rgb_fixture
        from tools.audit_port_weights import official_inventory, OFFICIAL_REVISION, OFFICIAL_REPOSITORY
    manifest=json.loads((model/'model.json').read_text());fixture=json.loads((model/'fixture.json').read_text())
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    exporter=sha256(ROOT/'tools/export_vae_encoder.py')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version']!=1 or manifest.get('component')!='vae-encoder' or manifest.get('source_sha256')!=exporter or manifest.get('ncnn_revision')!=lock['ncnn']['revision']:
        raise ValueError('Encoder exporter/schema identity mismatch')
    dimensions(fixture['width'],fixture['height']);w,h=fixture['width'],fixture['height']
    fixed={'component':'vae-encoder','text_tokens':0,'official_revision':OFFICIAL_REVISION,
           'diffusers_revision':lock['diffusers']['revision'],'posterior':'mode_first_32_channels_no_sampling',
           'packing':'pixel_unshuffle_2','encoder_bn':{'eps':1e-4,'affine':False},
           'decoder_inverse_bn_eps':1e-5,'boundaries':['mean','packed','normalized'],
           'gates':{'fp32':FP32_GATES},'wrapper_bitwise_equal':[True,True,True]}
    if any(fixture.get(k)!=v for k,v in fixed.items()) or type(fixture['text_tokens']) is not int or type(fixture['encoder_bn']['affine']) is not bool or any(type(x) is not bool for x in fixture['wrapper_bitwise_equal']):
        raise ValueError('Unreviewed encoder mathematical contract')
    official=ROOT/'models/official'
    cfg=official/'vae-config.json';source=json.loads((official/'vae-config.source.json').read_text())
    if source.get('revision')!=OFFICIAL_REVISION or source.get('sha256')!=sha256(cfg) or source.get('url')!=f'{OFFICIAL_REPOSITORY}/resolve/{OFFICIAL_REVISION}/vae/config.json' or fixture.get('vae_config_sha256')!=sha256(cfg):
        raise ValueError('VAE configuration source mismatch')
    config=json.loads(cfg.read_text())
    if config.get('latent_channels')!=32 or config.get('patch_size')!=[2,2] or config.get('batch_norm_eps')!=1e-4:raise ValueError('Unreviewed VAE configuration')
    dist=importlib.metadata.distribution('diffusers')
    origin=json.loads(dist.read_text('direct_url.json'))
    if origin.get('url')!='https://github.com/huggingface/diffusers/archive/'+lock['diffusers']['revision']+'.zip':raise ValueError('Unpinned diffusers source')
    for field,path in [('official_source_sha256','diffusers/models/autoencoders/autoencoder_kl_flux2.py'),('distribution_source_sha256','diffusers/models/autoencoders/vae.py')]:
        if fixture.get(field)!=sha256(Path(dist.locate_file(path))):raise ValueError('Official implementation source mismatch')
    labels=['encoder','quant','bn']
    official_inventory(official,[f'vae-{x}.safetensors' for x in labels])
    weights={x:json.loads((official/f'vae-{x}.manifest.json').read_text())['sha256'] for x in labels}
    sources={x:sha256(official/f'vae-{x}.manifest.json') for x in labels}
    if fixture.get('weights')!=weights or manifest.get('weights')!=weights or fixture.get('source_manifests')!=sources:raise ValueError('Official weight provenance mismatch')
    reviewed=REVIEWED_GRAPHS.get(f'{w}x{h}')
    if reviewed is None or any(sha256(model/name)!=digest for name,digest in reviewed.items()):raise ValueError('Unreviewed encoder graph/weights; independent validation required')
    names={'head.ncnn.param','head.ncnn.bin','fixture.json','conversion.json','trace.json','input.rgb','in0.f32','out0.f32','out1.f32','out2.f32'}
    if set(manifest['files'])!=names:raise ValueError('Incomplete encoder graph/fixture inventory')
    for name,digest in manifest['files'].items():
        if sha256(model/name)!=digest:raise ValueError('Encoder checksum mismatch: '+name)
    trace=json.loads((model/'trace.json').read_text());conversion=json.loads((model/'conversion.json').read_text())
    if trace.get('exporter_sha256')!=exporter or type(trace.get('threads')) is not int or trace['threads']!=2:raise ValueError('Trace exporter mismatch')
    if conversion.get('pnnx_sha256')!=lock['pnnx']['binary_sha256'] or type(conversion.get('return_code')) is not int or conversion['return_code']!=0 or conversion.get('unsupported_diagnostics')!=[] or conversion.get('unconverted')!=[] or conversion['command'][1:]!=['head.pt',f'inputshape=[1,3,{h},{w}]','fp16=0','device=cpu']:raise ValueError('Unreviewed conversion')
    if set(fixture['inputs'])!={'in0'} or set(fixture['expected'])!={'out0','out1','out2'}:raise ValueError('Wrong encoder input/output inventory')
    shapes={'in0':[1,3,h,w],'out0':[1,32,h//8,w//8],'out1':[1,128,h//16,w//16],'out2':[1,128,h//16,w//16]}
    for name,entry in {**fixture['inputs'],**fixture['expected']}.items():
        if entry.get('file')!=name+'.f32' or entry.get('shape')!=shapes[name] or any(type(x) is not int for x in entry['shape']) or entry.get('dtype')!='F32' or entry.get('layout')!='NCHW' or sha256(model/entry['file'])!=entry.get('sha256') or (model/entry['file']).stat().st_size!=int(np.prod(shapes[name]))*4:raise ValueError('Encoder tensor identity/size mismatch')
        if not np.isfinite(np.fromfile(model/entry['file'],'<f4')).all():raise ValueError('Nonfinite encoder tensor')
    rgb=fixture['rgb'];expected_rgb=rgb_fixture(w,h)
    if rgb.get('file')!='input.rgb' or rgb.get('shape')!=[h,w,3] or rgb.get('layout')!='HWC_RGB' or rgb.get('normalization')!='FP32 (v-127.5)*(1/127.5)' or rgb.get('sha256')!=sha256(model/'input.rgb') or (model/'input.rgb').read_bytes()!=expected_rgb.tobytes() or (model/'in0.f32').read_bytes()!=normalize_rgb(expected_rgb).tobytes():raise ValueError('Deterministic RGB/normalization identity mismatch')
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
