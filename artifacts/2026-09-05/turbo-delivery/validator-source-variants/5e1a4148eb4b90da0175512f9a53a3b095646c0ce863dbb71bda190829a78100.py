#!/usr/bin/env python3
"""Check native prompt-to-PNG against staged official modules with identical initial noise."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
from PIL import Image
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ernie_image.pipeline_ernie_image import ErnieImagePipeline
from export_dit_block import load_block,make_inputs,metrics,save_tensor
from export_dit_heads import load_heads
from export_vae import load_vae
from prepare_block import ROOT,sha256
from validate_text import real_reference

def reference(package,prompt,output,steps,device='cpu',initial_path=None):
    cfg=json.loads((package/'manifest.json').read_text())['config']
    h,w,text_tokens=cfg['packed_height'],cfg['packed_width'],cfg['dit_text_tokens']
    output.mkdir(parents=True)
    text_dir=output/'text'
    meta=real_reference([package/f'text/block-{i:02d}' for i in range(25)],prompt,text_dir)
    text_valid=torch.from_numpy(np.fromfile(text_dir/'expected.f32','<f4').copy()).reshape(1,len(meta['ids']),3072)
    text=torch.zeros(1,text_tokens,3072);text[:,:len(meta['ids'])]=text_valid
    initial=torch.randn(1,128,h,w,generator=torch.Generator().manual_seed(20260905))
    if initial_path is not None:
        values=np.fromfile(initial_path,'<f4')
        if values.size!=128*h*w or not np.isfinite(values).all():
            raise ValueError('Saved initial latent has the wrong shape or nonfinite values')
        initial=torch.from_numpy(values.copy()).reshape(1,128,h,w)
    save_tensor(output/'initial.f32',initial)
    synth,freqs=make_inputs(h,w,text_tokens,len(meta['ids']),20260905)
    fixture={'prompt':prompt,'config':cfg,'ids':meta['ids'],'steps':steps,
        'scope':'Native prompt-to-PNG versus staged pinned official FP32 modules; PE off, CFG=1; identical saved initial latent',
        'reference_environment':{'torch':torch.__version__,'pipeline_source_sha256':sha256(inspect.getfile(ErnieImagePipeline)),
            'dit_blocks_device':device,'heads_text_vae_device':'cpu','dtype':'float32','allow_tf32':False,
            'cuda_device':torch.cuda.get_device_name() if device=='cuda' else None},
        'inputs':{'initial':save_tensor(output/'initial-input.f32',initial),'text':save_tensor(output/'text.f32',text_valid),
                  'padded-text':save_tensor(output/'padded-text.f32',text)},'outputs':[],
        'source_sha256':sha256(__file__),'complete':False}
    for i in range(3):fixture['inputs'][f'constant-{i}']=save_tensor(output/f'constant-{i}.f32',synth[7+i])
    heads,_=load_heads()
    scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,shift=4.)
    scheduler.set_timesteps(sigmas=torch.linspace(1.,0.,steps+1)[:-1],device='cpu')
    sample=initial
    path=output/'fixture.json'
    path.write_text(json.dumps(fixture,indent=2,ensure_ascii=False)+'\n')
    # Stream exactly one official FP32 block on the selected device. Keep the
    # original CPU heads/scheduler/VAE, and forbid TF32 precision substitution.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    block_freqs=freqs.to(device)
    block_mask=synth[-1][None,None].to(device)
    for i,t in enumerate(scheduler.timesteps):
        start=time.perf_counter();captured={}
        hook=heads.final_norm.register_forward_pre_hook(lambda m,inputs:captured.update(x=inputs[0],c=inputs[1]))
        hook2=heads.adaLN_modulation.register_forward_hook(lambda m,inputs,out:captured.update(ada=out))
        heads(sample,t.reshape(1),text,torch.tensor([len(meta['ids'])]));hook.remove();hook2.remove()
        x=captured['x'].to(device);temb=[v.view(1,1,4096).to(device) for v in captured['ada'].chunk(6,dim=-1)]
        for index in range(36):
            block,manifest=load_block(ROOT/f'models/official/dit-block-{index:02d}.safetensors',index)
            m=json.loads((package/f'dit/block-{index:02d}/model.json').read_text())
            if manifest['sha256']!=m['weights_sha256']:raise ValueError('Official and converted DiT source differs')
            block=block.to(device)
            x=block(x,block_freqs,temb,attention_mask=block_mask);del block
            if (index+1)%6==0:
                print(json.dumps({'reference_step':i,'blocks_complete':index+1,
                    'elapsed_seconds':time.perf_counter()-start}),flush=True)
        x=x.cpu()
        patches=heads.final_linear(heads.final_norm(x,captured['c']))[:h*w]
        prediction=patches.transpose(0,1).reshape(1,h,w,128).permute(0,3,1,2).contiguous()
        sample=scheduler.step(prediction,t,sample).prev_sample
        fixture['outputs'].append({name:save_tensor(output/f'{name}-{i}.f32',value) for name,value in [('prediction',prediction),('step',sample)]})
        path.write_text(json.dumps(fixture,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'reference_step':i,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    mean=torch.from_numpy(np.fromfile(package/'vae/bn-mean.f32','<f4').copy()).reshape(1,128,1,1)
    var=torch.from_numpy(np.fromfile(package/'vae/bn-variance.f32','<f4').copy()).reshape(1,128,1,1)
    unpacked=ErnieImagePipeline._unpatchify_latents(sample*torch.sqrt(var+1e-5)+mean)
    del heads,block_freqs,block_mask,temb,x,captured,synth
    if device=='cuda':torch.cuda.empty_cache()
    vae,_=load_vae();decoded=vae._decode(unpacked,return_dict=False)[0]
    fixture['final']={name:save_tensor(output/f'{name}.f32',value) for name,value in [('final',sample),('unpacked',unpacked),('decoded',decoded)]}
    pixels=((decoded[0]/2+.5).clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
    Image.fromarray(pixels).save(output/'reference.png')
    fixture['complete']=True;fixture['reference_png_sha256']=sha256(output/'reference.png')
    path.write_text(json.dumps(fixture,indent=2,ensure_ascii=False)+'\n')
    return fixture

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'build/ernie-image')
    p.add_argument('--prompt',default='A red apple on a wooden table, soft daylight, realistic photo.')
    p.add_argument('--device',choices=['cpu','vulkan'],default='vulkan')
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp16')
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--vae-convolution',choices=['direct','sgemm'],default='direct')
    p.add_argument('--reference',type=Path,help='Reuse a complete, checksum-verified reference with identical configuration')
    p.add_argument('--reference-device',choices=['cpu','cuda'],default='cpu',help='Device for one streamed official FP32 DiT block; other official modules stay on CPU')
    p.add_argument('--latent',type=Path,help='Use these saved FP32 initial latents for both implementations')
    p.add_argument('--reference-only',action='store_true',help='Save the official fixture without launching the native candidate')
    args=p.parse_args()
    if args.output.exists() or not 1<=args.steps<=1000 or (args.device=='cpu' and args.precision!='fp32'):
        p.error('Use a new output path and supported device/precision/steps')
    if not args.reference and args.reference_device=='cuda' and not torch.cuda.is_available():
        p.error('CUDA reference requested but CUDA is unavailable')
    args.output.mkdir(parents=True)
    runner=args.output/'ernie-image.snapshot';shutil.copy2(args.runner,runner)
    # Gates are fixed before both reference and candidate execution. Pixel
    # gates diagnose a small fixture; they are not a perceptual quality claim.
    gates={'fp32':{'nrmse':.003,'global_rtol':.01,'atol':.0002,'pixel_mae':.1,'pixel_max':2},
           'fp16':{'nrmse':.15,'global_rtol':.25,'atol':.03,'pixel_mae':12,'pixel_max':80},
           'conditioning':{'nrmse':.0002,'global_rtol':.0002,'atol':.0002}}
    (args.output/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    ref=args.output/'reference'
    if args.reference:
        source=args.reference.resolve()
        fixture=json.loads((source/'fixture.json').read_text())
        config=json.loads((args.model/'manifest.json').read_text())['config']
        if (not fixture.get('complete') or fixture['prompt']!=args.prompt
            or fixture['steps']!=args.steps or fixture['config']!=config):
            raise ValueError('Saved reference configuration differs or is incomplete')
        entries=[*fixture['inputs'].values(),*fixture['final'].values()]
        for step in fixture['outputs']:entries.extend(step.values())
        for entry in entries:
            if (Path(entry['file']).name!=entry['file'] or sha256(source/entry['file'])!=entry['sha256']
                or (source/entry['file']).stat().st_size!=int(np.prod(entry['shape']))*4):
                raise ValueError('Saved reference tensor differs')
        if (sha256(source/'reference.png')!=fixture['reference_png_sha256']
            or sha256(source/'initial.f32')!=fixture['inputs']['initial']['sha256']):
            raise ValueError('Saved reference PNG or initial noise differs')
        if args.latent and sha256(args.latent)!=fixture['inputs']['initial']['sha256']:
            raise ValueError('Requested initial noise differs from saved reference')
        ref.symlink_to(source,target_is_directory=True)
    else:
        fixture=reference(args.model,args.prompt,ref,args.steps,args.reference_device,args.latent)
    if args.reference_only:
        print(json.dumps({'reference_complete':fixture['complete'],'output':str(ref)}),flush=True)
        return 0
    trace=args.output/'trace'
    command=[str(runner.resolve()),'--model',str(args.model.resolve()),'--prompt',args.prompt,
             '--output',str((args.output/'native.png').resolve()),'--device',args.device,'--precision',args.precision,
             '--steps',str(args.steps),'--latent',str((ref/'initial.f32').resolve()),'--trace-dir',str(trace.resolve()),
             '--vae-convolution',args.vae_convolution]
    result={'scope':fixture['scope'],'runner_sha256':sha256(runner),'validator_sha256':sha256(__file__),
            'package_manifest_sha256':sha256(args.model/'manifest.json'),'reference_fixture_sha256':sha256(ref/'fixture.json'),
            'device':args.device,'dit_precision':args.precision,'text_scheduler_vae_precision':'fp32',
            'command':command,'passed':False,'comparisons':[]}
    try:
        with (args.output/'native.log').open('w') as log:
            process=subprocess.Popen(['/usr/bin/time','-v',*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:process.wait(timeout=3600)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        result['return_code']=process.returncode
        if process.returncode:raise RuntimeError('Native generator failed')
        ids=[int(v) for v in (trace/'ids.txt').read_text().split()]
        if ids!=fixture['ids']:raise RuntimeError('Native tokenizer differs')
        entries=[(name,item,True) for name,item in fixture['inputs'].items()]
        for i,outputs in enumerate(fixture['outputs']):entries.extend((f'{name}-{i}',item,False) for name,item in outputs.items())
        entries.extend((name,item,False) for name,item in fixture['final'].items())
        for name,item,conditioning in entries:
            if sha256(ref/item['file'])!=item['sha256']:raise RuntimeError('Reference tensor checksum differs')
            expected=np.fromfile(ref/item['file'],'<f4');actual=np.fromfile(trace/(name+'.f32'),'<f4')
            if expected.shape!=actual.shape or not np.isfinite(actual).all():raise RuntimeError('Invalid native trace tensor')
            error=metrics(torch.from_numpy(expected),torch.from_numpy(actual));gate=gates['conditioning' if conditioning else args.precision]
            passed=error['nrmse']<=gate['nrmse'] and error['max_abs_error']<=gate['atol']+gate['global_rtol']*error['reference_max_abs']
            if name=='initial':passed=np.array_equal(expected,actual)
            result['comparisons'].append({'tensor':name,**error,'passed':bool(passed),'sha256':sha256(trace/(name+'.f32'))})
        native=np.array(Image.open(args.output/'native.png').convert('RGB')).astype('f8')
        expected=np.array(Image.open(ref/'reference.png').convert('RGB')).astype('f8')
        if native.shape!=expected.shape:raise RuntimeError('PNG dimensions differ')
        error=abs(native-expected);gate=gates[args.precision]
        result['png']={'shape':list(native.shape),'mae':float(error.mean()),'max_abs':float(error.max()),
                       'sha256':sha256(args.output/'native.png'),'passed':bool(error.mean()<=gate['pixel_mae'] and error.max()<=gate['pixel_max'])}
        # Independently check native float-to-PNG conversion, including channel order.
        decoded=np.fromfile(trace/'decoded.f32','<f4').reshape(3,native.shape[0],native.shape[1])
        quantized=(np.clip(decoded/2+.5,0,1).transpose(1,2,0)*255).round().astype('uint8')
        result['native_png_quantization_exact']=bool(np.array_equal(native,quantized))
        result['passed']=result['native_png_quantization_exact'] and result['png']['passed'] and all(c['passed'] for c in result['comparisons'])
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:result['failure']=str(error)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('passed','failure','png','native_png_quantization_exact')}),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
