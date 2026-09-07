#!/usr/bin/env python3
"""Run one native prompt-to-PNG job with saved inputs and bounded resource sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import math
from PIL import Image
from prepare_block import ROOT, sha256
from benchmark_request import (verify_benchmark_package, validate_native_report, requested_settings,
                               verify_noise, float32)

_IMAGE_SUFFIX = {'PNG': '.png', 'JPEG': '.jpg', 'BMP': '.bmp', 'TGA': '.tga'}

def png_chunk_types(raw):
    types=[];offset=8
    while offset+12<=len(raw):
        size=int.from_bytes(raw[offset:offset+4],'big')
        if size>len(raw)-offset-12: return []
        types.append(raw[offset+4:offset+8]);offset+=12+size
        if types[-1]==b'IEND': return types
    return []

def inspect_image_input(path):
    """Identify a CLI suffix and conservatively certify decoded RGB bytes."""
    raw=Path(path).read_bytes()
    with Image.open(path) as picture:
        image_format=picture.format
        suffix=_IMAGE_SUFFIX.get(image_format)
        if suffix is None:
            raise ValueError(f'Unsupported input image format: {image_format!r}')
        direct_rgb=(image_format=='PNG' and picture.mode=='RGB' and len(raw)>=26 and
                    raw[:8]==b'\x89PNG\r\n\x1a\n' and raw[24]==8 and raw[25]==2 and
                    not set(png_chunk_types(raw)) & {b'gAMA',b'cHRM',b'iCCP',b'sRGB'} and
                    'transparency' not in picture.info)
        decoded_sha256=hashlib.sha256(picture.tobytes()).hexdigest() if direct_rgb else None
    return suffix,decoded_sha256,('proven_lossless_opaque_rgb_png' if direct_rgb else 'unproven_native_decode_required')

def memory():
    fields={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','SwapTotal','SwapFree'):fields[key]=int(value.split()[0])
    return fields

def run_timed_command(command,timeout,log_path):
    """Run one process and retain timing plus a conservative failure classification."""
    result={'wall_started_monotonic_ns':time.monotonic_ns(),'timing_scope':'end_to_end'}
    process=None
    try:
        with Path(log_path).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:result['return_code']=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result['timed_out']=True;os.killpg(process.pid,signal.SIGKILL);result['return_code']=process.wait()
    except OSError as error:
        result['launch_error']=str(error);result['return_code']=None
    finally:
        result['wall_finished_monotonic_ns']=time.monotonic_ns()
        result['wall_seconds']=(result['wall_finished_monotonic_ns']-result['wall_started_monotonic_ns'])/1e9
    log_text=Path(log_path).read_text(errors='replace') if Path(log_path).exists() else ''
    code=result.get('return_code')
    if result.get('timed_out'): category='timeout'
    elif result.get('launch_error'): category='runtime_failure'
    elif code==0: category=None
    elif re.search(r'out of memory|allocation failed|cannot allocate memory',log_text,re.I): category='resource_exhaustion'
    elif code==-signal.SIGKILL: category='resource_unknown'
    elif isinstance(code,int) and code<0: category='crash'
    else: category='runtime_failure'
    result['failure_category']=category
    result['termination_signal']=-code if isinstance(code,int) and code<0 else None
    if isinstance(code,int) and code>0: result['termination_evidence']='positive exit status; no signal inferred'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'build/ernie-image')
    prompts=p.add_mutually_exclusive_group()
    prompts.add_argument('--prompt',default='A red apple on a wooden table, soft daylight, realistic photo.')
    prompts.add_argument('--prompt-file',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--precision',choices=['fp16','fp32','bf16'],default='fp16')
    p.add_argument('--device',choices=['vulkan','cpu'],default='vulkan')
    p.add_argument('--vae-device',choices=['cpu','vulkan'],default='cpu')
    p.add_argument('--vae-convolution',choices=['direct','sgemm'],default='direct')
    p.add_argument('--width',type=int)
    p.add_argument('--height',type=int)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--gpu',type=int,help='Native Vulkan device index; independent of NVIDIA sampling index')
    p.add_argument('--text-device',choices=['cpu'],default='cpu')
    p.add_argument('--text-down-vector',action='store_true')
    p.add_argument('--dit-weights',choices=['auto','device','host'],default='auto')
    p.add_argument('--gpu-reserve-mib',type=int,default=512)
    p.add_argument('--dit-cache-mib',type=int,default=0)
    p.add_argument('--ram-reserve-mib',type=int,default=3072)
    p.add_argument('--model-loading',choices=['default','stdio','mapped'],default='default')
    p.add_argument('--nvidia-sampling-index',type=int,default=0,help='Whole-device diagnostic only; not matched Vulkan allocation data')
    p.add_argument('--no-gpu-sampling',action='store_true',help='Disable optional NVIDIA diagnostics, e.g. for other GPU vendors')
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--trace',action='store_true',help='Diagnostic only; formal paired timing requires trace disabled')
    p.add_argument('--pe-model',type=Path)
    p.add_argument('--pe-max-tokens',type=int,default=256)
    p.add_argument('--pe-prompt-file',type=Path,help='Optional PE input; currently unsupported by the native CLI')
    p.add_argument('--latent',type=Path,help='Saved little-endian FP32 initial noise; required for formal comparison')
    p.add_argument('--noise-sha256',help='Expected frozen SHA256 of --latent; required for formal comparison')
    p.add_argument('--input-image',type=Path,help='Frozen img2img source image')
    p.add_argument('--input-image-sha256',help='Expected frozen SHA256 of --input-image')
    p.add_argument('--decoded-rgb-sha256',help='Expected SHA256 after decoding --input-image as RGB')
    p.add_argument('--strength',type=float,help='Img2img strength in (0,1]')
    p.add_argument('--resize',choices=['stretch','fit','crop'],default='stretch')
    args=p.parse_args()
    if args.output.exists() or args.timeout<1 or not 1<=args.steps<=1000 or not 0<=args.seed<=2**32-1:
        p.error('Use a new output and valid timeout, steps and seed')
    if args.device=='cpu' and args.precision!='fp32':p.error('CPU requires fp32')
    if not 1 <= args.threads <= 256 or (args.gpu is not None and not 0 <= args.gpu <= 63):
        p.error('Threads must be in [1,256] and GPU in [0,63]')
    if args.gpu is not None and args.device != 'vulkan' and args.vae_device != 'vulkan':
        p.error('--gpu requires a Vulkan generation or VAE device')
    if args.nvidia_sampling_index < 0 or any(not 0 <= v <= 2**32-1 for v in
            (args.gpu_reserve_mib, args.dit_cache_mib, args.ram_reserve_mib)):
        p.error('Invalid memory budget or NVIDIA sampling index')
    if args.device != 'vulkan' and (args.dit_weights != 'auto' or args.gpu_reserve_mib != 512):
        p.error('Explicit weight placement requires Vulkan')
    if ((args.dit_cache_mib or args.ram_reserve_mib != 3072) and
            (args.device != 'vulkan' or args.precision != 'fp32' or args.dit_weights == 'device')):
        p.error('Weight cache requires Vulkan FP32 with auto or host weights')
    if not args.dit_cache_mib and args.ram_reserve_mib != 3072:
        p.error('RAM reserve requires an enabled weight cache')
    if not 1 <= args.pe_max_tokens <= 2048:p.error('PE max tokens must be in [1,2048]')
    unsupported=[]
    if args.pe_prompt_file is not None:unsupported.append('--pe-prompt-file')
    if unsupported:
        args.output.mkdir(parents=True,exist_ok=True)
        result={'passed':False,'status':'incomplete','failure':'native runtime does not support requested functionality',
                'unsupported_arguments':unsupported,'trace':args.trace}
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return 2
    if (args.input_image is None) != (args.strength is None):
        p.error('--input-image and --strength must be provided together')
    if args.strength is not None and (not math.isfinite(args.strength) or not 0 < args.strength <= 1):
        p.error('Benchmark img2img strength must be finite and in (0,1]')
    if args.strength is not None:
        args.strength=float32(args.strength)
        if args.strength == 0:p.error('Benchmark strength must remain positive at native FP32 precision')
    try:
        manifest,cfg=verify_benchmark_package(args.model,args.width,args.height)
        width=cfg['packed_width']*16;height=cfg['packed_height']*16
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    args.output.mkdir(parents=True)
    runner=args.output/'ernie-image.snapshot';shutil.copy2(args.runner,runner)
    prompt_snapshot=args.output/'prompt.txt'
    if args.prompt_file:shutil.copy2(args.prompt_file,prompt_snapshot)
    else:prompt_snapshot.write_text(args.prompt,encoding='utf-8')
    latent_snapshot=None
    if args.latent:
        latent_snapshot=args.output/'initial.f32';shutil.copy2(args.latent,latent_snapshot)
        try:verify_noise(latent_snapshot,width,height)
        except ValueError as error:
            (args.output/'result.json').write_text(json.dumps({'passed':False,'status':'incomplete',
                'failure_category':'invalid_input','failure':str(error)},indent=2)+'\n')
            return 2
    input_snapshot=None;actual_input_sha256=None;actual_decoded_rgb_sha256=None;decoded_rgb_identity_status=None
    if args.input_image:
        temporary_snapshot=args.output/'input.snapshot'
        shutil.copy2(args.input_image,temporary_snapshot)
        image_suffix,actual_decoded_rgb_sha256,decoded_rgb_identity_status=inspect_image_input(temporary_snapshot)
        input_snapshot=args.output/('input'+image_suffix)
        temporary_snapshot.rename(input_snapshot)
        actual_input_sha256=sha256(input_snapshot)
    prompt_args=['--prompt-file',str(prompt_snapshot.resolve())]
    command=[str(runner.resolve()),'--model',str(args.model.resolve()),*prompt_args,
        '--output',str((args.output/'native.png').resolve()),'--seed',str(args.seed),'--steps',str(args.steps),
        '--device',args.device,'--precision',args.precision,'--vae-device',args.vae_device,
        '--vae-convolution',args.vae_convolution,'--width',str(width),'--height',str(height),
        '--threads',str(args.threads),'--text-device',args.text_device,'--dit-weights',args.dit_weights,
        '--gpu-reserve-mib',str(args.gpu_reserve_mib),'--dit-cache-mib',str(args.dit_cache_mib),
        '--ram-reserve-mib',str(args.ram_reserve_mib),'--model-loading',args.model_loading,
        '--report-json',str((args.output/'generation.json').resolve())]
    if args.gpu is not None:command+=['--gpu',str(args.gpu)]
    if args.text_down_vector:command+=['--text-down-vector']
    if args.trace:command+=['--trace-dir',str((args.output/'trace').resolve())]
    if args.pe_model:command+=['--pe-model',str(args.pe_model.resolve()),'--pe-greedy','--pe-max-tokens',str(args.pe_max_tokens)]
    if latent_snapshot:command+=['--latent',str(latent_snapshot.resolve())]
    if input_snapshot:
        command+=['--input',str(input_snapshot.resolve()),'--strength',str(args.strength),
                  '--resize',args.resize]
    pe_manifest=args.pe_model/'manifest.json' if args.pe_model else None
    pe_identity=sha256(pe_manifest) if pe_manifest and pe_manifest.is_file() else None
    actual_noise_sha256=sha256(latent_snapshot) if latent_snapshot else None
    noise_identity_proven=actual_noise_sha256 is not None and args.noise_sha256==actual_noise_sha256
    image_identity_proven=(input_snapshot is None or
        (decoded_rgb_identity_status=='proven_lossless_opaque_rgb_png' and
         actual_input_sha256==args.input_image_sha256 and actual_decoded_rgb_sha256==args.decoded_rgb_sha256))
    formal_eligible=noise_identity_proven and image_identity_proven and (args.pe_model is None or pe_identity is not None) and not args.trace
    result={'scope':'One native functional run; no full-resolution official denoising reference or perceptual quality gate',
        'passed':False,'quality_validated':False,'status':'pending','prompt':args.prompt if not args.prompt_file else None,
        'prompt_file':str(args.prompt_file.resolve()) if args.prompt_file else None,
        'config':None,'initial_package_config':cfg,'requested_settings':requested_settings(args),
        'selection_status':'pending_native_report','package_schema_version':manifest['schema_version'],
        'command':command,'runner_sha256':sha256(runner),'benchmark_sha256':sha256(__file__),
        'package_manifest_sha256':sha256(args.model/'manifest.json'),'system_memory_before_kib':memory(),
        'prompt_sha256':sha256(prompt_snapshot),'prompt_snapshot':str(prompt_snapshot.resolve()),
        'noise_sha256':actual_noise_sha256,'expected_noise_sha256':args.noise_sha256,
        'noise_dtype':'<f4' if latent_snapshot else None,
        'input_image_sha256':actual_input_sha256,'expected_input_image_sha256':args.input_image_sha256,
        'decoded_rgb_sha256':actual_decoded_rgb_sha256,'expected_decoded_rgb_sha256':args.decoded_rgb_sha256,
        'decoded_rgb_identity_status':decoded_rgb_identity_status,
        'strength':args.strength,'resize_policy':({'mode':args.resize,'width':width,'height':height,
            'filter':'bilinear','coordinate_transform':'half_pixel','antialias':False} if input_snapshot else None),
        'shape':[width,height],'shape_order':'WH',
        'pe_manifest_sha256':pe_identity,'formal_comparison_eligible':False,
        'formal_ineligibility_reasons':([] if noise_identity_proven else ['saved FP32 noise and matching frozen SHA256 are required'])+
            ([] if not args.pe_model or pe_identity else ['PE package manifest identity is required'])+
            ([] if image_identity_proven else ['input image bytes and native-equivalent decoded RGB must match frozen SHA256'])+
            ([] if not args.trace else ['trace must be disabled']),
        'trace':args.trace,'timing_scope':'end_to_end_external_process_launch_through_output_close',
        'package_preverification':'all package files/objects hashed before timing; file cache warmed but not controlled; native verification also timed',
        'gpu_sampling_scope':f'Whole NVIDIA device {args.nvidia_sampling_index}; mapping to selected Vulkan index unverified; includes other processes; 100 ms samples; not allocator/process VRAM',
        'gpu_sampling_enabled':not args.no_gpu_sampling and (args.device=='vulkan' or args.vae_device=='vulkan'),
        'precision':{'dit':args.precision,'residual':'fp32','text':'fp32','euler':'fp32',
                     'vae':'fp32 with FP64 CPU GroupNorm reductions' if args.vae_device=='cpu' else 'fp32'}}
    from source_inventory import source_files
    source_paths = source_files(ROOT)
    result['source_files_sha256']={str(path.relative_to(ROOT)):sha256(path) for path in sorted(source_paths)}
    (args.output/'request.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    sampler=None
    try:
        with (args.output/'gpu-device-memory.log').open('w') as gpu_log:
            try:
                if result['gpu_sampling_enabled']:
                    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits',f'--id={args.nvidia_sampling_index}','--loop-ms=100'],stdout=gpu_log,stderr=subprocess.DEVNULL)
                timed=['/usr/bin/time','-v','-o',str((args.output/'resources.log').resolve()),*command]
                result.update(run_timed_command(timed,args.timeout,args.output/'runner.log'))
            finally:
                if sampler is not None:sampler.terminate();sampler.wait(timeout=5)
        if result['return_code'] != 0:raise RuntimeError(f"Native inference failed: {result['failure_category']}; inspect runner.log")
        with Image.open(args.output/'native.png') as picture:
            picture.load();result['image']={'size':list(picture.size),'mode':picture.mode,'sha256':sha256(args.output/'native.png')}
        report=json.loads((args.output/'generation.json').read_text())
        cfg,binding=validate_native_report(report,args,manifest,width,height)
        for path,digest in ((runner,result['runner_sha256']),(prompt_snapshot,result['prompt_sha256']),
                (args.model/'manifest.json',result['package_manifest_sha256']),
                (latent_snapshot,actual_noise_sha256),(input_snapshot,actual_input_sha256),
                (pe_manifest,pe_identity)):
            if path is not None and sha256(path)!=digest:raise ValueError('Run input or runner identity changed during execution')
        if args.pe_model is None and report['prompt'] != prompt_snapshot.read_bytes().decode('utf-8-sig'):
            raise ValueError('Native consumed prompt differs from the immutable input')
        result.update(config=cfg,shared_source_binding=binding,selection_status='verified_native_report',
                      native_report=report,native_report_sha256=sha256(args.output/'generation.json'))
        result['passed']=result['image']['size']==[width,height] and result['image']['mode']=='RGB'
        result['status']='ok' if result['passed'] else 'failed'
        result['formal_comparison_eligible']=bool(result['passed'] and formal_eligible and not report['allocation_instrumentation'])
        if report['allocation_instrumentation']:result['formal_ineligibility_reasons'].append('allocation instrumentation must be disabled for speed rounds')
        if args.trace:result['trace_sha256']={path.name:sha256(path) for path in sorted((args.output/'trace').iterdir()) if path.is_file()}
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,subprocess.TimeoutExpired) as error:
        result['passed']=False
        result['status']='incomplete'
        result['failure']=str(error)
        result['formal_comparison_eligible']=False
        if not result.get('failure_category'):result['failure_category']='runtime_failure'
    result['system_memory_after_kib']=memory()
    samples=[int(line) for line in (args.output/'gpu-device-memory.log').read_text().splitlines() if line.isdigit()]
    if samples:result['gpu_device_total_mib']={'first':samples[0],'sampled_peak':max(samples),'samples':len(samples)}
    resources=(args.output/'resources.log').read_text() if (args.output/'resources.log').exists() else ''
    match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resources)
    if match:result['max_rss_kib']=int(match[1])
    # Human-readable output may contain a multiline enhanced prompt. Only the
    # structured producer record supplies stage timings and effective settings.
    if result.get('selection_status')=='verified_native_report':
        report=result['native_report']
        result['denoise_seconds']=[p['seconds'] for p in report['progress'] if p['stage']=='denoise']
        result['text_seconds']=next(p['seconds'] for p in report['progress'] if p['stage']=='text')
        result['total_seconds']=report['total_seconds']
        result['vae_and_png_seconds']=report['vae_and_image_encode_seconds']
    result['timing_scope']='external process launch through exit, including native verification, image/report close; Python preverification and postvalidation excluded'
    (args.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('passed','failure','image','total_seconds','max_rss_kib','gpu_device_total_mib')}),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
