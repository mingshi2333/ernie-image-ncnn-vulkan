#!/usr/bin/env python3
"""Run one native prompt-to-PNG job with saved inputs and bounded resource sampling."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from PIL import Image
from prepare_block import ROOT, sha256
from package_model import verify_package

def memory():
    fields={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','SwapTotal','SwapFree'):fields[key]=int(value.split()[0])
    return fields

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
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--trace',action='store_true',help='Diagnostic only; formal paired timing requires trace disabled')
    p.add_argument('--pe-model',type=Path)
    p.add_argument('--pe-max-tokens',type=int,default=256)
    p.add_argument('--pe-prompt-file',type=Path,help='Optional PE input; currently unsupported by the native CLI')
    p.add_argument('--input-image',type=Path,help='Optional img2img input; currently unsupported by the native CLI')
    p.add_argument('--strength',type=float,help='Optional img2img strength; currently unsupported by the native CLI')
    args=p.parse_args()
    if args.output.exists() or args.timeout<1 or not 1<=args.steps<=1000 or not 0<=args.seed<=2**32-1:
        p.error('Use a new output and valid timeout, steps and seed')
    if args.device=='cpu' and args.precision!='fp32':p.error('CPU requires fp32')
    unsupported=[]
    if args.pe_prompt_file is not None:unsupported.append('--pe-prompt-file')
    if args.input_image is not None:unsupported.append('--input-image')
    if args.strength is not None:unsupported.append('--strength')
    if unsupported:
        args.output.mkdir(parents=True,exist_ok=True)
        result={'passed':False,'status':'incomplete','failure':'native runtime does not support requested functionality',
                'unsupported_arguments':unsupported,'trace':args.trace}
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return 2
    manifest,_=verify_package(args.model)
    args.output.mkdir(parents=True)
    runner=args.output/'ernie-image.snapshot';shutil.copy2(args.runner,runner)
    prompt_args=['--prompt-file',str(args.prompt_file.resolve())] if args.prompt_file else ['--prompt',args.prompt]
    command=[str(runner.resolve()),'--model',str(args.model.resolve()),*prompt_args,
        '--output',str((args.output/'native.png').resolve()),'--seed',str(args.seed),'--steps',str(args.steps),
        '--device',args.device,'--precision',args.precision,'--vae-device',args.vae_device,
        '--vae-convolution',args.vae_convolution]
    if args.trace:command+=['--trace-dir',str((args.output/'trace').resolve())]
    if args.pe_model:command+=['--pe-model',str(args.pe_model.resolve()),'--pe-greedy','--pe-max-tokens',str(args.pe_max_tokens)]
    result={'scope':'One native functional run; no full-resolution official denoising reference or perceptual quality gate',
        'passed':False,'quality_validated':False,'status':'pending','prompt':args.prompt if not args.prompt_file else None,
        'prompt_file':str(args.prompt_file.resolve()) if args.prompt_file else None,'config':manifest['config'],
        'command':command,'runner_sha256':sha256(runner),'benchmark_sha256':sha256(__file__),
        'package_manifest_sha256':sha256(args.model/'manifest.json'),'system_memory_before_kib':memory(),
        'trace':args.trace,'timing_scope':'end_to_end_external_process_launch_through_output_close',
        'gpu_sampling_scope':'Whole NVIDIA device 0, includes other processes, 100 ms samples; not exact allocator/process VRAM',
        'precision':{'dit':args.precision,'residual':'fp32','text':'fp32','euler':'fp32',
                     'vae':'fp32 with FP64 CPU GroupNorm reductions' if args.vae_device=='cpu' else 'fp32'}}
    from source_inventory import source_files
    source_paths = source_files(ROOT)
    result['source_files_sha256']={str(path.relative_to(ROOT)):sha256(path) for path in sorted(source_paths)}
    (args.output/'request.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    sampler=None
    try:
        with (args.output/'runner.log').open('w') as log, (args.output/'gpu-device-memory.log').open('w') as gpu_log:
            try:
                if args.device=='vulkan' or args.vae_device=='vulkan':
                    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--id=0','--loop-ms=100'],stdout=gpu_log,stderr=subprocess.DEVNULL)
                timed=['/usr/bin/time','-v','-o',str((args.output/'resources.log').resolve()),*command]
                started_ns=time.monotonic_ns()
                with subprocess.Popen(timed,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as process:
                    try:result['return_code']=process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait();raise
                result['wall_started_monotonic_ns']=started_ns
                result['wall_finished_monotonic_ns']=time.monotonic_ns()
                result['wall_seconds']=(result['wall_finished_monotonic_ns']-started_ns)/1_000_000_000
            finally:
                if sampler is not None:sampler.terminate();sampler.wait(timeout=5)
        if result['return_code']:raise RuntimeError('Native inference failed; inspect runner.log')
        with Image.open(args.output/'native.png') as picture:
            picture.load();result['image']={'size':list(picture.size),'mode':picture.mode,'sha256':sha256(args.output/'native.png')}
        cfg=manifest['config']
        result['passed']=result['image']['size']==[cfg['packed_width']*16,cfg['packed_height']*16] and result['image']['mode']=='RGB'
        result['status']='ok' if result['passed'] else 'failed'
        if args.trace:result['trace_sha256']={path.name:sha256(path) for path in sorted((args.output/'trace').iterdir()) if path.is_file()}
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
        result['passed']=False
        result['status']='incomplete'
        result['failure']=str(error)
    result['system_memory_after_kib']=memory()
    samples=[int(line) for line in (args.output/'gpu-device-memory.log').read_text().splitlines() if line.isdigit()]
    if samples:result['gpu_device_total_mib']={'first':samples[0],'sampled_peak':max(samples),'samples':len(samples)}
    resources=(args.output/'resources.log').read_text() if (args.output/'resources.log').exists() else ''
    match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resources)
    if match:result['max_rss_kib']=int(match[1])
    log=(args.output/'runner.log').read_text()
    result['denoise_seconds']=[float(value) for value in re.findall(r'Denoise \d+/\d+: ([\d.]+) s',log)]
    for field,pattern in [('total_seconds',r'Total: ([\d.]+) s'),('vae_and_png_seconds',r'VAE and PNG: ([\d.]+) s'),('text_seconds',r'Text conditioned: \d+ tokens, ([\d.]+) s')]:
        match=re.search(pattern,log)
        if match:result[field]=float(match[1])
    (args.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('passed','failure','image','total_seconds','max_rss_kib','gpu_device_total_mib')}),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
