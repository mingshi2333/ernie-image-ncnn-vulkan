#!/usr/bin/env python3
"""Freeze Turbo delivery results while retaining failed precision gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from PIL import Image
from package_model import ROOT, sha256

CASES = {
    'apple-fp16': ('pipeline1024-portable-direct-v1', True),
    'long-fp16': ('pipeline1024-long-s64-v1', False),
    'long-fp32': ('pipeline1024-long-s64-fp32-v1', False),
    'chinese-fp16': ('pipeline1024-chinese-s64-fp16-v1', False),
    'chinese-fp32': ('pipeline1024-chinese-s64-fp32-v1', False),
}
COMPONENTS = ['pipeline64-cuda-reference-v1', 'vae-direct-64-v1',
              'vae-direct-1024-v1', 'text-s64-real-long-v1',
              'time-features-delivery-v1', 'diagnostic-long-step6-fp32-v1']


def recheck_metrics(run, result, fixture, gates):
    """Recompute the frozen verdict from saved bytes using NumPy, not the validator."""
    expected = {**fixture['inputs'], **fixture['final']}
    for i, step in enumerate(fixture['outputs']):
        expected.update({f'{name}-{i}': value for name, value in step.items()})
    if set(expected) != {row['tensor'] for row in result['comparisons']} or len(expected) != len(result['comparisons']):
        raise ValueError('Incomplete comparison inventory')
    verdicts = []
    for row in result['comparisons']:
        name = row['tensor']
        reference = np.fromfile(run/'reference'/expected[name]['file'], '<f4').astype('f8')
        actual = np.fromfile(run/'trace'/(name+'.f32'), '<f4').astype('f8')
        if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError('Invalid tensor while freezing evidence')
        maximum = float(np.max(abs(actual-reference)))
        nrmse = float(np.sqrt(np.sum((actual-reference)**2)/max(np.sum(reference**2), 1e-30)))
        if not (np.isclose(maximum, row['max_abs_error'], rtol=1e-10, atol=1e-15)
                and np.isclose(nrmse, row['nrmse'], rtol=1e-10, atol=1e-15)):
            raise ValueError(f'Saved metrics disagree with tensors: {name}')
        gate = gates['conditioning' if name in fixture['inputs'] else result['dit_precision']]
        passed = nrmse <= gate['nrmse'] and maximum <= gate['atol']+gate['global_rtol']*np.max(abs(reference))
        if name == 'initial':
            passed = np.array_equal(actual, reference)
        if bool(passed) != row['passed']:
            raise ValueError(f'Saved verdict disagrees with gate: {name}')
        verdicts.append(passed)
    native = np.asarray(Image.open(run/'native.png').convert('RGB'))
    reference = np.asarray(Image.open(run/'reference/reference.png').convert('RGB'))
    delta = abs(native.astype('f8')-reference.astype('f8'))
    if float(delta.mean()) != result['png']['mae'] or float(delta.max()) != result['png']['max_abs']:
        raise ValueError('Saved PNG metrics disagree with images')
    decoded = np.fromfile(run/'trace/decoded.f32', '<f4').reshape(3, *native.shape[:2])
    quantized = (np.clip(decoded/2+.5, 0, 1).transpose(1, 2, 0)*255).round().astype('uint8')
    quantization_exact = np.array_equal(native, quantized)
    gate = gates[result['dit_precision']]
    pixel_pass = delta.mean() <= gate['pixel_mae'] and delta.max() <= gate['pixel_max']
    if (bool(pixel_pass) != result['png']['passed']
            or bool(quantization_exact) != result['native_png_quantization_exact']
            or bool(all(verdicts) and pixel_pass and quantization_exact) != result['passed']):
        raise ValueError('Saved full verdict differs from independently recomputed verdict')


def write_report(output, rows):
    lines = [
        '# Turbo 本地交付与 1024 对照', '',
        '日期：2026-09-05。范围：Linux、ERNIE-Image-Turbo、batch=1、8 Euler 步、CFG=1、PE 关闭。', '',
        '原生提示词 → tokenizer → 25 层文本编码 → 36 层 DiT × 8 步 → BN/unpack → CPU VAE → PNG 已完整运行。'
        '独立模型包、运行前完整性检查、安装入口和回归验证已落地。'
        '1024 苹果对照通过全部门限；长英文和中文的精度失败按原始结果保留，不能将可用成图等同于全部数值门限通过。', '',
        '## 交付变更', '',
        '| 变更 | 实现与证据 |', '|---|---|',
        '| CPU VAE 内存 | 默认直接卷积，保留显式 SGEMM；GroupNorm 继续使用 FP64 归约，激活和 affine 为 FP32 |',
        '| 64-token 文本桶 | 独立导出目标图；规范化仅覆盖 13 处已审查 token 参数，完整图指纹与重建 FP32 权重散列必须匹配 |',
        '| 可移动模型包 | 136 个运行文件，23,271,870,723 字节（约 21.67 GiB），没有内部或外部依赖链接；32/64-token 包分别保存 |',
        '| 原生校验 | Rust SHA256 通过 C ABI 提供给 C++；所有文件散列/字节数、版本、配置和清单一致性检查，兼容旧 schema-1 开发包 |',
        '| 安装和构建 | 安装 ernie-image、使用说明和来源锁；Linux CPU/Vulkan 工作流已写入，本轮没有远程 Actions 运行 |',
        '| 官方参考 | 固定官方权重与代码，单块 CUDA FP32 DiT，禁用 TF32，其他模块 CPU FP32；子进程退出后再启动 Vulkan |',
        '| 证据封存 | 校验参考、native 张量、PNG、runner 和验证脚本散列，另用 NumPy 重算全部误差、门限和 PNG 量化 |', '',
        '清单用于检测文件缺失和损坏，不是发布者数字签名。模型包可搬移，安装二进制仍要求兼容的 Linux 系统库。', '',
        '## 完整图像对照', '',
        '每项都使用该项官方参考保存的同一份初始 latent。不同框架的相同整数 seed 不作为等价输入证明。'
        '各行是不同提示词/精度的单次结果，不能据此比较性能或宣称广泛质量通过。', '',
        '| 样本 | tokens / 桶 | 精度 | 张量通过 | PNG MAE / 255 | 最大像素差 / 255 | PNG 门限 | 全部门限 |',
        '|---|---:|---|---:|---:|---:|---|---|',
    ]
    for row in rows:
        lines.append(f'| [{row["case"]}](runs/{row["run"]}/result.json) | '
                     f'{row["tokens"]} / {row["config"]["text_bucket"]} | {row["precision"]} | '
                     f'{row["passed_comparisons"]}/{row["comparison_count"]} | {row["png_mae"]:.8f} | '
                     f'{row["png_max_abs"]:.0f} | {"通过" if row["png_passed"] else "未通过"} | '
                     f'{"通过" if row["passed"] else "未通过"} |')
    lines += ['', '每项 native float → PNG 量化均单独核对。详细指标、失败张量和输入配置见 [results.json](results.json)。', '',
              '| 样本 | 最终 latent NRMSE | 解码 NRMSE | 总耗时（秒） | 最大 RSS（GiB） |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["case"]} | {row["final_latent_nrmse"]:.9f} | {row["decoded_nrmse"]:.9f} | '
                     f'{row["total_seconds"]:.3f} | {row["max_rss_gib"]:.4f} |')
    lines += ['', '提示词如下；每条参考及其 precision 重试均保留原始记录。', '']
    seen = set()
    for row in rows:
        if row['prompt'] not in seen:
            lines.append('- '+row['prompt'])
            seen.add(row['prompt'])
    lines += ['', '### 固定门限与负结果', '',
              '门限在运行前写入各 run 的 gates.json，没有根据本轮结果放宽。'
              'FP16 张量要求 NRMSE ≤ 0.15，最大绝对差 ≤ `0.03 + 0.25 × reference_max_abs`；'
              'FP32 分别要求 ≤ 0.003 和 `0.0002 + 0.01 × reference_max_abs`。'
              'FP16 PNG 要求 MAE ≤ 12、最大差 ≤ 80；FP32 为 MAE ≤ 0.1、最大差 ≤ 2，单位均为 8-bit 像素值。', '',
              '文本/位置条件有独立门限；初始 latent 必须逐位相同。25 项张量中任何一项、PNG 比较或 PNG 量化失败，整体即失败。'
              '不能仅用平均像素误差较小将失败覆盖。', '']
    for row in rows:
        if not row['passed']:
            lines.append(f'- {row["case"]}：张量失败为 '+', '.join('`'+v+'`' for v in row['failed_tensors'])
                         +f'；PNG {"通过" if row["png_passed"] else "未通过"}。')
    lines += ['', '### 误差定位', '',
              '长英文的第七次预测（index 6）使用官方该步输入 latent、文本、时间特征、RoPE 和 mask 单独执行。'
              'FP32 NRMSE 为 7.4248e-5，最大差 0.003621，通过原有张量门限；'
              '自由运行同一步的 NRMSE 为 0.0034907，未通过。'
              '该结果支持“前序轨迹及条件的小差异被后续运算放大”的解释，尚未隔离到某一个算子或舍入来源。'
              '单步通过不能替代自由运行通过。证据见 [单步诊断](components/diagnostic-long-step6-fp32-v1/native/result.json)。', '',
              '时间特征包含全部 8 个 Turbo 时间点，41 个有效输入与 5 个非法输入均通过原有门限；'
              '最大 NRMSE 1.2245e-6。没有找到第七步特有的时间特征异常。', '',
              '## 内存与组件验证', '',
              '完整苹果生成最大 RSS 从旧 SGEMM 的 24,145,308 KiB（23.03 GiB）降至直接卷积的 '
              '6,104,472 KiB（5.82 GiB），约下降 74.7%。'
              '独立 1024 VAE 的最大 RSS 为 5,685,048 KiB（5.42 GiB），NRMSE 9.4142e-7，仍使用固定 2e-5 门限。'
              '直接卷积与高精度 GroupNorm 同时保留；不能只关闭 Winograd 后宣称归一化问题已解决。', '',
              '32/64-token 静态图的规范化完整指纹为 '
              '`67881d48031e178fcb5653f749d17895a9470d03adf5bf8098dd5bbff952ab36`。'
              '64-token 桶的 40-token 真实文本、25 层 CPU NRMSE 为 6.1913e-6。'
              '第一次 rebucket 调用了仅含 block zero 的目录，失败日志保存在 failures/，最终包来自完整 25 层的成功重试。', '',
              '旧苹果运行耗时 522.17 秒，新直接卷积运行 617.18 秒，后者增加了约 19.12 秒全模型校验；'
              '两者都包含 trace。不是受控性能实验，不宣称速度提升。'
              'FP32 单步诊断整卡显存 100 ms 采样峰值 7609 MiB，含其他进程；'
              'FP32 attention 当前构造完整注意力矩阵，显存成本高于 FP16，尚无精确 allocator 峰值。', '',
              '## 构建与回归', '',
              '- 原生 CPU：8/8 CTest 通过。',
              '- 原生 NVIDIA Vulkan：19/19 CTest 通过，详见 logs/ctest-delivery-hardware-final.log。',
              '- Python：34/34 通过，含 15 项模型包、4 项静态文本图检查。',
              '- 独立 GCC + 固定 glslang 构建成功；软件 Vulkan 16 项通过、3 项 BF16 因驱动不支持跳过。',
              '- 安装后的帮助入口和系统库解析通过；完整模型检查记录单独保存。',
              '- GitHub Actions 工作流只在本地准备和验证；没有本轮远程执行结果。', '',
              '这些构建测试使用小型算子或模型包占位文件，不下载真实权重，不替代上方全模型比较。', '',
              '## 使用与边界', '',
              '构建、转换、安装和运行命令见 [README](../../../README.md)、'
              '[运行说明](../../../docs/RUNNING.md) 和 [参考复现](../../../docs/REPRODUCE-PIPELINE.md)。'
              '本机推荐模型路径为 `models/turbo1024-s64-portable`；'
              'native PNG 和大型 trace 位于 `outputs/<results.json 中的 run>/`，未进入 Git。', '',
              '默认使用 Vulkan FP16 DiT、FP32 残差/Euler 和 CPU 直接卷积 VAE。'
              'FP32 可以降低本轮误差，但不能保证通过每个提示词的严格门限，且需要更多显存。'
              '模型限制为所选静态尺寸与 32/64-token 桶。PE、原版非 Turbo CFG 路径、量化、任意尺寸、'
              'Windows/其他 GPU 和可分发二进制尚未验收。', '',
              '后续优先级：独立提示词/种子数据集和跨步误差定位；FP32 attention 分块；'
              '减少逐步重复权重准备/上传；再分别评估 PE 原生 KV cache、量化和跨平台。'
              '联合 DiT 的隐藏状态每步变化，不能将跨步 K/V 复用作为精确优化。', '',
              '## 来源', '',
              '固定版本与运行环境见 environment/ 和每个参考 fixture。'
              'manifest.json 的 git_head 为生成本证据前的代码提交，source_files_sha256 固定实际文件。'
              '早期验证脚本的准确版本保存在 validator-source-variants/；后续运行保存 scripts/ 快照。'
              '未改写早期 [pipeline](../pipeline/README.md)、[组件](../components/README.md) 或 '
              '[单块](../dit-block/README.md) 报告，也未放宽其失败门限。', '']
    (output/'README.md').write_text('\n'.join(lines))


def freeze(output):
    if output.exists():
        raise ValueError('Use a new evidence directory')
    output.mkdir(parents=True)
    index = []
    required_logs = {'ctest-delivery-hardware-final.log': '0 tests failed out of 19',
                     'ctest-delivery-cpu-final.log': '0 tests failed out of 8',
                     'python-delivery-tests-final.log': 'Ran 34 tests',
                     'install-delivery-final-verify.log': 'Model verified:'}
    for filename, expected_text in required_logs.items():
        content = (ROOT/'outputs'/filename).read_text()
        if expected_text not in content or (filename.startswith('ctest') and 'Skipped' in content):
            raise ValueError(f'Required delivery check did not complete: {filename}')

    def copy(source, target):
        if not source.is_file() or source.stat().st_size > 2*1024*1024:
            raise ValueError(f'Missing or unexpectedly large evidence: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError(f'Conflicting evidence destination: {target}')
        shutil.copyfile(source, target)

    def metadata_tree(source, destination):
        if not source.is_dir():
            raise ValueError(f'Missing evidence directory: {source}')
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.log', '.cfg', '.param', '.txt', '.py'):
                copy(path, destination/path.relative_to(source))

    for label, (run_name, expected_pass) in CASES.items():
        run = ROOT/'outputs'/run_name
        result = json.loads((run/'result.json').read_text())
        fixture = json.loads((run/'reference/fixture.json').read_text())
        gates = json.loads((run/'gates.json').read_text())
        if result['passed'] != expected_pass or not fixture['complete']:
            raise ValueError(f'Unexpected or incomplete case: {label}')
        if sha256(run/'ernie-image.snapshot') != result['runner_sha256']:
            raise ValueError(f'Runner snapshot differs: {label}')
        validator = run/'scripts/validate_pipeline.py'
        if not validator.exists():
            validator = ROOT/'outputs/validator-source-variants'/(result['validator_sha256']+'.py')
        if sha256(validator) != result['validator_sha256']:
            raise ValueError(f'Validator snapshot differs: {label}')
        for script_name, digest in result.get('source_snapshot', {}).items():
            if sha256(run/'scripts'/script_name) != digest:
                raise ValueError(f'Source snapshot differs: {label}/{script_name}')
        package_path = Path(result['command'][result['command'].index('--model')+1])
        if sha256(package_path/'manifest.json') != result['package_manifest_sha256']:
            raise ValueError(f'Package manifest changed: {label}')
        if (sha256(run/'reference/fixture.json') != result['reference_fixture_sha256']
            or sha256(run/'native.png') != result['png']['sha256']
            or sha256(run/'reference/reference.png') != fixture['reference_png_sha256']):
            raise ValueError(f'Fixture or image changed: {label}')
        for row in result['comparisons']:
            if sha256(run/'trace'/(row['tensor']+'.f32')) != row['sha256']:
                raise ValueError(f'Native tensor changed: {label}/{row["tensor"]}')
        entries = [*fixture['inputs'].values(), *fixture['final'].values()]
        for step in fixture['outputs']:
            entries.extend(step.values())
        for entry in entries:
            if sha256(run/'reference'/entry['file']) != entry['sha256']:
                raise ValueError(f'Reference tensor changed: {label}/{entry["file"]}')
        recheck_metrics(run, result, fixture, gates)
        measured = {row['tensor']: row for row in result['comparisons']}
        log = (run/'native.log').read_text()
        rss = int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', log)[1])
        elapsed = float(re.search(r'Total: ([\d.]+) s', log)[1])
        row = {'case': label, 'run': run_name, 'passed': result['passed'], 'prompt': fixture['prompt'],
               'tokens': len(fixture['ids']), 'config': fixture['config'], 'steps': fixture['steps'],
               'precision': result['dit_precision'], 'comparison_count': len(result['comparisons']),
               'passed_comparisons': sum(r['passed'] for r in result['comparisons']),
               'failed_tensors': [r['tensor'] for r in result['comparisons'] if not r['passed']],
               'final_latent_nrmse': measured['final']['nrmse'], 'decoded_nrmse': measured['decoded']['nrmse'],
               'png_mae': result['png']['mae'], 'png_max_abs': result['png']['max_abs'],
               'png_passed': result['png']['passed'],
               'native_png_quantization_exact': result['native_png_quantization_exact'],
               'max_rss_kib': rss, 'max_rss_gib': rss/1024**2, 'total_seconds': elapsed,
               'native_png_sha256': result['png']['sha256'], 'gates': gates[result['dit_precision']],
               'metrics_independently_recomputed': True,
               'reference_environment': fixture['reference_environment']}
        index.append(row)
        metadata_tree(run, output/'runs'/run_name)
        # rglob intentionally does not follow the reused reference symlink.
        metadata_tree((run/'reference').resolve(), output/'runs'/run_name/'reference')

    for name in COMPONENTS:
        metadata_tree(ROOT/'outputs'/name, output/'components'/name)
    for name in ['text-block00-s64-v1', 'text-s64-v2']:
        metadata_tree(ROOT/'models'/name, output/'models'/name)
    for name in ['portable-turbo1024-s32-v1', 'turbo1024-s64-portable']:
        for filename in ['manifest.json', 'model.cfg']:
            copy(ROOT/'models'/name/filename, output/'packages'/name/filename)
    # The incomplete attempt used a directory containing only block zero.
    copy(ROOT/'outputs/rebucket-text-s64-v1.log', output/'failures/rebucket-text-source-path.log')
    for prefix in ['build-delivery', 'ctest-delivery', 'python-delivery', 'install-delivery', 'configure-delivery',
                   'package-turbo1024', 'export-text-s64', 'prepare-pipeline1024-s64', 'rebucket-text-s64-v2']:
        for path in sorted((ROOT/'outputs').glob(prefix+'*.log')):
            copy(path, output/'logs'/path.name)
    for name in ['sources.lock.json', 'requirements-reference.lock', 'tokenizer/Cargo.lock', 'tokenizer/Cargo.toml']:
        copy(ROOT/name, output/'environment'/name)
    copy(ROOT/'outputs/delivery-verification.json', output/'environment/delivery-verification.json')
    for path in sorted((ROOT/'outputs/validator-source-variants').glob('*.py')):
        copy(path, output/'validator-source-variants'/path.name)
    (output/'results.json').write_text(json.dumps(index, indent=2, ensure_ascii=False)+'\n')
    write_report(output, index)
    files = [ROOT/'CMakeLists.txt', ROOT/'sources.lock.json', ROOT/'.github/workflows/build.yml']
    for directory in ['src', 'tools', 'tests', 'probes', 'tokenizer/src']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.rs','.py','.param'))
    state = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
             'scope': 'Fixed 1024 prompts, with rejected FP16 and FP32 tensor gates retained; no broad perceptual quality claim',
             'runtime_platform': 'Linux, RTX 4060 Laptop 8 GB, Ryzen 7745HX, 32 GB RAM',
             'source_files_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))},
             'evidence_sha256': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(state, indent=2)+'\n')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2, ensure_ascii=False))
