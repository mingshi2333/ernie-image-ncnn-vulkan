from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('长英文：原实现', 'pipeline1024-long-s64-fp32-v1'),
    ('长英文：补偿 + 分块', 'pipeline1024-long-s64-fp32-chunked-v1'),
    ('中文：原实现', 'pipeline1024-chinese-s64-fp32-v1'),
    ('中文：补偿、完整矩阵', 'pipeline1024-chinese-s64-fp32-kahan-v1'),
    ('中文：补偿 + 分块', 'pipeline1024-chinese-s64-fp32-chunked-v1'),
]
DIAGNOSTIC = 'pipeline1024-chinese-s64-fp32-reference-text-v1'


def read(name):
    result = json.loads((ROOT/'outputs'/name/'result.json').read_text())
    if result.get('return_code') != 0 or 'failure' in result:
        raise ValueError('Require completed execution: ' + name)
    return result


def metrics(result, name):
    return next(row for row in result['comparisons'] if row['tensor'] == name)


def main():
    results = [(label, name, read(name)) for label, name in CASES]
    diagnostic = read(DIAGNOSTIC)
    gpu = json.loads((ROOT/'outputs/pipeline1024-long-s64-fp32-chunked-v1/gpu-memory.json').read_text())
    equivalence = json.loads((ROOT/'outputs/attention-chinese-chunking-equivalence-v1.json').read_text())
    lines = [
        '# FP32 注意力精度、显存与完整轨迹对照', '',
        '日期：2026-09-06。范围：Linux、RTX 4060 Laptop 8GB、Ryzen 7745HX、32GB RAM；'
        'ERNIE-Image-Turbo、1024×1024、batch=1、8 Euler 步、CFG=1、PE 关闭。', '',
        '**补偿累加降低了 FP32 注意力误差，查询分块降低了工作区；完整长英文和中文的严格数值门限仍未全部通过。** '
        '本报告保留原实现、候选、最终实现、无效诊断和运行失败。'
        '图片可以生成、平均误差很小、单步或单算子通过，均不等于完整质量验收通过。', '',
        '## 实现', '',
        '- Vulkan 非 Flash FP32 SDPA 对 softmax 分母和概率乘 V 使用 Kahan 补偿，乘法仍为 FP32。'
        '独立 FP64 oracle 先复现长累加失败，再验证修复。',
        '- 查询每次最多处理 128 行，每行保留全部 K/V 和对应 mask；分块间在 GPU 上复制并同步，'
        '没有下载中间激活。原生缓存追加/容量管理和低精度 Flash 路径继续由固定 ncnn 处理。',
        '- 两个派生 shader 由完整 SHA256 约束；源文件是 CMake 重新配置依赖。'
        '已在复制的夹具中验证修改任一个 shader 会阻止已有构建继续，恢复后通过。',
        '- 增加逐 head/block 跟踪、指定中间 blob 提取、官方文本条件诊断和独立结果复核。'
        '所有长运行保留二进制快照；早期长英文仅保存准确验证器版本，后续运行保存完整逐运行 Python 脚本快照。', '',
        '实现与复现方式见 [数值诊断说明](../../../docs/NUMERICAL-DIAGNOSTICS.md)。'
        'ncnn 子模块未修改，仍为 `6a1bf000f363714839a36793addc8c879d3d899e`。', '',
        '## 完整原生 prompt → PNG', '',
        '同一提示词的各个候选使用同一份已校验的官方参考、初始 latent 和模型。'
        '下列五行均运行 native tokenizer 与 CPU 文本编码器。FP32 指 DiT 精度；'
        '文本、Euler、VAE 激活继续为 FP32，VAE GroupNorm 继续使用既有 FP64 归约。', '',
        '| 样本 / 实现 | 张量通过 | PNG MAE / 255 | 最大像素差 / 255 | PNG 门限 | 全部门限 |',
        '|---|---:|---:|---:|---|---|',
    ]
    for label, name, result in results:
        count = sum(row['passed'] for row in result['comparisons'])
        png = result['png']
        lines.append(f'| [{label}](runs/{name}/result.json) | {count}/25 | '
                     f'{png["mae"]:.9f} | {png["max_abs"]:.0f} | '
                     f'{"通过" if png["passed"] else "未通过"} | {"通过" if result["passed"] else "未通过"} |')
    lines += ['', '| 样本 / 实现 | 最终 latent NRMSE | 解码 NRMSE | 总耗时 / 秒 | 最大 RSS / GiB |',
              '|---|---:|---:|---:|---:|']
    for label, name, result in results:
        resource = result['resources']
        lines.append(f'| {label} | {metrics(result,"final")["nrmse"]:.9f} | '
                     f'{metrics(result,"decoded")["nrmse"]:.9f} | {resource["total_seconds"]:.3f} | '
                     f'{resource["max_rss_kib"]/1024**2:.4f} |')
    lines += ['', '计时包含完整模型 SHA256 检查和 trace。以上是单次观测，不是受控速度比较。', '',
              '### 固定门限与剩余失败', '',
              '原门限没有修改：FP32 张量 NRMSE ≤ 0.003，最大误差 ≤ '
              '`0.0002 + 0.01 × reference_max_abs`；PNG MAE ≤ 0.1、最大差 ≤ 2（8-bit 像素单位）。'
              '文本与位置条件使用独立的 0.0002 门限；初始 latent 和 token IDs 必须相同，'
              'native float → PNG 量化必须逐位匹配。25 项张量、PNG 或量化任一失败，整体即失败。', '']
    for label, name, result in results:
        if not result['passed']:
            failed = ', '.join('`'+row['tensor']+'`' for row in result['comparisons'] if not row['passed'])
            lines.append(f'- {label}：{failed}；PNG {"通过" if result["png"]["passed"] else "未通过"}。')
    lines += ['', '最新长英文的 `prediction-7` 最大差为 0.11422062，原门限为 0.06847718；'
              '最终 latent 和 PNG 已通过，不能以此忽略最后一步预测的失败。', '',
              '附加误差定位显示：长英文最后一步 524,288 个预测值中有 3 个超过最大误差门限。'
              '中文补偿版有 1/4/19 个晚期预测值、334 个 decoded 数值超限，PNG 有 60 个像素超过最大差 2；'
              '官方文本特征诊断仍有 12 个 decoded 数值、3 个像素超限。'
              '这仅用于后续定位，不改变任何 verdict；位置与分位数见 '
              '[误差位置](evidence/outputs/attention-error-localization-v1.json)。', '',
              '苹果历史 FP16 完整对照通过全部门限；长英文/中文的历史 FP16 和 BF16 block 失败见 '
              '[此前交付报告](../../2026-09-05/turbo-delivery/README.md)。本轮不把未改动的低精度路径描述为新增完整验收。', '',
              '## 文本条件隔离实验', '',
              '单独调用 `--reference-embeddings`，跳过 native 文本编码，读取相同官方 FP32 文本特征；'
              '后续去噪和 VAE 仍为 native 自由运行。该实验标记为 `saved_reference_diagnostic`，'
              '不计作完整 native prompt → PNG 通过。', '',
              f'诊断张量通过 **{sum(row["passed"] for row in diagnostic["comparisons"])}/25**，'
              f'PNG MAE **{diagnostic["png"]["mae"]:.9f}/255**、最大差 **{diagnostic["png"]["max_abs"]:.0f}/255**。'
              '仅 decoded 张量失败，但 PNG 最大差仍超过 2，整体未通过。'
              f'详细结果见 [诊断记录](runs/{DIAGNOSTIC}/result.json)。', '',
              '它表明文本编码的小差异对这条中文轨迹有明显影响，且不是唯一来源。'
              '不能从这一个提示词推导各模块的通用误差贡献比例。', '',
              '## 注意力算子与工作区', '',
              '| 独立检查 | 修复前 | 修复后 |', '|---|---:|---:|',
              '| 4160 keys 合成 FP64 oracle NRMSE | 2.07227e-5，失败 | 5.66540e-8，通过 |',
              '| 常量 V 列最大差 | 4.56572e-5 | 1.19209e-7 |',
              '| 真实 Q/K/V 对 FP64 的 NRMSE | 3.894996e-6 | 1.376740e-6 |',
              '| 257 queries 受限工作区最大单次分配 | 528416 字节，超过 300 KiB | 263168 字节，通过 |', '',
              '真实算子参考使用官方 CUDA RoPE 后的同一份 Q/K/V。FP64 检查选取 64 个 image queries，'
              '每个查询保留全部 4160 keys；不是对所有 queries 执行 FP64 全矩阵。'
              '官方 CUDA 对同一 FP64 参考也有 1.246941e-6 NRMSE，说明跨后端最后几位舍入仍存在。', '',
              '分块与完整 Kahan 注意力在真实 32-head、4160-query/key 上输出 SHA256 完全相同：'
              '`3aa9423f46a88dcf800bd97daefe30dffd566e364ac6a11eb7f9fd3d82b30b87`。'
              '合成补充检查覆盖共享/逐头 mask、无 mask、GQA、Q/K 长度不同、V 宽度不同及最后一个不完整分块；'
              '硬件和软件 Vulkan 上均与完整修正版逐位一致。', '',
              f'完整中文轨迹额外比较了补偿版与补偿分块版：{equivalence["matched_tensors"]}/'
              f'{equivalence["total_tensors"]} 个张量 SHA256 相同，PNG 相同为 `{str(equivalence["png_exact"]).lower()}`。'
              '这条检查只说明分块保留已有算术结果，原有数值失败仍然保留。', '',
              '4160-token、32 头 FP32 分数矩阵从 2,215,116,800 字节（约 2.06 GiB）变为最多 '
              '68,157,440 字节（65 MiB）。这是单个分数矩阵，不是整个 GPU 内存。'
              '完整 mask、权重、Q/K/V 和其他中间结果仍有独立成本。', '',
              f'最新长英文整卡 200 ms 采样峰值为 **{gpu["peak_mib"]} MiB**，启动样本为 {gpu["first_mib"]} MiB。'
              '它包含桌面和其他进程；不能写成本程序 allocator 精确峰值。'
              '旧单步诊断曾采到 7609 MiB，但输入和采样范围不同，不能据此给出受控百分比。', '',
              '分块增加同步：4160 queries 分成 33 块，每个 DiT block 内部增加 32 次提交；'
              '36 blocks 共 1188 次提交/步，不含输入/输出 head、Euler 和诊断下载。'
              '当前实现优先降低显存并保持算术；减少同步仍是后续性能工作。', '',
              '## 负结果与无效诊断', '',
              '- 长英文的完整矩阵补偿版本执行到第三步时遇到 `vkAllocateMemory failed -2` 和 SIGSEGV，'
              '实际返回码 139，没有完整图片或数值 verdict。该次没有重叠的大型 GPU 任务；'
              '随后分块版完整执行成功。此失败与此前重叠 VAE/GPU 任务造成的失败分开保留。',
              '- CPU RMSNorm 的 FP64 候选将真实第一层 NRMSE 从 1.8079e-6 降至 4.2994e-8，'
              '但完整 25 层文本对官方 NRMSE 从 6.2544e-6 略增至 6.7566e-6。'
              '候选源码和结果归档，运行时未采用该改动；已有 VAE FP64 GroupNorm 不受影响。',
              '- 早期中间 blob 对照有一次轴顺序错误；早期 SDPA MemoryData 夹具漏设 `21=0`，'
              '错误读取带标签权重流。两者不作为有效质量证据，后续使用修正夹具。',
              '- 分块 shader 原型曾因工作组 specialization 定义冲突崩溃；另一原型虽然数值通过，'
              '仍有 push-constant 反射警告，不能作为最终有效 pipeline 证据。'
              '最终版本使用 ncnn 的 `parameter` 约定并检查 7 个 push constants，完整硬件测试通过。', '',
              '上述失败日志和 CPU 候选源码位于 `evidence/outputs/`。失败执行未加入完整图像对照表。', '',
              '## 构建、安装和复核', '',
              '- NVIDIA 硬件 Vulkan：23/23 CTest 通过；工作区测试扩展后又单独通过四种 mask/GQA 布局。',
              '- CPU：9/9 CTest 通过；本轮新增的 GPU 工作区探针也通过 CPU 配置编译。',
              '- 独立 GCC + 固定 glslang 构建成功；软件 Vulkan 20 项通过、3 项 BF16 不支持而跳过。'
              '扩展工作区布局在该配置下另行通过。',
              '- Python 既有契约回归 34/34 通过。新增诊断/封存工具通过语法检查和真实数据执行；'
              '独立审计能拒绝放宽门限和伪造通过 verdict。另有 5/5 项历史快照兼容回归通过，损坏或缺失的现代快照不会退回旧版本。',
              '- 最终安装程序的帮助入口和完整 136 文件模型包校验通过，校验用时 17.5481 秒。'
              '安装二进制 SHA256 与当前 build 一致。',
              '- 本地工作完成后保存 Git 提交；本轮没有推送、发布或远程 GitHub Actions 结果。', '',
              '`tools/collect_parity_evidence.py` 逐个核对参考/实际张量、token IDs、图片、模型清单、'
              '二进制及脚本散列，检查门限与原定义一致，再使用 NumPy 独立重算误差和 PNG 量化。'
              '完整执行的失败 verdict 同样纳入 [results.json](results.json)。早期长英文的脚本覆盖明确标为 `validator_only`，'
              '从旧证据目录按原始散列读取；后续运行标为 `per_run_scripts`。二者都已重新核对实际张量与原门限。', '',
              '`manifest.json` 固定实际源文件和全部小型证据（含本 README）的 SHA256；'
              '它的 git_head 指向封存之前的实现提交。大型模型、图片、trace 和二进制留在本机 outputs/models，'
              '不进入 Git。运行二进制之间仅有统计计数差异时，详见 '
              '[二进制身份记录](evidence/outputs/attention-binary-identities-v1.json)。', '',
              '## 与示例项目的区别', '',
              '示例默认 BF16，Z-Image 还按显存预算处理 host-memory weights 和 VAE 分块；'
              '它们因此避开了本项目 FP16 扩展路径中的一部分问题。'
              '公开 CI/示例图不提供与本报告相同的固定数值门限证据，也不能反向证明示例没有误差。'
              '本轮没有运行示例模型，不报告质量或速度胜负。'
              '固定源码、验证范围和发布记录见 [示例核对](../../../docs/REFERENCE-COMPARISON.md)。', '',
              '## 尚未完成', '',
              '长英文最后一步预测、中文晚期预测和解码的严格数值门限；独立多提示词/多种子感知质量数据集；'
              '受控性能、精确 allocator 峰值与更多硬件验证。PE、原版非 Turbo CFG、量化、任意尺寸、'
              'Windows 与可分发二进制仍按独立阶段验收。'
              '这次改动没有建立跨去噪步的 DiT K/V 复用。', '',
    ]
    (ROOT/'outputs/attention-report-v1.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
