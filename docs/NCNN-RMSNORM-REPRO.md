# Vulkan FP16 RMSNorm 的大数值输入问题

核验版本：上游 ncnn `6a1bf000f363714839a36793addc8c879d3d899e`，RTX 4060 Laptop。上游源码未修改。项目通过注册覆盖层，将 RMSNorm 的临时计算保持为 FP32。

## 可复现现象

```sh
cmake --build build --target ernie-rmsnorm-probe -j 4
# 原生实现：小输入通过，大输入失败，退出码为 1
build/ernie-rmsnorm-probe vulkan fp16 --native
# 项目覆盖层：相同输入通过
build/ernie-rmsnorm-probe vulkan fp16
```

探针不需要模型权重。两类布局是 `[8,4096]` 和 `[8,5,128]`，每行输入为 `scale * (0.5 + k/8)`，`k` 在 0 到 7 之间循环。全部输入值和 affine 权重都能被 FP16/BF16 精确表示。参考在独立的双精度计算中完成平方、求均值、开方和 affine 乘法。

| 实现 | 输入 scale | 最大绝对误差 | 判定 |
|---|---:|---:|---|
| 原生 Vulkan FP16 | 1 | 约 0.000452 | 通过 |
| 原生 Vulkan FP16 | 512 | 约 1.315 | 失败 |
| 项目 FP32 临时计算 | 1、512 | 见核函数原始日志 | 通过 |

## 原因与项目处理

[`rmsnorm_square.comp`](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/vulkan/shader/rmsnorm_square.comp) 在 float 中计算平方，但将结果写到 `sfp` 缓冲。[`RMSNorm_vulkan::forward_inplace`](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/vulkan/rmsnorm_vulkan.cpp) 的平方临时缓冲继承输入 elemsize。FP16 输入超过约 256 时，其平方可能超出 FP16 表示范围。之后改为 FP32 求和已经无法恢复丢失的数值。

项目的 `ErnieRMSNorm` 保留原生 CPU 实现和原生 Vulkan 的分组、packing、epsilon、affine 语义。在 Vulkan 低精度输入时，先在设备内转换为 FP32，调用原生 FP32 RMSNorm，再转回原存储精度。这会增加格式转换和临时显存，不是零成本修复。后续可研究直接读取低精度输入、使用 FP32 累加与归一化的融合核。

真实 ERNIE 第 0、1 两块串联也复现了这一差异：FP16 NRMSE 从 `0.672275889` 降至 `0.001470287`，CPU FP32 和 Vulkan FP32 结果保持不变。输入仍是合成 hidden states 和 shared AdaLN，所以这项结果不能代替真实文本条件下的完整模型验证。

相关实现：[覆盖层](../src/ernie_rmsnorm.cpp)、[探针](../probes/rmsnorm_probe.cpp)、[串联执行](../src/block_sequence.cpp)。目前尚未向上游发送 issue 或 PR。

## 输出 LayerNorm 的同类问题

接入官方 finalizer 后，其非 affine LayerNorm 在较大合成激活上也出现 FP16 平方溢出。输入为固定随机数乘 512，使用真实 `final_norm.linear` 和 `final_linear` 权重。仅将归一化临时计算改为 FP32 后，输出 NRMSE 从 `1.000087388` 降到 `0.000514459`，FP32 对照保持通过。

原因见 [`LayerNorm_vulkan`](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/vulkan/layernorm_vulkan.cpp)：中心化平方临时缓冲同样使用输入的 elemsize。项目的 LayerNorm 覆盖目前严格限定为 ERNIE 的 4096 维、非 affine 配置，没有泛化到其它模型。

独立双精度探针使用可被三种精度准确表示的输入，scale 为 1 和 1024：

```sh
build/ernie-rmsnorm-probe vulkan fp16 --layernorm --native
build/ernie-rmsnorm-probe vulkan fp16 --layernorm
```

原生实现的小值误差约 `0.000262`，大值误差约 `1.52753`，后者失败。项目实现通过 CPU FP32、Vulkan FP32/FP16/BF16 的相同参考检查。RMSNorm 和 LayerNorm 共用设备内精度提升与还原逻辑，仍由上游核函数完成实际归一化。
