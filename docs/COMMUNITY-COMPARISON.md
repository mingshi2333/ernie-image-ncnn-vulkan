# 同类实现与精度路线

核对日期：2026-09-13。本页比较公开代码、模型仓库与作者报告，没有执行其他项目的完整生成器。没有同机、同权重、同初始 latent、同精度的配对运行，因此不按不同作者的耗时和数值门槛给出总排名。

本项目定位是面向消费级显卡的原生文生图运行时，着重处理有限显存下的执行、可复核的模型转换和交付。逐块加载、预算内缓存与预取、GPU/RAM 权重选择、受限的 host buffer 回退和检查点重试构成当前实现的特点；这些能力与实测边界见[内存记录](NUMERICAL-RESULTS.md#内存执行改动后的完整回归)。

## 查到的实现

| 项目及核对提交 | 已公开的侧重点 | 对本项目有用的比较 |
|---|---|---|
| [futz12](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6)，`8dcd6e4` | 默认 BF16、host-memory weights，PE、可变分辨率与多组示例；[Discussion #6996](https://github.com/Tencent/ncnn/discussions/6996) | BF16 产品使用和示例展示值得参考。其 PE 已有 KV cache，不能把缓存本身当成本项目独有功能。双方没有完整同机性能横比 |
| [everythingfornothing](https://github.com/everythingfornothing/ernie-image-ncnn-vulkan/tree/77b4cbbd90bb49723efda95b9009c0bab4f0f6f5)，`77b4cbb` | FP32 CPU/Vulkan、分阶段误差定位、1024×1024 完整生成；[Discussion #6998](https://github.com/Tencent/ncnn/discussions/6998) | 与本项目最值得对照的是数值验证与 Vulkan 算子适配。其 README 明确将低精度、可变图像尺寸、PE 和性能优化列为尚未验证范围 |
| [touqiang-guang](https://github.com/touqiang-guang/ernie-image-ncnn/tree/953a98d578b01a24966cd12144c88234685c2a32)，`953a98d` | 固定 512×512，主要 CPU 推理，提供 Windows 可执行文件及 bat/vbs 输入入口 | 启动操作简单值得学习。README 的 GPU 20–30 分钟是预计值，不能当成实测；源码主要 DiT/VAE 路径关闭 Vulkan |
| [nihui/zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan/tree/c1938eef9cd7e03cb216da4f4c89a0c041543e7a)，`c1938ee` | 不同模型的工程参照：默认 BF16、按 heap budget 选择 host weights、VAE tiling、三平台发行包 | 发行和显存策略成熟，适合学习；模型不同，不能直接比较图片质量或速度。当前本项目已提供模型下载，程序还需要编译 |

搜索还返回 `nicccce/ernie-image-ncnn-vulkan`，其默认分支 commits API 在核对时返回空仓库（HTTP 409），没有据此纳入实现排名。

`everythingfornothing` 作者报告的 RTX 4080 SUPER、1024×1024、8 步 FP32 两例总耗时为 408.27 和 424.25 秒；Vulkan 对该项目 CPU 结果的 PSNR 为 57.16 / 64.43 dB。这是明确配置下的作者结果。本项目使用 RTX 4060 Laptop，且参考路径、提示词和测量范围不同，不能用两组数字宣称更快或更准。[原始报告](https://github.com/Tencent/ncnn/discussions/6998)

模型下载入口也做了匿名 API 核对：`wuyex/ernie-image-ncnn`、`Coderdw/ernie-image-ncnn-vulkan` 与 `chaseSun123/ernie-image-ncnn` 均存在。第三个仓库逻辑文件大小约 23.36 GB，与其 README 的约 500 MB 不符；此处只纠正文档大小，不据此否定运行结果。[模型文件](https://huggingface.co/chaseSun123/ernie-image-ncnn/tree/848fbdc1abbe1e9b3517ede03bf59a69ac05e263)

## 本项目已经支持什么精度

当前发布的是一套主模型，主要权重以官方 BF16 值保存，部分参数与派生数据保留 FP32。运行精度与文件格式独立，详见[模型说明](models/README.md#存储格式与运行精度)。

| 运行模式 | 当前用途 | 发布建议 |
|---|---|---|
| FP32 | 数值参考和 README 推荐的首跑配置 | 保持现有主包，无需另存膨胀后的 FP32 权重副本 |
| FP16 | Vulkan 默认的混合精度路径；FP32 残差、归一化与 master latent 保留 | 继续测量速度、显存和图像差异，不能把 `fp16` 标签解释成所有算子都执行半精度运算 |
| BF16 | 已实现并能出图，仍有独立数值偏差和设备能力要求 | 优先做稳定性与实际收益验证，通过后再作为推荐配置；目前维持实验标记 |
| INT8 / Q6 / Q4 | 本项目尚未交付量化版本 | 作为后续独立实验，先验证具体算子和后端，避免直接套用文件转换命令 |

512×512 同一苹果 fixture 的已有完整结果中，FP32、FP16、修正后 BF16 的 PNG MAE 分别为 0.000361125、0.235983531、1.294207255（0–255 通道单位）。它们说明该输入下的数值接近程度，不是通用画质评分，也不能证明 BF16 格式本身较差。[完整历史与边界](NUMERICAL-RESULTS.md)

## 后续优先级

1. 保持一套可下载主包，用运行参数区分 FP32/FP16/BF16，完成低精度路径的成图、耗时与内存对照。
2. 继续减少实际使用步骤，尤其是程序发行包。其他项目在双击启动与三平台发行方面已有可学习的做法。
3. 量化先从可选的 CPU PE 做小范围试验，再评估 DiT GPU 路径。保留 FP32 基准、原有官方对照和输出文本/图像，达到预先说明的可接受差异后再发布新包。

核对时 ncnn 最新 Git 为 `c0abf4830d8f8637efa5d34fce3985ba3d8642bd`。它的 `ncnnllm2int` 支持 4/6/8 位分块格式，其中 4/6 位是仅权重量化，8 位 CPU 路径为动态 W8A8，输出为 FP32。该格式不能与普通 INT8 路径混用：当前 Vulkan `Gemm::create_pipeline` 遇到 `weight_block_quantize` 直接报不支持，而另一条 `quantize_term` 路径仍可进入 `create_pipeline_int8`。因此不能承诺 Q4/Q6/Q8 文件转换后即可直接加速本项目的 Vulkan DiT。[ncnn 量化文档](https://github.com/Tencent/ncnn/blob/c0abf4830d8f8637efa5d34fce3985ba3d8642bd/docs/how-to-use-and-FAQ/quantized-int8-inference.md)、[Vulkan 源码](https://github.com/Tencent/ncnn/blob/c0abf4830d8f8637efa5d34fce3985ba3d8642bd/src/layer/vulkan/gemm_vulkan.cpp#L51)

本轮只做公开资料核对与交付文档更新。生产依赖仍为已验证的 `3b7bdba7`，没有因查询最新 Git 而改变源码或模型。
