# P3 功能与原生尺寸 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐参考项目的图生图、运行时尺寸、多格式和设备控制，使用户用一个原生程序与一套权重完成常见本地生成工作。

**Architecture:** 由模型包声明允许的形状契约；原生 ShapePlan 实例化经审查的图。推理库处理 RGB/latent，CLI 负责图像文件。图生图复用现有 denoiser，不复制完整生成器。

**Tech Stack:** C++17、现有 ncnn/pnnx 转换工具、AutoencoderKLFlux2 官方模块、CMake/CTest/Python unittest、固定版本图像编解码依赖。

## Global Constraints

继承 [总计划](2026-09-06-surpass-reference.md)。以下接口与命令均待实现。动态能力不能仅靠正则修改已导出的任意 reshape；不能让用户每换尺寸都启动 Python。P0 将记录对方范围，支持声明只覆盖本项目真实验证的图契约与硬件。

## Task F1: 共享权重包和原生 ShapePlan

**Files:** Create `src/shape_plan.h`, `src/shape_plan.cpp`, `tools/package_dynamic_model.py`, `tests/test_shape_plan.cpp`, `tests/test_dynamic_package.py`；Extend `src/model_config.h`, `src/model_config.cpp`, `src/pipeline.cpp`, `tools/package_model.py`, `tools/prepare_variant.py`, `tools/specialize_vae.py`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`。

**Interfaces:** `ShapePlan::create(contract, width, height, valid_text_tokens)` 产生 latent/patch 空间维度、有效长度、选择的文本桶、RoPE/mask 和每个需特化的参数；参数图来源 hash 与 schema 一起验证。schema-3 包共享权重，图模板只携带经过允许列表审查的形状参数；schema-1/2 继续走原有静态路径。

- [ ] 写 checked-arithmetic 和边界测试。目标轴长 16..2048 且为 16 倍数、面积 ≤ 2097152；有效文本 1..2048（含 BOS）。以下为接口示例，contract 由小型 fixture 构造：

```cpp
auto shape = ShapePlan::create(contract, 1376, 768, 1080);
require(shape.width == 1376 && shape.height == 768, "shape changed");
require(shape.valid_text_tokens == 1080, "padding became valid text");
require_rejected([&] { ShapePlan::create(contract, 1377, 768, 1080); });
require_rejected([&] { ShapePlan::create(contract, 2048, 2048, 1080); });
require_rejected([&] { ShapePlan::create(contract, 1376, 768, 2049); });
```

- [ ] 审计 DiT head/block/finalizer、VAE 的全部形状相关节点、position 与 packing 维度。先导出小形状和横竖两种独立参考，证明运行时改变量只影响允许列表；任何未知图 hash 都拒绝实例化。
- [ ] 保留独立导出的 32/64/2048 文本桶，按真实 token 数选最小可容纳桶。只有单独证明 mask/shape 等价后才新增中间桶；不得重命名同一静态图伪装新桶。
- [ ] 现有转换器的轴长 ≤1024、总 token ≤6144 是当前保护，不直接删除。先对大目标做单块峰值/图契约实验，再将 schema-3 的新范围纳入原生预算计划；未过资源检查不能发布范围声明。
- [ ] 对总计划十个 canary 尺寸独立执行官方参考与 native。对正式范围内最小 16×16、16×2048、2048×16 额外检查实际图执行；若模型对极端长宽比有更严限制，先记录目标失败再明确新版本支持范围，不能静默缩范围。
- [ ] 逐一破坏模板、shape 元数据、尾层权重和大小；Python/原生验证均拒绝。迁移 schema-1/2 包并验证无内部外链、没有每尺寸复制全部权重。
- [ ] 执行 `ctest --test-dir build -R 'shape_plan|pipeline_api' --output-on-failure` 与 `.venv/bin/python -m unittest discover -s tests -p 'test_dynamic_package.py'`；提交 `feat: instantiate validated runtime shapes from shared weights`。

**Acceptance:** 一个安装后的原生二进制和同一套权重在不调用 Python 的情况下生成所列尺寸；不依赖开发机的转换目录；超限在加载大权重前拒绝。新增形状使用 native text，不能只验预计算 embeddings。

## Task F2: VAE encoder 与图生图数学契约

**Files:** Create `tools/export_vae_encoder.py`, `tools/reference_img2img.py`, `tools/validate_img2img.py`, `src/img2img.h`, `src/img2img.cpp`, `tests/test_img2img_contract.cpp`；Extend `src/vae.h`, `src/vae.cpp`, `src/latent_ops.h`, `src/latent_ops.cpp`, `src/pipeline.cpp`, `tools/package_dynamic_model.py`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`。

**Interfaces:** `encode_vae(RgbTensor, options) -> encoded_latent`；`Img2ImgStart make_img2img_start(encoded, saved_noise, sigmas, steps, strength)` 返回初始 latent 与起始 step。RGB 归一化、encoder distribution 取值、patch packing、BN 归一化由固定官方配置及对方实现双向核对后写成显式 manifest，不以常见 Stable Diffusion 公式代替。

- [ ] 先检查对方 VAE encoder 是 mean/mode 还是采样，以及其 RGB 映射、posterior、quant conv 和 BN 正反变换。保存各阶段真实 tensor 与配置散列，再从小尺寸/分段导出 encoder，并用 F1 的受审查图契约覆盖目标尺寸；不重跑已知会超过主机预算的无界整图转换。只支持经验证的那种定义，额外随机变量必须由保存 bytes 输入。
- [ ] 使用固定官方 VAE encoder/decoder 和 Turbo sigma schedule 组合出独立 oracle。报告名称为“官方模块组合的图生图参考”，不称其为官方已发布的 ERNIE 图生图 pipeline。
- [ ] 对照对方源码固定强度语义：`strength=0` 时不加噪、不去噪，仍进行 VAE 编解码；大于零时 `n=min(steps,max(1,floor(steps*strength+0.5)))`，`start=steps-n`，初态为 `sigma[start]*noise + (1-sigma[start])*encoded`。Python 使用同一 positive-round 定义，避免 bankers rounding 差异。
- [ ] 写边界测试：强度 0 返回 encoded 与 `start=steps`；强度 1 在该 schedule 首个 sigma=1 时返回 noise 与 `start=0`；.25/.5/.75 对 8 步映射为 2/4/6 个去噪步；NaN/负值/大于 1 均拒绝。
- [ ] 用三张不同来源的自有/许可清晰输入：纹理照片、文字排版、几何色块，测试 0/.25/.5/.75/1。保留同一 RGB 解码结果与 noise，先验 encoder 的 mean/packed/normalized 边界，再验完整预测、latent、decoded 和像素。
- [ ] 明确强度 0 是 VAE reconstruction，通常不与原图逐像素相同；1 与同条件文生图需一致。各过程必须复用同一个 Euler/denoiser 实现。
- [ ] 执行 `ctest --test-dir build -R img2img_contract --output-on-failure`，再用新 `.venv/bin/python tools/validate_img2img.py --manifest outputs/port-corpus-v1/manifest.json --suite img2img --output outputs/img2img-dev-v1` 串行验证；提交 `feat: add validated VAE encoding and image-conditioned generation`。

**Acceptance:** 图生图强度与输入变换可复现，15 个输入/强度组合通过适用固定门槛；不会用文生图覆盖输入图像，包中 encoder 完整性检查与 decoder 等同。

## Task F3: 公共 C++ API 与多格式 CLI

**Files:** Extend `include/ernie/pipeline.h`, `src/pipeline.cpp`, `cli/options.h`, `cli/options.cpp`, `cli/main.cpp`, `cli/image_io.h`, `cli/image_io.cpp`, `tests/test_pipeline_api.cpp`, `tests/test_cli.py`, `cli/CMakeLists.txt`, `tests/CMakeLists.txt`；Create `tests/test_image_io.cpp`, `tests/test_request_validation.cpp`。扩展现有 PNG writer 并保留其需要的兼容入口。

**Interfaces:** 将 `RgbImage` 声明移到 `GenerationRequest` 前，新增 `std::optional<RgbImage> input_image`、`float strength=.5f`、`threads/gpu_index/text_device` 等显式设置；既有字段与 `generate()` 保持源兼容。推理库中不得出现图像编解码依赖或文件扩展名判断。

- [ ] 输入图像非空时强制尺寸与 RGB buffer 大小一致并做乘法溢出检查。宽高为零表示采用输入大小；显式目标尺寸不匹配时默认报错，用户指定 `--resize stretch|fit|crop` 才在 CLI 变换，具体插值与填色写入 trace。选择 `fit` 默认黑边，`crop` 居中；禁止静默 resize。
- [ ] 新建 `read_image/write_image`，支持 PNG/JPEG/BMP/TGA；PNG 继续现有依赖，其余选择一个固定版本且许可完整的轻量库，并在 `sources.lock.json` 留 hash。处理 RGB/灰度/RGBA：灰度复制 RGB，alpha 默认合成白底并允许 `--background #RRGGBB` 明示，行为记录在请求元数据中。
- [ ] PNG/BMP/TGA roundtrip 对自生成小图逐像素一致；JPEG 以固定质量 95、尺寸和解码误差检查，不使用无损断言。损坏文件、超大声明尺寸、路径含中文、输出已存在均有实际错误测试。
- [ ] CLI 增加 `--input`, `--strength`, `--threads`, `--gpu`, `--text-device`, 多格式输出，以及 `--diagnose` 展示实际设备/功能/包版本。运行时禁用的功能不显示为已支持，失败要指出可操作的参数/模型问题。
- [ ] 保留已有命令兼容和 prompt 原始 bytes。帮助信息以最少必需步骤展示文生图、图生图、PE 三个真实例子；推理仍离线，`--diagnose` 不联网也不加载整个模型。
- [ ] 独立小应用只包含公共头并链接 `ernie::pipeline`，内存 RGB 输入和输出皆可用；它不链接图像格式库。Windows 参数/文件名 UTF-8 边界在真实 Windows 下补测，不能仅靠 Linux unicode 文件测试代替。
- [ ] 执行 `ctest --test-dir build -R 'image_io|request_validation|pipeline_api|cli_contract' --output-on-failure`；提交 `feat: expose image inputs and complete native CLI controls`。

**Acceptance:** 使用说明可以直接照做；CLI 主函数继续只做读入、调用和保存，pipeline 只做编排。

## Task F4: 图生图与 PE 的组合和兼容冻结

**Files:** Extend `tests/test_cli.py`, `tests/test_package.py`, `tests/test_pipeline_api.cpp`, `tools/source_inventory.py`, `docs/RUNNING.md`, `docs/CODE-STRUCTURE.md`, `tools/README.md`。

- [ ] 覆盖文生图、PE 文生图、图生图、PE 图生图四种组合；记录最终实际 prompt、输入 RGB hash、增强 token、强度及选用尺寸。PE 结束后仍释放权重/cache，再进入图像编码/文本/DiT，不同时常驻全部模型。
- [ ] schema-1/2 没有 encoder 时图生图应给出缺少组件错误，原文生图命令保持可用；schema-3 与PE独立包均通过移动/完整性检查。
- [ ] 验证 CPU-only 构建的帮助与错误语义，以及 GPU 不支持目标精度时的明确行为。相同 API 不能产生和 CLI 不同的参数默认值。
- [ ] 更新结构图、模块所有权和完整用户命令，以真实运行输出修正文档。执行现有适用 CTest/Python 全部回归，并保存四种模式的小图/真实图记录；提交 `test: freeze generation modes and package compatibility`。

**Acceptance:** F/A 的 Linux 功能证据齐备，Windows/macOS实际交付由 P4/P5 验收。
