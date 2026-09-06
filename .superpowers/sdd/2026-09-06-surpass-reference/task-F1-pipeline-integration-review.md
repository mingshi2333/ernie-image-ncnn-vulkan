# F1 pipeline ComponentFiles 桥接独立审查

审查结论：当前冻结差异未发现 Critical / Important。一个 Minor 注释与实际链接关系不一致；可以继续进行受限 schema-3 包的真实端到端验证，但本审查不等于该验证已完成。

## 固定范围和证据

审查基线 HEAD `48a72a594d0fc6b0947439f9b02b51abed6cd3b3`。先冻结的 tracked diff 为同目录 `task-F1-pipeline-review.diff`，SHA256 `74def703378c2f9aa4715448efb0bd7953e31cafc6f84260c1069f263b9560c9`。新增 untracked component_files 两文件及随后添加的微型测试另以 `task-F1-pipeline-review-identity.json` 完整文件 SHA 固定；不能认为 tracked diff 已含新增文件。审查 source/test hashes 已写入报告配套身份清单。

范围：component_files/model_package.h、block_sequence/text_encoder/dit/denoiser.h/vae、pipeline、denoise_runner、CLI help、source_inventory 和相关 CMake/test 增量。只读模型包/Rust/shape_graph 既有实现用于追踪调用；没有再次将本人先前写的 package verifier 宣称为本轮独立全面审查。并行 image_encoder 改动不属于本报告的 root 桥接范围。

## Minor M1 — 构建注释仍称图契约没有接入生成器

`src/CMakeLists.txt:35` 写着 `Offline graph candidate verification only; not linked into the generator.`，但第 23 行 pipeline 现在链接 model-package，而第 44 行 model-package PUBLIC 链接 shape-graph。这会误导维护者对实际验证路径和证据覆盖的判断。建议改为已审查模板的原生图契约验证，并明确完整生成质量仍受独立 gates 限制。无需代码行为变化。状态：OPEN（已通知 root）。

## 调用链与兼容性

- 生产 text 全部 25 blocks、DiT 36 blocks 和 input/output heads、VAE 均由 `package.component` 取得拥有自身字符串的图与独立 weight path，最终 `load_component` 调用 `load_param_mem` + `load_model(path)`。未发现仍从旧目录拼接加载这些组件的生产分支。tokenizer 两文件、embedding、RoPE、BN 常量都走 `package.file`。
- 权重没有复制到内存 descriptor，也未按尺寸复制。ncnn 仍按既有层载入方式读各 weight path；“共享 CAS 文件”不意味着模型权重不展开或免除 GPU/CPU 权重内存。
- 固定 ncnn `src/net.cpp:2174–2202` 的文件和内存入口均委托 `load_param(DataReader&)`。本轮极小实算进一步验证文件图与内存图的相同输出，并在删除 param、清空 descriptor 后继续推理，覆盖图文本消费后的生命周期。
- 各 CPU/Vulkan `Option`、custom layer 注册、Net/Extractor/allocator 的作用域顺序未改变。串流 Net 销毁、外部 session allocator、提交等待和最终 Mat 保留逻辑没有随加载接口重写。text vector 模式仍先验证精确图及 weight 布局，再用派生内存图。
- 旧目录形式 run_text_blocks/run_block_sequence/run_dit/decode_vae 重载保留；probes 的重复 `--model` 与 `--input-head`/`--output-head` 仍映射相同 stem。`--timestep` 先返回，不需要模型目录文件。`DenoiseModel` 是内部 typed 结构变更，旧内部 C++ 字符串 aggregate 初始化需要迁移；不是公共 include/ernie/pipeline.h ABI 变更。仓库 probe 和 img2img 测试已迁移。
- 空模型/错误模型数量的 block/text 请求在 legacy path 转换前拒绝。`load_component({})` 在 ncnn 或文件访问前拒绝。完成 schedule 的 denoise 仍要求非空 descriptor、四个常量占位和合法 latent；不会读取 descriptor 指向的任何权重或解析图。微型测试使用不存在的 bin 与故意不可解析的非空图，start_step==steps 正常返回、stats 清空、observer 不触发。此结论不等于“空 DenoiseModel 允许零步”，当前接口明确不允许。
- Legacy run_dit/decode_vae wrapper 会先读取有界 param，再进入 typed 请求校验；错误请求可能更早收到 filesystem 错误。未见实际旧 CLI 合法调用因此失效，未将错误先后顺序提升为 Important。

## Schema、形状、verify 和来源范围

- generate 现在通过 ModelPackage 统一选择实例。schema1/2 仍走既有完整 verifier，配置来自与 model.cfg 验证一致的 manifest；显式 WH 必须匹配静态包。schema3 双实例必须显式指定 WH，单实例允许 WH=0 自动选择；没有扩大 6144 总 token 保护或宣布任意 WH 可生成。
- schema3 component 是固定已审查实例的 whole-graph hash 检查与内存文本实例化；源/目标 config 相同，未发现将未知图用任意 reshape 正则改写的旁路。当前两个独立实例的共享权重协议不能被表述成所有数学合法 shape 已验证。
- verify_model 的 schema3 分支不选择 WH，检查所有实例；读取/流式哈希权重不等于加载到 ncnn。旧 schema1/2 model.cfg 检查仍在；PE 包仍由独立接口负责。诊断/verify 和 generate 的证据性质需要继续区分。
- source_inventory 已纳入 schema3_contract.json。本轮执行检查确认所有审查中的源、测试、CMake 和该 JSON 都在 inventory 内；集合去重，不会重复收集。固定 ncnn 外部源码身份仍依赖原有 sources.lock/dependency 机制，本轮没有重做第三方全目录封存审计。

## 执行验证与限制

独立编译到 `/tmp`，未配置/写入共享 build-dev。仅链接现有小测试所需库；未加载实际模型、无 GPU 推理。

1. 当前 tests/test_component_files.cpp 与当前 src/component_files.cpp 独立编译运行，通过：2×2 InnerProduct 文件/内存加载位相等；删 param、清 descriptor 后结果仍 1 和 5.5；空 descriptor、NUL 图/路径、超长图、缺失 param、未知 stem 全拒绝。runner SHA `e21409b9cece67b304993fe41cabb6b918b1c86afe83035901f86f98c1d953a4`。
2. 当前 tests/test_img2img_contract.cpp 独立编译并链接现有 runtime，运行通过：端点、混合公式、正舍入、非法请求以及 completed-schedule no-denoise 路径。环境 OMP_NUM_THREADS=2；component test 本身 num_threads=1。
3. Python source inventory 精确成员检查通过。

编译使用 `/usr/bin/clang++ -std=c++17 -O0`，include 为 src、固定 ncnn/src、build-dev/ncnn/src；第一测试直接编译 component_files.cpp 并链接现有 libncnn.a，第二链接 libernie-runtime.a 和 libncnn.a，沿用当前 CMake link.txt 的 OpenMP/SPIRV/glslang 依赖。两测试只写临时目录，不依赖真实包/权重。

本轮没有运行完整 schema1/2/3 生成、GPU allocator 压力、两实例最终质量或性能对比。各 stage load_seconds 现在不包含 descriptor 构建阶段的 param 文件读取，不能直接把旧/new 子阶段计时差当加载优化；总 wall-clock 与固定 runner/输入身份仍是性能比较所需证据。未发现数学源码变化，不构成未经实跑的逐位相等承诺。

## 追加复核

root 将 src/CMakeLists.txt:35 改为 `Full graph allowlist used by offline checks and shared-package generation.`，已重新读取实际 diff，M1 CLOSED。同时重新读取 ModelPackage 失败信息新增 `Cannot open model package:` 前缀及三条 CLI expectation 对应变更：只统一错误上下文，不放宽失败条件。未把旧 executable 的通过计为此次证据；上述两个本 reviewer 的 CPU 测试是独立编译的。root 报告重新链接后的 8 项 focused CTest 全通过（1.77s，包括 CLI 26 子测试），作为 owner 提供的附加证据标注，非本人独立重跑。schema3 全图/生成仍待实际执行。
