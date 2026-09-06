# F2 图生图生产接入报告

## 已完成：schema-3 encoder 协议

提交 `638a58c` 保留旧 schema-3 `encoder: unavailable` 包的行为，并为固定源 manifest
`ef98859a...` 增加唯一受信任的 512×384 encoder。合同固定以下独立证据与运行 bytes：

- 官方 fixture `ee5ce5db...`、特化记录 `845826d9...`，以及 encoder/quant/BN 三份官方来源 manifest；
- encoder param `4858cc8b...`（8664 bytes）、bin `7fa2441a...`（137705600 bytes）；
- BN mean `9027fac5...` 与 variance `8b465682...`（各 512 bytes）；
- posterior mode、pixel-unshuffle 2、encoder BN eps 1e-4/no-affine 与 decoder 反变换 eps 1e-5。

`package_dynamic_model.py --schema3 --encoder DIR` 在建立输出目录前验证证据身份，随后将
encoder bytes 复制进 CAS；BN 必须与所选静态源已有绑定相同。Python 与 Rust verifier 都要求
available encoder 精确匹配编译期合同及对应 source instance，检查对象类型、大小、SHA 和无多余对象。
旧 unavailable 包不需要 encoder 对象；1024 或其他未审查源不能附加 encoder。C++
`ModelPackage::has_file` 可在加载前报告缺失组件，固定 encoder 图不做任意 ShapeGraph 改写。

## 已完成：API、CLI 与 pipeline

- 公共 stdlib-only 请求接受内存 RGB、strength、预处理模式和背景身份；旧字段和旧 positional
  aggregate 初始化顺序保留。
- CLI 实际读取 PNG/JPEG/BMP/TGA。无 `--resize` 时尺寸必须相同；`stretch`、保持比例并居中
  黑边的 `fit`、保持比例并居中裁剪的 `crop` 使用固定双线性变换。显式背景同时用于 alpha
  合成和 fit 填色；alpha 默认白、fit 默认黑。
- pipeline 从已认证包加载 encoder，强制 CPU FP32/direct。`strength=0` 在包校验和 encoder
  后直接 unpack/decode，不创建 DiT Vulkan context，不调用 PE/tokenizer/text/DiT，也不要求 prompt。
  为避免静默忽略，strength-zero 同时提供 prompt/PE/embeddings/text-down 会被拒绝；图生图请求
  `vae_device=vulkan` 也因 encoder 没有受审查 Vulkan 实现而明确拒绝。
- 正 strength 用同一 `FlowSchedule::turbo(steps)` 和已保存/生成 noise 调用
  `make_img2img_start`，再把绝对 `start_step` 传给既有 denoiser。进度为 suffix 内 1..N，耗时索引
  使用 `stats[i-start_step]`；trace prediction/step 文件仍用原 schedule 的绝对索引。
- trace 保存变换后的原始 RGB bytes、尺寸/strength/resize/background、encoder 三个边界、noise、
  初态、suffix 和最终 decoded。VAE 时间现在包含 encoder 与 decoder。

## 验证

- `python3 -m unittest tests.test_dynamic_package tests.test_vae_encoder_specialize`：24/24 通过。
- `cargo test --manifest-path tokenizer/Cargo.toml shared_package -- --nocapture`：3/3 通过。
- `ctest --test-dir build-dev -R 'image_io_contract|request_validation_contract|pipeline_api_contract|cli_contract|img2img_contract_cpu' --output-on-failure`：5/5 通过，1.74 秒。
- `ernie-image`、request validation、pipeline API、image I/O 四个目标以 `-j2` 增量构建成功。
- 对实际冻结 evidence 目录调用合同验证，返回 `available 512 384`，两个 encoder 运行文件身份均通过。
- 没有运行完整模型、GPU 或 formal15。

## 尚未闭合

本切片没有生成包含 encoder 的真实完整 schema-3 包，也没有运行 strength>0 的完整
PE/text/DiT/VAE 图生图或 15-case 验收。现有独立证据只证明一个 512×384 development
reconstruction；因此不能宣称 F2/F3 整体完成或图生图质量验收。Windows UTF-8 路径仍需真实
Windows 环境验证。完整包构建、真实 strength0/positive CLI 和独立代码审查应由 root 排队执行。
