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

## 生产包与真实 strength=0 补充执行

使用正常 `package_dynamic_model.py --schema3 --encoder` 路径建立了新的
`outputs/f2-production-img2img-package-v1`，没有修改既有模型目录或冻结资产。包的 manifest SHA-256
为 `2b212d7a...d923`，声明 1 个 512×384/text-2048 instance、80 个 CAS objects、总对象字节
23,409,328,994；native schema-3 verifier 对实际目录通过。encoder 合同仍是唯一受审查的
512×384 合同，param/bin/BN 四个对象分别是 `4858cc8b...`、`7fa2441a...`、`9027fac5...`、
`8b465682...`。

运行前将生产 `ernie-image` 与其完整实际源码清单冻结到
`outputs/f2-production-strength0-v1-execution`。runner SHA-256 为 `941a131b...b52d`，源码快照
HEAD 为 `f04b35f3...`，包含 schema source identity 修复和 resize 半像素边界修复。输入是与官方
512×384 reference 相同的确定性 RGB fixture；PNG SHA-256 为 `7b991c64...cb8c`，解码 RGB bytes
SHA-256 为 `c01747af...e2ea`。

实际 CLI 使用 `--device cpu --vae-device cpu --strength 0 --threads 2 --width 512 --height 384`，
在 systemd user scope 的 `MemoryMax=3G`、`MemorySwapMax=0` 下退出 0。wall time 42.70 秒，峰值
RSS 1,441,796 KiB，swap 0。执行包含约 19.51 秒的完整包认证以及约 23.14 秒的 encoder/decode；
没有运行 GPU。

六边界结果通过既有固定 gate：输入 RGB、encoder mean、packed、normalized/final、unpacked 的生产
散列与既有原生 512×384 证据逐字节一致。相对官方 reference，unpacked 最大绝对误差
`6.55651e-6`；decoded 最大绝对误差 `1.65403e-5`、平均 `8.19570e-7`；PNG 最大通道误差 1、
平均 `0.000125461`。新 decoded 与旧原生 probe 的最大差为 `4.76837e-7`，说明散列变化只是
最后位数差异，并未扩大原有官方误差。全部散列和逐项统计保存在 execution 的 `result.json`。

`strace openat` 记录证明包认证先读取全部 80 个 CAS 对象；认证后只有 source manifest、encoder
param/bin、BN mean/variance 和 decoder param/bin 再次打开。text/DiT 对象都只有认证读取，没有运行
加载；trace 也只有 input、encoder、final/unpacked/decoded 边界，没有 prompt、text、noise、initial、
prediction 或 step 文件。因此本次 strength=0 确实绕过 PE、tokenizer、text 和 DiT，而不是仅靠
计时推断。

本轮还关闭了独立审查的两个 Important：Rust verifier 现在要求 `encoder_source_manifest_sha256`
确实存在于 instances，不能借用别的 instance 的相同 CAS 自证；双线性缩放分别 clamp 原始 floor
坐标的两个邻点，2×1 红/蓝图缩放到 4×1 时左右边界保持纯红/纯蓝。Rust 3 个 package tests、
image-I/O 与 CLI 两个 CTest 均通过。

本次只证明真实生产 schema-3 的 512×384 strength=0 CPU 路径。strength=0.5 的相同输入、noise、
schedule 与官方 oracle 已可准备，但尚未执行；PE/text/DiT positive-strength、15-case 与其他尺寸仍是
pending，不能据此宣称整体 F2 或质量验收完成。

## Positive strength 0.5 真实执行（负面 parity 结果）

提交 `4edcfb3` 增加独立 suffix oracle，提交 `bbdba1c` 根据独立审查关闭认证缺口。修复后的
validator 将实际 `input.rgb/out0/out1/out2` 逐项绑定到固定官方 encoder fixture `ee5ce5db...`，
固定 encoder BN eps、原八步 sigma，并从受信 out2 与 saved noise 逐字节重算 FP32 start。它还绑定
固定源包 manifest `ef98859a...`，验证落盘 fixture 与返回对象相同、短 prompt/8 steps/start 4、恰好
4 个 prediction/step、六输入和三最终张量，共 17 张量的固定名称/shape/dtype/SHA/finite bytes，
以及最终 PNG。7 个小型 unit tests 通过；旧执行数据在修复后的 validator 下通过 17 分母 post-audit。

输入冻结在 `outputs/f2-positive05-512x384-v1/inputs`：短 prompt 是
`A red apple on a wooden table.`（9 tokens），PE off，native text reduction 为 Vector FP32；saved
noise 是新生成的 little-endian FP32 `[1,128,24,32]`，SHA `83226a3c...`。官方 normalized encoder
SHA `2f7c535a...`；八步原 schedule 的 sigma[4] 为 `0.800000011920929`，按固定乘后加顺序得到
start-4 SHA `958c63fe...`。没有使用 native encoder latent 作为官方 expected。

官方路径从固定官方 encoder 输出开始，执行官方 text、原八步 schedule 的绝对步 4/5/6/7、36 个
官方 DiT blocks、decoder inverse BN eps 1e-5 和官方 decoder。CUDA worker exit 0，wall 100.89 秒，
峰值 RSS 3,319,284 KiB，swap 0；suffix fixture SHA `378085a8...`，PNG SHA `e55552ef...`。

同一冻结 runner `941a131b...` 随后用同一 RGB、noise、prompt、包和绝对 suffix 执行 native Vulkan
FP32 + Vector text + CPU direct VAE。它 exit 0，wall 459.52 秒，峰值 RSS 1,783,604 KiB，swap 0；
text 169.105 秒，四步 denoise 分别 59.8422/58.9312/59.5022/59.4436 秒。trace 完整包含 encoder、
noise/start、conditioning、prediction-4..7、step-4..7、final/unpacked/decoded。

执行完整，但 parity gate **失败**。saved noise 逐字节相同，encoder normalized NRMSE
`7.06593e-7`，start NRMSE `1.51093e-7`；首个明显分歧在 Vector FP32 text，NMRSE `0.0362576`
（固定 conditioning gate `0.0002`）。prediction-4 NRMSE `0.0573655`，final `0.0502410`，decoded
`0.0978269`；PNG 最大通道差 86、MAE `3.72495`，超过 FP32 gate 2/0.1。完整逐边界统计保存在
`comparison.json`，`result.json` 明确记录 `status=parity_failed`、`complete_execution=true`、
`quality_gate_passed=false`。该负面结果不能成为 F2 acceptance，也不能与完整 text-to-image 25 项分母
混算。正式 15/72 输入未触碰。
