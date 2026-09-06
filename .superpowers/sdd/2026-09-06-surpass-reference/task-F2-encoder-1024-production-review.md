# 固定 1024 encoder 生产注册独立审查

对象：`ec5f24524f6a79e4bad61a6f718a94b033e5a3f2`。结论：未发现阻断这份精确注册与包的缺陷，允许作者执行已准备的 CPU strength=0 实测；不代表该实测已经通过，也不推广任意尺寸。无 GPU、无模型加载、无生产代码修改。

已独立完整核对 `outputs/f2-production-img2img-package-1024-v1` 中 source 对象 SHA 为 `72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1`，对应 packed 64×64、text bucket64、25 text/36 DiT 的既有固定实例。encoder param `d3207b56f558d65b9901ff73640b51ae2a0934143b43eaeab6cad45e275b9ceb`、bin `7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce` 和两份 BN 文件均完整 SHA/size 相符。没有重复审计全套大 DiT/text weights；这次增量身份核查覆盖源manifest和所有新增encoder对象。

官方有效 fixture `outputs/img2img-encoder-reference-1024x1024-v2/fixture.json` SHA `88f2e8b7ad63a47fd993b069282dcb8bca9042cf56a470efa84237b711d9f7d9`，以及 specialized-v2 conversion SHA `a78e001ccd2abf48206e41eff3597e6afa9c3c0c7d9258c73c13e3e08e985334` 均从文件重新散列匹配registry。对应先前独立三边界真实通过证据，未误用 invalid-wrapper-metadata-v1。官方执行来源/资源证明的历史局限继续有效，见前份独立 encoder review。

Python builder 按 trusted source 查registry并精确绑定 fixture、conversion、revision、encoder/quant/BN manifest、空间特化模板与两项变更、graph/bin字节；Python shared verifier 要求 encoder source 存在于实例。Rust对runtime encoder声明与registry匹配，要求source实例出现，实际CAS hash/size校验，且encoder/已有实例同名文件冲突拒绝。C++仍在调用者验证包与全图allowlist后进入数值组件；`encode_vae` 只新增1024×1024固定分支，原32×32、64×32、512×384保持，其他shape先拒绝。临时candidate函数已删除；正规生产与probe使用同一入口。数学、RGB FP32顺序、posterior/packing、CPU direct限制、finiteness/shape检查未被此提交改变。BN eps编码侧1e-4、解码逆变换1e-5的区别保持。

独立小验证：正确模块 `tests.test_vae_encoder_1024` + `tests.test_dynamic_package` 25/25通过，现有 `build-dev/ernie-image-encoder-contract` 实际运行通过。首次命令误用了不存在的 `tests.test_package_dynamic_model`，产生导入错误（另一模块4测试已通过）；更正后完整25测试通过，未隐瞒失败。Rust3项结果来自作者报告，本轮只读检查实际验证路径，没有另建Rust或GPU。

后续真实执行应使用已冻结的生产runner和确切包/输入，检查strength0无prompt/PE/text/DiT加载，encoder三边界与decoder/PNG固定门槛，并记录持续guard及实际退出。当前仅实现与准备完成，不记为完整1024 img2img生产通过。
