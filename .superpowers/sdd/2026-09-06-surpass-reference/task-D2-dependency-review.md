# D2 实际依赖补证独立复核

复核 `4dc3a44` 的实现与 `outputs/delivery-dependencies-v4`；数字说明修订 `ae39ef9`。只读 CPU 检查，没有重新构建、模型/GPU执行、下载或发布。独立复核脚本和结果为 `outputs/d2-dependency-review-v1/{check.py,result.json}`。

实际 173 个输出文件的完整路径集合、大小和 SHA 全部一致。独立遍历非 dev 依赖图得到 84 个包，其中 83 个 registry 包的完整 `.crate` 与 Cargo.lock checksum 相同，再逐一从归档重新读取全部 164 份 notice，逐字节认证输出；没有用文件名代替 archive 校验。四个排除包仅说明该 Linux 目标非可达，不外推其他 target。

独立使用另一份 SysV/GNU ar 解析器直接读取归档，不复用实现中的 `ar p`，认证 Rust 静态归档全部 572 个对象。19 个 Rust rlib 加 1 个 Oniguruma archive，共 20 个实际来源；全部 513 个关联对象的源/目标字节 SHA 相同。另核验 57 份项目依赖的 dep-info 身份，5 份 toolchain notice，以及 8 个 loader 实际解析的系统文件 SHA 和 ELF64/LSB/x86_64 头。完整 archive、D1 实际 CLI、原 build 命令文件、历史 rustc-info 和 Cargo.lock 身份都与记录相符。

`tests.test_release_dependencies` 实际 6/6 通过。结果支持本切片的依赖与文本来源说明，未发现未关闭 Important。归档成员是上界，未证明最终链接保留的每个 section；dep-info 关联也不等于通过独立可重现构建认证每个编译输入。动态链接器启动清单不涵盖 Vulkan ICD 后续加载。Rust 标准库供应商细项条款与目标环境/平台使用链仍需完成，`distributable=false` 和 `licenses_complete=false` 必须保留，原 release 草稿身份不变。

实际依赖结果 SHA：`2b40e0fe16d7a5e7cf927b9782768ed126b29ef1688340b5e19d46e7aac39e64`。
