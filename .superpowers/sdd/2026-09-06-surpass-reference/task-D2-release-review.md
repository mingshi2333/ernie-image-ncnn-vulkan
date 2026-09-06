# D2 本地归档独立复核

对象：`00cf1f4`，固定实际 `outputs/delivery-candidate-v2`。root 只读检查工具与实际归档，不运行 CLI/模型/GPU、不发布、不修改原安装目录。

独立重跑 `python3 -W error::ResourceWarning -m unittest tests.test_build_release tests.test_model_download tests.test_release_manifest -v`，33/33 通过。检查默认 CLI allowlist 与显式 SDK 分支、已有输出拒绝、符号链接与异常目标拒绝、错误 source/build/install 关联保留 blocker、固定 Cargo archive 与单文件 checksum、fixture 归档重现性。fixture 结果不扩展为实际 SDK 或 Windows 打包证明。

重新认证实际 archive SHA `73153c3c9100dd078659cec85709c887b28fcffb716816990589c7f52af9ad81`，直接读取 tar 中全部成员（无需解包执行），确认 181 个成员的完整路径集合与 files.json 一致、无软/硬链接，每项大小与 SHA256 全部符合。4 个选中安装文件与 D1 的真实 bytes/hash 记录匹配，source changed_files 为空，build 的 source_identity_sha256 与 source 身份一致。记录包含 161 份项目/upstream/Rust notice 和 8 个直接 ELF 依赖。

实际 release.json 保持 `distributable=false`、`published=false`、`licenses_complete=false`、`platform.verified=false`。四个 Cargo.lock 缓存缺口、zlib/libstdc++ notice 缺口、实际静态链接和系统动态库闭包、平台验证均作为 blocker 保留。没有把存在若干 LICENSE、同机 ldconfig 候选、或手填 platform label 当作发布/许可完成。

本次工具适合作为本地审查草稿生成器，没有未关闭的 Important finding。它还不是完成的可发行包生成链；系统共享库未捆绑，ELF 的直接依赖观察与 RPM 候选不证明实际 loader 或完整转依赖闭包。Cargo.lock 包清单也包括其他 target 和 build/optional crate，后续应基于固定 target/实际产物收窄并补 Rust 标准库及嵌套 native 依赖证据。
