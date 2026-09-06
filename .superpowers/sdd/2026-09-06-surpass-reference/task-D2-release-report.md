# D2 本地归档与许可准备切片

实现 tools/build_release.py、tests/test_build_release.py、docs/RELEASING.md，证据artifacts/2026-09-06/delivery-candidate。未改root CMake/运行CLI/包schema，未发布、未打包模型/环境/源checkout。root已明确批准只读D1 Vulkan安装前缀、无GPU的实际本地草稿。

默认仅4项安装文件：CLI与share/ernie-image的LICENSE/RUNNING/sources.lock。SDK只通过--include-sdk显式加入公共头、静态archive、CMake/pkg-config白名单；禁止未知SDK文件/符号链接。新输出目录、不覆盖安装，复制前后检查size/SHA，archive所有payload记录完整清单；tar+gzip固定时间/uid/gid和文件顺序，小fixture证明相同输入可重现archive哈希。发布/分发/platformverified始终false，只产生local_review_draft。不能通过传入“Windows verified”标签或自洽passed JSON取得正式身份。缺来源/notice/平台证据明确blockers，不把压缩成功当许可完成。

ELF readelf只读DT_NEEDED与RPATH，不执行输入。系统lib不复制；记录ldconfig所有host候选（包括本机multiarch候选）及RPM NEVRA/license字符串与可用/usr/share/licenses文本，声明不是实际加载记录。项目/stb保留来源hash，ncnn/glslang从sources.lock固定Git对象与glslang submodule pin提取LICENSE，不改上游。Cargo.lock88个包为保守清单（含本地bridge/构建/optional），认证本地.crate全SHA后只读提取Cargo.toml及license/notice；备用支持带逐文件checksum的registry source。没有因为名称/version匹配就相信任意本地notice。

11个小fixture tests全部通过：默认排除模型/secret/SDK、显式SDK、缺SDK/符号链接/旧输出拒绝、同输入archive重现、错误install/source/build记录不提升、真实install bytes字段关联、有效和损坏crate身份。CLI help通过。只在Linux执行测试，不虚构其他平台实测。

实际v1保留：shared outputs为symlink导致第一次命令按规则拒绝，之后用显式resolved父目录运行。初版Cargo cache缺.cargo-checksum所以明确列缺口，installed证据bytes与工具size字段差异导致保守绑定false。改进本地.crate认证与字段映射后新建v2，不覆盖任何v1。

真实v2 runner/command/输入metadata冻结于outputs/delivery-candidate-v2-execution。成功生成outputs/delivery-candidate-v2：4安装文件与D1实际结果匹配，94冻结项目源hash匹配、构建关联同70fa...sourceidentity；161项目/upstream/Rust notice，另系统RPM notices；181payload文件独立解包后逐项size/SHA与完整路径集合匹配。archive约10.672MiB，SHA73153c3c9100dd078659cec85709c887b28fcffb716816990589c7f52af9ad81。没有在packaging中调用CLI、模型或GPU。

仍为draft：r-efi/wasip2/wit-bindgen/zerocopy-derive本地cache缺失（不代表本机实际链接，需收敛有效依赖图），部分zlib-ng/libstdc++本机RPM没有可用license文本；实际静态notice归属/传递动态依赖加载/再分发依据未完成。licenses_complete=false、platformverified=false、distributable=false、published=false、公网URL null。系统候选许可证不自动证明最终可再分发，未替用户接受风险。D2全目标尚未完成，正式平台使用链与发布待后续。
