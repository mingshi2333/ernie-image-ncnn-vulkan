# D2 Linux 实际依赖与许可补证

完成独立 `tools/release_dependencies.py`、六项小合同和 RELEASING 接口说明。只读固定安装/构建、离线Cargo metadata与小型archive/ELF检查，没有模型运行、GPU、编译、网络下载或发布。原delivery-candidate-v2 archive不变；新增结果仍distributable=false、licenses_complete=false。

## 真实执行与结论

有效结果 `outputs/delivery-dependencies-v4/dependencies.json`，同目录保存完整cargo-metadata、loader输出、169份notice/toolchain文本、运行工具快照及files.json（173文件，1414429字节，清单自身不计入）。runner快照在执行后保存；工具在这次只读采集期间没有改动，不声称执行前完全密封。v1因只识别Make而实际构建为Ninja明确失败，v2/v3中间证据保留；最终v4识别并限制实际Ninja Cargo命令，没有覆盖历史。

输入为D1 fixed source v2、build-install-vulkan-v2和 `/tmp/ernie-sdk-vulkan-install-v2/移动 installation/bin/ernie-image`。命令记录真实CARGO_BUILD_JOBS=2、cargo build --release --locked、固定manifest/target-dir且无显式target或非默认feature参数；缓存rustc-info给出rustc1.98.0/commit88d9e12ae178fab0fb5cc050a94da85685d449ea、Fedora1.98.0-1.fc44、host x86_64-unknown-linux-gnu。独立metadata --locked --offline --filter-platform解析，Cargo.lock前后SHA不变。

- Cargo.lock88项中84项在当前非dev target闭包可达，包括项目本身、build/proc宏。4项不在闭包：r-efi5.3.0、wasip2 1.0.4+wasi-0.2.12、wit-bindgen0.57.1、zerocopy-derive0.8.56。这4项未缓存不再是本Linux构建的许可缺失依据。
- 83个registry包全部由完整.crate SHA匹配Cargo.lock认证，采得164份真实notice文本，含onig_sys/oniguruma/COPYING等嵌套原生组件许可。来源使用精确版本static.crates.io URL及archive/member/hash；本次已有本地缓存，无需网络获取。项目许可来自原delivery draft，不靠registry推断。
- 静态archive按完整artifact ID对应57个registry crate的实际dep-info；不是只看crate名字猜版本。项目bridge相对路径dep-info暂不归入该57，明确保留unassigned；不把它误称标准库。19个实际installed Rust rlib加1个libonig.a，共20条archive association逐成员重算SHA吻合：Oniguruma48个、compiler_builtins447个、其余18个rlib各1个真实对象。同名对象不同bytes会拒绝。
- 5份真实installed toolchain文件补入：Rust COPYRIGHT、Apache/MIT文本、rust-std-static cargo-vendor和COPYRIGHT-library.html，各保存SHA与RPM NEVRA/SOURCERPM。它们没有被当作完整Rust vendor条款审核的替代品。COPYRIGHT-library的外部依赖部分不提供足够完整的细项文本，保留阻断。

实际glibc loader --list模式不进入程序main，显式清除LD_*环境，精确解析8路径，ELF64/LSB/machine62全部吻合：ld-linux、libc、libm、libpng16、libz、libstdc++、libgcc_s，以及实际LLVM18路径的libomp。每个resolved文件SHA/架构/RPM所有者保存。i686候选全部排除；不私自捆绑任何系统库。Vulkan ICD等后续dlopen不在此startup闭包，仍是运行前提。

## 测试和限制

6/6 tests：dev排除/缺graph拒绝，loader未知/缺库拒绝，ELF32与64区分，真实tar内嵌native notice与hash损坏拒绝，路径穿越拒绝，真实ar成员+dep-info身份匹配与成员完整hash。最终实际采集exit0。没有Windows/macOS或其他Linux target支持声明。

这是把保守Cargo.lock清单收窄并补真实文本/对象身份；没有把archive成员误当最终链接保留section，也没有从一个RPM license expression推断可再分发。仍需审查Rust标准库供应商的精确外部依赖条款、最终链接闭包，以及目标发行环境/系统驱动运行前提。工具不提供任何把手填布尔值升级为distributable的入口。原归档不会因存在此补充JSON自动获批。
