# Windows 构建与验证范围

当前完成的是 Linux 上的 MinGW x64 交叉构建，以及 Windows 程序在 Wine 下的 CPU 检查和迁移安装验证。原生 Windows/MSVC 构建、Windows Vulkan 驱动和完整模型出图仍待实际验证；默认支持平台仍为 Linux。完整命令、失败记录和二进制身份见[本次记录](../artifacts/2026-09-07/windows-cpu-portability/README.md)。

## 编译器和 Rust target

C++、Rust 和图像库必须使用同一目标架构与兼容的运行库。本次实测为 MinGW GCC 16.1.1、Rust 1.98.0 的 `x86_64-pc-windows-gnu` target、libpng 1.6.58、zlib 1.3.2，运行环境为 Wine 11.0。Rust 工具链通过明确的 Cargo/rustc 路径选择，没有更改默认 Rust 工具链。该构建没有检测到 OpenMP，也没有启用 Vulkan，因此不提供 Windows CPU 性能或 GPU 支持结论。

交叉构建必须设置 `ERNIE_RUST_TARGET`。省略时 CMake 直接拒绝配置，避免把主机的 Rust 静态库链接进目标程序。MinGW x64 使用 `x86_64-pc-windows-gnu`；MSVC x64 对应 `x86_64-pc-windows-msvc`，后者仍需实际构建验证。原生构建使用 Cargo 默认的主机目标时可保持空值；如果通过 Cargo 配置选择其他 target，应同时明确设置本项目的 target，保证输出目录和 Rust/C++ 目标一致。[Cargo target 配置](https://doc.rust-lang.org/cargo/reference/config.html#buildtarget)

例如，已经配置 Windows C++ 工具链、目标 Rust 标准库和目标 libpng 后，可使用以下 CMake 参数；`WINDOWS_TOOLCHAIN.cmake` 由使用者的依赖环境提供：

```sh
cmake -S . -B build/windows-cpu \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/WINDOWS_TOOLCHAIN.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DERNIE_RUST_TARGET=x86_64-pc-windows-gnu \
  -DERNIE_ENABLE_VULKAN=OFF \
  -DERNIE_BUILD_TOKENIZER=ON -DERNIE_BUILD_GENERATOR=ON \
  -DERNIE_INSTALL_SDK=ON \
  -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF
cmake --build build/windows-cpu --parallel 2
cmake --install build/windows-cpu --prefix /new/windows-install
```

若同时安装了多个 Cargo/rustc，使用 `CARGO_EXECUTABLE` 和 `RUSTC` 明确选择；交叉链接器可通过 Cargo 对应的 `CARGO_TARGET_<TRIPLE>_LINKER` 配置。这里的示例不安装依赖，也不替代原生 Windows 验收。

Rust 静态库在 MinGW 下为 `.a`、在 MSVC 下为 `.lib`；CMake 现在根据目标平台命名，并在显式 target 时从 `cargo/<target>/release` 读取。Windows 系统库依赖通过链接接口保留到安装后的 SDK。切换 CRT 时，Rust 和 C/C++ 必须保持一致；可用 `--print=native-static-libs` 核对所需系统库。[Rust 静态库和 CRT 说明](https://doc.rust-lang.org/reference/linkage.html)

## 路径、安装与测试

公共 C++ API 的文本和路径字符串使用 UTF-8。Windows CLI 的独立入口 `cli/windows_main.cpp` 接收 UTF-16 参数并显式转换为 UTF-8；文件操作在边界转换为原生路径，ncnn 权重读取调用宽字符重载。模型、PE、提示词、输入/输出图像、原始张量和报告路径遵守同一约定，不依赖系统 ANSI 代码页。[Microsoft 编码边界说明](https://learn.microsoft.com/en-us/windows/apps/design/globalizing/use-utf8-code-page)

本次小型检查覆盖中文、俄文、空格和非 BMP 字符，使用独立创建的原生文件路径，避免仅因读写双方使用同一种错误编码而误判通过。图片检查包含实际 PNG/BMP/TGA 无损往返和 JPEG 编解码；组件检查实际加载并执行小型 ncnn 网络。它们不等于完整 ERNIE 模型推理。

安装测试移动带中文和空格的前缀，隐藏原仓库和构建目录、关闭网络，再以安装后的 `find_package(Ernie CONFIG REQUIRED)` 构建外部 C++ 程序。CLI 的 `--help`、CPU `--diagnose`、中文/emoji 模型路径的损坏包拒绝、Unicode 参数和提示词文件检查均有实际记录。测试使用独立 Wine 前缀；Windows 原生安装仍待验证。

源码构建产物需要相应的 C++、线程和 libpng/zlib 运行库。当前 MinGW/Wine 诊断显式核对并复制了所用 DLL，清单保留在证据中；尚未提供可公开分发的 Windows 归档。安装 SDK 不依赖 PNG/JPEG，CLI 使用图像库。

Wine 的 Windows 文件接口可能不暴露 Linux 创建的 POSIX 符号链接。本次原始三项跨系统链接检查失败保持原样；单独记录 Windows 接口创建的链接与包校验行为。不能通过跳过失败来宣称 Windows 的全部文件系统语义已经验收。
