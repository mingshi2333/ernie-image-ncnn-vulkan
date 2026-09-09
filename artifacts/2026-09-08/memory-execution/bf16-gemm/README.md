# BF16 Gemm cooperative 兼容修正

完整 v2 BF16 模型虽然完成，仍因 Gemm shader 使用设备未支持的 BF16 accumulator 类型而产生 5 条 VUID。`cmake/ErnieNcnnBf16.cmake` 分别认证 SDPA/Gemm 的原始 C++ 与 shader，仅让 BF16 storage 使用普通 Gemm，保留原生 BF16；FP32、FP16、INT8 的既有选择不变。派生文件在 ncnn target 内编译，原 submodule 保持干净。代价是暂时失去 BF16 Gemm 的 cooperative 加速。

新 `bf16_gemm_vulkan` 测试涵盖 4 种矩阵形状、常量 B 的转置、无偏置/向量偏置/矩阵偏置、输出转置和 pack1/4。每例分别请求 cooperative ON/OFF，强制检查原生 BF16 输入/输出，逐项精确对照独立 FP64 累加后 BF16 RNE 舍入，不使用可调误差容忍。无设备或不具备原生 BF16 storage 才跳过。

- 首版测试把 alpha/beta 写成 `0=1 1=1`，被 ncnn ParamDict 按 int 位模式读取，造成数值 0。`baseline-*`、`fixed-*` 和 `initial-test.cpp` 保留这次测试夹具错误；不能归因于生产模型。修正为浮点字面量，没有改变数据或数值门槛。
- 修正后的同一夹具链接经过认证的原始 Gemm 编译单元：4 例数值均精确通过，退出 0，却有 **8 条 VUID 匹配日志**，即 4 次错误及其规范引用。因此数值通过不足以证明 Vulkan 使用有效。命令、来源与日志见 `original-gemm.json`、`original-gemm-run.log`。
- 新兼容路径的本机全套 **62/62 CTest 通过、0 跳过、0 validation/VUID**。4 例 cooperative ON/OFF 均与独立参考精确相同。见 `final-ctest.xml`、`final-last-test.log`、`summary.json`。
- 13 项独立 CMake/编译契约检查全部通过，包含 CRLF、源码/shader 改动拒绝、编译单元缺失/重复拒绝、CPU 绕过、SDPA 派生字节保持不变及实际编译。见 `source-guards.json` 与 `guard-*`。

环境为 Linux / RTX 4060 Laptop，隔离加载官方 SDK 1.4.357.1 的校验层。Gemm 派生 SHA256 为 `dff4dc2ef0b7ec9df1769d51c144c50ed3d0d5e34593eb40ead01c3fce75a95e`。这些是小型算子验证；修正后的完整 BF16 结果须使用独立冻结的 v3 模型运行记录，不能直接沿用 v2 的无效结果。
