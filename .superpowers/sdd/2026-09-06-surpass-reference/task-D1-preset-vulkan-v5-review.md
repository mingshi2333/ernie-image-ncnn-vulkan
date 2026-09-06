# D1 Linux preset Vulkan v5 独立复核

范围：主仓库只读证据 `outputs/d1-preset-source-v5`、`d1-preset-vulkan-v5`、`d1-preset-vulkan-test-v5` 及两个同名 worker。没有重新编译、运行测试或使用GPU。

## 结论

无 Critical、Important 或 Minor。证据支持：同一冻结源码先完成 Linux CPU preset 27/27，再完成 Linux Vulkan configure/build 和完整 CTest 42/42；42是Vulkan构建的总测试分母，不应写成“27个Vulkan测试”。

## 身份与不可变性

- `source-identity.json` SHA为 `add223535f92c501c986b6bc3e6520ec746580a776bc7e143836ec9dd253d199`；独立重算其267个文件，0缺失、0散列差异。
- Vulkan build identity SHA为 `31c851397b9d71dd889e28f958c5c012316148c253685cb26ab798cacfccae74`，其source identity、worker SHA `771e5ea7...`、ncnn revision `6a1bf000...` 与实际文件/仓库记录一致。configure exit0 6.36秒、build exit0 281.29秒；结束后source changes和ncnn tracked changes均为空。
- test identity准确引用上述source/build identity，test worker SHA `1e35c853...` 与实际文件一致。独立重算其33个 `ernie-*` binary，0差异；test结束后的source/binary changes均为空。
- 复制保存的 `LastTest.log` 与build树最终日志逐字节相同，SHA `3ae5d2284f967beb19775168488fccb4320f2d83f37daae3e545c98049e73e24`。

## 测试分母与实际Vulkan

- 独立读取 `test.log`：恰好42个 `Passed`，结尾为 `100% tests passed, 0 tests failed out of 42`，wall 15.76秒；没有 `Skipped`、`Not Run` 或skip文本。
- 单独CPU preset证据为27/27、0失败，wall 2.13秒。因此正确表述为CPU preset 27/27及Vulkan preset完整42/42。
- `LastTest.log`包含真实 `NVIDIA GeForce RTX 4060 Laptop GPU`设备枚举及Vulkan调用结果；明确执行了native/runtime cache Vulkan、residual Vulkan FP32/BF16，以及GELU/RMSNorm/LayerNorm Vulkan FP32/FP16/BF16等测试。日志中的数值输出和命令参数表明确实进入Vulkan分支，不是仅靠测试名称或无设备skip判定。

复核没有扩大为安装归档内容审查、完整模型推理或跨平台证明；这里只认证本次冻结Linux CPU/Vulkan preset源码、二进制与测试运行的绑定关系。
