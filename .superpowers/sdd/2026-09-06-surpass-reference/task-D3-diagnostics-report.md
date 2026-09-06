# D3 无驱动诊断修复

已修复实际离线检查发现的程序行为：缺失 Vulkan driver 时，`diagnose()` 现在保留设备诊断和模型配置读取能力，返回无设备及原因；要求 Vulkan 生成的请求仍失败。可选 context 初始化失败先清理，不计入 active users，诊断跳过 ncnn 的自动创建查询。API 只追加标准库字段，现有源码可重编译，未声明 ABI 稳定。

实际 `/tmp/ernie-offline-img2img-1024-v2` 于 18.546 秒停止，未执行生成。完整日志归档于 `artifacts/2026-09-06/offline-linux-diagnostics/prior-offline-failure`。新版与旧 D1 v2 所有模型数学文件逐份相同；新源码 288 项，8 个明确覆盖，identity `3a112e7b…`。

全新 Clang CPU/Vulkan 都构建成功。CPU 五项和 Vulkan 六项受影响 CTest 全部实际通过，无 skip；涵盖正常/缺驱动诊断、严格推理拒绝、重复初始化失败后的状态、嵌套/并发/借用 context、CLI 元数据读取及外部安装消费者。Vulkan 又单独在完整源码根隐藏且断网下验证移动安装前缀，75 项文件绑定，安装 binary `d366c835…`。独立代码/安装审查 `55e58ef / 5de7279` 无剩余阻断。

新本地 archive `d5810af1…` 从该安装前缀构成，仍为 local_review_draft / distributable=false。首次以 worktree 的 outputs 符号链接作输出被打包器安全检查拒绝，在创建目录前停止；改用同一 outputs 的真实绝对路径后成功，未改工具约束。新的离线 case 只替换 archive/inventory 身份，checker、模型、prompt、PNG、noise、strength=.5 和原 native PNG 逐字节门槛完全不变。

当前 `/tmp/ernie-offline-img2img-1024-v3` 正在专属 10GiB/swap0 scope 执行；无驱动诊断已实际返回所需字段，正在完整图生图。最终生成结果另行记录。旧失败、未完成的其他 D3 场景、正式质量/性能/平台门槛均保留。
