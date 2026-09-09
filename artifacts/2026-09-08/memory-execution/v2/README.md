# 后续修正与最终验证

池复用修正按实际 VkDeviceMemory/VkBuffer 记录活动区间，只在连续空闲范围足够时绕过新增 backing memory 的预算检查。失败清理同步的任何非成功状态均 fatal，包括 host/device OOM；预取线程错误也计入 skipped。

本机新官方校验层1.4.357下全量61/61通过；随后局部预取失败计数修正的6项受影响测试通过，均无跳过或validation错误。失败清理注入先真正完成GPU工作，再返回受控错误；没有故意耗尽或破坏设备。

first-ci保存f9dbcf2的全部五个原始CI artifacts与作业身份：Windows37pass24skip，macOS57pass4skip，两个LinuxCPU各37pass；LinuxVulkan38pass19validation失败4BF16skip。它的旧1.3.275校验层无法识别支持的subgroup rotate结构。工作流固定官方LinuxSDK1.4.357.1的校验层/诊断工具及SHA256，系统loader/Mesa不变。sdk-*为实际下载、散列、原始官方发布条目及静态检查证据，sdk-info为本机实际枚举。

最终源码的远程CI与完整512模型正常/混合/BF16结果尚待执行，后续追加，不改写首轮失败和全RAM部分结果。恢复仅覆盖保留错误类型的buffer/command等路径；ncnn创建compute pipeline若将OOM折成通用-1仍会直接结束。
