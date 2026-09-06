# 固定1376×768 CPU组件实际对照

固定packed H48/W86、text bucket64、4192个tokens。两个CPU head共9个输出全部通过原FP32门槛，17,727,488个输出元素全部有限且逐项复算；输入head主输出NRMSE2.24885847e-7，其余conditioning输出最大NRMSE5.83469769e-7；输出head NRMSE1.30240545e-6。既有门槛NRMSE2e-5、最大差2e-4+2e-4×reference最大绝对值保持不变。

此前同尺寸VAE完整3170304元素已经通过：NRMSE9.3498124e-7，最大差4.76837158e-6。原生198481196字节bin不变，只特化两个已审reshape；见原VAE结果及root复算记录。

本轮四个CPU阶段串行执行：官方input、原生input、官方output、原生output，全部exit0，MemoryMax16GiB、swap0、CPU12/14两核预算，官方Torch两线程、原生ncnn四线程。3152个绑定包含307个冻结项目文件、已认证2822项runtime身份、六组官方head权重及实际head图/bin；各阶段前后完整验证。组件wrapper执行时与官方输出最大差严格0；第二份wrapper数组未单独保存。这里不声称新的实时loader枚举或不同机器复现已经完成。

使用现有export_dit_heads的reference-only路径，不执行pnnx。只给原工具增加可选official-root/reference-only/threads入口，并在仅input时跳过未请求的后续output计算；默认all转换入口兼容。三个小CPU合同通过。

这些是CPU组件对照；尚未通过4192-token的36个DiT block串联、Vulkan完整轨迹或1376最终图像。生产registry保持原范围，正式72例和15例没有运行。
