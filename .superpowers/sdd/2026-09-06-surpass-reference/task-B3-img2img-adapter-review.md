# B3 img2img benchmark adapter 独立审查

审查 b2053c3，只读实现与固定 performance-5 合同，CPU 12/14 运行 test_port_metrics.py：19/19 passed。未运行模型或 GPU。当前结论：有待修复 Important，不能以现状认定真实冻结 img2img case 已可正式执行。

1. 冻结 `outputs/port-corpus-v1/manifest.json` 中实际 input_image_path 是 `performance-5/input-image`，没有扩展名。adapter 以原 suffix 构造 snapshot，得到 `input`；CLI `extension()` 只接受 .png/.jpg/.jpeg/.bmp/.tga，实际请求在读取图像前拒绝。测试只用 input.png，不能检出。应从认证的实际格式选受支持 snapshot 后缀，保持原始文件字节与 SHA。
2. `Image.convert('RGB')` 忽略 alpha，而 CLI 默认对白背景按 `(channel*alpha+255*(255-alpha)+127)/255` 合成。透明红 PNG 在 adapter 被记为红，在原生为白；如果期望 SHA 同样来自 convert，formal_comparison_eligible 会错误为真。必须使用经核对的真实解码身份，或者让未证实一致的格式仅发展运行且正式不合格，不能将 Pillow 结果冒充 native RGB。
3. adapter 只记录 `resize_policy={mode:stretch}`，冻结 performance-5 完整记录 `antialias:false, coordinate_transform:half_pixel, filter:bilinear, width:1024,height:1024,mode:stretch`。port_metrics 严格逐字段比较整个 dict，因此原样接入一定被拒绝。实际 C++ stretch 确为 half-pixel/bilinear/noAA、边界 clamp、uint8 round，应该记录完整算法契约。

其余边界：实际 performance-5 strength=.5；本性能切片排除0不缺正式六项的需求。已复制 runner/prompt/noise/image 并让 argv 消费相应 snapshot，输入路径没有继续指回原文件。错误或缺失 expected SHA 允许发展运行且 formal=false 是可接受策略，但需负例测试。`source_files_sha256` 是 ROOT 当前源码观察清单，不能单独证明所传 runner 的构建来源或完整冻结执行依赖；父级执行计划仍须完成 binary/source 绑定。`quality_validated=false` 保持诚实，formal_eligible 不是 summarize_pairs 的质量/权重/身份验收。未取得模型性能结果。

三项发现已直接发作者 paired_metrics 和 root，待修复提交后补复核闭环。

## 修复闭环：8e86a8e + e242de5

结论更新：本 adapter 切片无剩余 Important。21/21 小测试独立重跑通过，未跑模型。原三项以及第二轮发现的“先读取原图再复制导致解码身份与实际 snapshot 分离”均已修复：先复制受控 input.snapshot，再检测该文件/计算 RGB 身份，最后 rename 到真实容器匹配后缀；正式解码认证严格限 PNG IHDR 8-bit truecolor、无 transparency 且无 gAMA/cHRM/iCCP/sRGB 转换元数据，其他格式发展可运行但不宣称正式 RGB 身份。完整缩放合同与冻结 stretch 字段一致。透明 PNG 负例证明即使用户提供 Pillow 丢 alpha 的匹配期望，仍 formal=false。

本审查直接对实际 `performance-5/input-image` 执行只读 inspect：suffix=.png；decoded SHA `0ce2b51640b9c95f19617f03eabf40c3f0368589cc1ee1190b70966165ac184f`，与冻结合同逐字相同。实际 PNG 无被拒绝的颜色转换/透明元数据，因此保守限制不会意外阻止本性能 fixture。source/runner 构建来源和完整 formal 质量/权重证明仍遵守前述父进程边界，不能把这次工具闭环当实际性能结果。
