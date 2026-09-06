# B3 img2img benchmark adapter 独立审查

审查 b2053c3，只读实现与固定 performance-5 合同，CPU 12/14 运行 test_port_metrics.py：19/19 passed。未运行模型或 GPU。当前结论：有待修复 Important，不能以现状认定真实冻结 img2img case 已可正式执行。

1. 冻结 `outputs/port-corpus-v1/manifest.json` 中实际 input_image_path 是 `performance-5/input-image`，没有扩展名。adapter 以原 suffix 构造 snapshot，得到 `input`；CLI `extension()` 只接受 .png/.jpg/.jpeg/.bmp/.tga，实际请求在读取图像前拒绝。测试只用 input.png，不能检出。应从认证的实际格式选受支持 snapshot 后缀，保持原始文件字节与 SHA。
2. `Image.convert('RGB')` 忽略 alpha，而 CLI 默认对白背景按 `(channel*alpha+255*(255-alpha)+127)/255` 合成。透明红 PNG 在 adapter 被记为红，在原生为白；如果期望 SHA 同样来自 convert，formal_comparison_eligible 会错误为真。必须使用经核对的真实解码身份，或者让未证实一致的格式仅发展运行且正式不合格，不能将 Pillow 结果冒充 native RGB。
3. adapter 只记录 `resize_policy={mode:stretch}`，冻结 performance-5 完整记录 `antialias:false, coordinate_transform:half_pixel, filter:bilinear, width:1024,height:1024,mode:stretch`。port_metrics 严格逐字段比较整个 dict，因此原样接入一定被拒绝。实际 C++ stretch 确为 half-pixel/bilinear/noAA、边界 clamp、uint8 round，应该记录完整算法契约。

其余边界：实际 performance-5 strength=.5；本性能切片排除0不缺正式六项的需求。已复制 runner/prompt/noise/image 并让 argv 消费相应 snapshot，输入路径没有继续指回原文件。错误或缺失 expected SHA 允许发展运行且 formal=false 是可接受策略，但需负例测试。`source_files_sha256` 是 ROOT 当前源码观察清单，不能单独证明所传 runner 的构建来源或完整冻结执行依赖；父级执行计划仍须完成 binary/source 绑定。`quality_validated=false` 保持诚实，formal_eligible 不是 summarize_pairs 的质量/权重/身份验收。未取得模型性能结果。

三项发现已直接发作者 paired_metrics 和 root，待修复提交后补复核闭环。
