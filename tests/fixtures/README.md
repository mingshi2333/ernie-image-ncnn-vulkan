# Fixture 来源

- `text-s64.ncnn.param`：本项目对固定官方文本 block 0 的独立 64-token pnnx 导出。
- `text-s2048.ncnn.param`：本项目独立 2048-token 导出，完整图规范化后的散列必须与 32/64-token 图相同；没有用字符串替换代替导出。
- `pe-chat-template.jinja`：固定官方 ERNIE-Image-Turbo revision `bc68c81e2a1730a394d5fc9fae70713dee940140` 的 [PE 模板](https://huggingface.co/baidu/ERNIE-Image-Turbo/blob/bc68c81e2a1730a394d5fc9fae70713dee940140/pe_tokenizer/chat_template.jinja)，逐字节保存用于无权重包校验测试。SHA256 为 `0c859484eecf01db103acd02c332610163ee425cd46866d6cb126ee1bee974ea`，保留官方模型的来源及许可。

这里不存神经网络权重、完整 tokenizer 词表或生成图片。真实权重及输出留在本地忽略目录，对应验收摘要和来源清单进入 `artifacts/`。
