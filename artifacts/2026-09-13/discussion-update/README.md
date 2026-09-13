# ncnn Discussion 内容更新

2026 年 9 月 13 日按用户“更新 discuss 文档”的要求，同步项目文档和已有的 [Discussion #6985](https://github.com/Tencent/ncnn/discussions/6985)。更新于 `2026-09-13T16:04:05Z`，修改原帖正文，标题、作者、分类和三张演示图保持原样。

文档提交 [`186c52e`](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/commit/186c52e29bd391e9b8207fbbbf6fcde50f239b2a)先推送到 `codex/surpass-reference`，再更新原帖。内容包括：

- DiT 单行 padding mask 的原理、四条 shader 适配和占用表，注明 #6998 的参考来源。
- 64×64 完整 CPU 回归的 25/25 张量、PNG MAE 0.024495443 和最大通道差 1。
- 256-token 图的完整 25 层文本结果，以及 PE chunk16 的完整 26 层 token/logits 结果，两者保持候选状态。
- 下载后原生模型校验、设计索引，以及最新框架 CI 的 244 通过、40 能力跳过、0 失败和 130 项 HTTP/清单检查。

正文保留项目创建日期 2026 年 9 月 5 日、腾讯犀牛鸟第三阶段标注、历史数值结果和精度说明。存储节省、数值对照与性能结论分别陈述，不用本轮数据声称速度或画质提升。

发布前重新核对原帖未被改动，确认作者和可更新权限。保存一次更新请求及响应，之后独立读取 GitHub 正文，确认与准备内容完全一致；渲染 HTML 包含三张原图和六个表格。未登录 HTTP 访问返回 200，也能看到新日期、mask 表和最新 CI。

详细回执与散列见 [receipt.json](receipt.json)，原文与新正文分别保存在 [before-body.md](before-body.md) 和 [body.md](body.md)。[readback.json](readback.json)为独立回读，[anonymous-check.json](anonymous-check.json)为匿名访问检查。[本轮实现与实验](../design-loop/README.md)保留原始模型结果，文档更新没有重跑或替换它们。
