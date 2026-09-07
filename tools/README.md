# 转换与验证工具入口

原生推理不运行本目录。下载、转换和官方参考使用项目 `.venv` 及 `requirements-reference.lock`；完整性检查和独立打包只使用标准库。所有大模型实验串行执行，使用新的输出目录保留失败记录。

| 要完成的工作 | 入口 | 说明 |
|---|---|---|
| 准备基础官方组件 | `fetch_pipeline_components.py`、`fetch_tokenizer.py` | 固定 revision，按组件下载；完整流程见 [复现说明](../docs/REPRODUCE-PIPELINE.md) |
| 新尺寸或更长文本 | `prepare_variant.py` | 独立导出目标图、复用受散列约束的权重、验证文本/DiT/head/VAE 后组包 |
| 把本地链接包变成独立包 | `package_model.py` | 复制全部 136 个实际运行文件；`--link` 明确保留开发链接 |
| 共享已审查实例的权重 | `package_dynamic_model.py` | schema3 CAS 对象与完整源包身份；支持原生实例选择及已登记的空间目标（1024/s64 源可生成 1376×768/s64），encoder 按尺寸单独认证 |
| 准备可选 PE | `fetch_pe_components.py` → `export_pe_block.py` → `validate_pe_block.py` → `build_pe_model.py` | 独立验证真实 PE block 和缓存，再构建完整 26 层包 |
| 官方 PE 与原生 PE 对照 | `reference_pe.py` → `validate_pe.py` | 参考进程先退出，再启动原生端；检查每个 greedy token 和全部 logits |
| 文本对照 | `validate_text.py` | 原生文本路径对官方真实权重，支持 UTF-8 prompt 文件 |
| 完整文生图对照 | `validate_pipeline.py` | 相同保存的初始 latent，固定张量/PNG 数值门槛，可连接官方 greedy PE oracle；共享包绑定原源配置与实际空间目标，并使用已审查的完整参考 |
| 官方图生图 suffix 参考 | `reference_img2img_positive.py` | 认证独立官方 encoder、保存噪声和原始 schedule suffix；当前仅固定 512×384 开发案例，不替代完整文生图 oracle |
| 测量一轮原生生成 | `benchmark_pipeline.py` | 支持固定/共享包及运行时尺寸，核对原生完成记录、实际文本桶与内存设置；trace 默认关闭，保留独立计时和设备采样范围；不是质量验收 |
| 核对固定对照项目权重 | `audit_port_weights.py`、`audit_port_relations.py` | 分块规范化实际权重、按固定图连接核对命名角色，并单独报告派生常量与数学差异；不自动证明全图等价或速度 |
| 归档通过及失败证据 | `collect_parity_evidence.py` | 重新计算误差、核对散列，冻结小型报告；不改原始门槛 |

`export_*` 导出独立官方图与 fixture；`build_*` 构建真实权重包；`validate_*` 判定误差；`diagnose_*` 定位已发现的差异；`collect_*` 固定证据。底层助手如 `prepare_block.py`、`rebucket_*.py`、`specialize_vae.py` 由上述入口组合使用。

`pipeline_package.py` 统一旧包和共享包的配置/实例绑定；`pipeline_reference.py` 限定完整边界分母与版本控制中的参考来源。共享包的验证器和证据收集器都核对真实 CAS 对象及源 manifest。新的官方参考应先完成独立执行和来源审查，不能仅写入一组彼此匹配的散列。诊断用 teacher forcing、saved embeddings 和图生图 suffix 各自保留不同的范围，不进入完整文生图通过数。

运行时尺寸的官方参考可通过 `validate_pipeline.reference(..., runtime_size=(width, height))` 从已认证的原始 schema-2 包执行，不修改源包或创建伪造的固定尺寸包。调用方仍需禁用梯度并限制线程及资源。旧来源若缺少官方权重散列，可显式传入 `source_weights_package`；该包也必须命中固定来源散列，且全部非图运行文件与原包同一，才能借用其权重来源记录。参考 fixture 同时保存原配置、目标配置和两个来源身份；原生用户入口不调用此 Python 参考。

现有脚本保持可直接执行及原来的同目录导入，以兼容历史运行快照。新功能优先扩展任务入口，不增加另一套并行命名或隐式下载行为。
