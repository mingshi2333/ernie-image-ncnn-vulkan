# PE development batch: 12/12 independently verified

本目录封存已经执行完成的 CPU FP32 greedy PE 开发批次，来源 `outputs/pe-development-v1`。此次只读审查未加载模型、未重新执行推理、未构建或使用GPU。原始journal、contract、snapshot保持原字节；contract中的pending是运行前冻结状态，实际执行结果由journal和本次audit给出。

独立审查重新验证58份Python源码快照、固定runner以及每case运行的runner副本、validator/reference脚本身份、原生模型manifest及其全部60个文件、28个官方权重组件、官方Ministral3源码hash和tokenizer来源文件。逐case复核合同prompt/尺寸/输出预算、canonical输入IDhash和计数、退出码/状态/时钟/日志hash、reference/result之间的身份链接，以及所有原始logits/IDs/text/status文件的SHA。模型文件仅用1MiB流式hash读取，无模型对象。

全部1742步的131072维FP32 logits已逐对重新计算，shape/finite/两侧argmax与实际生成ID均通过。固定判定为NRMSE≤2e-4，最大绝对误差≤2e-4+2e-4×max(abs(reference))，每步重算值与原result精确一致。输入IDs、生成IDs、增强文本保持逐字节一致。这里的最大绝对误差不是单独的2e-4门槛；同时应用上述固定绝对+相对门槛。

| case | input tokens | actual output tokens | EOS | worst step NRMSE | worst step max abs |
|---|---:|---:|---|---:|---:|
| pe-dev-00-en | 139 | 64 | False | 3.344549542e-06 | 2.819299698e-05 |
| pe-dev-01-zh | 153 | 96 | False | 4.853223696e-06 | 4.291534424e-05 |
| pe-dev-02-ja | 154 | 96 | False | 7.922753374e-06 | 5.620718002e-05 |
| pe-dev-03-quotes | 147 | 128 | False | 1.227920175e-05 | 0.0001170635223 |
| pe-dev-04-newline | 141 | 128 | False | 5.140216657e-06 | 4.851818085e-05 |
| pe-dev-05-whitespace | 142 | 64 | False | 4.137661906e-06 | 6.103515625e-05 |
| pe-dev-06-empty | 131 | 32 | False | 3.76429284e-06 | 5.054473877e-05 |
| pe-dev-07-control | 150 | 64 | False | 5.070499551e-06 | 6.115436554e-05 |
| pe-dev-08-wide | 141 | 256 | False | 9.626743867e-06 | 7.200241089e-05 |
| pe-dev-09-portrait | 141 | 256 | False | 5.795360328e-06 | 5.531311035e-05 |
| pe-dev-10-long-output | 145 | 526 | True | 5.953702827e-06 | 4.839897156e-05 |
| pe-dev-11-near-capacity | 2048 | 32 | False | 5.681658142e-06 | 5.102157593e-05 |

实际最长输出是**526 token且EOS**；近容量样本实际**输入2048 token、输出32 token**，并没有执行2048-token输出或4096-token满cache。标称容量不能当作已测容量。本批只验证greedy；随机采样同seed不是跨实现数学oracle。结果属于开发质量覆盖，不是正式72-case图像质量或性能胜出。

每case精简JSON保留完整原始文件hash/size清单及最大指标；batch-files.json覆盖顶层driver/bootstrap/log与58份源码快照。原始大logits和runner保留于outputs，不加入Git。audit.json记录独立复核范围与边界。review.py可从项目worktree用项目.venv执行，参数为原批次目录和**新的输出目录**，读取证据并重新验证；它拒绝覆盖现有输出。版本封存不改变原始artifact。
