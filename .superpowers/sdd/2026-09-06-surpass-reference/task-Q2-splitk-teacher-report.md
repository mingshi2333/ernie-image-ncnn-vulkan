# Q2：block15 下投影补偿候选的完整 teacher 诊断

## 当前状态

CPU 实现、独立编译与输入封存完成；GPU 尚未执行，等待前序图生图任务释放。局部 screen 的通过及历史 compile/invalid-layout 失败见 `task-Q2-splitk-screen-report.md`、提交 `4c35148`；root 的独立复核为 `4d1a2de`。这里不将 screen 结果推广为完整层、完整轨迹或正式质量通过。

## 唯一候选与完整分母

输入沿用已认证 official block14 hidden 与其余九个 conditioning tensor，原始 block15 oracle 不变。仅在固定 param 的 `Gemm gemm_6 1 1 87 88` 将类型替换为 `Q2SplitDown`，其余字节不变；原图 SHA `e8ed770153823bc60b2f27cfac14b9a58f534eb6bd9f19dc692d650f7faddd13`、原始完整权重流 SHA `0321f883d79cab6606ac3d2736344043625f3a0fed9bb1f5fcfdd33d1acd07da` 均强绑定。原始权重不改写、不复制到生产包。

完整输出分母为 4160×4096 = 17,039,360 个 FP32 元素；保存完整 blob87、候选 blob88 和 out0。候选不运行 text/head/scheduler/VAE。不得把这次单层 teacher 称为中文自由运行修复。

独立类继承固定 ncnn `Gemm` 的参数和二进制读入，严格拒绝非 K=12288、N=4096、M=4160、transB=1、alpha/beta=1、constantB=1、无 C、无 quantize 的配置。读出的 B[N,K] 完整转置后，再逐位比较 screen 已认证的全部 50,331,648 个 KN 元素，随后才上传。这样不会只依据模型文件大小猜测矩阵方向或忽略权重流位置。

算法与 screen v3 相同：32 段，每段 384 项；Neumaier FP32 累加并保留 FMA 乘积残差，保留两条 limb，再补偿合并。新增着色器相对 screen 仅增加行起点偏移，不改变每个 dot 的计算顺序。输入显式转换为 FP32 scalar packing，固定检查实际维度；权重使用 flat scalar 上传。

## 分行缓冲与有效性前置条件

每块最多 16 行，临时缓冲 16 MiB，总共 260 块。C++ 逐行 coverage 必须每行恰一次；每块调用 submit-and-wait/reset，保证所有 GPU 读结束后才复用临时缓冲。scalar input、weight、output、temp 的 VkMat 所有者覆盖整个循环。后续残差仍由原图执行。

结果分析前必须同时满足：

1. 完整 blob87 与原 teacher 的 blob87 字节一致，证明改变没有越过下投影边界。
2. 候选 blob88 在固定五行（0、22、4095、4096、4159），全部 20,480 个元素逐位复现 screen v3 的 official-side 输出。
3. 所有 17,039,360 个 out0 元素逐位等于原完整 blob75 + FP32(gate×候选 blob88)，检查残差、packing 和全覆盖。
4. 所有保存 tensor 大小正确、全有限；输入/来源/模型/工具的 bound SHA 在执行前后均不变。

即使以上成立，也仅报告原 teacher/候选对相同 official oracle 的完整 L2/max 及彼此差异，不自动应用正式 gate 或晋级。

## 已完成小验证与封存

独立编译两次 exit0，第一次成功开发入口保留为 `runner-before-weight-identity-check`，第二次增加完整 KN 权重身份检查；两份源码和编译日志均保留。没有改 root CMake 或重新构建生产目标。复用此前冻结的 ncnn 静态库及五个项目数学 wrapper，未从并行 live source 重拷数值代码。

`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p 'test_splitk_*.py'`：6/6 通过。覆盖完整/尾块逐行分母、非法行计划、未知图拒绝、缺失末行拒绝以及先前 FP64 指标分母测试。这些是小 CPU 合同测试，不是 GPU 布局或数值结果。

执行目录：`outputs/q2-block15-splitk-teacher-v1`。
runner SHA：`46237c14f25bfd962d07a7dbc45c5dd3dc71c67242a3fd24126e91e78c37b915`。
plan SHA：`6bdfa3d9a4d117afa621152e01cf48e4e576bd93dd559306ffea31c7a0805369`。
worker 独立身份保存在 `execution/execution-identity.json`，运行入口为封存 `execution/worker.py`，其导入工具也来自该目录。

资源限制：固定 CPU 0、2 两物理核；保留 baseline GPU-block 的内部 num_threads=4，但实际亲和性只有两核，worker 元数据明确二者区别。递归进程树采样 RSS 9 GiB、整 GPU 6144 MiB、host available 至少 3 GiB、2400 s timeout；执行前与过程中均检查资源。GPU 排队期间没有启动候选。
