# Q2：复用 native81 的官方剩余实验，专属无swap scope CPU准备

状态：**仅CPU准备，未启动GPU或官方模型**。保留0c9247d记录的原swap停止及v2全部文件。新尝试不会重跑native，直接使用经完整SHA/finite/分母验证的native81/out0独立副本。

## 数值合同保持不变

唯一剩余工作为官方baseline完整block一次，若81不同，再调用官方MLP一次。baseline的out0/75/87/88必须全部逐位复现先前已认证官方oracle，否则不接受81、不运行matched支路。81相同则省去MLP。输入布局、全部conditioning、权重、FP32/backend/TF32设置及完整分母均保持原合同。

`diagnose_mlp81_official_resume.py`与原封存官方payload的唯一差异是结果计数：native_full_block由1改为0，并添加native_replay_reused=true；数学、hooks、旧四SHA前置及matched替换点没有变化。复用native81 SHA `da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54`、nativeout0 SHA `175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3`，各完整17,039,360元素。复用来源是资源中止尝试中已经完成且独立认证的native阶段，不将整个旧尝试改写为成功。

## 新scope入口及硬限制

最终新输出为`outputs/q2-block15-mlp81-official-v5`，新scope名`ernie-q2-mlp81-official-v5.scope`。外层launcher先固定SHA验证guard和plan，再用systemd用户scope启动guard：

- `MemoryMax=10G`，实际memory.max必须恰为10,737,418,240 bytes。
- `MemorySwapMax=0`，实际memory.swap.max必须0，scope当前swap必须0。
- `CPUQuota=200%`，实际cpu.max必须quota/period恰为2。
- taskset affinity必须恰为CPU0/2；每PID采样同时检查亲和性。仍传AllowedCPUs=0,2；环境未暴露该scope的cpuset.cpus.effective，**不虚称已证明独立cpuset控制器限制**，实际亲和性与cpu.max均验证。
- 保留更严格的旧递归RSS上限9GiB、整GPU6144MiB、host available至少3GiB、2400s、0.5s采样。未提高原门槛。

在任何模型子进程前，guard读取自己实际/proc cgroup路径，要求精确专属scope后缀，写preflight-cgroup.json，然后检查实际控制器值、亲和性、host/RSS/GPU。元数据只是声明，实际cgroup文件才决定是否可启动。所有身份文件再次认证后才创建模型进程。

持续遥测包含scope内所有PID的Name、VmRSS、VmSwap、实际cgroup、affinity，同时递归查询后代，拒绝逃逸其他cgroup的后代；保存scope memory.current、memory.swap.current、memory.events、memory.swap.events、cpu.max、memory.max/swap.max以及host可用量与整GPU内存。当前样本先写盘，再验证持续限制，因此越界样本也可追溯。OOM/max事件增长、swap、RSS/GPU/host/timeout越界及监测失败都停止；错误写入worker-result或启动日志，不吞掉失败。

launcher用独占创建launch-record.json阻止重复启动；guard也拒绝已有attempt.json。scope启动前user bus失败会写scope-launch.log/exit，不允许隐式fallback到非隔离运行。user bus由当前uid的`/run/user/<uid>/bus`显式指定，避免原工具进程未提供环境时跳过隔离。

## 真实CPU探针及保留的准备历史

首次CPU探针因DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR缺失而在创建scope前失败，没有模型/GPU；记录保留。验证uid1000及实际/run/user/1000/bus后，明确环境的CPU探针成功。

最终探针实际观测：scope `/user.slice/user-1000.slice/user@1000.service/app.slice/ernie-q2-mlp81-cpu-probe-v4c.scope`，memory.max=10737418240、memory.swap.max=0、cpu.max=`200000 100000`、memory.swap.current=0，PID2102388 VmSwap0、affinity[0,2]、RSS15,503,360 bytes，host available15,819,386,880 bytes，OOM/max事件0。这是小CPU探针，**不是官方模型资源验收**。

v4准备未运行模型；之后补充“初始cgroup读值在检查失败前落盘”，最终封存v5。v4源/plan/探针保留，不回写旧SHA。该变化不修改控制器检查逻辑或数学；实际scope探针的控制器证据仍适用，v5执行时仍必须重新读取并通过本次实际scope检查。

## 最终身份与审批入口

- launcher SHA `01002b79ddd0f319b1ef6fef4f3aa5049454c31306e3d1d72248646cf58dcf6f`
- plan SHA `38f2f50d52cf7c67e942ae7753509f1d91a5128275fdb531bbbed43bd9f5e5f9`
- guard SHA `7f1b77a0658d3db904734b4cca1620aa6d83666cf871af93582373a2d4ec347e`
- official-only payload SHA `d370f120313c36410d805820f9c3ced05c5b5532295710d9c2ce3308080ca116`
- 5905个bound文件CPU重验通过，包含旧完整证据/实际权重/执行源码/新payload及复用native字节。

单向身份关系为：独立审批launcher SHA → launcher核guard+plan → guard核plan中全部payload/输入。没有循环自哈希。Root审核并另行放行GPU后才可执行：

```sh
.venv/bin/python -c 'import hashlib,pathlib; p=pathlib.Path("outputs/q2-block15-mlp81-official-v5/launcher.py"); b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()=="01002b79ddd0f319b1ef6fef4f3aa5049454c31306e3d1d72248646cf58dcf6f"; exec(compile(b,str(p),"exec"))'
```

本轮4项小测试通过：错误/过大内存、允许swap、错误quota或已有swap均拒绝；错误scope/亲和拒绝；scope前置失败时不创建模型；篡改guard时不创建scope。payload/guard/launcher语法检查通过。没有执行上面GPU入口，无官方81、matched MLP或新质量结果。
