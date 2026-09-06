# Q2 official-only scope 准备独立复核

Root 对 `2d019b3` 的 v5 CPU 准备执行只读独立检查，未运行模型。结论：该次独立尝试的准备无剩余阻断，可进入下一独占 GPU 时段；没有把 CPU 探针或先前部分 native 结果标为官方质量通过。

- 完整重验 plan 中 5905 项路径的 SHA，3492 个不同 inode，全部一致。只对同 inode、size、mtime 的相同对象复用散列；完成后检查对应 size/mtime 未变化。
- 外部批准 launcher `01002b79…` 精确绑定 plan `38f2f50d…` 和 guard `7f1b77a0…`；后者验证全部 payload/输入后才启动模型，没有自哈希循环。独占创建 launch-record 和 attempt 防止重复启动。
- 新 official-only payload 与原已审官方 payload 完整文本对比，只有 native forward 计数 1→0、native_replay_reused=true 两处记录变化，模型、hooks、原四个 SHA 前置、完整 native81 替换及最多一次 matched MLP 均无变化。
- 原有效 native81/out0 是封存副本，身份受新 plan 约束；本次不重跑 native。旧的22.6867秒 swap 停止及官方缺失边界状态保留。
- guard 首先保存实际 cgroup 读值，再检查专属 scope、10GiB memory.max、swap.max=0、cpu.max 比率2及CPU0/2，失败不启动模型。逐PID/cgroup/affinity和控制器遥测记录后再检查越界；max/OOM事件增长、进程swap、RSS9GiB/GPU6144MiB/host3GiB/2400s均保持停止条件。启动失败由外层 launch-record/log/exit 保留。
- 环境未暴露 cpuset.cpus.effective，报告准确区分已验证亲和性/CPU quota 与未证明的 cpuset 控制器。历史当前 app scope 观察未被倒推为旧已退出PID的执行证据。

实际重跑小测试：scope guard 3/3、launcher 篡改拒绝 1/1 全过；检查错误限制与已有swap、scope/affinity不符、前置失败不创建模型、guard篡改不创建scope。未重复大型原始实验。

资源守卫按0.5秒采样，不是分配器精确峰值或任意短暂进程逃逸的证明；systemd控制器提供硬内存/swap限制，独立调用仍须验证本次实际scope。若scope OOM杀死guard，外层exit/log保留失败，不能要求guard在被杀后写出正常完成JSON。正式质量、S/M仍未评估。
