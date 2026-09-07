# Windows RAM 余量查询与缓存准入边界

2026-09-07，在 `host_memory` 组件中加入原生 Windows 的内存余量读取。每次查询取可用物理 RAM、进程剩余 commit 容量与可用虚拟地址空间的最小值；page-file 对应字段只限制额度，不作为额外 RAM 加到物理内存上。含义来自 [Microsoft MEMORYSTATUSEX 文档](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex)。查询失败仍禁止新增缓存，既有流式权重执行不受影响。缓存默认仍为 0，未修改模型数学或精度。

实际测试改变了实现方向。第一版在配置为 4 GiB 的 Linux 进程组内经 Wine 返回 **18,668,998,656 字节**可用额度；这不是有效的该进程组剩余额度，不能拿来决定缓存容量。第一版 Linux 2/2 和 Windows/Wine 2/2 测试通过，只证明接口可执行，不证明这个数值适合缓存准入。因此保留第一版全部输出和源码补丁，并增加以下限制：

- Wine 返回 unavailable，继续逐块加载。最终实际原始 Windows API 报告 17,542,742,016B，而生产缓存查询正确返回 unavailable。
- 进程在 Windows Job 内时也继续流式执行。查询 Job 关系失败同样禁止准入，查询物理内存前后都检查关系。尚未建立完整 Job 祖先额度读取；[QueryInformationJobObject 的官方说明](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject)明确 NULL 只查询直接所属 Job，不能将它当作完整祖先约束。
- 测试使用实际 Windows API 创建并分配 Job，独立确认成员关系变为 true，再检查原先取得的 reader 会禁用缓存，避免沿用旧额度。它是当前测试进程自己的 Job，不修改其他进程。

最终 MinGW Windows CPU 和 Linux CPU 两种构建均通过；分别执行 `host_memory_cpu`、`weight_session_cpu`，共 **4 次 CTest 全通过**。Windows CLI `--help` 退出 0，专用 Wine 服务退出完成。Windows 测试实际验证 Wine/Job 的拒绝准入分支，**没有证明原生 Windows 的无 Job 缓存路径、Windows Vulkan 完整模型或真实内存压力回收**。这些仍需原生 Windows 验证；macOS reader 仍未实现。Linux 原有读取和缓存状态测试保持通过，无需因 Windows 条件编译改动重复完整 Linux 图像。

首个执行器在 Wine 初始化之后，额外的 `wineserver -p60` 返回 2 且无诊断，未进入 Windows 测试；原始失败保留。随后去掉不必要的持久化辅助命令，使用已建立的专用前缀完成实际测试并显式退出。未把辅助命令失败当作产品故障或删除该记录。

`v1/`、`v2/` 分别保存两个候选的原始补丁、来源散列、构建与测试日志。最终程序和测试可执行文件保存在 `/var/tmp/ernie-windows-headroom-v2/binaries`，散列见 `final-identity.json`，大二进制未入 Git。CPU 工作固定在独立的 12/14 核、2 核额度、4 GiB 进程组和 swap 0；共享 768/1024 图像批次始终使用其既有冻结源码及 92559ea4 程序。无公开推送或发布。
