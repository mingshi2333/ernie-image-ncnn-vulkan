# Q2 81 实验：swap 守卫停止，保留 native replay，未获得 matched-input 结论

唯一授权运行已结束：session89906，22.68666456s，状态 `aborted_resource_guard`。守卫观测进程swap **9,306,112 bytes** 后杀死所有已跟踪后代。没有重试、补跑、放宽守卫或改变数学。GPU 已核无任务模型进程并立即通知 Root 释放。

准备提交dc0336d、入口身份修复a599abf经独立1bc3817通过后，执行者先在内存中按独立批准值核验launcher SHA `07c879d246e92b12d220cbb88b549504e35179166f9f73a2afb41ed4503b3962`，再执行原封存launcher。它认证原worker `2ac4e0…` 和plan `c7447c…`。CPU0/2、原所有守卫不变。此次执行未替换封存脚本。

## 已得到与未得到的证据

native observer 实际完成一次完整 block15。WHDC均为[4096,4160,1,1]；两个输出各68,157,440 bytes、17,039,360 FP32元素，CPU后验全部finite：

- 新81 SHA：`da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54`
- 完整out0 SHA：`175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3`，**逐位复现原native teacher**。

这是81 observer不改变原native结果的实证，仅限该固定输入和完整输出。新81可保留为已认证native边界；**没有官方81可比，因此不解释它的误差，不进行matched-input分解**。

官方进程随后启动，但被守卫中止：没有torch-config.json、gpu-before-forward-runtime.json、baseline-observation.json、official输出目录或result.json。冻结执行源码在唯一block forward之前写config和runtime快照；当前没有到达该forward的证据，官方baseline和matched MLP均未完成。不能将计划最多三次调用写成三次已执行，也不把资源失败归因于MLP数值。

停止后重新完整验证plan的5885个bound文件，全部一致。`outputs/q2-block15-mlp81-v2/execution-audit.json` SHA：`a103f7505b0167e872c388e0325b05ab2b8939e310419ea9dd6adf07e30c1943`。实际日志/输出/身份与失败状态保留原位。

## 资源事实和测量边界

43个采样。递归RSS总和最高836,218,880 bytes，整GPU采样最高3977MiB，host available最低17,722,568,704 bytes。第42个附近样本22.12786s仍swap0；22.68384s样本总swap9,306,112 bytes，host available18,137,624,576 bytes。随后守卫退出，wrapper return_code−9。**18GB可用内存不排除某个进程已经有swap映射**。

PID序列依据封存sequence的串行调用与样本：

- 2087828：sequence；0.0476s出现。
- 2087961：native observer；5.3946s出现，9.7916s已退出。
- 2088325：official子进程；15.1288s出现，最后样本与sequence同在。

守卫代码调用每PID的 `psutil.memory_full_info().swap` 求和，只保存**总和**及PID列表。它没有保存逐PID `/proc/.../status` VmSwap、smaps/smaps_rollup 或cgroup记录。停止后2087828/2088325均不存在，无法读取历史/proc。因此**不能把9,306,112 bytes可靠归到其中某一个PID或特定历史cgroup**；也不能将当前系统swap计数当作那个进程的值。

源代码确证：launcher/worker只使用普通subprocess、start_new_session、CPU affinity和轮询守卫，没有创建/加入专属cgroup，没有设置或验证memory.swap.max=0。start_new_session隔离会话，不提供内存或swap隔离。

额外CPU只读探针当时继承`/user.slice/user-1000.slice/user@1000.service/app.slice/app-org.chromium.Chromium-14995.scope`，memory.max=max、memory.swap.max=max，memory.swap.current=3,853,959,168 bytes，memory.events的OOM计数均0。这证明**当前诊断入口环境**允许swap，不是已退出模型进程cgroup的直接历史记录。未将该当前观测冒充执行时测量。

## 下一独立实验前需修的入口

若Root另行批准新尝试，先创建专属cgroup并在任何子进程启动前认证其memory/swap限制，至少实际读取确认memory.swap.max=0；记录每个实际模型PID的cgroup、逐PID VmSwap/smaps信息以及cgroup memory.current/swap.current/events。需要确认所有后代加入该组，而非只给外层wrapper设置限制；保留现有递归RSS/整GPU/host/timeout守卫和失败证据。

本次只读定位到**隔离入口未禁止swap且缺少逐PID归属遥测**，没有证明内核为何选择swap，也不把host available当作原因排除依据。先前审批允许的一次尝试已消费，不再运行该worker，不以新目录绕过禁止重试。本轮最终结论为资源停止、native observer replay有效、official/matched结论pending，质量字段为null，正式验收false。
