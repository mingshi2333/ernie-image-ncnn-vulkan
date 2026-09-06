# Q2 up84：两forward只读观察，CPU冻结准备

状态：CPU小编译、合同/来源绑定及scope探针完成；**未运行新GPU或模型**。依据已批准提案e53235a继续实现，保留所有审过的v2/v5与负结果，无生产数学/CMake/默认修改。

## 唯一执行流与先验条件

新native完整block **一次**，同一extractor/allocator/session依次保留81、84、out0，再下载完整FP32。五个数学cpp、全部对应头与sdpa_shader.h及libncnn.a逐文件等于旧封存observer来源；只有观察代码增加输出。源码仍加载原完整block，不重新造MLP子图。输入是原exact-official block14及原九个conditioning/constants。

native81必须SHA `da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54`，out0必须`175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3`。81/out0各完整17,039,360，84完整51,118,080；native端核WHDC/elempack/elemsize，CPU sequence再核字节数、finite、身份和布局成功标记。任一失败，不进入官方。

新官方**仅MLP一次**，输入为独立保存、完全认证的native81 BSH→SBH。只读hooks捕actual81、up_proj输出84、pre-down87及down88。actual81必须匹配native81；87/88必须分别等于v5实际matched `9a34cfce2296c4f2b3ef7f48aad848faf7362d349ee3ae4b7fcb0719ce0ba3c1` / `5b0f3edbb3c641a4826da737016b59f4d6eee1f708b919298a883234d4e2eba7`，才可采信新增84。每hook恰好1次，使用同一原模块/权重/FP32/backend，实际torch-config逐字段等于v5。

最大总数为**native完整1 + 官方MLP1 = 2 forwards**，官方完整baseline **0**。原v5 baseline复验身份而非重跑。sequence/launcher均拒绝已尝试目录，不重试，失败保留观察与日志。

精确84映射、局部假设及“没有86不能量化乘积因果贡献”的边界沿用task-Q2-up84-proposal-report.md；本次没有改变解释标准或提出算子替换。

## CPU编译及资源准备

小编译在实际systemd scope `ernie-q2-up84-compile-v1.scope` 下完成：memory.max4294967296（4GiB）、swap.max0、cpu.max200000/100000、affinity0/2，启动编译前逐项核验。exit0，cgroup memory.peak585,961,472 bytes，swap0，max/OOM事件0。没有使用root CMake/build目录做重配置；编译输入与archive均在新输出内。

执行scope沿用10GiB memory.max、swap.max0、CPU200%/affinity0,2；旧RSS9GiB、整GPU6144MiB、host floor3GiB、2400s、0.5s采样不变。native与official子进程串行，逐PID VmRSS/VmSwap/cgroup/affinity及scope current/events持续记录，初始控制器读值先落盘再检查。前置失败不创建模型子进程。

新guard纯CPU探针实际通过：scope `ernie-q2-up84-cpu-probe-v1.scope`，memory.max10737418240、swap.max0、cpu.max200000/100000、affinity0/2、PID/组swap0；RSS15,564,800 bytes，无模型/GPU。探针实际所有映射库均在final plan绑定内。

## 来源与运行库

Final plan **6133 bound** 已全部CPU重验，含全部旧输入/实际官方组件/param/bin、旧v5 matched与baseline执行身份、前后官方runtime实际路径、新源码/runner、原静态archive及原包含依赖。新源码没有增加头文件，相同旧依赖清单与新source分别绑定。编译命令、CPU限制与返回值记录完整，当前compiler路径/SHA另记录在evidence。

原native runtime只绑定ldd不覆盖dlopen，因此本准备增加：native runner和所有64位Vulkan ICD、NVIDIA辅助库及其ldd闭包（177库初始目录），Vulkan implicit/explicit layer JSON与可解析库闭包，以及guard自身psutil扩展。实际执行采样记录每PID映射库路径；任何实际映射的库不在bound或SHA变化均中止，不将未知运行库下结果判为有效。官方还保留前后实际import/mapped runtime对象快照与逐字节复验。

两条已安装但不能解析库路径的layer声明（ShaderDebugger相对路径、Vinegar）在layer-catalog及identity中明确保留。没有启用/禁用或修改系统layer；**若实际映射到未绑定库，尝试必须失败**。不声称CPU目录等于已经观察到完整GPU运行库集合；当前实际GPU映射仍pending。此设计不允许绕过未知项。

封存前补充guard自身导入及layer目录绑定；两份预备plan/launcher保留为`*-before-guard-import-binding*`和`*-before-layer-binding*`，没有执行。最终身份以下列值为准，未回写任何既有已审输出。

## 最终身份和审批后入口

目录：`outputs/q2-up84-v1`。

- launcher：`62bc6bf52e11428039843ab31f9dafc4c3802e21a357c24db26e7d0e53860358`
- plan：`621ef5ab8cb3c800f65306e416b37d94defa54e39761d1312190d966f5238e53`
- native runner：`ec5de0277fb94beab136877253b44873aa44e0285a145c50598cc51981c5f720`
- guard：`90c4fd6d9556e7675806d58fe5634f6f20bc6c6d856fb31ed930cc9b098ad1cc`
- sequence：`6c44698d50cb9cab988cb877605683c9660d8b3125b533983add5b5972ffc578`
- official payload：`ba08f5af9914f475bb93b9a011c03be4707ce6e154c4bd67d447c6bd79d0f630`

独立审查并获新GPU槽后，执行者外部验证launcher，再由它验证guard/plan，guard验证所有输入/代码/运行库；单向哈希链无循环：

```sh
.venv/bin/python -c 'import hashlib,pathlib; p=pathlib.Path("outputs/q2-up84-v1/launcher.py"); b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()=="62bc6bf52e11428039843ab31f9dafc4c3802e21a357c24db26e7d0e53860358"; exec(compile(b,str(p),"exec"))'
```

三项合同测试通过：旧matched实际81/87/88任一错SHA拒绝、native错误阻止第二个子进程、native截断分母拒绝。Python语法检查和实际小编译均通过。尚无新增84数值，不宣称改善任何局部或完整质量指标。
