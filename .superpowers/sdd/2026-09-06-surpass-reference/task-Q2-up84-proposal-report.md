# Q2：84 ungated up_proj 同输入边界，CPU-only 方案

状态：只完成方案及已有来源的CPU身份绑定。**尚未生成新observer/payload执行包，不可直接启动GPU**。未运行任何新模型，没有改数学/默认/门槛。

提案：`outputs/q2-up84-proposal-v1/proposal.json`，SHA `96d00d6b2569f37e1455f7fe13aaea6242142bdf92182bdc82d18b6b7a788ea6`。

## 所检验的精确假设

v5全分母证明87的81传播项与同输入局部项同量级且明显反向抵消。这不定位具体算子。下一步只观察无GELU的up分支，先区分“ungated投影的输出本身相同”与“这个线性投影已产生同输入实现差异”。

固定图：conditioning后81→Split的83→gemm_4→84。gemm_4为K4096/N12288、transB=1、无bias；完整权重已认证为`layers.15.mlp.up_proj.weight` [12288,4096]，canonical FP32 SHA `0039d65f89e8f12d679f9d9ead7ad927ab64a7b21be8a3c1b9e0f3ff77d73fbb`。另一路82→gemm_5→85→GELU→86，最后84×86→87。84不包含GELU。

新84磁盘BSH=[1,4160,12288]，native WHDC=[12288,4160,1,1]，官方SBH=[4160,1,12288]，完整51,118,080元素。不能拿75或未调制输入替代81。

## 最少新增 actual forwards：两次

1. **native完整block一次**。复用原exact-official block14和九个conditioning/constants、原图/权重/FP32选项/数学源。observer只增加84，并同时保留已有81及out0作为前置不变性证据。完整81必须等于v5复用的native81，完整out0必须等于原teacher175bfa…；否则新增84无效。
2. **官方MLP一次**。直接使用v5已认证的完整native81作为原官方block.mlp输入，新增up_proj output hook取得84，同时捕获实际输入81及既有pre-down87/down88。实际81、完整87/88必须逐位匹配v5 matched MLP，才接受84。权重、加载路径、FP32/backend/TF32配置不变。

无需再运行官方完整baseline：当前目标只有同native81下MLP的84，v5已完成且通过四旧SHA的官方完整baseline作为已有认证背景重新验SHA；新官方调用由v5实际matched87/88约束，不谎称它重做了完整baseline。也不新建一个native简化MLP子图，以免引入未经证同的新图/装载/选项路径。

## 可复用的实际数据

- native81 `da8bc040…` 与nativeout0 `175bfa…`：v2完成的native observer，0c9247d已验证，再由v5原样复用；整个v2资源停止仍为负结果。
- matched87 `9a34cfce…` 与matched88 `5b0f3edb…`：v5官方MLP实际一次forward所得。
- v5官方baselineout0/75/87/88已逐位复现原oracle，result SHA `f62eb1d8ae79579ea135afc016290a14143ef4661c8dfccc701a36c28a855b54`。
- 既有原固定model param/bin、官方组件及原hook helper来源全部列在提案bound，实际CPU重哈希。

## 反证条件与解释上限

若N84与M84全字节相同，ungated up输出不能解释当前matched87差异，剩余定位转向gate/GELU分支或最终乘法。若N84不同，则证实这里确有同输入线性投影差异，先完整报告L2/max/NRMSE/不同元素数。

**仅84不能量化它对87的影响，也不能给up/GELU分支排因果百分比**，因为另一个乘数86尚未获得。若后续需要归因乘积，须按这个结果再选新边界。可在取得新84后提出预声明行的CPU FP64数值检查，但不把数学更准自动当作官方parity更好。当前没有批准/准备该额外计算或算子替换。

若任何81/out0或87/88前置复现失败，整个新增边界失去有效性，停止解释、不换阈值。计划沿用已真实验证的专属10GiB/swap0/CPU200%/affinity0,2 scope、9GiB RSS/6144MiB GPU/host3GiB/2400s守卫。需要下一阶段实现并冻结observer/payload/launcher后独立审查，当前提案本身不是GPU执行授权。
