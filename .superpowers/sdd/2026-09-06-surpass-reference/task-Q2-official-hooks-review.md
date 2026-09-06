# Q2 official block15 hook 实际证据独立复核

审查b083fc6：没有发现阻断本诊断解释的缺陷。只读CPU12/14、BLAS2执行独立hash/数组/分块FP64检查，无模型forward或GPU作业。结果不能晋级为完整中文轨迹通过、全模型精度策略或正式门槛变更。

独立检查器和结果保存在 `outputs/q2-official-hooks-independent-review-v1/{check.py,result.json}`。4/4合同小tests通过。

## 身份、只读性与完整分母

逐项核对plan的5472条bound路径SHA（按相同inode去重读取）；官方forward前后runtime清单的实际源路径和封存对象也逐项核对，共3074个唯一文件inode被读取。实际Python仍从记录的安装路径运行，归档中的硬链接/复制对象是保存及校验用途，不是另一个密封导入环境。安装文件后续若原地修改，硬链接也可能改变；hash验证会检测变化，不能将硬链接本身叫不可变存储。

原封存export_dit_block.load_block直接创建eval/requires_grad_false的官方ErnieImageSharedAdaLNBlock；没有用ExportBlock替代。实际ERNIE源0f1814…中三个hook位置清楚：adaLN_mlp_ln前为注意力残差75，linear_fc2前为gated MLP输入87，其输出为88。hook只有计数和detach所有者保存，隐式返回None；源码从这些边界继续使用out-of-place RMSNorm、乘法和加法，没有改写被保存的storage。各hook恰一次，保存发生在原forward完成后。

75/88/out0各17039360元素，87为51118080元素。独立完整size/hash/有限性校验通过。实际out0与历史 `diagnostic-chinese-step0-stages-v1/oracle/fixture/block-15.f32` 完整SHA同为 `a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197`，不是只接受脚本内的常量或subset。所有已认证input/角度/源/权重绑定未变；没有重新生成文字或timestep后替换teacher输入。

## 独立数值复算

我用每32行FP64累计重算六组完整native/candidate对official边界误差，全部L2/max/NRMSE与analysis一致。所有4160行上，official、native、candidate各自out0都逐位重建为FP32 residual75 + FP32(gate×88)。因此candidate完整out0 L2从.1987765增至.370414、max从.006591797增至.020507813是真实恶化，且down88处已存在，不是猜测残差布局造成。

对明确六行 `[0,22,652,4095,4096,4159]`，以K=256分块FP64乘法独立重新计算（与作者整矩阵FP64 BLAS路径不同），24576输出的各局部误差L2/max在1e-6相对/1e-10绝对容差内复现，cos(native_local,official_local)=.8373470069085。上游传播与两侧局部误差cos仅约.021，支持作者在该六行内否定“上游抵消主导”的假设。

额外独立从固定official dit-block15 safetensors读取linear_fc2的BF16矩阵[4096,12288]，每128行按位扩展FP32并转置，对全部50331648权重元素比较，逐位等于分析用weight-kn.f32[12288,4096]。FP64解释没有换不同矩阵轴或数值权重。

报告明确六行是事后选择（包括最大误差行652），不是全张量FP64或无偏抽样。FP64本身仍是有限精度近似；报告没有用它替代官方oracle。范数项非正交，不能从范数直接推出因果百分比；相关性也不证明native/official使用相同kernel归约次序。该解释与既存“局部FP64更准但官方一致性更差”的负结果相容，没有过度外推。

## 真实执行与历史范围

唯一v3 forward exit0，worker记录22.132832654秒，41条采样独立重算：RSSsum峰1218793472、整卡峰3938MiB、host最低17126092800字节，swap均0且守卫未触发。worker按约.5秒加查询耗时采样，约束递归子进程RSS/整卡采样/host下限；这不是精确allocator峰或硬cgroup/RSS保证，也没有系统OOM事件审计。报告已如此限定。

v1/v2是CPU准备协议修订，未执行GPU，不应被称GPU数值失败；v3才是唯一实际官方forward。活动日志排除修复、配置解析后runtime清单和post-identity后result写入顺序符合报告。最终out0通过是内部边界解释的前提；脚本仍保存invalid状态并非零退出处理失配，没有把倒推down当官方真值。此切片可以作为该局部候选拒绝的证据，后续完整生成及D3交付验证仍独立待办。
