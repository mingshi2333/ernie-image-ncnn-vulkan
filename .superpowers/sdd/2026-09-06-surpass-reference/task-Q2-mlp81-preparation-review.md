# Q2 block15 MLP81 CPU preparation independent review

审查对象：提交`dc0336d`，`outputs/q2-block15-mlp81-v2`；只读CPU检查，未执行GPU或模型forward，未修改作者实现。

## Important

### I1：正式执行入口尚未从外部认证worker身份

`execution/worker.py`固定plan SHA `c7447c...`和sequence SHA `b4e9ee...`，sequence又在执行前后验证plan中的5885项bound；但plan与contract都不包含worker本身，worker也不能安全地用自身内容自证。因此当前文档给出的直接命令若在执行前worker被改写，资源guard、最多forward数或实际子命令可改变而仍使用同一个受信plan。

这不要求制造plan↔worker循环依赖。修复应由固定外层launcher/authorization identity在启动前验证worker SHA `2ac4e0c...`和plan SHA `c7447c...`，并让launcher自身SHA成为审批对象；等价地，root可用已记录的外层执行器在同一不可分割启动步骤先核两个SHA再exec。补一个改写worker后拒绝、未创建`started.json`的CPU负例。完成前不应启动GPU。

## 已核验成立

- plan 1,245,261 bytes、SHA `c7447c...`；5885项bound逐文件重算，0缺失、0不匹配。runner SHA `3689bfc...`，worker当前SHA `2ac4e0c...`。
- 81映射正确：learned RMSNorm(75,eps1e-6)，乘`1+in5(scale_mlp)`，再加`in4(shift_mlp)`；`in6`只用于MLP后的residual gate。
- MLP顺序与实际official源码和固定ncnn尾图一致：up projection与gate projection并行，`up * GELU(gate)`形成87，再经down projection形成88。
- 四个关键native权重从完整436,241,436-byte bin流解析，offset/count/dtype/transpose与官方具名tensor的canonical SHA相符；没有靠shape猜转置。
- native磁盘布局BSH `[1,4160,4096]`；runner实际报告WHDC `[4096,4160,1,1]`，official读取后转置到SBH `[4160,1,4096]`。81和out0各17,039,360元素，87为51,118,080元素，均检查完整字节数和finite。
- sequence严格先运行一次native full block并要求完整out0逐位等于旧SHA，再运行一次official full block；official的75/87/88/out0四个旧完整SHA全部逐位重现后才允许接受81或执行matched MLP。81逐位相同则省略尾调用，否则只执行一次`block.mlp(native81)`。
- hooks只detach并读取75/81/87/88，各自计数必须为1，随后remove；没有hook替换输入或写模块输出。matched调用直接以完整native81作为MLP输入，不重跑attention/conditioning，不修改权重、dtype、backend或TF32状态。
- 最大调用分母显式为native full block 1、official full block 1、conditional official MLP 0或1；结果分别记录hook counts和forward counts，不把额外调用隐藏。
- worker的CPU0/2是作者实际执行分配；root已澄清CPU4/6只约束本独立审查进程，因此不是finding。worker另有递归RSS9GiB、整卡6144MiB、host floor3GiB、process swap0、2400秒与0.5秒采样限制。本审查没有启动worker。

## CPU验证

```text
python3 -m unittest tests.test_mlp81_contract -v
Ran 3 tests in 0.000s
OK

independent bound rehash
bound 5885 bad 0
```

结论：数学映射、固定输入/模型身份、layout、旧基线前置条件和完整调用分母均可支持这一单一诊断。正式GPU执行仍需先关闭I1；即使后续完成也只是matched-input诊断，不是production/formal acceptance或中文自由运行质量通过。

## 修复复核

提交`a599abf`新增单向外层launcher，没有改写原v2 worker、plan、输入或数学。launcher SHA `07c879d...`作为独立审批身份；其内容在内存中先由外层命令核SHA再执行。launcher在创建任何子进程前固定验证worker `2ac4e0c...`与plan `c7447c...`，避免plan/worker循环自证。独立核对identity三个SHA均与实际文件一致；两项CPU负例分别篡改worker和plan，均在`subprocess.run`调用前拒绝，2/2 PASS。I1已关闭，准备态无剩余Critical/Important；GPU仍未由本审查启动。
