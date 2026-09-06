# B2 剩余八项权重关系独立证明

结果：8项局部关系完成实际CPU检查，7项逐字节一致；Timesteps频率2048项中88项与当前固定官方CPU实现不同，max abs 5.960464477539063e-8、NRMSE 2.0191671970545926e-8。未改主weight-audit-all-v6及其1121 direct/8 unmatched口径。状态仍 `bounded_relations_checked_not_full_graph_proof`，`allowed_to_close_S=false`。不是完整模型图数学证明，也不是图像质量或性能验收。

## 独立身份与资源

工具 tools/audit_port_relations.py 不读v6的offset/value-match结果作为证明输入。固定24份实际源/资产/官方元数据SHA，重新认证assets-manifest及其revision140a052f.../URL/大小与4份param+bin实际SHA。随后独立重解析finalizer、preprocessor、encoder、decoder四个完整bin至EOF：270条序列化记录、0parser gaps。官方dit-final_norm与vae-bn实际safetensors整文件重新散列，验证固定repository/revision、manifest、完整header范围及dtype，再读取指定名字/行切片；不把任意自签audit JSON作为信任根。

固定manifest SHA：147e7710a43acc1bcf0b1d6ebd44dc114173720343a8e0891f6e9ce6e765780d。
官方final_norm文件：12fa6e315102b34387b8790eb4f59adcfbe83187bcc8a734f5146b4646ee2067。
官方BN文件：994f11f035c926c3be5a59802b401ab5304bbab7d53053995d4737b76b8411a6。
全部完整pins和每条实际offset/角色/值散列保存在 outputs/reference-port-v1/weight-relations-v1/relations.json。

实际执行4.020706秒；递归后代RSS采样峰627662848 bytes，低于3GiB守护；CPU亲和性0/2、torch intra/inter-op均2线程，无GPU或模型加载。torch只用于官方频率函数的2048元素前缀。没有接管下载，没有重跑全102官方组件/28peer资产扫描。

## 四项 finalizer 关系

| peer | 官方逻辑张量与切片 | 结果 |
|---|---|---|
| gemm_0 B | final_norm.linear.weight [8192,4096]，rows[0:4096] | exact cbdef59a…7167310 |
| gemm_0 C | final_norm.linear.bias [8192]，rows[0:4096] | exact 4a26257d…540429 |
| gemm_1 B | 同weight rows[4096:8192] | exact effe7241…6ba0260 |
| gemm_1 C | 同bias rows[4096:8192] | exact 67939b84…0687b2e |

矩阵方向不是按大小猜测：固定ncnn revision f6f734f... 的 gemm.cpp:15-29读取transB/N/K，:86-108在transB=1时以K×N即row-major[N,K]读B，C的broadcast type4读N个值；:130-180逐输出行使用B对应行。固定param SHA3175e429...中两个Gemm均N=K=4096/transB=1，无转置规范化。官方原始BF16权重按指定连续行展开FP32，与peerFP32字节严格相同。

角色由接线证明：splitncnn_0把同一conditioning分给两个Gemm；gemm_0输出4经add_0加标量1，与非affine LayerNorm输出6相乘；gemm_1输出5在add_2加到乘积。因此gemm_0是scale，gemm_1是shift。BinaryOp的ADD=0、MUL=2由固定binaryop.h:26-28认证。官方transformer_ernie_image.py:282-293声明Linear(hidden,2*hidden)，chunk得到scale,shift，应用norm(x)*(1+scale)+shift；与上述切片顺序一致。工具额外精确验证上述节点接线、4096维度、transB、LayerNorm eps1e-6/affine=false，拒绝换线/改方向。

此证明仅覆盖final_norm两个仿射映射的角色、参数与连接，不证明全36块DiT、所有算子数值实现和最终图输出。

## 一项 Timesteps 频率关系与明确负证据

固定preprocessor param SHA611af908...：MemoryData pnnx_fold_109为2048×1，乘输入timestep后依次sin/cos并拼接。官方transformer:328为Timesteps(4096,flip_sin_to_cos=False,downscale_freq_shift=0)；固定embeddings.py:56-62生成 `exp((-log(10000)*arange(2048,FP32))/2048)`，然后乘timestep。工具从已认证官方函数AST提取原始前缀执行到torch.exp，未替换计算顺序；生成的源码前缀和torch2.12.1+cu130 CPU环境完整记录。

peer频率SHA c4c9af3f7a86fc46fb66ca8bb96df090e14669fa93612c09eaa56f9757dcd314。
官方CPU频率SHA e5d3c4cb2987c81afc48ab2b2f560ea7eacf51a6ff0e3254b2715c2223191b36。
**88/2048不逐位相同**，独立再读rawbytes核对；逐项索引/值/bit模式保存在 independent-frequency-check.json。只确认固定图中的频率角色及接近该官方函数数值；没有peer原导出器及其数学库执行证据，不能声称已证明这些88差异的历史来源，也不能称该项exact match或据此推断完整生成质量。

## 三项 VAE 常量重新认证

- decoder convdw_103 depthwise1×1/channel128 scale：逐位等于sqrt(FP32展开官方BF16 running_var+1e-4)，SHA e89b48bf…84d199。
- encoder bn_0 slope：128个FP32 1，SHA02722f12…05d4ac。
- encoder bn_0 bias：128个FP32 0，SHA076a27c7…f36560。

固定decoder首层param、encoder尾部Crop首32→Reorg2→BN eps1e-4由whole graph SHA绑定；官方Flux2 config latent_channels32、patch_size2、eps1e-4及autoencoder_kl_flux2.py:138-143 affine=false由实际文件固定SHA绑定。固定ncnn BatchNorm load_model:24-36依次读取slope/mean/variance/bias，:49-53计算归一化系数，证实上述常量角色。此处重新读取真实文件，没有复用v5成功JSON作证明。

必须保留的不等价边界：peer decoder depthwise scale使用1e-4，而官方ERNIE pipeline:378-382 inverse BN明确加1e-5。本工具把前者证明为其实际值，不把它误写成后者数学等价。官方pipeline源SHA另记录在独立复核JSON。

## 测试与产物

四项小测试通过：错误source pin拒绝；finalizer错接线/维度/转置拒绝；1ULP不等保留且非finite/shape错误拒绝；官方张量shape/dtype约束。初次测试发生包导入路径错误，已补标准tools/脚本双入口导入后4/4通过；该setup失败没有产生实际审计结果。

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 taskset -c 0,2 .venv/bin/python -m unittest tests.test_port_weight_relations -v
```

实际审计目录 weight-relations-v1 保留工具快照、relations、执行资源、日志、官方频率和独立bit差异复核。频率差异继续显式存在；全模型逻辑拓扑/歧义仍由根独立审核，S不关闭。
