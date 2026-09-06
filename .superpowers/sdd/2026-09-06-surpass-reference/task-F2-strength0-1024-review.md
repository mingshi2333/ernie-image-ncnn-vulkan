# 固定1024 strength-zero实际生产独立复核

最终状态：native与补足实际来源记录的official-v2均独立审查通过；未发现阻断这份固定1024 strength0实测的未解决缺陷。独立脚本与小结果位于 `outputs/f2-strength0-1024-independent-review-v1/{check.py,result.json}`；本审查不加载模型、不使用GPU。

## 已查证

native `outputs/f2-production-strength0-1024-v2-execution` 保存runner ca2ed9dd12c9e3fc1a4d7b8e0869ef1db51ed20e75b4392801b00e769484c75e与249源快照；所有源文件SHA重新核对一致。identity.json实际SHA为bd19c303afc168e6f9a73143723c7ef7aea32570042cbd32a00bb7bb33ddc319，result中的source_inventory_sha256实际指此包含runner/input/model的identity文件。模型manifest为2c1d0cdf39fe4cc94f7133d6dfac97a048123a8ebc8a4a5072e4d5e48368a5ad；前一独立注册审查已经核对72bb源与新增encoder四个对象，此轮不重复全量大权重散列。运行日志显示实际原生包完整验证完成。

命令没有prompt、PE、embeddings，strength0、CPU、FP32、threads2、direct CPU VAE、1024×1024。冻结pipeline源的无denoise分支在PE/text/DiT执行之前return，日志只有模型verify与VAE/image，没有文本或DiT阶段。认证包时读取权重字节作hash不等于加载对应模型执行。trace恰八文件，六个f32边界另加input.rgb和img2img.txt，没有预测步/文本trace。

8个trace全部完整SHA和size匹配，input.png SHA符合冻结身份；实际PNG解码RGB逐字节等于native trace与官方encoder input.rgb。final.f32逐字节等于encoder-normalized.f32。六边界使用FP64累加、完整分母重新算NRMSE/max，分别五个524288元素与decoded3145728元素，参考shape匹配且双方全部finite。

|边界|NRMSE|max绝对误差|
|---|---:|---:|
|encoder_mean|7.283859634807e-7|6.914138793945e-6|
|encoder_packed|7.283859634807e-7|6.914138793945e-6|
|encoder_normalized|7.331021137810e-7|3.933906555176e-6|
|final|7.331021137810e-7|3.933906555176e-6|
|unpacked|7.299470294188e-7|6.914138793945e-6|
|decoded|2.306728676786e-6|2.157688140869e-5|

固定门槛NRMSE2e-5、max<=2e-4+2e-4*max(abs(reference))全部通过，没有改门槛或漏元素。PNG完整1024×1024×3比较max1、MAE0.00010585784912109375，双方PNG哈希匹配result。审查脚本初次把参考dtype当成< f4/float32（实际标签F32）导致两次断言失败，检查fixture后按明确F32=little-endian FP32契约更正，完整重算通过。

native supervisor实际8GiB memory.max、swap0被观测，scope被看到、返回0、last memory.events无OOM；50ms循环持续检查hostavailable>=3GiB、1800s超时，并在失败时kill scope。native采样memory.current峰6083784704 bytes、hostavail最低11252105216、父墙钟151.467900722秒。官方v1相同8GiB/swap0、CPU4/6两线程、持续guard：采样峰4990722048、host最低12905791488、51.471840578秒、返回0。这里是scope内存采样峰，不是每进程RSS或精确硬件allocator峰；没有逐样本时间序列，不能据此声称无采样空隙。启动前失败的native-v1单独保留，不计为实际推理。

## 官方参考来源局限与待闭环

official reconstruction-v1的255本地源快照全部匹配；命令执行live tools/reference_img2img_large.py而非snapshot worker。该工具从有效88f encoder fixture的out2继续，精确encoderBN1e-4、decoder逆BN1e-5+pixelshuffle2后执行官方decoder._decode。load_vae在运行时检查decoder/postquant权重hash、prefix与sources.lock固定revision，reference.json记录两权重hash。但v1未记录实际安装AutoencoderKLFlux2源码SHA/path、distribution版本/direct_url、vae config SHA和component manifest文件SHA。本地工具快照不能代替这些运行库身份。作者已接收发现，保留v1并以补足身份的新工具准备official-v2，审查尚不把v1称完整冻结官方来源。

这是单张固定1024发展fixture的strength0重建；不代表positive-strength、任意shape、正式15/72case或模型广泛质量通过。

## c708baf / official-v2 闭环

作者保留v1，提交c708baf并实际重跑 `outputs/img2img-reference-reconstruction-1024x1024-v2`。reference.json SHA fc915a48ce5e2921d899ed7e8f33878f92686987866640e9e7d9181dbac00616 已由production result显式关联。独立v2审查脚本位于 `outputs/f2-strength0-1024-independent-review-v2`，再次完整重算六边界/PNG全部通过；官方v1/v2 unpacked、decoded、PNG三个文件逐字节相同。

v2记录实际安装diffusers 0.41.0.dev0、direct_url固定7643c4826609c47755e3da0e5b768e8070468f49、AutoencoderKLFlux2类路径与SHA7d9a976c1e4f42615e8c422f1643d86b49c4339221bd04b67f518b718ebd6c2d。独立从该实际路径重新散列一致；vae config SHA4d5ba5e01de06d589dd46e2955ab97e2b0968703dce31b9ebe1d2a38c141836d及decoder/postquant manifest文件SHA均重新核对。v2保存255本地源快照全部匹配。它仍是live-tools/安装库执行，未声称全依赖 hermetic；新增实际身份消除了v1缺少可核对安装来源的问题。

v2实际父墙钟55.247524137秒、采样scope峰4876550144bytes、host可用最低11154571264bytes，观测memory.max8GiB与swap0，cgroup seen、完整退出0、memory.events无OOM。持续guard代码与v1相同范围。最终结论只认可该单张固定1024输入的生产strength0重建通过，不扩大到positive-strength、其他分辨率或正式质量语料。
