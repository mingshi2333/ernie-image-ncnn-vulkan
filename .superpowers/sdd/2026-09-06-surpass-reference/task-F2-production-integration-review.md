# F2 生产集成独立审查

审查对象：`638a58c`、`6a30dae`；只读实现。另识别后续 `582c463` 的预处理身份修复，不将其误算为原提交已经正确。审查期间作者已接收下面两项问题并开始修复；本文保存原提交的可复现结论，不替代修复后验收。

**结论：原两提交需要修改。发现两个明确的 Important/P2 缺陷：Rust available encoder 缺少所选 source instance 的存在性检查；双线性上采样的左/上边界邻域选择错误。没有运行完整模型或重散列大型包。**

## R1 — Rust 接受未绑定实例的 available encoder（P2）

位置：`tokenizer/src/shared_package.rs` 原102–117行。`encoder_source` 只用于决定是否向某个 ResolvedPackage 加入 encoder 文件，没有在遍历实例后检查该 source 是否真的在 `instances` 中。对于一个没有对应512 source的旧包，只要在metadata声明受信任512 encoder、对象集合仍仅包含原实例绑定，Rust最终 `used == objects` 仍通过。Python `verify_shared_package` 明确检查 `matches[0] in selected`，会拒绝相同语义。

隔离复现：复制6a30dae的Rust package/shared_package源到 `outputs/f2-production-review-v1/tokenizer/`，仅在已有合成 corruption fixture 测试中把 encoder registry 的 key 改为一个未被选中的 source；生产验证函数保持原样。使用既有Rust依赖编译单独test，不加载模型。结果：

```
UNBOUND_ENCODER_ACCEPTED=true RESOLVED_FILES=136
available encoder must bind an actually selected instance
exit 101
```

影响是原生verify-model错误认证可用性声明和Python/Rust合同不一致；当前C++ has_file仍找不到encoder，因此没有证据表明它能在1024上成功加载错误encoder，不夸大为越界推理。建议在完整instance遍历后要求 `encoder_source` 存在于 `seen`，并增加只包含另一实例、但声明available encoder的负例。作者已收到。

## R2 — 双线性上采样边界混入相邻像素（P2）

位置：`cli/image_io.cpp` 原291–297行。对于half-pixel坐标 `fx < 0`，代码先把 `floor(fx)` clamp成0，再使用 `bx=ax+1`；而权重仍为 `fx-floor(fx)`。这让图像左边缘错误地与第二个像素插值，顶部同理。正确edge-clamp应分别clamp未截断的floor邻域两端，或先clamp连续坐标。

用6a30dae源码编译独立2像素测试：输入 `[red, blue]`，2×1 stretch至4×1。实际输出：

```
64,0,191
191,0,64
64,0,191
0,0,255
```

首像素应为 `255,0,0`，边缘检查exit1。该错误会改变进入encoder的真实RGB，包括stretch和放大后的fit/crop，不只是展示细节。已有常量图stretch测试无法发现。建议增加水平、垂直双色边缘回归。作者已收到。

## 已确认的执行边界

| 检查 | 审查结论与证据边界 |
|---|---|
| 旧包缺encoder | schema1/2仍只解析原required_files，schema3 unavailable声明继续接受；txt2img仅在input_image分支检查encoder，不强迫旧包增加文件。img2img明确报缺reviewed encoder。 |
| 唯一生产shape | 编译期registry只将512×384 encoder绑定到固定512 source。ModelPackage选择所请求WH，图hash及bin SHA/size由CAS认证，固定encoder不经过任意shape改写。独立encode_vae还有32×32/64×32小诊断入口，但这些不在生产schema3 trust表。 |
| BN数学 | registry锁定encoder epsilon1e-4、无affine、posterior mode、pixel-unshuffle2；解码unpack_for_vae仍使用sqrt(var+1e-5)。未把两端epsilon误合并。 |
| strength0 | generate在encoder后unpack/decode并return，控制流位于enhance_prompt、tokenizer/text和run_dit之前；GpuContext开关排除该路径，CPU encoder/direct/FP32独立设置。不提供prompt合法，提供prompt/PE/embeddings/text-down显式拒绝。全包完整性校验仍可读取PE以外的text/DiT对象进行SHA，不能把“不加载推理模型”说成“完全不读其文件”。 |
| positive strength | make_img2img_start保留完整FlowSchedule，四舍五入half-up选suffix，极小正strength至少一步；denoise从绝对start_step到steps-1，重新使用同一完整schedule，不将suffix再归一化。 |
| progress/trace | stats按执行步push，callback收到绝对i；pipeline进度使用i-start_step+1与stats[i-start_step]，trace仍prediction-i/step-i绝对编号。静态路径核验成立，尚无本次真实positive全链运行证据。 |
| 默认与无prompt | 保持device=vulkan/precision=fp16、strength=.5、threads4等既有默认；CLI只有input+strength0不要求prompt，正strength仍需prompt来源。API允许空字符串prompt是既有行为，未将其当作新回归。 |
| public/legacy | 公共pipeline.h仍仅stdlib，新增字段尾部追加，旧aggregate初始字段不移位；legacy probe路径未在这两提交改动。 |

原6a30dae将alpha合成背景和resize填充背景共用单一trace字段；显式background且无resize还未写入request。此问题已由后续 `582c463` 分开 `input_background` / `input_resize_background` 并传播身份，不列为仍未处理的问题。

## 执行的小型验证

- `ctest --test-dir build-dev -R '^(image_io_contract|request_validation_contract|pipeline_api_contract|cli_contract|img2img_contract_cpu)$'`：5/5通过，1.31s。只执行既有小CPU二进制，未构建build-dev；这些既有用例仍未捕获R1/R2。
- `python3 -m unittest tests.test_dynamic_package tests.test_vae_encoder_specialize`：24/24通过。
- 两个独立缺陷复现分别exit101/exit1；文件、原提交source快照、编译日志、运行日志位于 `outputs/f2-production-review-v1/`。
- 本次没有执行正式encoder/decoder、PE、text、DiT或完整GPU模型，也没有验证formal15。作者准备的真实strength0以及positive图生图完整质量均属于后续独立证据。

O1 GPU放行后临时执行了本agent的allocator小合同，不属于F2质量验证；其单独报告保留失败与修复记录。

## 修复后闭环复核 — f04b35f

作者提交 `f04b35f` 后，本review再次从该commit导出源码到独立 `outputs/f2-production-review-v2`，没有使用正在编辑中的生产文件替换失败快照。

- Rust在实例遍历后检查 `encoder_source` 是否属于 `seen`。原同一未绑定source的独立测试现在输出 `UNBOUND_ENCODER_ACCEPTED=false`，隔离Rust3/3通过。
- Python在合成manifest上执行同语义的未绑定available encoder检查，输出 `UNBOUND_ENCODER_ACCEPTED=false Unreviewed schema-3 encoder`。跨语言拒绝边界一致。
- resize分别clamp原floor坐标的两端，原2→4双色测试现在输出 `[255,0,0],[191,0,64],[64,0,191],[0,0,255]`，exit0。顶部使用相同对称修复逻辑，代码审查确认不再从clamped起点推导第二邻点。
- v1/v2各有完整源/二进制/日志散列清单 `identity.json`，原失败证据未覆盖。

**R1、R2在f04b35f下均已独立闭合，本次审查没有剩余具体实现阻断。**这仍不是正式图生图质量验收：真实生产strength0、positive完整运行与formal15由其他实际证据承担，本文没有把小合同外推为完成F2全部目标。
