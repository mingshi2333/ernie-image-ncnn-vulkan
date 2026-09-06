# O1 fixed64 ON/OFF 执行准备独立审查

审查范围：`outputs/execution-metrics-pipeline64-o1-v1`。本轮只读脚本、冻结文件、现有构建及小型 CPU 反例；不加载模型，不使用 GPU，不重建。先前 runtime 实现审查 `420f6af` 不替代本次执行计划审查。

## 初始发现（需修复后才能执行）

1. **Important，监督器必然触发 worker 拒覆盖。** 初始 supervisor `baba17d3…` 在 Popen worker 前打开/创建 `samples.jsonl`；worker `1bdee9fe…` 的拒覆盖列表包含该文件。因此身份检查通过后仍会退出 `refuse overwrite`，不会进入 native。应在 supervisor 自身启动前检查其日志/过程文件，而 worker 仅检查自身输出。修前准备不曾执行模型。
2. **Important，源码与构建来源闭包不完整。** 初始 plan `df0325ac…` 仅记录 80 文件，缺 tokenizer/CMakeLists、Rust 源/Cargo lock/schema 合同；build_source_note 比较 `rust` 路径而非实际 `tokenizer`。未绑定实际 pinned ncnn 与 ON 派生副本/生成 shader 身份。因此已有证据不足以声明仅统计开关变化的同数值来源，需补实际来源与已有构建链。
3. **Important，等长缺失张量可假通过。** 初始 compare `c33eb2b8…` 只要求 27 个名称和两侧 bytes 相等；不检查 25 个 FP32 的固定长度/finite 或 PNG 结构。独立临时 CPU 反例把所有 trace 与 PNG 置空、填入其余所需合成字段，实际 exit 0 / status passed，PNG SHA 为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。需补固定 shape/size/finite、合法 64×64 PNG、trace initial 对 frozen initial 的实际字节绑定。该反例仅证明验证器缺口，不是模型结果。

另外要求 `memory.events` 必须含 oom/oom_kill 字段，不能把缺失字段默认成零。

## 已独立核实的初始准备

- 80 个冻结 source 文件 SHA 全部符合 plan；两份 frozen runner 分别逐字节 SHA 等于现有 stage-on-v2 / stage-off-v2 构建输出，两个复制 CMake cache SHA 正确。
- 规范化构建目录后 cache 差异仅 metrics ON/OFF、ON 派生 ncnn/glslang source、hook link 与 ON 所需 Python 检测；这不能单独代替完整依赖来源验证。
- prompt/initial/historical IDs 的大小与 SHA 均符合 plan；233 model 项存在且大小符合。本审查未重新扫描全部大型 weights，冻结 worker 已设计在实际执行前逐文件完整流 hash。
- 两命令数值参数相同：64×64，8 steps，Vulkan FP32，CPU direct VAE，threads 2，同一模型、prompt 和 saved latent；ON 额外 metrics 输出。trace-on，所以性能与内存 formal eligibility 为 false。
- 监督器采用独立 systemd scope，MemoryMax 10 GiB、swap 0、taskset 4,6；50 ms host available 3 GiB 守卫、1800 s timeout。实际 scope observation 将在执行后验证，目前只能称准备。记录的是 sampled memory.current 峰；不能称精确 RSS 或全卡 GPU 峰。
- 阶段比较要求 partial_known_intervals、host interval 非重叠声明、GPU/传输 bytes 未知为 null，五个可测阶段存在，阶段和不超过 CLI host 时间。这是受限诊断，不会建立 S/M。

## 状态

作者已收到全部发现，正在追加修复与冻结。当前尚无实际 ON/OFF 生成结果；本报告初始版不批准执行。

## 修复复核：准备态通过

最终身份：plan `d20b6b2ec68a286dc551eec730604cbf98b5be5a9678bc2036cc5c1efe1aec39`；worker `aee653c47c0ec3cab3404d12e6ec7a70c28be627d02efec9cd2b9c1915e3a1b7`；supervisor `b1e660e6fc8391bad9ab394a1c47c49aa24c8be93b9d52305f5fba1ee49a822f`；launcher `3cdd40d7fe7362c2907dd2095c416aad78170d4c14a9da78b6d229baff5a5abd`；compare `3f9555a27339748522074a4f5ee4138c6b7713f7e0d3be4d25780a70c29baefb`。已按 artifact-identity 重新独立散列，全部一致。

- Supervisor 在创建自身日志前检查其输出未存在，worker 仅检查自身 native/metrics/driver/trace 输出；samples 冲突已解除。旧准备身份及 executed=false 保存在 invalid-preparations.json；并未执行失败模型或覆盖真实结果。
- 冻结项目 inventory 已扩为 305 项（含 tokenizer/Rust/schema），逐文件大小及 SHA 独立通过。ncnn base 7791 项、derived 7793 项的实际文件全部 SHA 重新读取核对通过。固定 base commit `6a1bf000f363714839a36793addc8c879d3d899e`。独立比较清单，ON 仅变化 src/allocator.cpp，并新增 allocation-metrics-provenance.json / allocation-metrics.patch，无删除；补丁内容属于已审观察钩子。`git diff b08db3e..771c8ac` 的真实编译目录（含 tokenizer）为空。
- 两份 binary 与实际 build 输出 SHA 匹配，配套 cache 只出现上述预期差异。依赖 manifest 在运行前会认证；其 manifest 内 live 源逐文件 SHA 已由本次独立审查确认。它们是既有构建的冻结来源记录，并非新做可重现构建实验。
- 27 trace 现在逐个固定 size，25 FP32 全元素 finite，ON/OFF 全字节相同，initial/prompt/IDs 均与冻结输入绑定。PNG 比较要求全字节一致并审计 chunk 完整边界、CRC、64×64 IHDR、IDAT 与唯一末尾 IEND；此处是容器验证，不另作解码器质量测量。
- Actual allocation 必须是 vk_device_memory/observed_ncnn_allocator_lifetime，正分配数、正峰值、live=0；device index0 存在、allocator 非空且均 inactive/live_handles0。不能用空 session 的布尔字段冒充真实 allocator 事件。
- 独立 CPU 临时小 fixture：一个合成正例通过；截断 tensor、两侧相同 NaN、两侧截断 PNG、缺 OOM 字段、空 allocator 列表、两侧相同但错误 initial 共六个反例均拒绝。未导入 Torch，未使用模型或 GPU。

最终结论：三项 Important 已收敛，没有剩余执行准备阻断项。可以在 root 授予的独占 GPU 队列中执行固定 ON/OFF 各一次；这不是实际运行验收。审查时 on/off 目录仍为空。运行后仍需审查有效 cgroup/守卫、全部输出与 allocation cleanup；只观察到的 sampled memory.current 和限定 VkDeviceMemory domain 可报告，正式 S/M 不成立。失败日志不得重试覆盖。
