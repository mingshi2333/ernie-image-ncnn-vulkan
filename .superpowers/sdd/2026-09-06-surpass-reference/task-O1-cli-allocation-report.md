# O1 CLI allocation wrapper — implementation slice

2026-09-06. 此片不关闭 O1；没有运行正式语料或 GPU 模型。

## 实现与生命周期

私有 CLI `--metrics-json NEW.json` 仅在 ON 编译可用。默认 OFF 不链接 observer、不生成派生 ncnn，也不包含 GPU 查询或 session。ON 即使不请求报告也明确提示其时间是 diagnostic，不能当正式速度 runner。

Session 在 CLI 输入图片准备和 generate 前创建，仅用不会创建实例的 get_gpu_instance 检查初始状态。generate 返回/抛异常后，其原有 RAII 已释放图/allocator/GpuContext；图片写出并 close 后再 snapshot。既不改变 pipeline 生命周期，也不销毁借来的全局实例。初始或最终存在实例、仍有 live memory/allocator、身份缺失、observer 失效均不完整。单次独立 CLI 进程是范围边界，并非线程/多 pipeline 隔离承诺。

新固定派生补丁在 VkAllocator 构造时从其已有 vkdev->info 复制 index/vendor/device/name/API/driver/pipeline cache UUID。UUID 字段明确是 pipeline_cache_uuid，并非物理 deviceUUID。原始固定源码未改；原 VkAllocateMemory 成功点和真实 vkFreeMemory 后 callback 数不变。此版本需要新的派生目录，旧证据保持。

JSON schema1 的 domain 是 vk_device_memory；total.peak_bytes 是所有同时存活实际 VkDeviceMemory 的合计峰值，绝不能加各类别峰值。按 device/type/heap/flags/host_import 分组，allocator 有角色与 generation；无观测 total=null。coverage 仅限 observed_ncnn_allocator_lifetime，排除其他驱动分配。CPU RSS、GPU 时间、阶段时间均为 null。

本地 host_nanoseconds 从输入准备到图清理和 image close，是 cli_generation_and_image_write，包含验证/加载但不含进程启动、参数解析及最终 JSON 写入。不能冒充 end_to_end；正式父进程必须自己固定 runner/config/source/input 等身份并用外部时钟。报告不接受自证 hash 标签，formal_speed_eligible/formal_memory_eligible 恒 false。

## 失败处理

输入准备、generate、图片写出分别记录失败阶段和原始 error。报告只以 O_EXCL 新建，避免覆盖已有文件；write/close 失败独立报错。主失败与次级报告失败并存时先输出原错误，再输出报告错误。大 JSON 不进入 stdout。发生报告写出失败可能留下部分新文件，需按进程失败处理，不能当完整 measurement。

## 已执行证据

- `cmake -S tests/allocation-cli -B build-o1/cli-contract -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/clang++`; 独立小 target -j2。
- CTest allocation_cli_cpu 1/1，内部 13/13 Python cases 通过。使用真实 main/options/prompt_file/report/hook，以 fake inference/I/O 覆盖成功、generate/image/report/双错误、OFF拒绝、trace、原文件保护、设备身份缺失/冲突、未释放分配、借用实例、无分配 null。这不是实际 Vulkan 运行。
- 固定 candidate ncnn patch tests 3/3，包括完整源码清单 identity、未知源码拒绝、3 个成功 allocate/10 个 free 位置保持。
- `git diff --check` 通过。
- 日志：outputs/allocation-cli-o1-v1/{contracts-test.log,patch-test.log,contracts-build-v2.log}。

## 正在执行与待完成

真实 ON CLI 在独立 build-o1/cli-on 配置成功，Release clang/system glslang，固定 candidate ncnn；配置 cgroup peak309.9MiB/swap0。完整构建 unit ernie-o1-cli-on-build，unified session18646，-j2、MemoryMax3G/MemorySwapMax0；日志 outputs/allocation-cli-o1-v1/on-build.log。构建开始后尚未确认完整链接，不能称真实 CLI 验证成功。OFF CLI 完整链接和真实 CPU 拒绝/失败路径、ON/OFF GPU 同输入输出、完整模型清理与身份验证仍待串行执行。旧 v2 allocator probe 成功不能自动覆盖新设备身份 hook 的实际验证。

## 完整链接后续核验

上述构建现已完成：ON unit/session18646 exit0，2m24.403s、峰1.4GiB/swap0；OFF独立 build-o1/off 复用已编译原始 ncnn 后启用 generator，session39751 exit0，1m1.663s、峰888.5MiB/swap0。两份 runner 已复制封存为 outputs/allocation-cli-o1-v1/ernie-image-{on,off}，identity.json 保存 binary/log/config/provenance/source inventory hashes。

实际 ON CLI 常规小合同30/30通过。该既有集合包含 --diagnose 设备枚举，未加载模型或执行 kernel；此枚举已经单独告知 root，不算正式 GPU 合同。随后专用 CPU FP32 missing-model 失败：generation_failed、原错误保存、initial/final instance 均 false、total=null、coverage_complete=false。第一次该命令漏传 precision，CPU请求被参数检查拒绝，日志 real-on-missing-model.stderr 保留；修正命令的 v2 报告单独保存。

实际 OFF CLI --metrics-json 在模型处理前明确拒绝，没有报告文件；nm 检查 OFF binary 不含 AllocationMeasurementSession 或 allocation_device_identified 符号。尚未运行新 ON/OFF 的实际 GPU 合同或 full-model，不宣称完整 O1/真实性能统计闭环完成。所有构建已退出，无待监督会话。
