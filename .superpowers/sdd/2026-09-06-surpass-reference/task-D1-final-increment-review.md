# D1 收尾增量独立审查

相对c708d14已关闭的安装修复，审查root尚未提交的workflow/presets/source inventory/CLI与fixture测试增量。只读源与已保存日志，CPU小tests；没有构建或GPU执行。结论：本增量未发现未关闭Important。远程CI和未实际运行平台不算通过。

## 真实CPU preset v5

`outputs/d1-preset-cpu-v5/result.json`明确configure/build/test三项exit0，墙钟分别6.5073/238.5830/2.1387秒。test.log实际27/27通过，包含installed_cpp_consumer。identity使用cmake --preset linux-cpu、cmake --build --preset linux-cpu、ctest --preset linux-cpu --no-tests=error，固定ncnn6a1bf000.../glslang780b6a...，资源记录memory.max4GiB/swap0/2CPU，结果source_changes_after为空、ncnn tracked变化为空。

独立重新核对 `outputs/d1-preset-source-v5/source-identity.json` 的267个冻结文件，全部SHA一致。新增CMakePresets.json、.github/workflows/build.yml、tests/fixtures/text-s32.ncnn.param均在此冻结清单内；当前source_inventory函数也确实枚举这三项。使用这份实际CPU-only构建runner独立运行tests.test_cli+tests.test_text_bucket，34/34通过，没有skip。

v4历史25/27失败保留：两个旧CLI假设默认Vulkan可用，所以CPU-only build先报能力错误，无法到达模型校验；以及冻结snapshot缺旧artifact中的32-token图。修复没有放松模型数学门槛。矩形请求测试现在显式CPU FP32，依旧要求Cannot open model package；BF16测试从真实runner --diagnose读取vulkan_compiled与gpu_count，分别要求Built without Vulkan / No Vulkan device / Cannot open model package，不能把任意非零错误当通过。能力分支diag本身必须exit0且布尔字段有效。

新tests/fixtures/text-s32.ncnn.param完整SHA912dc8a3f8839cca841004f8c5afb4df5c9d5015d4fedbb4deb77d67a62e2523，与原artifacts/2026-09-05/pipeline/models/text-s32-v1/block-00/text.ncnn.param逐字节来源一致。test只是将依赖从旧artifact移入可冻结fixture，仍对32/64/2048独立图做已审查维度归一后的固定graph hash，norm epsilon和GQA头数负例保留。

## 安装与CI范围

安装损坏包检查现在除了退出1还要求Unexpected schema-3 fields，防止错误启动/错误能力失败冒充schema拒绝。沿用已复核的恶意ncnn_DIR回归、移动prefix消费者和CLI验证；CTest通过ERNIE_INSTALL_BUILD/CONFIG绑定当前build，非skip。普通CTest installed消费不启用bwrap，其范围不同于此前真实--isolate-source的CPU/Vulkan v2证据，不能混称本次27项全在沙箱执行。

CMakePresets版本3要求CMake>=3.21，两个Linux preset固定静态SDK、generator/tokenizer、metrics OFF、system glslang OFF、INT8/weight quant OFF，两job；CPU/Vulkan只差显式能力开关。隐藏父preset有Linux host条件，没有声明Windows/macOS preset。

workflow在Ubuntu24.04的CPU/Vulkan矩阵启用SDK，并固定Rust1.98.0，现有pinned checkout SHA保留、credentials=false。CTest含实际installed_cpp_consumer，之后另install/help；新增download stdlib loopback测试。仓库远程workflow未由本审查运行，固定Rust安装、Ubuntu软件Vulkan及35分钟预算能否实际完成仍须远程日志，不从本机Linux结果推导。root后续同源Vulkan preset仍在排队，不能把此前不同源安装Vulkan-v2替代本次preset验证。
