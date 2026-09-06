# O1 CLI allocation wrapper 独立审查

范围：`fb65395` 的 CLI wrapper、private allocation report、派生 ncnn device identity hook 和小型
tests，以及 `6aea9e5` 补充的 ON/OFF 完整链接与 CPU 证据。只做代码审查、已保存 JSON/hash 核验和
fake inference CTest；没有模型或 GPU 执行，也没有修改实现。

## 结论

未发现 Critical、Important 或 Minor 实现问题。此结论只覆盖 CLI 生命周期和报告边界，不关闭 O1
的真实 Vulkan/full-model 验证。

## 已核对合同

- `--metrics-json` 只在 ON build 接受；OFF 在加载模型前拒绝，且保存的 OFF binary 不含 observer
  symbols。报告路径在参数解析和 report constructor 检查，最终以 `O_EXCL`/0600 打开；write 循环处理
  EINTR/短写并检查 close。已有文件不会被覆盖。
- session 在输入图片准备前、generate 前创建。generate 抛错时 pipeline 的 RAII 已先展开；catch 才
  snapshot。成功时 image write/close 完成后 snapshot。wrapper 不创建或销毁 ncnn 全局 GPU instance。
- generation/image/report 三个失败阶段分开。主失败先原样写 stderr；报告失败作为第二错误追加。报告
  write 自身失败不会递归覆盖主错误或把部分文件宣称有效。
- `coverage_complete` 同时要求 hook available/valid、无 initial/final borrowed instance、设备身份齐全、
  全部 memory live bytes 为零、allocator 不 active 且无 live handles。没有 allocation 时 `total=null`，
  不虚构零峰值。冲突身份、observer 异常、borrowed instance 和残留 allocation 都 fail closed。
- total peak 是同时存活 VkDeviceMemory 总量；分类和 allocator peaks 单列，没有相加成伪总量。设备字段
  明确记录 process handle、index/vendor/device/API/driver/name 和 pipeline-cache UUID；没有把后者称为
  物理 device UUID。
- host clock 的 scope 明确是 `cli_generation_and_image_write`，formal speed/memory eligibility 恒 false；
  CPU RSS、GPU time 和 stage times 均为 null。JSON 对引号、反斜杠和控制字符进行转义，大内容不写
  stdout。

## 独立验证

`ctest --test-dir build-o1/cli-contract -R '^allocation_cli_cpu$' --output-on-failure`：1/1 CTest
通过，内部 13/13 cases，0.06 秒。保存的 `identity.json` 绑定 ON/OFF runner、build logs、CMake cache、
source inventory 和 patch provenance。实际 ON CPU missing-model v2 报告保留原错误，initial/final
instance 均 false、`total=null`、`coverage_complete=false`；OFF 拒绝没有创建报告。

## 未闭合

新 hook 尚未用真实 Vulkan allocation/full-model 路径验证设备 identity、所有 allocator generation 与
最终清理。旧 allocation probe 不能替代这一步。当前 JSON 也明确要求 frozen parent binding，不能直接
作为正式性能或显存预算结果。需要 root 串行安排同输入 ON/OFF GPU 运行并验证输出不变后，才能继续
O1 的真实性闭环。
