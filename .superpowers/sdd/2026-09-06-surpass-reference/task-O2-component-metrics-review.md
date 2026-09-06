# O2 component metrics 独立窄审查

审查b008bfc及修复a1486c1，只读源码和小CPU合同，无O2模型/GPU运行。结论：两项Important已在代码上闭合，可以准备独立实际对照；不能据此声称默认完整模型逐位等价或O2性能完成。

1. 原b008bfc的CPU/Vulkan text、streamed block及Vulkan head在extractor/command仍活着时reset Net，改变原生命周期。pinned ncnn option.cpp默认启用local pool，net.cpp Extractor::clear释放缓存，Net::clear先删除pool/weight allocator会产生生命周期风险。a1486c1为这些路径加内层作用域，extractor与command均在Net reset前销毁。各自原有command/extractor相对声明顺序保留；并非所有路径都具有相同相对顺序。CPU head/VAE原先已在内层作用域关闭extractor。
2. 原pipeline text分支只在25块全部返回后冲刷stats，第二块失败会丢掉第一块已完成记录；text各边界亦未标failed。a1486c1 param/model/extract三处补failed捕获，pipeline失败catch调用metrics.blocks后立即重抛，成功路径单独调用。不会同时走成功/失败两次冲刷；已完成块和当前已观察失败边界保留。

component/absolute_step/block标签从实际block索引及原schedule绝对step传播；后缀不重置绝对编号。record_component只向独立component_intervals追加，不加stage_times；失败中的未完成head/VAE父区间仍可能缺失，JSON声明partial_known_intervals。model_load_composite含load_model实际读取、解包、pipeline创建及上传/等待，extract包含host dispatch/内部等待；不是纯磁盘或GPU计时。未观察的销毁/拒绝边界不得补0。无公开API变化。

小CPU独立重跑：build-dev execution_metrics_cpu 1/1通过；build-o1/cli-stage-contract allocation_cli_cpu 1/1通过（13内部CLI用例）。这两项主要检查collector/JSON，未注入真实文本第二块失败；失败闭合依据控制流审查，不冒充模型级测试。

剩余非阻断口径：关闭collector时不分配详细记录，但新增若干Clock::now、Net堆对象形式仍存在；不能写“完全零额外成本”或“OFF二进制不变”。现阶段允许固定真实ON/旧参考输出对照准备，实际源/runner必须重新冻结。此审查未扩大为新性能实验。
