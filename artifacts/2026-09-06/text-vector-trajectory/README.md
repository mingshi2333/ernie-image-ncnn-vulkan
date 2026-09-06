# 文本 down projection 的完整图像诊断

相同官方参考、相同512×384模型、315个有效token、相同已保存FP32噪声、8步Vulkan FP32 DiT与CPU FP32 direct VAE。这里只将条件输入换为25层原生诊断产生的vector down候选；未替换官方embedding。候选SHA为433ae46eaa1c8b831fa273f68b8345646d6ddfdb56695b4d3286b4964bca294a。

原完整native PE→text→image历史24/25（decoded最大误差失败），本次保存候选输入诊断25/25且PNG通过。runner、执行目录中封存的62个Python脚本、固定门槛、输入复制与token身份由轨迹工具重新审核；快照不表示所有62个脚本均被导入执行。正式 native_acceptance_eligible=false：这次没有在生成程序内部执行PE/text，不能替代原生整链验收。

| 边界 | 原 NRMSE | 候选 NRMSE | 原最大误差 | 候选最大误差 |
|---|---:|---:|---:|---:|
| text | 5.99987048e-06 | 3.136188651e-06 | 0.003784179688 | 0.0009155273438 |
| final | 0.0001882558957 | 9.332130594e-05 | 0.005950927734 | 0.002828598022 |
| decoded | 0.0001002783495 | 5.303196947e-05 | 0.01110547781 | 0.005784600973 |

候选PNG MAE=0.001559787326，max=1，原生PNG量化逐位一致。decoded固定max门槛0.01099487681388855未调整。

本次native总时长712.131秒、RSS 1548376KiB包含权重检查与trace；当前还并行CPU PE和下载，不能作为受控性能比较。候选文本vector在独立原生实验中增加计算成本，后续速度验收必须如实纳入。

trajectory.json同时保留原native失败、saved official-text诊断与saved candidate-text诊断三条轨迹。官方text注入曾使某些NRMSE变差，本次candidate在所列final/decoded边界同时改善；没有“文本是唯一误差来源”的结论。原始大tensor/PNG留在outputs，不进入Git。
