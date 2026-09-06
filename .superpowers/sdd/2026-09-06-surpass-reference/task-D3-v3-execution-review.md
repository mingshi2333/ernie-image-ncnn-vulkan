# D3 v3 新归档与离线执行独立审查

当前结论：准备与安装/构建身份无阻断；完整生成仍由 root 执行，结果待复核。本次只读 CPU 12/14 哈希/元数据审查，没有模型/GPU执行。

## 执行前身份

- 新 case 与 `/tmp/ernie-offline-img2img-1024-v2/case.json` 全键比较：仅 `archive` 与 `inventory` 变化，model、inputs、expected_png、request、hidden_paths、resources 均保持原样。期望 PNG 仍是通过正式固定门槛的发展样本原文件 d426ac…，不是依据本次输出重新确定。
- 新 archive SHA `d5810af124bac9ebc8b02874e03dfb2a2dd0f5f8b1ede6bb807eb1d2265582d1`，11,193,462 bytes；inventory SHA `9081c867244b45ed8b9422314d486242415e82e4816a5353cf77932d1d8b6610`，30,118 bytes。case SHA `a10712f9a85c098a033c1abcf95a1d7d1f5fe69ad825becfb1cb47e8f0213e28`。
- 逐项独立重算 `/tmp/ernie-offline-img2img-1024-v3/preparation.json` 的全部188项SHA；tar全部181 regular payload的大小/SHA/完整集合，以及候选目录和实际解包目录对应文件SHA，全部一致。不重散列大型模型包；model manifest 与原case不变，完整模型前后验证由冻结实际执行流程承担。
- 实际执行 CLI `d366c835fe8d8a7de06db804cae094a4b81b1c64ccdbb005551d8942aee7c267` 与已审新安装前缀/33binary Vulkan build证据相同；checker仍为原已审 `4f959a945ff7496f03084b59832ae47fda0e7b8235c0a1258667ce47a7de57f3`。release.json内嵌 Vulkan build identity/result 与 `outputs/d3-runtime-build-vulkan-v3` 两原文件内容完全相同。
- 源身份仍 `3a112e7b9df9bb5be02ff21529773966a596d6563dfb694bf50892cec9fa3a27`，288源已在无驱动审查全部认证。新 CPU build日志三份和33binary亦独立重算匹配，实际5/5针对CTest无skip通过（pipeline API、request validation、GPU context CPU分支、CLI、installed consumer）。Vulkan6/6及新安装75文件已在 task-D3-missing-driver-review.md认证。

归档保持 local_review_draft、distributable=false、无公开URL，不含模型，既有许可/平台限制未被小合同或准备成功消除。旧v2实际无驱动失败保持历史，v3实际最终PNG及进程/资源结果未在本节宣称通过。
