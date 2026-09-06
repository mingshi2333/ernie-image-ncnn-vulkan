# D3 Linux 离线生成准备

新增 `tools/check_release.py` 与 `tests/test_check_release.py`，独立审查 `a881060`。16/16 小测试实际通过，含本机 bubblewrap 重绑模型、隐藏原目录、清除 Python 环境与不同网络 namespace。审查发现并修复隐藏 PAX metadata 未计入提取上限、可修改输入映射跳出准备目录，以及 tarfile.isfile 意外接受 sparse/contiguous 类型；均有拒绝回归。合法长 Unicode PAX 文件仍可提取。最初测试构造了超过本机单个文件名 255 字节的目录，修正为仍需 PAX 且合法的 150 字节文件名；没有修改运行工具以接受非法文件名。

固定 case 在 `outputs/d3-offline-img2img-1024-v1/case.json`。使用原始 D2 v2 archive `73153c3c…` 的完整 181 文件清单，安装 native CLI SHA `90c66f72…`，来源为已受审查的 D1 Clang Vulkan v2。不是复制开发 build-dev。模型为实际通过 fixed F2 strength.5 的 schema3 包，输入 PNG、带 LF prompt、保存噪声均复用原字节，固定 CPU Vector text / Vulkan FP32 四个后缀步骤 / CPU direct VAE。PNG 必须逐字节等于原 native SHA `d426ac2a82574ed330d1c0f56061f7a5aa335fc2fb87c5bf31083204a50c9e16`，不能看到差异后改门槛。

最终实际准备目录 `/tmp/ernie-offline-img2img-1024-v2`，独立认证 188 项文件，checker SHA `4f959a945ff7496f03084b59832ae47fda0e7b8235c0a1258667ce47a7de57f3`。v1 准备保留不执行。输出位于整个源码根之外；完整源码根在 namespace 中成为空 tmpfs，模型只读重绑到新的中文空格路径，结果单独可写，原始模型不移动。执行仅清洁环境下的 native CLI，没有激活开发 .venv。

实际执行仍 pending GPU 队列。限额 10 GiB / swap0 / 两核；host available 连续采样不得低于 3 GiB，每条命令 1800 s 上限。拟执行真实 model verify（前后）、help、diagnose、无 driver diagnose、缺模型、完整固定生成、拒绝覆盖原图。监控与 Python 在 namespace 外，系统库/驱动沿用本机；这不是最小 rootfs 或跨发行版 ABI 证明。速度、正式质量、所有 D3 负例/PE/其他平台及可再分发状态不由本切片关闭。
