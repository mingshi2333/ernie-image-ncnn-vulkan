# D3 离线安装运行检查工具与实际准备独立审查

结论：本轮发现均已修复；允许固定development case进入Root顺序GPU执行队列。只读审查tools/check_release.py/tests/test_check_release.py以及实际准备，未运行ERNIE模型/GPU，不能称D3实际生成已通过。

最终工具SHA `4f959a945ff7496f03084b59832ae47fda0e7b8235c0a1258667ce47a7de57f3`。独立16/16 tests通过，包含真实小bubblewrap实验，无跳过。最终不可变准备 `/tmp/ernie-offline-img2img-1024-v2`，preparation SHA `7d099bf5742b5a8730000ab7a7a33ff08a0b02bce8baeed9b3aaa2f0e7805b32`，188项准备文件hash逐项吻合。v1保留未执行。

## 发现与闭环

1. 原实现直接tarfile解析，在member.size校验前PAX/GNU扩展已经解压读入，能绕过payload大小上限。独立小反例：192字节声明payload、1397字节压缩包，含1MiB隐藏PAX comment，原工具接受。修复增加gzip原始512字节header预扫，解释metadata前限制单项64KiB、累计4MiB、总解压量1GiB+16MiB、header数及尾零填充。限制PAX键，拒绝sparse/link语义；合法长Unicode path测试仍通过。
2. 原run校验prep.files，却从未约束的prep.inputs取实际CLI路径，改成外部同bytes文件可以绕过准备树身份。修复从固定case和复制且认证的files.json重建精确文件集合和canonical输入路径，逐项检查size/hash；外指映射和删除runtime清单记录负例均拒绝。
3. 后续审查发现TarInfo.isfile()包含GNU sparse和contiguous类型，不能用于“仅普通文件”契约。预扫和提取均改为显式REGTYPE/AREGTYPE，并新增S/7拒绝测试。隐藏源目录证明也补目录存在及tmpfs文件系统检查，避免ls错误空stdout伪装空目录。

## 实际准备核验

case为已独立验证的1024×1024、strength.5、八步总schedule、CPU text vector/CPU direct VAE/Vulkan FP32，PE关闭；使用此前F2实际图像作为固定PNG目标。归档仍delivery-candidate-v2，SHA `73153c3c9100dd078659cec85709c887b28fcffb716816990589c7f52af9ad81`；files.json SHA `51f2dae95382eb929fad540dd4fb8373e857f4ad8a95fdec97e7515665bf2768`。

独立逐文件核验181份payload的完整size/SHA和精确集合，迁移目录为“移动 installation”，4份映射文件分别为31字节含LF prompt、2097152字节保存noise、原PNG和expected PNG。expected PNG SHA `d426ac2a82574ed330d1c0f56061f7a5aa335fc2fb87c5bf31083204a50c9e16` 来自此前实际F2原生结果，不是工具生成的自我参考。模型manifest2c1d与既存固定schema3一致。v1/v2 case、inventory和payload未变，仅检查器及准备身份更新。

我另在实际v1准备目录执行纯/bin/sh隔离试验：原整个仓库根目录为空，重绑“模型 shared/manifest.json”可读且SHA2c1d，VIRTUAL_ENV/PYTHONPATH为空，网络namespace不同于父进程。未执行原生binary、Python模型或GPU。v2加强的namespace证明代码与同类真实小测试通过。

## 证明边界与执行待办

正式run必须使用v2冻结检查器，并在确切10GiB memory.max/swap0 cgroup内执行，持续50ms host≥3GiB守卫，timeout为每个受监督命令的上限。CLI生成无trace；缺GPU、模型缺失与既有输出拒绝是单独命令。PNG检查为RGB8容器CRC/shape加与已认证native PNG全文件字节相同，不能单独将容器检查当完整通用PNG解码验证。

源码隐藏覆盖整个仓库（含原build和项目venv），模型先只读重绑再隐藏原目录。宿主/usr等运行依赖仍可见，Python环境变量清空并直接启动native CLI，不声称机器不存在Python或提供对恶意程序的完整系统沙箱。新网络namespace证明普通IP网络隔离；不是对host Unix socket恶意逃逸的安全审计。模型无symlink并通过原生verify-model前后全验；case/archive/inventory的可信来源仍由冻结父流程与既存D1/D2证据提供，不靠JSON布尔值创造认证。

当前v2为prepared_not_executed且无result.json，资源参数是计划，不是本轮测得峰值。接下来真实help/diagnose/model验证/生成/PNG对照/拒绝分支及cleanup记录通过后才能关闭该固定离线交付样本。工具始终distributable=false，full_quality_gate与speed_comparison不评估；不能泛化formal、性能、跨平台或发行许可。
