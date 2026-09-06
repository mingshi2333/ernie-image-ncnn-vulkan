# D2 独立下载与清单切片

本片实现 `tools/download_model.py`、`tools/release_manifest.py` 及两个专属测试模块。未编辑推理CLI、包schema实现、root CMake；没有网络发布、外网下载或大模型运行。

## 实现与范围

入口：`python3 tools/download_model.py --manifest FILE --output DIR`，只有Python标准库依赖，可连同同目录release_manifest.py独立使用。清单schema_version=1，必须包含固定40/64位小写hex model_revision、graph_schema、required_capabilities、conversion的source/script_sha256、license的identifier/source/notice，以及非空files。每项严格path/url/size/sha256，拒绝未知字段、重复JSON键、bool冒充整数、绝对/穿越/Windows保留名/反斜杠/冒号/大小写冲突/父子文件冲突/内部断点名冲突。URL必须HTTPS（本地fixture允许HTTP loopback），初始URL路径必须含该固定revision独立段；拒绝认证信息和fragment。重定向允许HTTPS CDN，禁止TLS降级到外部HTTP。许可记录只是可审查声明，不证明再分发权限。

下载前显示总bytes和目标目录；顺序1MiB流式，重新完整SHA认证已完成文件后跳过。未完成文件使用`.part`与绑定精确文件清单/ETag/Last-Modified的`.part.json`。206严格检查Content-Range的起点、末端、总长；200将断点从零重取，不追加；版本变动明确失败并保留旧断点。截断/HTTP错误/错误尾字节均不能发布正式文件；完整partial也必须重新完整SHA后才完成。SHA通过后通过同文件系统hard-link原子创建最终名，再删除part，保证不会覆盖并发创建的既有最终文件。该实现要求目标文件系统支持硬链接；不支持时明确失败保留已认证part，不静默退回可覆盖rename。

拒绝输出路径祖先或目标/断点符号链接、特殊文件、共享硬链接断点；拒绝已有错误最终文件并保持其字节。输出目录用O_EXCL `.download.lock` 防同工具并发；正常异常清除自己的锁，不删除另一个进程的锁。进程被强制杀死留下锁时，操作者应确认旧进程已退出后处理；工具不按年龄擅自抢锁。可信本地输出目录是前提，未声称对恶意本地用户持续替换父目录的race进行沙箱隔离。

容量检查采用保守剩余完整文件大小（以允许Range200完整重取），下载响应确认后再次检查。这可能在仅足够有效206后缀空间时提前拒绝，但不会为了省空间假定服务器一定支持Range。OS实际写入失败继续向上传播并不发布目标；这不是预留磁盘配额。

## 验证

`python3 -W error::ResourceWarning -m unittest tests.test_model_download tests.test_release_manifest`：20/20通过，约0.52秒。ThreadingHTTPServer实际loopback HTTP测试覆盖新下载、完整文件skip、Range/If-Range续传、200重启、ETag版本变化、截断后再次续传、错误末字节、错误Content-Range、503失败、完整partial无网络恢复；文件系统测试覆盖容量不足（注入disk_usage）、错误既有目标保留、符号链接、硬链接断点拒绝、锁互斥、原子发布竞争不覆盖。清单测试覆盖path/revision/url/types/duplicate keys。CLI help实际通过。早期503测试暴露HTTPError连接ResourceWarning，已显式close并用警告转错误重跑全部通过。

## 未完成项

返回结果仅 `download_integrity_verified` 与 `model_semantics=not_verified`。没有调用native `--verify-model`或PE语义校验，不宣称规范化全权重审计、模型质量、运行能力或完整D2结束。暂无获确认的正式发行资产URL，未生成虚构manifest；现有example.invalid仅测试字符串，不是发布地址。平台归档、许可证法律依据/完整notices、发行草稿与迁移回归由后续切片完成。旧组件Range/safetensors工具与peer下载历史已阅读作参考，保持不变；此工具不依赖其中的专用模型提取逻辑。

## Root 独立审查修复

初始67cdafb保留。root独立20/20测试之外复现uppercase内部文件/锁名与Windows禁止字符仍被接受：LOCK.PART、.DOWNLOAD.LOCK、a?.bin、a|b.bin。此为清单路径契约缺口，现按casefold检查内部保留后缀/名字，拒绝Windows禁止字符<>:"\\|?*，补CONIN$/CONOUT$/CLOCK$与COM/LPT superscript1/2/3设备别名（含扩展名和设备基本名尾空格）。测试包含同目录payload/data.bin与payload/DATA.BIN.PART冲突。22/22原有加新增测试通过；仅Linux上执行路径合同测试，没有虚构Windows实际下载验收。
