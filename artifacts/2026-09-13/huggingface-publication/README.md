# Hugging Face 模型发布

发布于 2026-09-13T17:04:26Z：[akashimio/ERNIE-Image-Turbo-ncnn](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn)，公开版本 `9924de97ebe85c540ce09e207142a6efee614be7`。

| 交付物 | 文件数 | 字节数 |
|---|---:|---:|
| 文生图共享包 turbo/ | 89 | 23,271,740,211 |
| 可选 PE 包 pe/ | 61 | 7,680,869,430 |
| 模型卡、许可证、来源清单等全部发布文件 | 157 | 30,954,256,107 |

两个包是已有转换产物的独立复制。本轮没有修改权重、模型清单、运行时代码或生产 ncnn pin。上传前对完整暂存包执行 Python 与原生校验；上传后核对全部文件身份和大小，其中大文件使用远端 LFS SHA-256，普通 Git 文件实际下载后计算 SHA-256。

公开后，匿名 API 与模型页面返回成功。使用项目下载器在新目录实际下载两组小文件及 PE tokenizer；另从最大权重读取首尾各 1 MiB，与本地字节一致。两个完整固定下载清单也对已有暂存包完成校验，主包通过下载器串联的原生校验。这里没有重复下载全部约 31 GB，也没有把包校验写成一次新的完整成图测试。

Git LFS 属性按实际远端存储逐路径整理，避免小型共享对象被误认为 LFS 指针。这一步只改变 .gitattributes、provenance.json 与 files.json，原权重和模型清单不变。

使用入口见 [模型下载说明](../../../docs/models/README.md)，固定下载清单为 [turbo-v1.json](../../../docs/models/turbo-v1.json) 与 [pe-v1.json](../../../docs/models/pe-v1.json)。README、运行说明和现有 Discussion 的模型下载入口随本次发布更新。同题实现与精度选择的来源核对见 [比较说明](../../../docs/COMMUNITY-COMPARISON.md)。

`publication.json` 保存发布身份，`package-verification.json` 保存上传前校验，`remote-inventory-check.json` 保存全文件远端核对，`download-checks.json` 保存匿名下载与完整清单检查。复核脚本在本目录，使用 `/var/tmp/ernie-hf-release-20260913-v1` 的本次暂存目录；未保存凭据或带签名的下载地址。
