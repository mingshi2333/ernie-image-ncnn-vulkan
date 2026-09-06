# D2 实际 D3 链接 map 独立复核

审查 `991a5f2`、`outputs/d2-linkmap-v1` 及 `artifacts/2026-09-06/linked-binary-dependencies`。只读 CPU12,14、小文件/完整流散列和独立 GNU map/ELF64/ar 解析；未重链接、未重新编译、未执行产物或模型、未使用 GPU。结论：没有发现阻断此次固定证据的 Important；本结果不构成公开再分发批准。

## 实际命令、产物和资源

独立用 shlex 拆原链接命令，去掉 shell 空操作前后缀，与 plan argv 比较：仅 `--dependency-file` 目的路径和 `-o` 目的路径变化，末尾新增 `-Map` 与 `--cref`。没有源文件编译参数，也没有构建命令。原35个链接依赖与新 link.d 解析后35个真实路径集合完全相同。

40个记录输入逐项重新读取大小/完整 SHA，全部等于 plan 运行前值和 result.post_sha256；包含原对象/归档、原 D3 binary、build.ninja/cache、编译器/链接器及依赖库。6个运行输出大小/SHA 也通过。新 `ernie-image.relinked` 为33,742,088字节，完整 SHA `d366c835fe8d8a7de06db804cae094a4b81b1c64ccdbb005551d8942aee7c267`，正是已独审固定离线出图 D3 程序。原对象和已测程序没有被改写。

实际 result exit0、failure=null、0.661426816s；preflight 真实 scope memory.max=4GiB、swap.max=0、cpu.max=200000/100000、affinity8,10。六份 samples 所见 memory.max/swap/cpu 控制一致，memory.events 所有值0、swap.current0。重新计算 sampled memory.current 最大143,155,200字节、host available 最低17,469,009,920字节，符合 result。该峰是整个 scope 的采样值，不是精确进程 RSS；没有把本次重链接作为模型性能测量。

## 独立分母与字节重算

独立解析时先按 marker 将 archive selection、discarded input、最终 map 和 cross-reference 分开。最终输入贡献仅接收带地址/大小和真实 object/archive-member 来源的记录，避免将符号和 discarded 段计入。读取 ELF64 section headers 的 SHF_ALLOC/type，并由 map 当前 output-section 名称对应 ELF section 分类，避免 TLS NOBITS 与普通 VMA 重叠歧义。

- included archive members348；最终正大小贡献 archive members348，direct objects9，无 included-only 成员。
- 最终输入段25,429条（含零大小）；357个实际正贡献来源逐项 positive_section_count / positive_map_bytes / alloc bytes / NOBITS bytes 与 analysis-v2.json **全部相同**。
- 357来源都有 SHF_ALLOC 贡献；31来源还含 nonalloc；0来源仅nonalloc。
- ncnn206、Rust80、glslang35、runtime16、pipeline6；PE/model-package/shape-graph/C++ tokenizer/libc_nonshared各1。
- alloc输入贡献21,531,404字节，NOBITS82,905，file-backed21,448,499；nonalloc10,165,061。它们是输入段贡献统计，不能解释为去重后的ELF大小、运行内存、符号精确所有权或许可义务。

初版仅按VMA求交的21,531,985多计581，已保留在 analysis.json 及对应 analysis-v1.py。最终 v2 用 section-name匹配得到本审查独立算法相同结果；原 map/binary 未改变。没有通过改输入掩盖统计错误。

## Rust 归档条件复用

独立 GNU ar parser 读取完整572成员，整包 SHA `6276b2862ebde256a0d1722d06f4f4c4767455b8b574be69641b05bb32af13c5` 与已审 delivery-dependencies-v4 完全相同。实际入选80成员：32个分别匹配32个已审 project crate 的 artifact身份，46个完整成员 SHA 匹配旧 runtime archive associations，2个本地 bridge 对象；无未解释的第三方归类猜测。46项中15来自libonig、18来自compiler_builtins，其余来自13份Rust运行库对象。这里只复用字节相同的Rust归档历史身份，未声称整个旧D1程序等于新D3。

## 封存与边界

manifest 的15项 raw_evidence 与12项 archived_files 均逐文件重算大小/SHA通过；analysis-v2脚本 SHA、analysis-v2.json和原 raw map/binary 绑定正确。解析器源码的 GNU regular ar / ELF64 x86-64 ET_EXEC 固定格式限制、薄归档/BSD名字拒绝与此实际输入一致。它是固定证据分析器，不作为任意链接格式的通用接受器。

`distributable=false`、`licenses_complete=false` 保留。库在链接依赖列表中出现与成员实际被提取是不同证据；未提取成员不自动豁免许可证。系统动态库仍是外部运行前提，精细 toolchain/vendor notices、Vulkan runtime、其他离线场景和 Windows/macOS仍未验收。本次审查已完成，无需重新运行已通过的 actual relink。
