# F1 1376×768 native 静态实例缺口（只读准备建议）

结论：现有数学 ShapePlan 可计算该尺寸，但没有对应生产静态实例、可信 source manifest 或官方 heads/VAE 形状证据。此次未读正式72/15样本、未导出/运行模型、未改代码。

尺寸应严格区分：输出 WH=1376×768；VAE latent WH=172×96；packed WH=86×48；image_tokens=4128；FP32 packed latent=528384元素/2113536 bytes。s64 总序列4192；s2048 总序列6176。目前后者超过6144上限，不能从512×384/s2048直接更换WH后执行。

## 现有挂点与硬限制

- 实际 builder 是 `tools/package_dynamic_model.py::build_shared_package` / `--schema3`，没有 materialize_shared_package.py。它消费已认证 portable static source，以完整runtime bindings建立CAS，不执行新图导出。当前仅1–2实例；新增第三个同时安装还需明确调整数量合同，不能仅往JSON加对象。
- `tokenizer/schema3_contract.json::source_manifests` 仅登记 ef988…(512×384/s2048) 与72bb…(1024²/s64)。Python builder/verify、`tokenizer/src/shared_package.rs` 原生选择器都按完整source digest/config校验；`tools/audit_shape_contract.py::PINNED_MANIFESTS` 另固定三份历史源。新实例必须有独立完整schema2 source inventory，再审核注册其SHA，不能从既有hash推导真实性。
- `src/model_config.cpp::reviewed_shape_config` 仅许可64²、512×384、1024²三组；`src/shape_graph.cpp::dimensions` 必须先过它，故现有模板实例化会明确拒绝1376×768。`ShapePlan::create` 的数学面积/整数检查不等于这些生产批准。
- s2048的6176还会被 `src/model_config.cpp` 两处、`src/conditioning.cpp`、`tools/package_model.py`、`tools/export_dit_heads.py` 的6144合同拒绝。下一小切片优先s64，不借修改所有caps绕过独立形状验证；如最终需要长文本，另审查6176及mask/RoPE/内存边界，再一致调整。

## 最小准备顺序

1. 以1024²/s64完整源为起点，保持25层s64 text param/bin及全部真实权重逐字节身份。空间变化不要求重新导出文本bucket；s2048需要独立已有文本模板，不能把s64改常量伪造另一个bucket。
2. 使用完整规范图哈希规则准备候选图：36块DiT的Gemm静态M、reshape12–15及unsqueeze20–21改为4192；input head reshape7 image_tokens=4128、text M=64；output head M=4192、slice image_tokens=4128、reshape5 W86/H48。精确规则在 `src/shape_graph.cpp` 和 `tools/audit_shape_contract.py`；只能动已列字段，权重不变，全图规范SHA必须仍匹配。先做候选fixture工具/合同，不提前放宽生产白名单。
3. 官方 input/output heads独立验证复用 `tools/export_dit_heads.py`：这里height/width为packed维度，应H48/W86、text64。该工具捕获官方embedder/conditioning/六份modulation，output包含大幅激活final norm/projection；通过 `tools/validate_dit_heads.py` 全部输出形状/finite/SHA及固定数值门槛。这仍是组件合成输入证据，不是整链质量。
4. VAE官方reference需要latent H96/W172→RGB H768/W1376。`tools/export_vae.py` 与 `tools/specialize_vae.py` 当前都将每轴限制128，因此目前连reference-only/两reshape specialization都拒绝W172；需独立固定shape扩展。禁止全VAE大尺寸pnnx重新求值（现工具本就拒绝latent面积>4096的非reference-only）。保留完整基图SHA，修改decoder reshape99 flatten=16512、reshape100 W172/H96，所有bin身份不变；执行官方该尺寸decoder及native CPU direct组件对照后才能升级固定入口。当前export_vae参考是synthetic unpacked latent，不包含BN逆变换/unpack；需额外证明 packed128×48×86→unpacked32×96×172、decoder BN eps1e-5和RGB写出。不要误用encoder eps1e-4；纯text-to-image本切片不需要新增encoder注册。
5. 官方source/revision/config、实际runner与执行依赖快照、每边界完整分母/固定gate和资源guard通过后，生成并审核完整新schema2源manifest，才同步Python/Rust registry及C++ reviewed_shape_config。随后以新静态CAS实例跑发展集完整text→八步DiT→VAE闭环；组件通过或planner通过均不能关闭正式F1质量验收。

资源排队由root安排。这里只指出可复用路径和先后依赖，没有执行上述准备命令或承诺该新尺寸峰值可容纳。
