# F1 schema3 validation 独立审查

结论：2 项 Important OPEN，未发现 Critical。当前真实参考 metadata 的分母完整、两层 source snapshot 均匹配；发现的是验证器 fail-closed 缺口，不能把它们反推成当前运行已使用伪造输入。未修改实现、未导入 torch、未散列完整包权重、未运行模型/GPU。

审查源 SHA：pipeline_package.py `82ac6d3724ec499d19e107669c5abe6fa733dfd77918dd03cda19bd6b2dd3c14`；validate_pipeline.py `19c51c4f9b82926b0a80fa7c05f53ac6e7371ddd05b947b4f46c16ae1f441f46`；test_pipeline_package.py `277c48bbb685412cc28165e3f70465406d1f6834fadeed8cead2c318ab28459f`。行号针对这些冻结内容。owner 后续修复需单独复核。

## Important I1 — existing oracle 的官方来源与所选 shared instance 未绑定

位置：tools/pipeline_package.py:26–43；tools/validate_pipeline.py:225–242。

shared package 自身会完整认证 CAS 与固定 source_manifest/config；这部分不是只比同名 config。问题发生在另一侧：validation_package 只要求 reference 非 None，就返回 selected source_manifest 身份；existing-reference 路径只比较 complete/prompt/steps/start_step/config、tensor 自洽 hash/size 和 PNG/initial hash。没有把 oracle 来源与 selected source_manifest 关联，也不验证 reference_environment 的官方 pipeline source/dtype 或 fixture source_sha256。

无大文件复现：读取当前实际 reference/fixture.json 的 metadata 副本，把 source_sha256 改为64个0，把 reference_environment 改为 dtype=float16/pipeline_source_sha256=64个0，其余字段不动。当前实际入口 predicate 仍返回 False（不拒绝）。这些字段随后没有任何读取校验，因此 tensor/PNG 原文件完全不需要重写或重散列即可通过后续相同数据检查。任意另一套同 config/prompt oracle 也只需自洽重签 tensor 便可进入验证，而 result.package_binding 仍报告所选固定 source_manifest。

建议：提供明确的可信 oracle binding（例如受版本控制的 fixture SHA → source_manifest/official source/precision 关联，或校验已有外部 provenance 链）；不能只让调用方再填一个自洽 source_manifest 字段。旧 fixture 缺少 package_manifest_sha256，需通过已有真实运行 provenance 补侧车绑定而非改写历史 tensor/manifest。生产包通过认证不等于对照 oracle 也已认证。

## Important I2 — 完整分母由可删减 fixture 自己决定

位置：tools/validate_pipeline.py:228–232、307–309、328。

新增 start_step==0 检查可以拒绝明确标注 suffix 的 oracle，但没有检查 outputs 长度是否恰好等于 steps、每步 prediction/step 键是否齐全，亦没有精确要求6个 inputs 和3个 final 键。最终比较列表完全由这些可删减字典生成，all(...) 不验证应有25项。因此把 complete=true/steps=8/start_step缺省0保留、outputs改为空列表后，入口不拒绝；后续只比较6 inputs+3 final，共9项，仍可能输出 passed=true 与 native_acceptance_eligible=true。也可只移除失败步的 prediction 键使其不参与分母。

建议：在 symlink/native 启动前明确验证完整 schema，要求类型正确的 complete/steps/start_step，len(outputs)==steps，每步精确 prediction+step，固定 inputs/final 集合及匹配形状/布局；将 expected comparison names 冻结并在判定时检查集合和数量。仅检查 result 中已有条目都通过不足以保证完整轨迹验收。

## 无 torch 的小反例

以下脚本直接编译当前 main 中真实 reference 条件 AST；不执行 main，也不导入工具及任何模型库。实际执行得到 actual=False、no-step-records=False、forged-source=False（False表示入口不拒绝）。后两者分别对应 I2/I1；后续代码只遍历剩余记录，来源字段不再被读取。

```python
import ast, json, pathlib, types
p = pathlib.Path('tools/validate_pipeline.py')
tree = ast.parse(p.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
block = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.reference')
guard = next(n for n in block.body if isinstance(n, ast.If))
expr = compile(ast.Expression(guard.test), str(p), 'eval')
f = json.loads(pathlib.Path('outputs/pipeline512x384-pe-fp32-v1/oracle/reference/fixture.json').read_text())
env = dict(reference_prompt=f['prompt'], config=f['config'], args=types.SimpleNamespace(steps=8))
for label, fixture in [
    ('actual', f), ('no-step-records', {**f, 'outputs': []}),
    ('forged-source', {**f, 'source_sha256': '0'*64,
        'reference_environment': {'dtype': 'float16', 'pipeline_source_sha256': '0'*64}}),
]:
    print(label, eval(expr, {}, {**env, 'fixture': fixture}))
```

## 已确认的正确边界

- 4项现有 tests/test_pipeline_package.py 全通过（0.008s）。它们覆盖 legacy WH、shared 必须 existing reference、多实例准确选择及 CAS verifier 错误传播；目前没有覆盖上述两项 oracle 反例。
- shared selection 先 verify_shared_package 后筛选 WH；其底层固定 source_manifest 全字节 hash、真实源 config、完整136 runtime binding、大小及 CAS hash、whole-graph允许列表均须通过。未发现仅凭相同 config 冒充 native source 或未认证CAS进入native的新增旁路。本轮只读此调用链，没有重复执行大包校验。
- 双实例不传 WH 拒绝、轴交换拒绝，只有已认证匹配实例进入 config。native command 使用 fixture config 的明确 WH，且此前已匹配 selected config。
- FP32/FP16/conditioning gate AST 与 HEAD 既有值完全一致：FP32 NRMSE.003/max .0002+.01*refmax/PNG MAE.1 max2；没有发现 gate 漂移。
- 当前真实 reference metadata：outputs长度8，每步prediction+step；inputs为initial/text/padded-text/constant-0/1/2；final为final/unpacked/decoded，即25项。没有发现此次传入了 suffix oracle。此项是 metadata 读取，不是本轮重新散列全部reference tensor。

## native/source 与 validation-source 分层

`outputs/f1-shared-native-worker-v1`：native snapshot 的213项 source、execution.json 的215项 validation_sources，逐项小源文件哈希均匹配；execution.native_snapshot_sha256 与 snapshot.json 文件 SHA 一致。两个source层的差异恰好为修改 tools/validate_pipeline.py，并新增 tools/pipeline_package.py、tests/test_pipeline_package.py。不存在把未进入 native 编译的 validator 修改说成 native runner 更新的必要性；两个身份层已经分开记录。

实际 command 指向 `validation-source/tools/validate_pipeline.py`；Python 正常 script 目录查找导入相邻 validation-source/tools，prepare_block ROOT 为 validation-source 根。代码收集 scripts 使用 Path(__file__).parent，而非资产根，符合这次封存路径。run_guarded.py 没有添加 live tools 路径或运行时模块替换。metadata 中两runner SHA 为 image `6a3481e558a003e3f6fad6d8a9bc2ff903216d539c7ce2b86b352c19befaecbc`、denoise `f16a9689615b3023cb83f24404dda1a4f551ae934b22bd31483602904c3e55a2`；本轮按限制未重新读取大二进制，仅核对source与metadata链接，不能把该SHA称为本轮独立重散列结果。

scripts 目录封存所有 *.py，并不意味着全部模块都实际导入。source snapshot 的目录完整性与编译/导入依赖覆盖是不同结论，应继续保持此前报告的表述边界。当前运行未结束，本报告没有给其最终质量/性能 verdict。

## 第一轮修复复审

执行前路径的 I1 CLOSED、I2 CLOSED。独立运行 pipeline_reference 6项 + pipeline_package 4项，共10/10 PASS。实际封存fixture通过 full_reference_contract，返回25；原 outputs=[] 反例现在被完整合同拒绝，原 source_sha256=0 反例被 whole-fixture registry 拒绝。明确 suffix/bool step、缺键、步骤长度、文件顺序、shape/dtype/ID 等已检查。registry 的 8d7638... → ef98859... 映射亦与 artifacts/2026-09-06/native-vector-pipeline/result.json 的实际二元身份一致，不只是凭字段命名推定来源。旧已运行验证器未改写；此处为修复后独立 metadata/代码复审。

### 新增独立审计 I3 Important — shared manifest 的自洽绑定仍不等于固定 source 认证

位置：tools/pipeline_reference.py:81–92；tools/collect_parity_evidence.py:60–65（该轮内容）。

新 reviewed_shared_reference(path,binding,package_manifest) 正确认证 oracle 整份SHA，但 shared package侧仅验证 manifest hash与binding一致、selected config与fixture一致、runtime_bindings与binding一致。没有读取并认证 source_manifest 对应CAS对象，也没有验证其完整files与 runtime_bindings一致。独立 collector 只调用此函数，未调用 verify_shared_package。

实际小反例：使用真实已registry认证的fixture，临时manifest只有schema_version=3、一个instance，source_manifest_sha256写可信ef98859...、config复制fixture、runtime_bindings={'invented-file':'0'*64}；没有objects目录。binding使用同一runtime_bindings并重算 shared_manifest_sha256。当前 reviewed_shared_reference(...) 成功返回reviewed身份。collector前置manifest hash也只来自同一result，可同步重签。故该独立审计不能证明“真实共享绑定来自固定源”，即使入口validator本身仍有完整包验证而不受影响。

建议最低限度独立认证 source CAS metadata：根据可信source_manifest SHA定位对象，核对其原始bytes SHA，再要求selected config与完整runtime_bindings严格等于固定source manifest的config/files，并检查完整预期库存和声明大小。若独立audit不重新散列全部weight bytes，应明确标注“source metadata绑定已验证；当前CAS内容认证沿用/另行执行”，不能称已独立重新认证所有CAS。原始包无需修改。

无torch/无权重的复现核心：

```python
import hashlib, json, pathlib, tempfile
from pipeline_reference import reviewed_shared_reference
oracle = pathlib.Path('outputs/pipeline512x384-pe-fp32-v1/oracle/reference/fixture.json')
f = json.loads(oracle.read_text())
source = 'ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59'
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d)/'manifest.json'
    instance = dict(source_manifest_sha256=source, config=f['config'],
                    runtime_bindings={'invented-file':'0'*64})
    p.write_text(json.dumps(dict(schema_version=3, instances=[instance])))
    binding = dict(source_manifest_sha256=source,
                   shared_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   runtime_bindings=instance['runtime_bindings'])
    print(reviewed_shared_reference(oracle, binding, p))  # 本轮修复仍接受
```

I3 OPEN，已即时报告 root；本轮无实际共享包损坏证据。

## 第二轮修复复审 — I3 CLOSED

重新读取当前实现：reviewed_shared_reference 在绑定 oracle 后，强制读取 objects/<registry可信source SHA>，拒绝 source/objects symlink、缺文件及大于4MiB文件，验证真实原始 bytes SHA，并比较 schema2/portable/config/完整files。所选 bindings 不能再由新的自洽metadata替代固定source inventory。

独立重跑原真实registry反例的两个阶段：缺objects被 `Missing bounded source manifest object` 拒绝；将实际可信39KB量级source manifest复制到临时objects后，仍带 invented-file 的binding被 `Shared instance differs from the pinned full source inventory` 拒绝。真实共享包的轻量helper正例通过，oracle仍为8d7638...。未读取模型weight文件。

职责现在明确分开：轻量helper认证whole-fixture registry与选定固定source manifest的元数据绑定，不声称重验所有权重；collect_parity_evidence.audit 的schema3分支已先实际调用 verify_shared_package(package)，再调用helper并检查完整比较分母，因此完整collector执行时会重新认证全部CAS bytes。该collector全量调用在本轮只做源码复核，没有因运行中资源约束再执行一次大包重散列。

11项小tests再次独立通过（0.003s）。原I1/I2关闭状态保持，I3现关闭。当前修复后的独立audit与旧冻结运行validator身份仍必须区分，不能改写已启动job的source snapshot来表示它使用了修正版本。
