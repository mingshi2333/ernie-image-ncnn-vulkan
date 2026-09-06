# B2 fixed graph named weight role review

2026-09-06；只读审查根尚未提交的 tools/port_graph_contract.py、audit_port_weights.py 集成及 test_port_graph_contract.py。未改实现，未重新扫描学习权重，未执行模型/GPU。

结论：在固定六份完整 param SHA 的入口与当前范围下，未发现能使错误 learned-weight role 通过的阻断缺陷。独立读取 v6 audit/official-tensors.json 和六份小 param 得到859/859 named_graph_weight_match、gaps=[]。分母为 DiT36×11 + text25×9 + PE26×9+finalnorm + 三份 embedding/LM。text graph 声明958而实际956 producer，恰2 spare；其他图0。未定义输入、重复输出与逆拓扑边被拒绝。

审查确认：roles 先从固定 block 顺序、Q/K/V→SDPA、O→residual、前后 normalization、gate activation/product/down、DiT conditioning 和 PE cache 对应确定，然后用官方命名键检索 shape/value。不是从 hash 相等反推角色。Gemm constant B/transA/transB/C 条件与[N,K]存储轴固定；embedding和LM具备独立输入输出检查。正式图 pin 不允许通过替换 params 或绕线复用已有权重冒充同一图。核心输出仍明确不证明 kernel 语义、runtime conditioning 或端到端 parity，allowed_to_close_S=false。

部分scope独立检查：只传 pe/lm_head 行时，只review该图且只map1，expected_complete=859保持，S仍false；因此调用者必须读取 reviewed_graphs/count，不能把 gaps=[] 单独解释为全模型通过。集成将图缺口放在 graph_weight_mappings.gaps，而原扫描缺口保留顶层 gaps；两者都要看。顶层status仍unproven，不存在当前自动错误关闭S路径。

非阻断测试覆盖弱点：test_qk_gate_and_residual_rewiring_is_rejected 的 Swish gate→up mutation 引用了尚未产生的 up，首先被 Graph 的拓扑检查拒绝，因此该子例本身没有独立证明 gate-role 检查。复核额外把 gate 改接到已定义 q，确实由 MLP gate projection role 拒绝，生产检查有效。现合成fixture仅text，DiT modulation/PE cache有真实固定图正证据但缺独立合成负例；建议补充，不能把现test数量称完整语义覆盖。

执行：test_port_graph_contract+test_port_adapters 共15/15通过；独立859全映射、LM-only partial、已定义错误gate负例通过。证据 outputs/reference-port-v1/graph-review-v1/result.json 保存完整结果及三份被审查文件SHA。此结论不把旧v6权重字节扫描算成新执行，也不扩展到非固定param或完整图parity。
