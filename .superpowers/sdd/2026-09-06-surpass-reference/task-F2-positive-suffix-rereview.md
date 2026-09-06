# F2 positive suffix 修复独立复审

审查 bbdba1c 和后续260f222，限定原I1/I2与复审发现的canonical filename缺口。静态+独立小CPU反例，无模型/GPU执行，不修改被审实现。

## I1 CLOSED

bbdba1c新增真实RGB/三个encoder边界对固定受信fixture的hash绑定，明确BN与原8步sigma数组，重算FP32 start逐字节验证。原来全零start重签、真实out2换成全零、sigma/BN改写三个反例均已被拒绝。

初复审另发现contract可使用alternate noise/start文件，而run固定复制saved-noise/start-4：合法alternate混合被接受，但输出边界复制旧bytes。已立即报告；260f222要求noise/start/prompt canonical文件名，独立原反例现被 `Positive-strength input filenames are not canonical` 拒绝。这关闭已复现的输入与输出副本不一致路径。

## I2 CLOSED（原完整分母问题）

bbdba1c新增validate_suffix，核对内存返回fixture与落盘一致、prompt、固定512×384/2048文本bucket、steps8/start4、恰好prediction/step4/5/6/7、全部17必需张量的名字/shape/dtype/size/hash/finite、initial=start及PNG hash。固定包manifest/config亦在run调用前绑定。原complete=true空outputs/final反例被拒绝。已有真实历史fixture的结构仍为17项，但结构完整不代表它是正确请求的oracle，见下节。

该helper是输出结构与bytes合同；它不独立认证官方执行环境/整图数学。固定官方执行source、worker和实际导入路径仍须作为发行证据独立保留，不把source_sha256自报字段视为执行证明。

## 必须保留的新历史身份纠正

复审期间根/paired定位了原wrapper `.rstrip("\\n")` 去掉LF：输入文件和native保留末尾LF，旧官方suffix prompt没有LF；对应最后token为native1626、旧oracle1046。这不是encoder或DiT数值失败的可信比较输入。

260f222改为精确read_text，准备contract.text也保留LF。独立检查当前request prompt与新official-text fixture都是 `A red apple on a wooden table.\n`，新文本末token1626。旧官方目录完整迁移到 official-invalid-prompt-trimmed。旧suffix按当前正确prompt送validate_suffix被拒绝；仅显式使用其旧prompt可通过17项结构验证。

因此初审“实际17张量/混合hash正确”的结论只覆盖这些边界bytes和分母，不意味着原prompt-to-output配对正确。原text NRMSE .03626/PNG max86不能用来判定正确输入下的encoder或pipeline质量。新的正确prompt完整suffix仍需根后续执行/验证，不能凭此次修复或新文本CPU结果宣称其质量通过。

## 独立复现与测试

保存 q2-review-scratch/f2_positive_rerepro.py，不覆盖初审反例脚本。实际结果：

- arbitrary_start REJECTED：Saved start does not equal the reviewed FP32 mixture。
- wrong_encoder_bytes REJECTED：Official encoder normalized bytes differ。
- wrong_sigma_and_bn REJECTED：Recorded encoder or schedule contract differs。
- alternate_names REJECTED：Positive-strength input filenames are not canonical。
- historical_trimmed_prompt REJECTED：Suffix fixture identity or prompt differs。
- historical_structure_only_denominator 17（明确传入旧prompt，不是当前请求验收）。
- empty_suffix REJECTED：Suffix must contain four complete denoising steps。

tests.test_img2img_reference独立8/8通过，0.012秒。初次复跑时历史official目录恰好被迁移，实际正例读取FileNotFound；随后按新历史路径修正独立脚本并完成以上验证，没有重写历史数据。
