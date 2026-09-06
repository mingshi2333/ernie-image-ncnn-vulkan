# F2 strength=1 端点准备独立复核

审查fe08621，结论：固定1024×1024开发样本准备通过，可以进入root已授权的顺序执行队列；这不是实际生成或29边界质量通过。只读CPU审查与小测试，无模型/GPU作业。

证据 `outputs/f2-positive1-1024x1024-v1`：input-contract SHA `a269a01cae3ad669d985caf5b040408335722cc87b4b706ed668aa34b3ff3943` 独立重算吻合；13份prep-source的SHA与字节数均核验。8份input.png/input.rgb/prompt.txt/saved-noise.f32/out0.f32/out1.f32/out2.f32/encoder-fixture.json与已验收strength.5样本逐字节相同，包含精确LF prompt与固定88f官方encoder fixture。

start-0.f32完整524288 FP32元素的字节与saved-noise.f32相同。原8步sigma表完全保留，start_step0、denoise_steps8，调用官方reference明确传0；suffix验证按绝对编号0..7要求8对prediction/step，6输入+16过程+3最终=25，另encoder3+noise=29。预期分母不是当前实际结果。原shape registry仍只接受已固定512×384和1024×1024；对外准备/运行入口的strength限制为.5和1，没有开放任意strength/shape。validate_suffix低层函数接受显式start_step，正式run先经固定validate_inputs；不能把低层函数单独当成任意输入可信入口。

独立13/13 unittest通过；真实旧strength.5 inputs与已有suffix17重新验证通过；复制小输入后将strength改为.75/0/1.1均被拒绝。端点噪声末元素改变一ULP的测试被拒绝。未重建大模型或复算权重包。

prep-identity仍executed=false，status为pending review/execution，没有result.json或实际模型输出，状态诚实。13个扁平prep源码是审查快照，不是接下来完整官方执行所需hermetic工作目录；实际执行需另冻结完整目录/runner并保留身份。资源10GiB/swap0/host≥3GiB/50ms/1800秒目前是计划，不能当实测。下一次post-audit需扩展到29项：现有strength.5专用21项工具不能未经修改用于此端点。
