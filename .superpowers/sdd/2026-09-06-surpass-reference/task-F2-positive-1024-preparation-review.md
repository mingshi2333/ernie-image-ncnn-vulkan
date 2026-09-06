# 固定1024 strength .5准备独立审查

对象130cf7347cd6d0ed7af575ee4630136523351561；只读tools/reference_img2img_positive.py、tests/test_img2img_reference.py及outputs/f2-positive05-1024x1024-v1准备证据。结论：未发现阻断这份精确准备输入的缺陷，可以在root明确GPU排队许可后执行有界真实参考/原生后缀。准备通过不是17边界实际运行通过。

独立核对10个input和10个prep-source完整SHA全部匹配prep-identity。source schema2 manifest实际完整SHA72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1，对应latent1×128×64×64、text bucket64、25text/36DiT。profile固定有效official encoder88f2e8b7...，不使用旧无效wrapper metadata。真实input.png为1024×1024，解码RGB逐字节等于固定input.rgb和已认证encoder输入。prompt实际31bytes `A red apple on a wooden table.\n`，包含一个LF，不按可视文本丢弃换行。

8步原schedule，strength .5保留绝对4..7四步，shift公式对base .5得到FP32 sigma .8。独立使用numpy显式multiply/subtract/add FP32分别运算重新生成524288个start元素，全部字节相同；没有借用validate_start作为唯一oracle。复制临时输入后修改一个start元素、同时更新其sha形成自洽伪声明，validate_inputs仍按独立encoder/noise mixture关系拒绝，错误明确reviewed FP32 mixture。已保存noise本身定义跨运行共同输入，不以跨框架同seed冒充同noise。

后缀形状审查：sequence=64×64+64=4160，constant0/1为[1,4160,128]，constant2为[4160,4160]；latent与四个prediction/step都[1,128,64,64]；text按真实IDs长度、paddedtext64；unpacked[1,32,128,128]、decoded[1,3,1024,1024]。required共6输入+8步输出+3末端=17，每个固定filename/dtype/shape/hash/元素数/finite都检查，initial必须等于冻结start hash。尚未产生真实suffix时这些仅合同，不能计为执行分母已通过。

run必须精确source manifest SHA与shape config匹配后才能调用reference，输出目录既存即拒绝；prepare也先拒绝既存目录。准备后的validate_inputs会绑定固定fixture内RGB/mean/packed/normalized hashes，不能仅修改自建contract逃过。执行时来源快照需要补全实际工具依赖及官方安装类身份；目前10个prep-source是准备记录，不能称完整hermetic执行闭包。大GPU可行性仍待实际guard，4160注意力长度不能从512×384成功推导为已通过。

独立运行tests.test_img2img_reference 9/9通过，另上述真实输入核验及自洽错误start负例通过。没有模型加载、没有GPU执行、没有修改作者代码。
