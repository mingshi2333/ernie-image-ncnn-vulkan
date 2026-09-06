# F1 固定1376×768原生VAE实际对照

原生单组件实际通过，尚非完整1376生成。原授权launch/plan/controller不变，session50449 exit0，外层94.831849856秒，模型槽已立即释放；没有重试或覆盖。实际目录`outputs/f1-native-vae1376-preparation-v1`。

全[1,3,768,1376]、3,170,304个FP32输出独立重算，与原验证器完全一致：max_abs_error **4.76837158203125e-6** ≤ 原max gate **0.00041317281723022463**；NRMSE **9.349812403869265e-7** ≤ 原 **2e-5**。两侧所有元素finite。原生完整SHA `8c70c49e11fc59a21f75797babfe43ece9271652107f0646ca94662b8e4244a5`，官方 `e8c5daefae6b065680769560f7a91f10140cd38a9a10dacee3b9ac6a1634b12d`。569项冻结文件执行后完整size/SHA再次通过，实际复制runner保持dbd6d9d…；图仍只有两reshape变化、权重未改。

实际16GiB/swap0/cpu.max200000/100000、affinity12,14，内部ncnn4线程。1837连续采样最大间隔0.056464133秒；scope memory.current sampled峰 **5,895,925,760 B**，host minimum **12,355,031,040 B**；memory.events全部0，退出时仅controller自身PID2248772，随后外层返回。`/usr/bin/time`记录native wall92.12秒、MaxRSS5,734,740KiB；该RSS与scope采样口径不同，不能互换。控制器含认证/验证收尾94.756623711秒。不将此计为S/M。

`actual-evidence.json`完整绑定外层/guard/validation产物与重算指标，SHA `cbca2b658f0e5f1372a29979abe5ca32c2101d68e6c0401d10237aa660b9d86f`。没有GPU、PNG、BN/unpack或encoder；仅synthetic unpacked latent→decoder。

下一项最小工作：复用已准备4192-token图候选，在packed H48/W86、text64上做官方与native input-head（8输出）/output-head（1输出）单组件对照；随后同输入验证36-block串联，复用原text64和完整权重；组件通过后才接固定1376发展样例的完整官方/native轨迹。无需大VAE whole-pnnx，不读取正式72/15。生产shared source registry仍未添加1376，不能把现有通用shape planner当完整runtime已验收；最终固定实例登记需独审通过的source/shape关系。下一步不新增层层身份框架，沿既有冻结与资源guard执行组件。
