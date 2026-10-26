# DFlash2 与 DSpark：运行时成熟度、公开对照和 AMD 适用性

核验日期：2026-10-22。仅检索公开主来源，没有下载权重、运行模型或执行 GPU 实验。目标场景为 Ryzen AI Max+ 395（gfx1151）低并发本地展示，以及现有 Qwen3.6-27B AWQ-vLLM 工程。

## 结论先行

不能把两者排成所有场景固定不变的第一、第二名。DFlash2 在作者发布时的 Qwen3.8-27B/H200 同控实验中明显优于当时的 RadixArk DSpark；换成后续更好的 DSpark checkpoint、不同量化和不同 kernel 后，公开结果可以变成接近持平，甚至按任务分别胜出。

对现有项目的决策顺序应是：**匹配 target 的有效权重 → 当前 runner/量化/ROCm 能否正确执行 → 每轮提交 token 数除以每轮真实耗时 → 长上下文与连续服务稳定性。** 算法名与单独接受率都不足以决定展会方案。

如果继续固定当前 Qwen3.6-27B，且已经核验相匹配的 DSpark 草稿，而 DFlash2 候选的 target/feature/权重语义尚不明确，先实施 DSpark 的小规模 A/B 更有把握。如果愿意另外建立 Qwen3.8-27B 路线，DFlash2 应进入第一批对照；它已有直接 gfx1151/Vulkan 运行与深度测试，但这些结果不是我们的 Qwen3.6/AWQ/HIP 复现。

## 1. 两者的上游实现成熟度

### vLLM

DFlash2 的主体 `#52816` 于 2026-08-21 合并进 main；v0.28.0 于 2026-08-26 正式发布并列出 DFlash2。实现包含 model、selector、Triton path walk 与测试，并走 Model Runner V2：旧 V1 DFlashProposer 缺少 candidate selector，不能靠改模型路径直接得到正确 DFlash2。[主体 PR](https://github.com/vllm-project/vllm/pull/52816)、[v0.28.0](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)

`#53797` 于 8 月 27 日合并，提供 speculators 格式的 DFlash2 loader；明确配置算法名 `dflash2` 在 vLLM 中映射为 `method: dflash`。服务参数写 dflash 并不表示它只运行第一代 DFlash。[格式支持 PR](https://github.com/vllm-project/vllm/pull/53797)

判断：DFlash2 已有正式主线服务支持，不能称为纯研究原型；但“正式发布”不能替代我们的 0.20/0.23 本地分支、AWQ 和 gfx1151 的迁移验证。DSpark 的当前源码与旧工程改动由主线程另审。

### SGLang

`#35371` 于 2026-08-19 合并进 main，覆盖 block-local grouped convolution、candidate selector、贪心/随机采样与实际服务评价。[SGLang PR](https://github.com/sgl-project/sglang/pull/35371)

判断：也是正式上游路径；具体量化 lm_head、target GDN 验证、ROCm 架构和 graph 仍需逐项确认，不应从 NVIDIA 成果直接推断所有 AMD 配置。

### llama.cpp

`#27342` 首先合并到 `xsn/dflash2` 分支，真正进入 master 的入口是 `#27816`，合并日期同为 2026-08-27。这一区别需要保留，不能只看到第一个 PR 的 Merged 就断言进入主线。[中间集成 PR](https://github.com/ggml-org/llama.cpp/pull/27342)、[master 合并 PR](https://github.com/ggml-org/llama.cpp/pull/27816)

判断：可用于独立 GGUF 实验；它与 AWQ/vLLM/HIP 项目的运行时、量化权重和性能基线不同。

### Speculators 的训练成熟度不能与推理混为一谈

DFlash2 的训练/导出 `#1006` 于 8 月 27 日合并，但官方算法文档明确：作者未发布训练目标，Speculators 当前采用实验性组合目标，不能视为复现作者训练与参数规模。已有 pretrained checkpoint 的服务支持，并不等于新增 target 上的训练配方完全确定。[训练 PR](https://github.com/vllm-project/speculators/pull/1006)、[官方算法说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md)

这点是 DSpark 在“我们要自行重训一个特定 target 的 drafter”时的重要比较维度。不能把 DFlash2 训练文档中的 experimental 反过来误写成所有 DFlash2 推理都是实验性。

## 2. 可信对照一：vLLM 官方 PR，DFlash2 比当时 DSpark 快约 25%

来源为 DFlash2 实现者在上游 `#52816` 提交的真实服务结果，不是转载。

条件：`Qwen/Qwen3.8-27B`、单 H200、同一容器依次服务各方法；每点在四个独立 H200 shard 测量后平均，不相加。K=7、GSM8K、T=1、top-p=.95、top-k=20、xhigh、最大输出 4096；预热请求剔除。比较 drafter 是当时 `RadixArk/Qwen3.8-27B-DSpark`。

- 并发 1：DSpark 178.5、DFlash2 224.6 token/s，后者约 +25.8%。
- 并发 8：966.2 对 1211.7，约 +25.4%。
- 并发 32：2220.9 对 2759.4，约 +24.2%。

这是同 target、同硬件、同引擎和采样的强证据。它证明该批 checkpoint/实现中的差距，不能证明后续 DSpark 草稿、Qwen3.6 或 RDNA 都存在同样差距。[数据与完整协议](https://github.com/vllm-project/vllm/pull/52816)

同一 PR 的微基准显示 conv+selector 占单轮约 0.67–0.84%，依赖 H200 和编译/graph；不能原样写成“395 上新增开销也不到 1%”。作者说明 grouped conv 是许多小调用，编译和 kernel launch 方式会改变成本。

## 3. 可信对照二：SGLang 官方 PR，同期结果方向一致

条件：同 Qwen3.8-27B、单 H200、FA3、block8 / K7、T1/p.95/k20、xhigh、最大 4096。GSM8K 并发 1 为 DSpark 185.3、DFlash2 236.1 token/s，约 +27.4%。[SGLang 同控结果](https://github.com/sgl-project/sglang/pull/35371)

这不是上一节 vLLM 的同一组数字，两个引擎不可混用分母。PR 中“并发 64 提升约 24%”对应 Muse-Glimmer、四 H200、DFlash1 对 DFlash2；不是 Qwen27B 的 DSpark 对 DFlash2。

## 4. 可信对照三：后续 DSpark 权重使 Apple 同控结果接近持平

社区 `ARahim3/mlx-dspark` 提供新 RedHatAI DSpark 与 DFlash2 的配对实验：M4 Pro 48GB、同进程按 prompt 交错两臂、每次 200 输出 token、三次中位数、相同 cap7；数据于后续 kernel 和 DSpark 权重更新后重测。

- target 8bit：DSpark 31.1、DFlash2 30.8 token/s，约相等。
- target 4bit：DSpark 38.6、DFlash2 40.0 token/s，DFlash2 约 +3.6%。
- 4bit chat 的加速比为 DSpark 2.25×、DFlash2 2.05×；代码/数学则 DFlash2 更好。

这些是项目作者自行测量的主来源，且比单次跑分更注意配对；仍是 Apple/MLX，不能作为 AMD 数值。它最有用的结论是：更新 drafter 后，历史“DFlash2 大胜 DSpark”可收敛到几个百分点。[配对结果](https://github.com/ARahim3/mlx-dspark/blob/main/README.md)

## 5. 可信对照四：GB10 上 DFlash2 与 DSpark v2 中位数持平

`hasso5703/dgx-spark-qwen38` 的 2026-08-29 实验使用 ASUS GX10 / GB10、Qwen3.8-27B NVFP4、同 SGLang 镜像、空闲机器、相同模板与 262144 配置上限。这里 262144 是 max context，不代表每条 prompt 实际输入都达到该长度。

八类任务中位数为 DFlash2 两轮 30.4 / 29.4，DSpark v2 29.5 token/s。部分 prose/reasoning 单元 DFlash2 有优势，但 DFlash2 的同一 code 单元两次波动达 42.0→27.8，只有单次每配置，不能精确排序小差距。[原始说明](https://github.com/hasso5703/dgx-spark-qwen38/blob/main/BENCHMARKS.md)

这说明即便统一内存硬件与模型相同，也必须区分 checkpoint 版本、负载与测量波动，不能把 H200 数字直接折算成 GB10 或 Strix Halo 数字。

## 6. AMD 直接证据：DFlash2 确实能跑，但没有两者同控胜负

### MI355X / ROCm

vLLM `#61761`（2026-10-15）报告单 MI355X、Qwen3.8-27B + z-lab DFlash2、K7、MRV2、ShareGPT、512 输出、T0：main 上并发 1 无 spec 91.7、DFlash2 239.4 token/s；并发 64 无 spec 2855.4、DFlash2 2641.3。作者另展示补丁改善，高并发可反转成收益。[ROCm 实测与计划](https://github.com/vllm-project/vllm/issues/61761)

这个结果没有 DSpark 对照，也不是 gfx1151。其意义是“运行时后端选择与状态存储可决定成败”，不是证明 DFlash2 算法优劣。该 profiler 中 grouped conv 只占 0.3%，大头是 GEMM 与 GDN；也不能先入为主把新增卷积当唯一瓶颈。

### Ryzen AI Max+ 395 / gfx1151 / Vulkan

Nathanw1014 的固定提交 `1185a3d2bbcbc0ce650facbc83278b661a199780` 记录 Framework Desktop 395、64GB、Qwen3.8-27B UD-Q4_K_XL、f16 KV、Q8_0 DFlash2、Vulkan/RADV，llama-benchy 自然 prose、输出128、两轮：

- 实际上下文深度 0 / 8K / 32K，无 spec：11.81 / 11.44 / 10.54 token/s。
- DFlash2 width4：26.39 / 21.58 / 21.11。
- DFlash2 width7：25.18 / 21.46 / 16.32。
- DFlash1 width5：21.09 / 12.87 / 10.75。

作者公开误差与原始工具输出；同一模型下，32K 时 width4 明显优于 width7。这里没有 DSpark 臂，也不是本项目 Linux ROCm AWQ 算子链。[395 固定版本实测](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/1185a3d2bbcbc0ce650facbc83278b661a199780/docs/dflash2-strix.md)

这是真正接近我们的硬件证据，支持“DFlash2 值得在 395 试验”，不支持“它已证明比我们的 DFlash 或 DSpark 快”。不要拿这份未调优 GGUF baseline 与我们 28.3 token/s 峰值直接拼接比较。

## 7. 对低并发 RDNA3.5 的分析判断

以下是系统推断，不是已测性能：

1. 395 的统一内存带宽有限，目标权重重复读取代价高，因此以更少 target forward 生成更多最终 token 有价值；两种算法都利用这一点。
2. 相同验证宽度下，DFlash2 如果提高接受率，确实可能得到较直接收益；但其效果取决于 target/drafter 分布、量化、模板与领域。
3. DSpark 的 confidence 可以驱动长度选择，减少低收益后缀验证；实际引擎是否启用了成本模型、变长图和适配的 attention/GDN 状态处理，必须源码确认。
4. DFlash2 也不必永远固定使用全部训练块长；可以外部扫描或按成本选 cap，但这与 checkpoint 自带 DSpark confidence head 不是同一种接口。
5. RDNA 上 small-M WMMA、verify attention 的 Split-K、kernel launch、top-k fallback、GDN 状态写回，都可能比算法的小模块开销更关键。当前公开 H200 的百分比无法预测这些项。
6. 我们现有 DFlash 已经针对长上下文与 HIP 内核优化；应以它作为强对照，而不是仅与未调优 target-only 比。新方法赢普通 baseline，并不保证赢我们当前系统。

## 8. 建议实施顺序与足够回答胜负的最小实验

展会主链继续使用已验收的现有 DFlash。另建候选环境，不覆盖当前环境：

1. 固定现有 Qwen3.6 target：验证匹配 DSpark 的 loader、特征层、anchor/块长、GDN rollback 与 AWQ target 行为；通过后测短上下文。
2. 仅在找到正确匹配的 DFlash2 权重时加入同 target 对照；如采用 Qwen3.8，则把它标成独立模型路线，不能和 Qwen3.6 混合宣称算法提升。
3. 每种方法在实际已填充 2K / 8K / 16K / 32K、并发 1 / 2 / 4 下测 cap2/4/7 或训练支持范围；先贪心，再用展会实际采样。
4. 三组都保留相同 target 权重、量化、template、功耗、预热、输出长度、prefix caching 与 graph 配置。动态长度单列，不要同时改变所有变量。
5. 报告 TTFT、端到端 token/s、decode TPOT、每步提交 token、draft/verify 分项、显存与连续多轮失败。每点至少三个配对重复，并轮换顺序，避免温升和缓存顺序偏差。
6. 同时观察 target-only 的强制全拒绝对照、部分接受、EOS、长输出及多轮前缀复用；“无损”先按目标推理路径实际验证，区别浮点执行路径差异和真正状态错误。

推荐措辞是：**DSpark 是当前保持原 target 的优先兼容性候选；DFlash2 是值得同台测试的高收益候选。最终按我们的 395/AWQ/真实工作负载选择默认，架构上保留可切换。** 在匹配权重未核验和本地 A/B 未完成前，不将任何一者写成确定的性能赢家。
