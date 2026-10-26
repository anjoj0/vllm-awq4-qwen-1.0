# GitHub DSpark 实现与成熟度检索记录

核验日期：2026-10-22。只读检索；没有安装软件、下载大模型或执行任何 GPU 基准。这里的“上游已合并”“有真实服务测试”“匹配我们的平台”是三个不同条件，不能互相替代。主线程另行审查 vLLM 当前源码和 Qwen3.6-27B 候选权重。

## 直接结论

GitHub 上已经存在正式发布或合并进主线的 DSpark 实现，不能说 DSpark 全部停留在论文或个人实验分支。与我们的 vLLM + Qwen3.6-27B AWQ4 + gfx1151/gfx1100 项目最接近的路线，仍然是在现有 vLLM 系统内接入其上游 DSpark，并保留现有 DFlash；其他引擎适合做交叉验证或学习参考。

本轮没有找到同时覆盖“雷神 Ryzen AI Max+ 395 / gfx1151、Qwen3.6-27B、当前项目 AWQ4 权重、自研 HIP 内核、DFlash/DSpark 可切换、长上下文回归与实际端到端提升”的可直接复用公开成品。这是本轮检索的范围结论，不能表述为全球不存在。

按我们的用途排序：

1. vLLM 上游运行时：最适合复用我们已有工程。
2. DeepSpec + Speculators / TorchSpec / SpecForge：分别提供作者参考、训练/导出、模型适配验证；不是替代 vLLM 的同一种运行时。
3. SGLang：最值得研究信心调度、变长验证、成本模型，作为第二参考实现；当前 Strix Halo 适配仍有未合并 PR，切换整体引擎会丢失我们当前内核接入与验收路径。
4. llama.cpp：已有主线 DSpark 和 AMD HIP + Qwen3.6-27B 的用户案例，适合 GGUF 独立复现；不能直接接入我们 AWQ-vLLM 项目，也不能由 Qwen3 的支持直接推断所有混合 GDN 模型路径。
5. Apple vllm-metal：有认真公开生命周期测试与数值失败记录，适合参考双 drafter 抽象和验收方式；不适合 AMD 部署。
6. 社区 fork：有运行证据，但应先审查上游是否已包含同等功能，避免再承担完整分叉维护。

## 1. vllm-project/vllm：正式上游具备 DSpark 与 AMD 路线

维护主体：vLLM 项目。v0.26.0 发布页显示 2026-07-27 发布，提交标识 `568afb3`；发布说明明确包含 DeepSeek-V4 的 AMD DSpark `#47419`。这证明 AMD DSpark 不再只是私有补丁，但这项支持主要对应 DeepSeek-V4 / Instinct，不能替代对 RDNA 和 Qwen3.6-27B 的单独验证。[v0.26.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.26.0)

当前源码存在 `vllm/models/deepseek_v4/amd/dspark.py`，文件说明明确为 gfx950 ROCm 适配：复用 AMD decoder layer、HCHeadOp 分发与 AITER MegaMoE。这个入口针对特定大 MoE 模型，不代表给单卡 Strix Halo 的 Qwen DSpark 自动提供了全部算子支持。[AMD DSpark 模型源码](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/amd/dspark.py)

判断：可作为正式工程基础；“DSpark 功能已发布”成立，“我们当前硬件与权重量化组合已验证”仍待本地验收。最值得复用其 proposer、拒绝采样、模型 loader 和调度契约；先固定块长，再考虑动态长度。

## 2. vllm-project/speculators：DSpark 已进入正式训练发布

维护主体：vLLM 项目及 Red Hat AI 等贡献者。`v0.7.0` 于 2026-07-30 发布，提交 `1c2895f`，正式加入 DSpark、Markov / confidence heads、训练参数及已发布模型；发行说明同时提示与 vLLM 0.27.0/0.27.1 的已知兼容问题。[v0.7.0 发布](https://github.com/vllm-project/speculators/releases/tag/v0.7.0)

官方算法页将 DSpark 实现为 DFlash 派生模型，并明确推理由 vLLM 的 `method: dspark` 承担。其训练 head 有 vanilla、gated、rnn 等变体，实际 checkpoint 与 loader 应逐一匹配，不应仅凭 DSpark 名称认为可互换。[算法文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)

判断：成熟的上游训练与模型导出路线之一；它本身不是现成 ROCm Qwen3.6-27B 展会镜像。官方发布涉及 GLM、Gemma 模型，也不能推导出我们的目标权重已经完成支持。

## 3. deepseek-ai/DeepSpec：作者官方训练与评价参考

维护主体：DeepSeek。公开仓库覆盖数据生成、target cache、训练、投机接受长度评价；README 提供 Qwen3-4B/8B/14B 和 Gemma4-12B 的作者 checkpoint。默认训练示例按单机八卡设计，数据缓存也可能很大。其定位是研究与训练评价全栈，而非面向用户 HTTP 服务的生产调度器。[DeepSpec](https://github.com/deepseek-ai/DeepSpec)

判断：算法与 checkpoint 语义的优先依据；不是一条在 395 笔记本上开箱即用的服务路线。作者 checkpoint 的目标模型、non-thinking 训练模式与我们的 Qwen3.6/AWQ 分布不同，不能跨模型直接替换。

## 4. lightseekorg/TorchSpec：训练实用路线，有公开采用案例

维护主体：LightSeek Foundation。框架以推理/训练物理分离、Mooncake 流式传输 hidden states 为核心；README 列出 vLLM 的一级支持和其他引擎适配，也提供从 DFlash checkpoint warm-start DSpark 的导入步骤。这是重新训练匹配 drafter 的候选工具链，不意味着无需训练即可把新增 head 变成有效 DSpark。[TorchSpec README](https://github.com/lightseekorg/TorchSpec)

判断：值得复用训练与模型转换流程。其公开安装说明与示例较多围绕 CUDA/Mooncake，具体 ROCm 依赖和 gfx1151 验收要另做；不能把“torch-native”直接解释为所有 ROCm 配置已经验证。

## 5. Dogacel/vllm 的 dspark-local：有真实指标，属于个人 fork 证据

Dogacel 发布的 Qwen3-8B-DSpark 模型卡直接指向这个分支，并说明实测平台为单张 H100 80GB、SPEED-Bench coding、温度 1、生成 2048 token、probabilistic drafting、K=7。这提供真实服务对照证据，但目标是 Qwen3-8B，不是 Qwen3.6-27B，也不是 AMD。[模型卡与复现](https://huggingface.co/Dogacel/Qwen3-8B-DSpark)

源码为 vLLM 的个人 fork。[dspark-local 分支](https://github.com/Dogacel/vllm/tree/dspark-local)

判断：可参考训练导出格式、head 参数和测量方法；不建议把整个 fork 作为我们下一版唯一依赖。先与上游 DSpark 对比，确认是否仍有必需差异。

## 6. SGLang：已有成体系的服务集成，动态验证设计非常值得借鉴

维护主体：LMSYS / sgl-project。官方在 2026-07-06 发布 DSpark 集成文章，对应 `#30261`。文章介绍 confidence / STS 校准、按请求分配验证长度、static / compact / cap-accept 三种验证方式、总 token 数分档的 ragged graph、离线成本表和 overlap 调度。这比“加上 Markov head”覆盖更完整的系统收益路径。[SGLang 官方文章](https://www.lmsys.org/blog/2026-07-06-dspark-sglang)

### 与 Qwen3.6-27B 的直接联系

SpecForge 主仓提供 Qwen3.6-27B DSpark 的三段验证：target 生成数据、单样本过拟合、HF 导出并在 SGLang 服务，要求整块 16 token 被接受且与 target 前缀一致。教程使用 BF16、FlashInfer、关 graph、关 radix；这是功能 gate，不是泛化接受率或 AMD 性能报告。[Qwen3.6-27B gate 教程](https://github.com/sgl-project/SpecForge/blob/main/scripts/gates/README.md)

### AMD 证据与边界

AMD ROCm/MAD 主仓有 SGLang Kimi-K3 DSpark 基准入口，平台限定八张 MI350X/MI355X、TP8、gfx950。其说明特别区分 MI355X 已报告数据和 MI350X 未验证调优。这是数据中心 ROCm 的公开支持证据，不是 RDNA 单卡证明。[ROCm/MAD SGLang 基准](https://github.com/ROCm/MAD/blob/develop/benchmark/sglang/README.md)

Strix Halo 的 `sgl-kernel` 支持仍有 `#31137` Open PR。该 PR 作者报告使用一组未合并 companion patches 在 gfx1151 上运行多模型，包括 BF16 Qwen3.6-27B，但其验证矩阵并没有声明我们的 AWQ + DSpark 组合。不要把作者的普通服务测试当成 DSpark 正确性 gate。[gfx1151 PR #31137](https://github.com/sgl-project/sglang/pull/31137)

Qwen3.6 混合线性注意力 DSpark 另有 `#36481`：作者报告 compact graph 在 B300 上出错，而 static/cap-accept 可运行。这说明动态验证是独立的兼容层，不能从固定宽度可跑直接推断 ragged graph 完备。[compact 验证问题](https://github.com/sgl-project/sglang/issues/36481)

判断：优先研究系统原理，作为第二实现交叉验证。在展会前迁移整个项目到 SGLang 的风险和工作量通常高于给现有 vLLM 添加 DSpark，尤其当前我们已有 AWQ/HIP、GDN 状态和长上下文优化。

## 7. ggml-org/llama.cpp：上游正式合并，可作为 GGUF 独立路线

维护主体：ggml-org。Qwen3 DSpark PR `#25173` 于 2026-07-28 合并，复用 DFlash 的 target feature extraction、KV 注入与验证流程，增加 DSpark 模型/head、顺序采样和转换。DeepSeekV4 的 MTP + DSpark `#25784` 于 2026-08-02 合并。这些合并已构成长期上游支持基础。[Qwen3 DSpark #25173](https://github.com/ggml-org/llama.cpp/pull/25173)、[DeepSeekV4 #25784](https://github.com/ggml-org/llama.cpp/pull/25784)

当前官方文档列出 `draft-dspark`、HF→GGUF 转换、块长限制、confidence 阈值截断，并说明 dense drafter 支持的 backbone 范围。文档中的阈值截断不应等同于 DeepInfer/SGLang 的完整成本驱动动态调度。[官方 speculative 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

### Qwen3.6 + AMD 的直接公开案例

`#26339` 于 2026-07-30 报告 Linux HIP、两张 MI50、Qwen3.6-27B Q4_1 + satgeze DSpark GGUF，作者称 layer split 可运行，而 tensor split 失败。该 issue 后以 stale/not planned 关闭，不能把关闭当作已修复。它证明“AMD + Qwen3.6-27B DSpark”存在用户实践，也显示平台/并行模式限制；不是本项目 gfx1151/gfx1100 的复现证据。[MI50 问题 #26339](https://github.com/ggml-org/llama.cpp/issues/26339)

`#26230` 的 Qwen3.6 讨论中，2026-10-18 又有 35B-A3B GGUF DSpark 的运行报告，但平台为 RTX 2080 Ti，且不同任务收益不同。35B 与 27B 也应分开判断。[Qwen3.6 支持讨论](https://github.com/ggml-org/llama.cpp/discussions/26230)

判断：可以准备独立 GGUF 试验；不属于对现有 AWQ-vLLM 模式的低成本替换，会更换量化布局、执行后端和大量性能基线。展会前如准备备用程序，仍要对精确机器与权重重新验收。

## 8. vllm-metal：主线已合并，但官方明确 experimental

维护主体：vLLM Apple Metal 项目。`#986` 在 2026-10-04 合并，复用 DFlash 的 target feature commit 和 scheduler KV 生命周期，同时保留 DSpark 独立 embedding、Markov head 与 K 槽语义。PR 同时公开 Apple M5 Max 的真实 checkpoint/HTTP 验证及数值不一致，明确尚未建立完整 bitwise losslessness，不声称已取得性能提升。[PR #986](https://github.com/vllm-project/vllm-metal/pull/986)

判断：不适合 AMD 运行，但其双 drafter 共享生命周期、独立槽位契约、失败 qualification 不隐藏的做法，很适合我们借鉴。单凭测试数量或 merged 状态不能把它称为已验证的成熟性能方案。

## 9. 单卡 Strix Halo 的社区 DSpark：确实存在，但不能照搬收益

### Luce-Org/lucebox

社区 HIP/GGUF 引擎的当前部署文档明确列出 DeepSeek-V4-Flash-0731 + DSpark 的 `ds4-strix` profile，使用 gfx1151、fused verifier 和 confidence 驱动宽度，并指向测量 PR。这里的模型是 DeepSeek-V4、格式是 ROCmFPX/GGUF，整个引擎不同。[推荐部署文档](https://github.com/Luce-Org/lucebox/blob/main/server/docs/RECOMMENDED_SETUPS.md)

判断：AMD 端工程参考，尤其验证 kernel、内存布局与动态宽度；本轮未审查其完整数值等价性 gate，不能代替我们的损失无损与性能验证。

### acupof-ai/cachedMoE

这是 Linux/RADV Vulkan/Slang 引擎，目标为 128GB Strix Halo 和 DeepSeek-V4.1，作者将 DSpark 标成实验模式。其 top-K 接受规则是近似验证，并明确不保留目标采样分布；README 公开了若干质量门限失败。因此它证明的是实验探索，不能当作严格无损 DSpark 的成熟成品。[cachedMoE README](https://github.com/acupof-ai/cachedMoE)

## 10. 哪些证据足以把我们版本称为“成熟方案”

在两块目标架构上逐一验证，而不是只运行一条 demo：

1. 模型加载与语义：target revision、drafter revision、tokenizer、feature layers、mask ID、anchor 布局、head variant 均固定。
2. 对照：target-only、当前 DFlash、DSpark 固定块长；动态 confidence 调度单列。
3. 状态：全拒绝、部分接受、EOS、取消、长上下文、GDN rollback、prefix reuse、抢占/重算和请求复用。
4. 性能：同 prompt/template/采样/长度、同 graph 与 kernel 设置；预热后测 wall time、TTFT、TPOT、接受长度、drafter 和 verifier 分项。
5. 稳定性：展会预期时长的多轮连续运行、温度/功耗/共享内存占用、失败恢复。
6. 文档：精确 OS/ROCm/vLLM commit、补丁、模型下载 revision、启动配置和完整原始结果。

这套证据比“某仓库已合并”“有 DSpark 权重”“在某张 GPU 更快”更接近我们需要交付的展会和面试材料。
