# DFlash2 权重与 DSpark 对照核验（2026-10-22）

这是主任务的研究材料。只读取公开资料、配置和 safetensors JSON 头，没有下载张量数据或运行推理。结论按当前 Qwen3.6-27B AWQ4 + gfx1151/gfx1100 项目来判断。

## 1. 结论先行

**两种算法都已有可直接研究的 Qwen3.6-27B 社区草稿，不能以“DFlash2 没有匹配权重”直接排除它。**

- DSpark：`abstract-extraordinary/Qwen3.6-27B-DSpark` 和 `...-49k` 已有作者报告的服务实验；特征层与我们现有 DFlash 一致，接口切入较直接。
- DFlash2：已发现并核验 `vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1`；它是标准 HF BF16 格式，真实包含动态卷积与 candidate selector，**不是只能用在 MLX 的文件**。但该 checkpoint 的作者明确尚未进行服务速度和下游 Agent 质量测试。
- 近期开会演示的可靠主线仍是我们已有的 DFlash；新增后端先做同机 A/B。若按结构推断，DFlash2 的候选表查找避免逐位置全词表 Markov W2 计算，且全部 SWA2048，对 395 有吸引力；DSpark 的现有同目标社区权重实验信息较充分。这是“硬件结构潜力”和“具体权重成熟度”的不同排序，不能合成一个无条件赢家。
- 任何来自 H200、GB10、Apple Silicon 的 tok/s 都不能替代当前 AWQ4 target、ROCm、HIP 算子的实测。

## 2. 官方到底发布了什么

[z-lab/dflash README](https://github.com/z-lab/dflash#supported-models) 当前 DFlash2 一栏列 Muse-Glimmer-30B 与 Qwen3.8-27B；Qwen3.6-27B 位于 DFlash 一栏。HF 的 Inco AI 组织还公开了 [Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)，该 MoE 模型与当前 dense27B 不同。

[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 明确是 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) 的镜像，属于同一作者路线。不是社区仅用 DFlash2 名称包装出来的另一种算法。

本次 HF API 按 `dflash2` 搜索（limit1000）返回278条模型名。检索到的直接 Qwen3.6-27B 候选是下文 vvsotnikov 的 BF16 与 MLX4bit 两份。这个名字搜索仍不等于对全部 HF 进行了语义穷举。

尝试 `z-lab/Qwen3.6-27B-DFlash2` 的公开 API 返回401；这个响应不能区分不存在、私有或其他访问条件，因此不对该 ID 作不存在判断。可确认的事实只是：官方当前发布表没有列出可读的这个 target 配对，而社区已经有公开候选。

## 3. 当前 Qwen3.6-27B DFlash2 社区候选

### 3.1 固定版本与目标

[vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1](https://huggingface.co/vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1)，实读 revision：

```text
a233ec768abb2ed9a2a97a294b5808d0ad10932e
```

[固定 config](https://huggingface.co/vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1/blob/a233ec768abb2ed9a2a97a294b5808d0ad10932e/config.json)：

- `architectures=["DFlash2DraftModel"]`，`model_type="qwen3"`，BF16。
- hidden5120、intermediate17408、5层、32Q heads、8KV heads、head_dim128、target64。
- `dflash_config.block_size=8`，`conv_kernel_size=2`，`conv_group_size=16`，`selector_rank=256`，`selector_top_k=16`。
- 特征层 `[5,19,33,47,61]`；mask token248070。
- 5 层全部 `sliding_attention`，`sliding_window=2048`。
- `vocab_size=draft_vocab_size=248320`，未做49k输出词表裁剪。
- `rope_theta=10000000`，配置最大位置262144；最大位置只是配置，作者没有验证这个上下文长度。

**与我们现有 DFlash 的特征层 [1,16,31,46,61] 不同，必须重配 target capture。** 这是加载另一个 checkpoint，不是只增加一个 selector 后仍沿用原有所有配置。

### 3.2 实际权重头

本次只读8,960bytes的 safetensors JSON头。确认81个BF16张量、1,924,404,480个载荷参数，没有读取任何张量数值：

```text
candidate_selector.hidden_projection.weight          [256,5120]
candidate_selector.predecessor_codebook               [248320,256]
candidate_selector.successor_codebook                 [248320,256]
layers.{0..4}.attention_conv.base_kernel              [2,2,5120]
layers.{0..4}.attention_conv.kernel_projection.weight [1280,5120]
layers.{0..4}.mlp_conv.base_kernel                    [2,2,5120]
layers.{0..4}.mlp_conv.kernel_projection.weight       [1280,5120]
```

文件名 `model.safetensors`，大小 **3,848,817,920bytes**（约3.584GiB，十进制3.85GB）。HF LFS SHA256：

```text
52d968b9a9f7cc077580327074119a3815f74f48296c5f05b1aa1e0713f7b1f8
```

导出不包含 `embed_tokens`、`lm_head`、DSpark式 `markov_head`、`confidence_head` 或 `d2t` 张量。它需要运行时提供/共享 target 的输入输出权重。不能因 DSpark checkpoint 含 embedding/head 有8.8GB，而 DFlash2 没有它们仅3.85GB，就宣称后者少占5GB运行内存。

按此头计算，selector物化权重约245MiB，其中两个codebook约242.5MiB；动态卷积权重合计约125.39MiB。因此DFlash2并非完全没有新权重；优势主要在访问方式与计算组织，要通过 profiler 确认实际驻留与读量。

### 3.3 训练与成熟度

[固定模型卡](https://huggingface.co/vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1/blob/a233ec768abb2ed9a2a97a294b5808d0ad10932e/README.md) 说明：

- 目标是 `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，BF16，thinking disabled。
- 从已经训练过的 Qwen3.8/blend 谱系草稿继续适配，而非从零训练；本次计划500步，只发布step100。
- 本次仅400个训练calls；目标再生成的 agentic coding 轨迹，CE加耦合selector CE；target embedding和head冻结。
- 作者检查了导出81个张量与训练checkpoint逐值一致，但**明确没有本checkpoint的服务bench或Agent质量评估**，也未报告可据此判定部署效果的heldout接受率。

因此可以列为“现成研究候选”，不能列为“已证明优于DSpark的Qwen3.6权重”。thinking enabled、AWQ4部署和长上下文接受率都仍待测。

## 4. 官方 Qwen3.8 DFlash2 的结构、版本与实验边界

本次核验版本：

- z-lab mirror：`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`。
- incoai original：`015e795645c74b1a0eeef3b570031fb62e769bc5`。

两者都只读8,936bytes的JSON头，均81个tensor；结构与上面的3.6社区适配相同。官方3.8配置的特征层也为[5,19,33,47,61]，5×SWA2048，block8、selectorK16/rank256，完整词表248320。通过API的LFS内容哈希还可检查两份是否同一权重；本记录没有用下载整文件的方式重新计算hash。

[官方3.8模型卡](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2#evaluation) 报告在单H200、SGLang、FA3、官方sampling参数、七个proposal下，与内置MTP和RadixArk社区DSpark比较。表中DFlash2在列举的工作负载上更快，但这是具体checkpoint加具体runtime的作者实验，不能证明所有DSpark训练方案更差，更不能直接套用到3.6AWQ+AMD。

注意官方卡的早期vLLM命令安装PR52816，技术博客的后续命令则直接安装vLLM。文档时点不同；实际最低版本与适用runner应以当前源码/已验证commit为准，不能照旧卡把一个PR分支当成永久依赖。

## 5. DFlash2 与 DSpark 的机制差别

[Inco AI 技术博客](https://inco.ai/blog/dflash2/) 和当前[模型实现](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dflash2.py)可以互证：

- DFlash1：一次backbone并行生成block的独立位置预测，位置内缺乏选中token之间的关联。
- DSpark：backbone仍并行，但按位置用前token的低秩Markov头修正整个词表logits；另有confidence，供验证长度控制使用。
- DFlash2：backbone的attention和MLP前后加入分组动态短卷积；先从完整unary logits取每位置topK，再计算相邻候选间的转移分数，最后沿已算好的边顺序选出一条路径。

核心分数：

```text
S_t(a,b) = U_t(b) + dot(A(a) * H(h_t), B(b))
```

其中a为前token候选，b为当前候选，H由当前hidden得到。代码 `successor_table[candidate_ids]` 和 `predecessor_table[predecessor_ids]` 只gather候选行，再做 `einsum` 评分。顺序路径选择仍存在，只是没有在每一位置重新修正完整词表。DFlash2也仍需计算完整unary head/topK；不能把它说成彻底不扫全词表。

对gfx1151低并发，DSpark fullW2的反复读取可能相对昂贵；DFlash2将这一部分改成少量候选访问，但也增加卷积projection、topK与小kernel launch。W7900带宽/算力结构不同，且两种后端的融合成熟度影响结论。因此架构比较只能形成实验优先级。

当前DFlash2候选没有DSpark confidence头。若要有类似DeepInfer的接受率感知动态验证预算，还需额外设计/训练信号或独立统计控制，不能因都使用低秩转移头就默认完全等价。

## 6. 训练代码是否齐全

[Speculators DFlash2说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md) 明确区分作者推理架构与其自身实验训练目标；原始公开实现没有完整披露训练objective，speculators使用unary loss加topK selector CE。其新版还允许miss时replace或strict策略，不能把它说成已复现原作者全部训练配方。

该实现当前要求完整target词表，拒绝pruned draft vocabulary；因此DSpark49k的输出裁剪格式不能照搬到DFlash2 checkpoint并期待loader自然兼容。

[SpecForge DFlash2指南](https://sgl-project.github.io/SpecForge/concepts/DFlash2.html) 提供 Qwen3.6-27B 配置与训练示例。它复用DFlash的feature schema及训练strategy，新增卷积与selector；并支持identity/zero初始化，使新模块先保持原草稿函数。但“有训练配置”与“已有成熟权重”仍应分别标注。

原作者博客中“selector新增约2M参数”的消融统计与当前checkpoint中物化的完整词表codebook，口径不同。Speculators也提醒自由codebook训练参数化不等于原作者所称参数量；不要拿当前两张248320×256表直接证明博客错误，也不要把博客的小参数数字当成运行时表大小。

## 7. 与我们项目做公平对照时的特别注意

1. **anchor语义**：这份DFlash2的block8通常是anchor+7个proposal，vLLM命令用`num_speculative_tokens=7`；当前DSpark block8按其sample-from-anchor语义可输出8个proposal。直接把两边的数字都设成8会错配训练长度。应记录实际草稿数和target query形状。
2. **M边界**：低并发时7proposal常对应target验证M8，而8proposal常对应M9。我们既有HIP路径就在这一带有形状性能差异；这可能影响胜负，必须把实际AWQ后端路由与M分布一起报告。也要做各自最佳配置和相同实际proposal数的补充对照。
3. **capture层**：DSpark候选与现有tap相同；DFlash2候选换成[5,19,33,47,61]。不要把旧tap缓存输给新草稿。
4. **窗口布局**：DFlash2是5SWA；DSpark候选5full；现有DFlash是4SWA+1full。长prompt的draft attention与KV内存不在相同条件，应该报告这个事实。
5. **target稳定**：保持同一AWQ4模型revision、tokenizer、量化算子路由和sampling。3.8 target不能拿来证明3.6后端替换收益。
6. **模式**：3.6DFlash2候选只适配nothink；若展会需要thinking，先测试模式变化的接受率和实际wall time。
7. **指标**：对照no-spec、现DFlash、DSpark full/49k、DFlash2；测TTFT、稳定decode、端到端wall、各位置prefix survival、峰值内存、draft/target/sampler分项。输出质量验证相对部署的量化target进行，不能把投机无损说成AWQ量化本身无损。

针对“哪个更好”的当前可支持答案：**395的性能探索可将DFlash2列为高优先级候选；Qwen3.6现成权重的已报告服务证据，DSpark更充分；展会正式展示继续以已验证DFlash为主。最终择优须取决于同机、同target、同提示集的端到端A/B。**

## 8. 本地证据

- [hf_dflash2_metadata.json](./hf_dflash2_metadata.json)：HF目录名字搜索及两个官方3.8仓库的config、版本和tensor header。
- [hf_dflash2_qwen36_metadata.json](./hf_dflash2_qwen36_metadata.json)：匹配3.6社区草稿的config、LFS元数据和tensor header。
- [read_hf_dflash2_metadata.py](./read_hf_dflash2_metadata.py)：有限大小的只读核验脚本，不下载tensor。
- [DSpark核验记录](./dspark_checkpoint_notes.md)：对应DSpark full/49k配对、参数形状及来源冲突。
