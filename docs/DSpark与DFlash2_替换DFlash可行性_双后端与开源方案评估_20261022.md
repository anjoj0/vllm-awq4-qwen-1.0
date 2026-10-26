# DSpark 与 DFlash2：替换现有 DFlash、双后端设计及 AMD 适用性

研究日期：2026-10-22。面向本项目 Qwen3.6-27B AWQ4、Strix Halo/gfx1151 与 W7900/gfx1100；尤其考虑即将使用的 Ryzen AI Max+ 395 展会机。

本文是源码、权重元数据与公开实验的静态评估。没有下载完整模型、执行远程模型代码或开展 GPU 基准。引用他人数据时保留其模型、设备和测试条件；工程工作量是范围估计，不是已经提交的 diff。

当前项目快照为 `5f05a501ac51a0bff646f4f98b731f74b40e08b8`。新 vLLM 主线源码固定为 `aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee`，另核查了 v0.20.0、v0.23.0、v0.24.0 和 v0.25.1 发布标签。本文按实际仓库使用的 **Qwen3.6-27B** 讨论；Qwen3.7/3.8 的配对不能仅凭同为 27B 直接混用。

配套公开材料：[Strix Halo 部署基线](STRIX_HALO_ORIGINAL_README.md)；[DeepInfer 固定版本源码](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)。

## 1. 先回答：能换、存在权重、能切换，但“更好”取决于目标

**可以把 DSpark 接成第二种投机后端，也可以进一步加入 DFlash2。** 上游 vLLM 已有实现，没必要从 DeepInfer 整体移植一个新引擎。但当前展会用的 v0.20.0 补丁栈不能只改一个 `method` 字符串就完成接入。

Hugging Face 已有直接面向 Qwen3.6-27B 的 DSpark 社区权重，也找到 Qwen3.6-27B 的 DFlash2 社区候选。后者明确是早期训练检查点，不能把“找到了文件”解释为已完成成熟服务验证。

建议按目标选择：

- **近期展会主路径：保留目前已验证的 DFlash。** 新后端先在独立环境验收，不能把检索结果当成整机性能证明。
- **保持当前 Qwen3.6-27B target，尽快做可比较的研究：优先完成 DSpark 功能 A/B，再加入 DFlash2。** 当前匹配的 DSpark 候选已有更多服务实验记录；这是现成权重的准备程度判断，不是算法胜负。
- **愿意更新到官方配对的 Qwen3.8-27B，并投入新引擎适配：DFlash2 很值得优先评估。** 它有作者发布的配对权重、上游服务集成和公开 AMD/395 实践，且其局部混合/候选选择可能更适合低并发带宽约束。更新 target 后必须重建质量与性能基线。

**目前不能严谨宣布“395 上 DSpark 一定胜过 DFlash2”，也不能宣布相反结论。** 缺少针对我们同一 AWQ target、同一机器、同一输出协议的受控实测。下文给出能缩小不确定性的具体证据与验证顺序。

## 2. 三种方法到底差在哪里

### 2.1 当前 DFlash

当前草稿是 `z-lab/Qwen3.6-27B-DFlash`，目标模型特征注入后，草稿 backbone 一次处理多个位置，由 target 批量验证。我们已经做过非因果 attention、SWA、验证阶段 3D split-K、量化与缓存等适配。

这份草稿的配置是 5 层、`4×SWA(2048)+1×full`、hidden5120、target64 层、特征层 `[1,16,31,46,61]`。[原始草稿配置](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash/raw/main/config.json)

### 2.2 DSpark：并行 backbone + 顺序 Markov 小头 + 可选置信度调度

DSpark 保留 block-parallel backbone，在输出 logits 上增加前一已选 token 的低秩修正。以 vanilla 头示意：

```text
backbone 一次得到 h[0...N-1]
z[i] = LM_head(h[i])
m[i] = W1[token[i-1]]
q[i] = softmax(z[i] + W2 × m[i])
token[i] ~ q[i]
confidence[i] = sigmoid(proj(concat(h[i], m[i])))
```

Transformer 主体仍可一块并行计算；顺序依赖主要留在较轻的采样头。置信度预测当前候选的接受概率，可给验证预算控制器使用。**它不是简单把草稿的最大 softmax 概率改名叫 confidence。**

上游模型直接继承 DFlash backbone：

```python
class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov / confidence heads."""
```

[源码：qwen3_dspark.py，L192–230](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dspark.py#L192-L230)

运行时则逐位置执行 Markov 修正、采样并更新 `prev`，而不是独立采样所有位置。[DSparkSpeculator，L156–207](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L156-L207)

已有 DFlash 权重缺少训练好的 Markov/confidence 参数，不能仅重命名变成 DSpark；新头可以通过训练得到，也可以直接使用已配对的公开 DSpark checkpoint。作者训练参考见 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)；通用训练实现见 [Speculators DSpark](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)。

### 2.3 DFlash2：局部卷积混合 + 小候选集合中的依赖选择

当前上游 DFlash2 在草稿层的 attention 和 MLP 周围加入输入相关的局部 grouped convolution，让块内位置之间传递更多信息；输出阶段先得到每个位置的候选 token，再用 predecessor/successor codebook 和 hidden projection 计算相邻位置的候选边分数，最后沿候选集合生成序列。

它也不是“每个位置完全独立、最后随便选 Top-K”。依赖在 backbone 的局部混合和候选选择两处体现；选择过程仍有前后关系，但可避免每个位置都重复扫一次完整词表的 Markov 投影。

当前源码的候选边分数为 unary logits 加低秩双线性项：

```python
return unary_logits[:, :, None] + torch.einsum(
    "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
)
```

[候选选择源码，L290–358](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dflash2.py#L290-L358)；[卷积与 decoder，L144–287](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dflash2.py#L144-L287)。

候选截断改变的是草稿 proposal 分布。要保留 target 的采样语义，验证器必须使用实际 proposal 及其概率；不能将“只要落入 target Top-K 就接受”的近似规则当成同一算法。

Speculators 提供 DFlash2 训练支持，但其文档明确区分自己实验性的训练目标与作者原始训练方法；不能把任意训练复现都视为作者权重的等价替代。[DFlash2 训练说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md)

## 3. DSpark 的公开权重：有直接对应当前 27B 的候选

### 3.1 完整词表：先做功能验证

[abstract-extraordinary/Qwen3.6-27B-DSpark](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark)，本次核查 revision：

`36ccb76a7bf87a7f33cc3b85260e952685a8a7d0`。

核查配置与 safetensors JSON 头，确认：

- `Qwen3DSparkModel`、hidden5120、target64、vocab248320。
- 特征层 `[1,16,31,46,61]`，与我们现草稿接口相近。
- 5 层**全为 full attention**；block8；Markov rank256；启用 confidence。
- 真实存在 `markov_head.markov_w1/w2`、`confidence_head.proj`，不是只有配置标签。
- 包含自己的 embedding 与 LM head；文件张量载荷约 **8.196 GiB**，不等于运行时额外峰值内存。

[固定配置](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark/blob/36ccb76a7bf87a7f33cc3b85260e952685a8a7d0/config.json)

配置匹配是可接入的结构证据，不能代替对目标 revision、量化母本、tokenizer、特征定义及 AMD 数值路径的验证。

### 3.2 49k：降低输出头与 Markov 读取量

[abstract-extraordinary/Qwen3.6-27B-DSpark-49k](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark-49k)，revision：

`a3d00995272f95093311c638cb5913db7be3c033`。

主要结构相同，输出词表缩为 49,167；输入 embedding 与 W1 仍覆盖完整 target 词表，LM head 与 W2 减少行数，并携带 `d2t` 映射。张量载荷约 **6.202 GiB**。

不能把所有 token ID 都改成 49k，或把 `d2t` 当成一个随意可省的文件。上游 loader 明确要求缩减词表必须提供 LM head 和映射；采样后按实际映射恢复到 target ID。[加载与映射检查，L312–371](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dspark.py#L312-L371)

建议先完整词表确认功能，再用 49k 测带宽与接受率的权衡。该模型卡的 GB10/NVFP4 实验不能直接给出我们 AWQ/ROCm 的提升百分比。

### 3.3 其他权重及不可混用的配对

[satgeze/Qwen3.6-27B-DSpark](https://huggingface.co/satgeze/Qwen3.6-27B-DSpark) 也是社区候选，block15。当前模型卡明确描述从 DFlash warm-start 后训练；其他模型卡曾把其对照叫“untrained”，存在版本/描述冲突，本文不沿用为事实。

该卡还记录过 RoPE 导出问题：草稿训练用 full-dimension RoPE，旧配置却继承 target 的 `partial_rotary_factor=0.25`，导致极低接受率。需检查最终解析后的草稿参数，不能自动继承 target 的同名字段。

DeepSpec 官方发布表列有 [Qwen3-4B](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)、[8B](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7)、[14B](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) 与 Gemma4-12B 草稿。当前表未列 Qwen3.6-27B；这与“已有匹配社区权重”并不矛盾。14B 即使同 hidden5120，其词表、层数和特征层仍不同，不能跨接。

DeepSeek-V4/V4.1 的 DSpark 属于各自模型配套模块，不能拿它的草稿参数直接用于 Qwen。DeepInfer 内部类名仍叫 `DFlash`，也不能据类名判断它只实现 DFlash1。

## 4. DFlash2 权重：官方配对与当前模型的实验候选要分开

### 4.1 作者发布的 Qwen3.8-27B 配对

[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 与作者 [DFlash 仓库](https://github.com/z-lab/dflash) 是优先参考。

本次核查的官方 revision 为 `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`：5 层全部 SWA2048、特征层 `[5,19,33,47,61]`、block8、selector Top-K16/rank256。它面向 Qwen3.8，而非当前 Qwen3.6。

该 checkpoint 不含 embedding/LM head，需要运行时从 target 共享相应参数。因此不能只按它约 3.85 GB 文件与 DSpark 的 8.8 GB 文件相减，宣称运行时一定少占对应内存；还要看实际共享、target 高精度 LM head、布局和 cache。

### 4.2 当前 Qwen3.6-27B 的社区候选确实存在

[vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1](https://huggingface.co/vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1)，本次核查 revision：

`a233ec768abb2ed9a2a97a294b5808d0ad10932e`。

它是标准 HF `DFlash2DraftModel`，不是只供 MLX 的文件。真实 header 包含卷积、candidate selector、两份全词表 codebook 与 hidden projection；无 embedding、LM head、confidence、d2t 参数。5 层全为 SWA2048，特征层是 `[5,19,33,47,61]`。

**成熟度限制很具体：**作者卡说明从已训练 head warm-start，针对 non-thinking Qwen3.6 BF16 完成 100 次 optimizer update、处理 400 条训练 call 样本/轨迹片段，并明确尚未做 serving benchmark 或下游 Agent 质量评估。400 是其报告的数据量，不能据此推出 target 推理调用次数；也不能用初始化模型的成绩代替这个新检查点的成绩。

这使它成为合理的研究候选，但还不足以取代我们已有展会配置。接入时应按其配置提取特征，不能沿用旧 DFlash 的 feature-layer 列表；AWQ target 下的接受率、采样和质量也需要重新验证。

## 5. 版本门槛：先选新运行时，再谈切换开关

本次逐标签读取 `vllm/config/speculative.py`：

- `v0.20.0`、`v0.23.0`、`v0.24.0`：未包含 DSpark 配置支持。
- `v0.25.1`：已有 DSpark 方法、模型识别和 V2 runner 路径。
- DFlash2 后来进入上游；v0.28.0 发行说明已列出其支持。当前固定主线还包含后续格式、采样和性能修复。

这不是对所有私有分支的判断；某个写着“0.23”的开发容器可能带额外主线补丁，应以实际 commit/源码确认。也不将 v0.25.1 当作所有 49k、adaptive 和新格式均可用的保证。

[v0.25.1 配置源码](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/config/speculative.py)；[v0.28.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)。

当前源码明确：DSpark 仅由 **V2 GPU model runner** 实现。其工厂已分别创建 DFlash、DSpark 和 DFlash2 speculator；因此双后端的框架抽象已经存在。[V2 限制，L3033–3059](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/config/vllm.py#L3033-L3059)；[方法工厂，L27–52](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/v1/worker/gpu/spec_decode/__init__.py#L27-L52)。

**建议建立独立的新运行时研究镜像，保留原 DFlash 镜像。** 将自研 gfx1151/gfx1100 AWQ 算子注册、数值与必要语义修复迁入新版本，逐项检查哪些旧补丁已经上游解决。不要在 v0.20.0 上机械拷贝几段新类，或把原字符串替换脚本对新版本全部执行。

## 6. 双后端/三模式如何设计

### 6.1 将“目标计算后端”与“草稿算法”分开

`AWQ_MMQ_DECODE_BACKEND=triton/hip` 控制 target 的量化算子；`SPEC_BACKEND=dflash/dspark/dflash2` 控制投机算法。这是两个维度，可以分别 A/B。

对外可统一：

```dotenv
SPEC_BACKEND=dflash
VLLM_DRAFT_MODEL_ID=z-lab/Qwen3.6-27B-DFlash
VLLM_DRAFT_REVISION=已验收的固定revision
VLLM_SPEC_TOKENS=8
VLLM_DRAFT_TP=1
VLLM_SPEC_ADAPTIVE=0
```

以上为**拟新增的项目配置**，当前旧 Compose 不识别这些变量。启动器负责校验组合并生成上游配置，不能把它们写入旧 .env 后就宣称已支持。

映射关系：

- `dflash` → vLLM `method="dflash"` + DFlash checkpoint。
- `dspark` → vLLM `method="dspark"` + DSpark checkpoint。
- `dflash2` → 当前上游仍用 `method="dflash"`，由 `DFlash2DraftModel` 等架构选择 DFlash2Speculator。不能假设要填 `method="dflash2"`。
- `none` → 完全不传 speculative-config，保留 target-only 基线。

### 6.2 N 与 block_size 不能统一照抄

对当前 DSpark block8 候选，`sample_from_anchor=True`，anchor 位置的 hidden 也用于预测未来 token，因此可产生 8 个 proposal；DFlash2 block8 含一个不作为 proposal 的 anchor，通常对应 7 个草稿。不同导出格式还可能选择另一布局，应读取配置及对应版本的 runtime 契约。

研究配置示意（不是当前旧镜像的可直接运行命令）：

```json
{"method":"dspark","model":"abstract-extraordinary/Qwen3.6-27B-DSpark","num_speculative_tokens":8,"draft_tensor_parallel_size":1}
```

```json
{"method":"dflash","model":"vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1","num_speculative_tokens":7,"draft_tensor_parallel_size":1}
```

启动前固定权重 revision，校验 block、anchor/mask、词表映射、特征层、RoPE 和草稿精度。先关闭动态验证，以同 target 的纯文本 greedy 任务建立正确性基线；温度采样、图、长文、多模态和工具调用分开加入。

### 6.3 “切换”推荐先做启动时选择

单实例启动选择一种草稿，切换时重新创建并预热进程。这个模式最容易维护，也避免同一 APU 同时常驻多个草稿/目标副本。

保持两个服务常驻再由 router 分流，可以做无请求中断的入口切换，但往往重复加载 target、布局和 KV，并争用同一 GPU。不能把 W7900 双 TP=4 的资源条件直接套到一台 395。

同进程按请求热切换，则要管理多草稿模型、独立 cache、特征提交、graph、采样状态与 scheduler。这属于新的 runtime 工程，不是增加一个环境变量。第一阶段无需承担这一复杂度。

## 7. 需要改多少代码

### 7.1 复用已有新 vLLM：改动以集成为主

如果新版本能正确运行现有 AWQ target，DSpark/DFlash2 的模型、loader 和采样已有上游实现，通常不需要重写其核心。

项目侧预计涉及 4～6 类文件：新研究 Dockerfile、版本/权重配置、统一启动器、Compose/profile、模式校验与记录、已有 benchmark 的后端参数。这一层可先按**约百行到数百行**规划，属于设计估算，不能当成已测出的总代码量。

另须审查现有 HIP 注册和旧补丁的迁移量。它可能小于运行时改造，也可能因 API/布局变化扩大；真实量应在固定新基线完成首轮 build/load 后，以 diff 统计。

### 7.2 固守 v0.20.0：不是只补两个 head

至少触及配置/注册、模型 loader、特征捕获、块布局、顺序采样与 proposal 概率、V2 执行链、cache/回滚，以及可能的图与量化适配。应按一个独立兼容性项目评估，不能承诺几十行完成。

当前主线的 DSpark 模型、speculator 和 loader 三个直接文件合计约 799 行，但它们复用 DFlash、V2 和采样基础设施。**复制这 799 行不等于完成回移。**

### 7.3 重新训练是另一笔工作

若公开候选不满足质量/接受率要求，需要训练或继续训练匹配目标的草稿。新增 Markov/confidence、卷积/selector 参数必须学习；不能只随机初始化后期待获得文章中的收益。训练数据、target 特征生成、算力预算、导出与回归评价不属于“改几处启动参数”的工作量。

## 8. 对 395 / AMD 的关键成本：不能只看接受率

### 8.1 DSpark 的 Markov 头可能变成带宽负担

由真实权重形状可算出：完整词表 W2 的 BF16 大小约 **121.25 MiB**；49k 版约 **24.01 MiB**。若 8 个顺序位置各重新读取一次，理论读量分别约 970 MiB 和 192 MiB。缓存、融合、batch 和实际 kernel 会改变结果；这是按每位置一次全矩阵读取的条件估算，不能当作 profiler 读量或严格上界。

对低并发 395，rank256 不代表完全没有成本。49k 或上游的 top-k Markov 投影优化有研究价值，但后者需要对应版本、dtype 与数值路径。

当前源码有针对 **NVIDIA CUDA + BF16 activation + NVFP4 权重**的快速 gathered-bias 分支；非 CUDA 和其他组合使用不同路径。因此不能把 NVFP4 作者实验的头部开销直接算到 AMD 的 BF16 DSpark 头上。[平台条件与回退，L95–161](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dspark.py#L95-L161)

### 8.2 DFlash2 的结构潜力与新增成本

DFlash2 先在完整 logits 中选择候选，再只对这些候选读取 codebook、计算转移分数。相较每个位置的全词表 Markov 投影，可能减少顺序阶段的大范围权重读取。它的局部混合又可能改善块后部接受率。

但每层 attention/MLP 的卷积、Top-K、候选 gather、selector 和 sampling 也都有成本。尤其 eager 下很多小调用可能暴露 launch 开销；图/编译模式会改变结论。

当前 grouped-conv 内核主要使用 Triton load/store、浮点运算与静态 tap 循环，没有在所读这段代码中发现写死 NVIDIA SM/TMA 的要求；这只是迁移可行性的线索，**不是已通过 gfx1151 编译与数值测试**。其 `hidden_states.is_cuda` 在 ROCm PyTorch 上也可能为真，不能把它误读成排除 AMD 的条件。[卷积内核，L25–112](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/model_executor/models/qwen3_dflash2.py#L25-L112)

### 8.3 长上下文：当前候选的 attention 类型差异很大

我们已恢复的 DFlash 是 4SWA+1full；选中的 DSpark 是 5full；选中的 DFlash2 是 5SWA。它们的 drafter cache、长上下文读取与接受率曲线不同。

不能强行把 DSpark 的 full 层改成 SWA来沿用现有最佳参数，那会改变草稿行为。也不能仅从全 SWA 推断 DFlash2 一定有最佳接受率：全历史信息可通过 target features 进入，但训练和执行契约仍决定质量。

比较总内存时记录：target 权重和双布局、草稿独有/共享权重、target/draft KV 与递归状态、graph、workspace、CPU 侧缓存和 OS 余量。统一内存使这些项共同竞争实际 RAM，文件大小只覆盖其中一部分。

### 8.4 我们已知的 M=9 边界会影响结果

在常见固定宽度验证中：

```text
当前 DFlash：N=8 草稿，target 验证常见 M=9
候选 DSpark：block8 可对应 8 草稿，target 常见 M=9
候选 DFlash2：block8 含 anchor，7 草稿，target 常见 M=8
```

实际还受 batch、chunk、请求状态和 runtime 布局影响，应从 profile/shape 统计确认。DSpark 的 draft query 使用 N 槽，不表示 target 的 N+1 验证瓶颈自动消失。

如果采用已有 M≤8 专用路径，DFlash2 的自然宽度可能避开 M=9 边界；当前默认 Triton 或其他路由下是否仍有差异，需要实测。这也是为什么不能将“全部参数都写 8”当成公平实验。

建议同时做两类对照：各自推荐且被支持的配置，衡量实际最佳效果；相同实际 proposal 数，尽量隔离验证形状差异。某个旧 DSpark runtime 若要求完整 block，不要为追求相同 N 强行修改配置，先更新实现或单列限制。

### 8.5 完整 DSpark 动态调度与当前 eager 配置存在门槛

DSpark 的静态起草可以先独立接通；confidence 驱动的自适应验证是下一阶段。

当前 vLLM 主线 `enable_adaptive_verification` 默认为 false。启用后明确要求 full graph 模式，并对 LoRA、PP 等组合设限制：

```python
if not self.compilation_config.cudagraph_mode.has_full_cudagraphs():
    raise ValueError(
        "Adaptive verification requires full CUDA graphs. Use cudagraph "
        "mode FULL, FULL_DECODE_ONLY, or FULL_AND_PIECEWISE, or disable "
        "adaptive verification."
    )
```

[验证条件，L3095–3121](https://github.com/vllm-project/vllm/blob/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee/vllm/config/vllm.py#L3095-L3121)

我们 Strix Halo 展示基线仍保留 `--enforce-eager`，因此不能同时承诺直接获得这套动态验证收益。需先在新栈独立验收 HIP Graph、GDN 状态、动态 metadata 与成本曲线。代码中的 CUDA graph 命名沿用框架术语，AMD 上仍要验证实际 HIP 实现和支持范围。

DeepInfer、SGLang 与当前 vLLM 的控制器也不是同一个实现；接通 DSpark head 不等于完整复现某一篇文章的动态调度效果。

## 9. 公开速度证据能说明什么

### 9.1 上游 H200 同控：DFlash2 明显领先过一版 DSpark

vLLM [PR #52816](https://github.com/vllm-project/vllm/pull/52816) 报告同 Qwen3.8-27B、单 H200、GSM8K、K7、T1/p0.95/k20、xhigh、最大输出4096的服务对照，并剔除预热。

并发1：当时 RadixArk DSpark 为 **178.5 token/s**，DFlash2 为 **224.6 token/s**，后者约快 **25.8%**。这是强于跨帖子跑分的对照，但只支持相应 checkpoint/实现/平台，不能直接套用到现在的 DSpark、Qwen3.6 或 AMD。

SGLang 作者 PR 也有同期同方向结果；它与 vLLM 是不同引擎的数字，本文不混用其分母。

### 9.2 更新 DSpark 后，也有接近持平或分任务胜负的结果

[mlx-dspark 配对实验](https://github.com/ARahim3/mlx-dspark/blob/main/README.md) 在 M4 Pro 上以更新的 RedHatAI DSpark、相同 cap、同进程交错测量：8bit target 的两者约持平，4bit 总体 DFlash2 只领先几个百分点，chat 又可由 DSpark 领先。

它属于 Apple/MLX 社区测量，不能预测 AMD；其有效启发是**草稿训练质量与版本足以改变历史差距**。算法名不构成固定排名。

### 9.3 AMD ROCm：DFlash2 已有真实服务报告

vLLM [issue #61761](https://github.com/vllm-project/vllm/issues/61761) 报告单 MI355X、Qwen3.8-27B、DFlash2、K7、V2、ShareGPT、T0/512输出：并发1有显著收益，但同一主线配置并发64又低于 target-only，后续补丁会改变结果。

它证明 ROCm 存在真实路径，也说明量化、GDN、存储与调度会决定系统结果。该报告没有 DSpark 对照，更不是 gfx1151，因此不能据此判定本机算法赢家。

### 9.4 395：已有 DFlash2 报告，但不是我们的 AWQ-vLLM 栈

[固定提交的 395 实验](https://github.com/Nathanw1014/strix-halo-llamacpp/blob/1185a3d2bbcbc0ce650facbc83278b661a199780/docs/dflash2-strix.md) 使用 Framework Desktop 395/64GB、Qwen3.8-27B GGUF、Vulkan、自然文本、两轮测试。

32K 实际上下文处，DFlash2 width4 为约 **21.11 token/s**，width7 约 **16.32 token/s**。没有 DSpark 对照，也不是当前 AWQ/HIP 后端。它支持“值得试验”与“更长草稿不一定更快”，不能拿来和我们 28.3 的短文峰值直接比较。

## 10. GitHub 上有哪些值得采用的方案

### 首选：vLLM 上游，复用现有工程

[vllm-project/vllm](https://github.com/vllm-project/vllm) 已有 DSpark 与 DFlash2 模型、加载、采样和服务路径。DSpark 的 AMD DeepSeek-V4 适配进入过正式发布，但它主要针对 Instinct/gfx950、大 MoE 与配套内核，不能当作 Qwen/RDNA 的无条件支持。

DFlash2 主体 #52816 已合并，v0.28.0 发行说明列出支持；后续 #53797 支持 Speculators 格式。这里已不只是个人 fork。**最适合我们的选择是复用正式上游，再验证自有 target/内核，而非长期维护一套从 DeepInfer 拼出来的完整执行链。**

### 训练与语义：DeepSpec、Speculators、TorchSpec、SpecForge

- [DeepSpec](https://github.com/deepseek-ai/DeepSpec)：作者训练/评价和配对权重，是理解 DSpark 数值与训练语义的首选。
- [Speculators](https://github.com/vllm-project/speculators)：上游训练、导出与多种草稿格式；DSpark 已有正式训练发布。DFlash2 训练配方仍要与作者原始目标区分。
- [TorchSpec](https://github.com/lightseekorg/TorchSpec)：提供训练设施与 DFlash warm-start DSpark 路线；ROCm、流式通信依赖及具体设备需另行确认。
- [SpecForge 的 Qwen3.6 gate](https://github.com/sgl-project/SpecForge/blob/main/scripts/gates/README.md)：有 target→训练→导出→SGLang 验收流程，但单样本过拟合通过不等于通用服务质量和接受率。

这些不是与 vLLM 完全相同职责的 HTTP 引擎，不能简单相互替代。

### SGLang：动态验证的第二参考

[SGLang DSpark 介绍](https://www.lmsys.org/blog/2026-07-06-dspark-sglang) 涵盖置信度校准、变长验证、ragged graph、成本表与重叠调度；DFlash2 也有正式上游实现。

但 [gfx1151 适配 PR #31137](https://github.com/sgl-project/sglang/pull/31137) 在本次核查时仍 Open，不能把普通 BF16 服务测试当成我们 AWQ+DSpark 的验收。迁移完整引擎需要重新接入 HIP 算子和性能工具，近期成本高于在现有 vLLM 路线上研究。

### llama.cpp：可运行的独立 GGUF 路线

DSpark [#25173](https://github.com/ggml-org/llama.cpp/pull/25173) 已上游合并；DFlash2 通过 [#27816](https://github.com/ggml-org/llama.cpp/pull/27816) 进入 master。注意此前的 #27342 先合并到开发分支，不能仅据那次 Merged 判断主线状态。

还有 [MI50 + Qwen3.6 DSpark 用户案例](https://github.com/ggml-org/llama.cpp/issues/26339)，同时报告不同 split 模式的限制。这说明不能笼统说“AMD 没有 DSpark”。不过 GGUF 引擎与当前 AWQ-vLLM 是两套量化/执行路径，适合独立交叉验证，不是替换一个配置即可保持所有基线。

### 社区 fork 与 Strix 专用引擎

Dogacel 的 DSpark fork、Apple vllm-metal、Luce-Org/lucebox 等有各自的运行证据。优先审查上游是否已包含相关能力，再决定是否承担分叉维护。某些 Vulkan 实验使用近似 top-K 接受、并未保持 target 分布，不能与严格验证方案合称为“无损 DSpark”。

**本轮没有找到覆盖我们完整组合的现成成品**：395/W7900 + 当前 Qwen3.6 AWQ4 + 自研 HIP + 多草稿切换 + 长上下文/质量验收。这个检索范围结论不等于对所有公开/私有项目做了不存在证明。详细来源见 [GitHub 调研记录](research/dspark_dflash2_20261022/dspark_github_notes.md)。

## 11. 怎样做出足以回答“谁更好”的最小实验

### 11.1 用有效提交速度评价，接受率作为解释指标

可用近似分解：

```text
每轮有效速度 ≈ E[实际提交 token 数] /
             (T_draft + T_verify + T_sampling + T_metadata/同步)
```

还要纳入首次 prefill、排队与请求结束的 wall time。某方法接受更多 token，却花了更多 draft/采样时间，完全可能更慢。

### 11.2 分离“运行时升级”与“草稿方法”

先在旧环境保留已验证基线；在同一个新环境分别跑：

1. 同 AWQ target-only。
2. 同 target + 当前 DFlash 权重。
3. 同 target + DSpark full。
4. 同 target + DSpark 49k。
5. 同 target + Qwen3.6 DFlash2 实验权重。

如果换成 Qwen3.8 target，则以上另开一组，不能将“新模型+新引擎”的差值全部记到 DFlash2 账上。

### 11.3 先验证语义与状态，再测速度

固定 target/draft revision、tokenizer、chat template、thinking 模式、RoPE、feature层、mask/anchor、KV dtype。DFlash2 的当前 3.6 候选针对 non-thinking，应先在该条件测试。

覆盖全拒绝/部分接受、EOS、长输出、GDN rollback、prefix reuse、请求取消与恢复。纯文本通过后再加图像、工具与 Responses API；不能认为旧 DFlash 多模态通过就代表新 V2 路径全部通过。

greedy 下检查输出与数值；随机采样下正确处理实际 proposal 概率、映射与拒绝重采样，评估分布/质量。浮点执行次序造成的差异与真正的状态错误分别定位。

### 11.4 最小性能网格

建议真实输入深度 2K、8K、16K、32K，并发1为主、2/4作为补充；输出长度固定为256/512或现场实际长度。每个方法扫描该 checkpoint/runtime 支持的宽度；不超出训练和配置契约。

保留相同 target kernel、功率模式、graph、采样、prefix-cache条件；交错 A/B 至少三轮，记录波动。profiler 插桩运行与正式测时分开。

每个点至少记录：TTFT、端到端 wall token/s、TPOT、draft/verify/sampler 分项、平均提交长度、真实 M 分布、峰值 UMA、频率/温度、失败与回退。

DSpark adaptive 单列：先静态通过，再确认 full graph 与成本曲线正确，不能把它与 DFlash2 eager 在不同设置下直接得出纯算法结论。

## 12. 最终决策建议与交付边界

就目前证据，建议按三条线推进：

1. **展会交付线：当前 DFlash。** 延续已经优化的 AWQ、SWA、3D attention 与稳定参数，提前完整演练。
2. **保持现模型的研究线：DSpark full → 49k → DFlash2 v0.1 同机 A/B。** DSpark 先做兼容性是因为当前权重服务信息更充分；DFlash2 进入测试后不预设它更慢。
3. **下一轮架构优化线：重点研究 DFlash2，同时保留 DSpark。** DFlash2 的局部混合、稀疏候选与 SWA 对 395 有值得验证的结构优势；DSpark 的较成熟训练配方、49k 头和 confidence 调度也可能在某些任务/上下文获胜。

如果现在必须决定把哪一个设为“默认新方案”，答案是：**先不改默认；把可切换的工程接口做好，用本机数据决定。** 这不妨碍将 DFlash2 作为重点优化对象，也不妨碍 DSpark 成为先落地的第二后端。

本次完成了检索、源码审查、固定权重配置与小型 tensor header 核验，没有实现双后端切换、训练草稿或测得新的性能。研究原始材料：

- [DSpark 权重核验](research/dspark_dflash2_20261022/dspark_checkpoint_notes.md) 与 [配置/tensor metadata](research/dspark_dflash2_20261022/hf_dspark_metadata.json)。
- [DFlash2 权重核验](research/dspark_dflash2_20261022/dflash2_checkpoint_notes.md)。
- [GitHub 成熟度](research/dspark_dflash2_20261022/dspark_github_notes.md) 与 [运行时/公开对照](research/dspark_dflash2_20261022/dflash2_runtime_comparison_notes.md)。
- [固定新 vLLM 本地源码](https://github.com/vllm-project/vllm/tree/aa31fa038f2cbfbc681ec9f7b7e39c9c728502ee) 与 [项目部署快照](https://github.com/anjoj0/vllm-awq4-qwen-1.0/tree/5f05a501ac51a0bff646f4f98b731f74b40e08b8)。
