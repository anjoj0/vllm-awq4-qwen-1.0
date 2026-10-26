# DSpark 草稿权重核验记录（2026-10-22）

本记录是主任务的只读研究材料。没有下载完整权重、执行远程模型代码或运行 GPU 推理；只读了公开模型卡、配置、版本元数据和两个 safetensors 的 JSON 头。

## 一、可交付结论

- Hugging Face 确实存在 DSpark 草稿权重，包括与当前项目名称直接对应的 **Qwen3.6-27B 社区草稿**。因此“没有公开 Qwen27B DSpark 权重，必须先自行训练”这个判断不成立。
- 优先候选是 `abstract-extraordinary/Qwen3.6-27B-DSpark`（完整词表）及 `abstract-extraordinary/Qwen3.6-27B-DSpark-49k`（截断词表）。可先以完整词表版本验证功能，减少 d2t 映射变量，再评估 49k 的带宽收益。
- 这两个模型与我们的既有 DFlash 接口具有相同的 target64、hidden5120、vocab248320、特征层 [1,16,31,46,61]，是可行性的有力结构证据；但这不等于 AMD AWQ4 下已经验证可运行，也不保证更快。
- 两个候选都使用 **5 层 full attention**，并非我们目前 4×SWA2048+1×full 的优化布局。直接把我们的 SWA 配置补丁套上去会改变草稿语义，需要单独评估或训练。
- DSpark 的训练特征对应具体 target checkpoint。Qwen3.6 名称相同也要核对实际量化母本、revision、tokenizer 和 target 层定义；Qwen3.7 或 Qwen3.8 不能只因同为 27B 而直接混用。

## 二、直接匹配社区权重的独立核验

### 2.1 完整词表版本

[模型仓库](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark)；本次实读版本：`36ccb76a7bf87a7f33cc3b85260e952685a8a7d0`。

[固定版本 config](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark/blob/36ccb76a7bf87a7f33cc3b85260e952685a8a7d0/config.json) 实读字段：

- `architectures=["Qwen3DSparkModel"]`，`model_type="qwen3"`。
- `block_size=8`，`num_hidden_layers=5`，`num_target_layers=64`。
- `hidden_size=5120`，`intermediate_size=17408`，`num_attention_heads=32`，`num_key_value_heads=8`，`head_dim=128`。
- `target_layer_ids=[1,16,31,46,61]`。
- `markov_rank=256`，`markov_head_type="vanilla"`。
- `enable_confidence_head=true`，`confidence_head_with_markov=true`。
- `vocab_size=248320`，无截断词表字段。
- `partial_rotary_factor=1.0`，嵌套 `rope_parameters.partial_rotary_factor=1.0`，`rope_theta=10000000`。
- `mask_token_id=248200`，`dtype="bfloat16"`，`transformers_version="5.14.1"`。
- `layer_types` 全是 `full_attention`，`sliding_window=null`。

只读了 safetensors JSON 头 6,872 bytes，得到 64 个张量项。关键权重真实存在：

```text
confidence_head.proj.weight        BF16 [1, 5376]
confidence_head.proj.bias          BF16 [1]
embed_tokens.weight               BF16 [248320, 5120]
lm_head.weight                    BF16 [248320, 5120]
markov_head.markov_w1.weight       BF16 [248320, 256]
markov_head.markov_w2.weight       BF16 [248320, 256]
```

5376=5120+256，吻合“backbone hidden 与前 token 的 Markov embedding 拼接后预测接受率”的实现。最后一个张量的结束 offset 为 8,800,300,034 bytes，约 8.196 GiB；此为文件张量载荷，不是运行时峰值显存，实际还受加载共享、布局和缓存影响。

### 2.2 49k 截断词表版本

[模型仓库](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark-49k)；本次实读版本：`a3d00995272f95093311c638cb5913db7be3c033`。

[固定版本 config](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark-49k/blob/a3d00995272f95093311c638cb5913db7be3c033/config.json) 的主体结构同上，增加 `draft_vocab_size=49167`，仍保留 `vocab_size=248320`。并没有把所有输入 token embedding 缩成 49k。

只读了 safetensors JSON 头 6,920 bytes，得到 65 个张量项：

```text
d2t                               I64  [49167]
confidence_head.proj.weight        BF16 [1, 5376]
confidence_head.proj.bias          BF16 [1]
embed_tokens.weight               BF16 [248320, 5120]
lm_head.weight                    BF16 [49167, 5120]
markov_head.markov_w1.weight       BF16 [248320, 256]
markov_head.markov_w2.weight       BF16 [49167, 256]
```

实际 tensor header 与作者所说的 row-gather 策略一致：缩减输出 `lm_head` 和 `markov_w2`，保留输入 `markov_w1` 的完整 target token ID 索引。`d2t` 内容没有读取，需让已支持此格式的加载器验证 offset-map 语义，不能把其值直接当成 target token ID。载荷结束 offset 为 6,659,400,314 bytes，约 6.202 GiB。

[49k 模型卡](https://huggingface.co/abstract-extraordinary/Qwen3.6-27B-DSpark-49k/blob/a3d00995272f95093311c638cb5913db7be3c033/README.md) 的速度结果来自作者的 GB10 + NVFP4、并发 1、小样本固定输出实验，只可视为候选筛选信息，不能当成我们 Ryzen AI Max+ 395 + ROCm + AWQ4 的提升比例。长提示场景作者推荐完整词表；我们仍需测端到端 wall time、TTFT、接受长度及峰值内存。

### 2.3 Markov 头的带宽含义

由上面的真实形状可以算出：完整 `markov_w2` 的 BF16 载荷约 121.25 MiB，49k 的约 24.01 MiB。若一个低并发 step 的 8 个顺序 Markov 位置都从显存重新读一次 W2，分别对应约 970 MiB 与 192 MiB 的理论读量；是否能缓存、融合或批处理要依据 profiler 实测。不能只按增加了一个“小 rank256 头”就假设其开销可以忽略。

## 三、另一个 27B 草稿，以及来源冲突

[satgeze/Qwen3.6-27B-DSpark](https://huggingface.co/satgeze/Qwen3.6-27B-DSpark) 是第三个可研究候选。当前模型卡自述从 z-lab DFlash warm-start，并通过 DeepSpec 在线方式训练；[config](https://huggingface.co/satgeze/Qwen3.6-27B-DSpark/blob/main/config.json) 是 `Qwen3DSparkModel`、5 层 full attention、相同特征层和 hidden/vocab，但 `block_size=15`。

当前 satgeze 卡还记载了重要格式问题：旧版本继承 target 的 `partial_rotary_factor=0.25`，与草稿训练时的 full-dimension RoPE 不一致；现已明确写为 1.0。只删字段可能被配置类注入家族默认值，应该校验最终解析后的 config。该案例与 DeepSpec issue [#71](https://github.com/deepseek-ai/DeepSpec/issues/71) 的训练/推理 RoPE 不一致问题有关，但 issue 是报告，不能据此断言所有当前版本仍有 bug。

**来源冲突必须保留**：abstract-extraordinary 模型卡称其对照 satgeze 为 “upstream, untrained”；而 satgeze 当前卡明确自述已训练。可能比较的是旧 revision，或者标签不准确。没有固定当时 baseline revision 前，不应把“satgeze 未训练”和“v2 比未训练 baseline 快 1.59×”叙述成独立核实事实。

## 四、官方权重与其他社区模型分别是什么

### 4.1 DeepSpec 官方公开独立草稿

[DeepSpec Released Checkpoints](https://github.com/deepseek-ai/DeepSpec#released-checkpoints) 当前列出独立 DSpark 草稿：

- [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)
- [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7)
- [deepseek-ai/dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7)
- [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7)

这张官方发布表未列 Qwen3.6-27B，准确结论是“官方当前表内没有列出”，不代表对整个 HF 做了不存在证明。官方表说明这些模型针对对应 target 的 non-thinking 再生成语料训练，thinking/领域分布改变时需重新评估。

[官方 Qwen3-14B draft config](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7/blob/main/config.json) 虽然也有 hidden_size5120，但 vocab151936、target40层、40Q heads、features[1,10,19,28,37]、rope_theta1e6，与 27B 方案不同。因此“同 hidden5120，直接换上 14B drafter”不成立。

### 4.2 DeepSeek-V4 与 V4.1 的集成草稿

[deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) 官方卡明确说明：这是同一个 V4 checkpoint 加上 speculative decoding 模块，并非一个通用小草稿。它不能与 Qwen 的 target features/tokenizer/attention/MoE 随意交叉配对。

[DeepSeek-V4.1-Flash-1017 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) 记录 `num_nextn_predict_layers=3`、`dspark_block_size=5`、`dspark_target_layer_ids=[37,38,39]`、`dspark_markov_rank=256`，其词表是129280，和 Qwen 的248320不同。DeepInfer 的类名仍叫 `DFlash`，但代码内已组合 Markov、confidence 与 semi-AR sampling；类名不能用来判断其只实现原始 DFlash。

### 4.3 当前还存在更多独立 DSpark 草稿

[vLLM speculators 的 DSpark 文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md) 提供 [RedHatAI/GLM-5.2-speculator.dspark-preview](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark-preview)，且说明使用 vLLM `method="dspark"`，并提供小 Qwen 训练例子。其存在证明训练与服务生态已经开始通用化，不能当作 Qwen27B 兼容证据。

[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) 是相邻 27B 版本，作者用 Qwen3.8 的 NVFP4/FP8 target 测试，特征层[5,19,33,47,61]，与当前 Qwen3.6 不同。仅可用来研究 SpecForge/SGLang DSpark，不能直接替代当前配对。

## 五、模型层机制与代码移植范围

[DeepSpec Qwen3DSparkModel](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/qwen3/modeling.py) 具有 DFlash 式特征融合和 block backbone，外加：

1. `build_markov_head(config)` 创建低秩前 token 依赖，用 `markov_w1` 获取上一个 target token embedding，再通过 W2 修正当前 draft logits。Transformer backbone 仍可一块并行计算；token 选择的小头按位置顺序推进。
2. `confidence_head` 接收 hidden，或 hidden+Markov embedding，预测当前位置接受概率。
3. “能生成 DSpark draft token”与“引擎按 confidence + 当前硬件成本曲线分配动态验证长度”是两个阶段，前者接通不代表复现了 DeepInfer 的整个变长调度收益。

所以已有 DFlash 权重不能只把 `method` 改名就拥有训练好的 DSpark 头；而我们已找到真实包含这些头的同目标社区 checkpoint，可省去先训练这一步。现有 AWQ target 算子通常属于 target compute 后端，DFlash/DSpark 属于投机算法后端，两层应独立配置与对照。

## 六、本地研究证据

- [hf_dspark_metadata.json](./hf_dspark_metadata.json)：固定 revision 的 config 与两个 safetensors JSON 头。只包含 tensor metadata，没有 tensor payload。
- [read_hf_draft_metadata.py](./read_hf_draft_metadata.py)：此次只读核验脚本；元数据读取上限2MB，safetensors JSON头上限262136bytes，并在读完头后关闭连接。没有模型下载、反序列化执行或 GPU 调用。
