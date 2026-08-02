# W7900 未使用多卡技术实验补充

## 1. 实验动机

本阶段面向 8 张 AMD Radeon PRO W7900（`gfx1100`，48 GiB/卡）的 Qwen3.6-27B BF16 推理服务，考察原有 TP、KV dtype 与双 TP=4 之外的多卡技术。核心问题有两个：

1. 单个 256K 科研长文请求能否进一步降低首 token 时延；
2. 多个长请求同时到达时，能否提高聚合吞吐并避免重复扫描相同上下文。

候选技术包括 Decode Context Parallel（DCP）、Prefill Context Parallel（PCP）、自定义 All-Reduce、Sequence Parallel、GEMM-通信融合、torch.compile + HIP Graph、Prefix Caching、Prefill/Decode 解耦以及 `TP=4 × PP=2`。实验不以“能通过参数校验”为完成标准，而以端到端可运行、数值路径明确且存在可重复收益为准。

## 2. 实验环境与统一配置

| 项目 | 配置 |
|---|---|
| GPU | 8 × Radeon PRO W7900，`gfx1100` |
| 模型 | Qwen3.6-27B BF16 |
| vLLM | 容器 0.23 系列；历史优化工作树与 2026-08-01 上游工作树 |
| 主注意力后端 | Triton unified attention，prefill tile=16 |
| 主长文配置 | TP=8，auto KV，`max_model_len=262144` |
| 页大小 | 优化 profile 使用 `block_size=784` |
| 输出长度 | 并发实验均为 64 token/请求 |
| 测试文本 | `combined_papers_for_llm_L.txt` |

为区分上游版本影响，报告中的 A/B 只在相同工作树、runner、页大小和 prompt 构造下计算加速比。跨工作树数据只用于说明兼容性和趋势，不直接归因于单个开关。

## 3. 技术路线结论总览

| 技术 | 256K 单请求 | 多请求 | 实验结论 |
|---|---|---|---|
| Prefix Caching + 784 页 | 显著有效 | 显著有效 | 正式推荐，用于同一长文连续问答 |
| torch.compile + HIP Graph | 已是 BF16 主路线 | 4×64K 明显有效 | 正式推荐，不能再将其视为“尚未使用” |
| `TP=4 × PP=2` | 不适合低时延 | 4×60K 明显退化 | 仅增加 KV 容量，不作为速度 profile |
| Sequence Parallel + 融合通信 | 未形成可运行路径 | 未形成可运行路径 | ROCm 编译通路被 `AsyncTPPass` 缺失阻塞 |
| 自定义 All-Reduce | 未激活 | 未激活 | 当前 ROCm 构建最终仍选择 PYNCCL |
| DCP | 理论上针对长 decode | 尚不可测 | 缺少满足 paged-KV/LSE 契约的 ROCm FlashAttention 后端 |
| PCP | 理论上针对长 prefill | 尚不可测 | 当前实现仅支持 MLA，Qwen3.6 非 MLA |
| Prefill/Decode 解耦 | 可能改善服务吞吐 | 尚不可测 | 容器未安装 NIXL/LMCache，不能声称性能收益 |

## 4. Prefix Caching：本阶段最重要的有效优化

### 4.1 为什么默认前缀缓存会使首问变慢

Qwen3.6 同时包含 Attention 与 Mamba/GDN 状态。启用 prefix caching 后，vLLM 将混合缓存切换到实验性的 `align` 模式，默认选择约 400-token 对齐页。最新版工作树中，261K 首问无论是否开启 prefix caching，wall time 都约为 294–296 秒；因此慢点并非“写入缓存”的成本，而是新版混合 KV 页布局和内核路径的变化。

历史优化工作树在关闭缓存时会选择 784 页，已有 261K wall time 156.47 秒。实验因此尝试组合 V1 model runner、`mamba_cache_mode=align`、prefix caching 与显式 `block_size=784`。V2 runner 会在 KV 初始化阶段安全拒绝该组合，错误为“V2 尚未支持 align 模式的显式页大小”；V1 runner 则能启动并通过短请求、长请求和缓存命中验证。

### 4.2 261K 单请求与连续问答

| 配置 | 首问 TTFT | 首问 wall | 后续问题 TTFT | 后续问题 wall |
|---|---:|---:|---:|---:|
| 最新工作树，默认 400 页，V1，prefix on | 292.76 s | 294.31 s | 1.60 s | 3.17 s |
| 历史工作树，784 页，V1，prefix on | **150.72 s** | **153.95 s** | **1.43 s** | **4.67 s** |

784 页将首问 wall time 相对默认 400 页降低 47.7%，同时保留了长前缀复用能力。后续问题生成长度为 64 token，因此 wall time 不宜与前一组较短输出直接比较；关键指标是 261K 前缀扫描已由约 151 秒降至 1.43 秒 TTFT。

8K 短测中，首问 TTFT 为 2.78 秒，第二个不同问题为 0.17 秒，说明命中来自共同文档前缀，而不是完全相同请求的结果复用。

### 4.3 4 个 261K 问题并发

测试先建立同一 261K 文档前缀，再同时提出 4 个不同问题，每个问题生成 64 token。无缓存对照使用完全相同的工作树、V1 runner、TP=8、tile=16 和 784 页。

| 配置 | 4 请求 wall | 平均 TTFT | 平均请求 wall | 聚合输出吞吐 |
|---|---:|---:|---:|---:|
| Prefix cache 关闭 | 616.82 s | 383.28 s | 615.46 s | 0.415 tok/s |
| Prefix cache 命中 | **13.09 s** | **6.15 s** | **13.02 s** | **19.56 tok/s** |

共享前缀使 4 请求总 wall time 缩短 97.9%，端到端加速 47.1 倍；平均 TTFT 下降 98.4%，聚合输出吞吐提高到原来的 47.1 倍。

无缓存日志解释了这一数量级差异：4 个请求并非立即完成 4 路 prefill，而是从 1 路运行、3 路等待，逐步扩展到 4 路，反复读取同一 261K 文档。缓存命中后只处理每个问题的短后缀和 decode，消除了约 104 万个重复 prompt token 的主计算量。

该结论只适用于共享长前缀的工作负载，例如同一论文、代码库或病例上的多轮和多用户问答。对于 4 篇完全不同的 261K 文档，prefix caching 不会产生该收益。

## 5. torch.compile + HIP Graph：长文多请求的默认执行方式

在相同历史工作树、V1 runner、TP=8、784 页、prefix cache 关闭条件下，对 4 个约 63.5K prompt、每请求 64 token 进行 A/B：

| 执行方式 | 4 请求 wall | 平均 TTFT | 聚合输出吞吐 |
|---|---:|---:|---:|
| `enforce_eager=True` | 121.99 s | 74.90 s | 2.10 tok/s |
| torch.compile + HIP Graph | **78.18 s** | **48.31 s** | **3.27 tok/s** |

图执行使 wall time 下降 35.9%，聚合吞吐提高 56.0%。因此 BF16 长文服务应默认启用编译与图捕获。此前部分 AWQ4/DFlash profile 使用 eager，是为了规避自定义量化算子和 speculative 路径的图兼容问题，不能推广为 BF16 TP=8 的默认设置。

编译缓存还使相同 profile 的服务重启由约 5 分钟缩短到约 83 秒，但这是部署冷启动收益，不应与请求 TTFT 混为一谈。

## 6. `TP=4 × PP=2`：容量增加，但速度明显退化

### 6.1 V2 与 V1 的功能差异

Qwen3.6 能完成两段流水线的权重切分。V2 model runner 在 261K prefill 后、首 token 采样阶段失败，4 个 PP0 worker 均报：

```text
IndexError: index_fill_(): Expected dtype int64 for index
vllm/v1/worker/gpu/model_states/mamba_hybrid.py
```

切换 V1 model runner 后，短请求能够生成，说明模型的 PP 结构本身成立；错误来自 V2 混合 Mamba 状态的索引类型。

### 6.2 多请求性能

| 配置 | prompt/请求 | 4 请求 wall | 聚合输出吞吐 | KV token 容量 |
|---|---:|---:|---:|---:|
| TP=8，PP=1 基线 | 63.5K | 109.71 s | 2.333 tok/s | 约 205 万 |
| TP=4，PP=2，V1 | 60.0K | 295.61 s | 0.866 tok/s | 约 398 万 |

尽管 PP=2 的 prompt 略短，wall time 仍为基线的 2.69 倍，吞吐仅为 37.1%。其 KV token 容量约提高 94%，原因是每个流水段只保存部分层的权重和 KV 状态；但每个 microbatch 需要跨 PP 段传递激活，且长请求下 pipeline bubble 与点对点通信无法被当前并发充分掩蔽。

因此 `TP=4 × PP=2` 是容量 profile，不是低时延或高吞吐 profile。当前节点已有的双 TP=4 是两个独立服务，短请求 8 并发吞吐较单 TP=8 提高 23.1%；它与单 engine 的 PP=2 不是同一种并行方式。

## 7. 当前无法形成性能结论的路线

### 7.1 Decode Context Parallel

Qwen3.6 有 4 个 KV heads。对 GQA 而言，在 TP=8 下只有 DCP=2 具有有效分片关系。实际启动检查表明：

- `FLASH_ATTN + FP8 KV` 不支持该 KV dtype；
- `FLASH_ATTN + auto KV` 检测不到 vLLM 所需的 ROCm paged-KV/LSE 扩展；
- 现有 Triton/ROCM_ATTN decode 路径不提供 DCP 合并所需的 LSE。

DCP 需要各卡返回局部 softmax 的 log-sum-exp，再进行数值正确的跨卡归并。绕过后端检查可能产生静默数值错误，因此本阶段没有用不完整实现生成性能数字。

### 7.2 Prefill Context Parallel

当前 V2 实现只对 MLA 模型开放 PCP，Qwen3.6 属于非 MLA 混合 Attention/Mamba 模型，配置层会明确拒绝。PCP 在理论上适合 100K+ prefill，但不能把其他模型的结果外推到本项目。

### 7.3 自定义 All-Reduce

TP=2 启动时已移除 `--disable-custom-all-reduce`，但最终 engine config 仍为 `disable_custom_all_reduce=True`，通信调度只选择 `PYNCCL`，候选 `CUSTOM` 未激活。对于本机 PCIe-only 拓扑，TP=4/8 也不满足 vLLM 自定义 All-Reduce 的全连接要求。因此没有伪造“开/关 A/B”；当前可用主线仍是 RCCL/PYNCCL。

### 7.4 Sequence Parallel 与 GEMM-通信融合

V2 runner 会拒绝 Sequence Parallel。V1 中将 `enable_sp=True`、`fuse_gemm_comms=True` 且 `sp_min_token_num=1` 后，最终 engine config 确实保持启用，但 AOT 编译在所有 worker 上失败：

```text
NameError: name 'AsyncTPPass' is not defined
vllm/compilation/passes/pass_manager.py
```

这说明参数入口存在，但当前 ROCm 构建没有完整注册异步 TP pass。该路线尚未到达可比较性能的阶段。

### 7.5 Prefill/Decode 解耦

容器中未安装 `nixl` 和 `lmcache`。P/D 解耦还需要验证 GPU KV 传输、混合 Mamba 状态迁移和 ROCm 通信正确性；仅启动两个服务并不能构成 P/D 解耦。因此本阶段不报告该路线的吞吐数字。

## 8. 最终工作负载路由结论

本阶段没有得到“一种并行方式覆盖所有请求”的结论，而是进一步明确了 W7900 节点的四类 profile：

| 工作负载 | 推荐配置 | 依据 |
|---|---|---|
| 同一 64K–261K 文档连续或并发问答 | BF16 TP=8，V1，tile=16，block=784，prefix cache，compile+HIP Graph | 4×261K 缓存命中相对无缓存加速 47.1 倍 |
| 不同的 100K–261K 文档，单大请求优先 | BF16 TP=8，auto KV，tile=16，compile+HIP Graph | 原有 100K+ 与 261K 低时延主线 |
| 不同的约 64K 文档并发 | BF16 TP=8，compile+HIP Graph | 相对 eager wall time 下降 35.9% |
| 短上下文多租户 | 双 TP=4 独立服务 | 既有 8 并发吞吐提高 23.1%，优于单 engine PP=2 |

最重要的新认识是：对科研长文服务，优化目标不能只写成“支持 256K”。第一次读入长文仍由 TP=8、tile=16 和 784 页负责；同一长文的后续分析则应从“重复执行 256K prefill”转化为“复用长前缀，只计算问题后缀”。这使多卡优化从单请求算子调优扩展为面向真实科研工作流的状态复用与请求路由。

## 9. 证据位置

- 原始日志与 JSON：`multicard_frontier_20260802/`
- 统一结果：`multicard_frontier_20260802/consolidated_results.jsonl`
- 多卡启动器：`codes/vllm-awq4-qwen-1.0-main/w7900_optimization/scripts/start_bf16_multicard_variant.sh`
- 共享前缀并发 harness：`codes/vllm-awq4-qwen-1.0-main/w7900_optimization/scripts/bench_prefix_concurrency_stream.py`

实验结束后已停止 vLLM 服务，保留 `xdhpc` 容器；8 张 GPU 的已分配显存均为 0%。
