# W7900 多卡推理与 DFlash 优化

该目录包含 Qwen3.6-27B BF16/AWQ4 在单节点 8× AMD Radeon PRO W7900（`gfx1100`，48 GiB/卡）上的代码、vLLM 补丁、启动 profile、科研长文数据集与实验结果。

W7900 路线不是 Strix Halo UMA 参数的直接迁移。它围绕 RDNA 3 内核形状、离散显存 KV Cache、多卡通信、短上下文 speculative decoding 和 64K–261K 科研长文服务重新设计。

## 当前结论

系统采用按上下文长度、前缀复用关系和并发模式路由的多 profile 设计：

| 工作负载 | 推荐配置 | 决定性证据 |
|---|---|---|
| <=14K、batch=1 | AWQ4 target TP=4 + DFlash N=4，draft TP=1 | 8K/12K 相对 target-only 快 27.3%/7.7% |
| >14K AWQ4 | target-only TP=4 | 16K 起 DFlash 越过交叉点；32K 慢 28.7% |
| 短上下文多租户 | 双 TP=4 独立服务 | 8 并发 159.66 tok/s，较单 TP=8 提高 23.1% |
| 不同的 64K 文档并发 | BF16 TP=8 + compile/HIP Graph | 4 请求 wall 121.99→78.18 s，吞吐提高 56.0% |
| 不同的 100K–261K 文档 | BF16 TP=8 + auto KV + tile=16 | 100K+ 单大请求低时延主线 |
| 同一 64K–261K 文档连续/并发问答 | BF16 TP=8 + V1 + block=784 + prefix cache + HIP Graph | 4×261K wall 616.82→13.09 s，加速 47.1× |
| KV 容量优先 | FP8 KV 或 TP=4×PP=2 容量 profile | FP8 KV 容量约 1.99×；PP=2 KV 容量约 205万→398万 token |
| 4 个独立 64K 请求、8 卡 | dual TP=4 replicas | batch wall 120.06 s；P/D TP=4+4 为 229.93 s |

不存在一个对所有请求都最优的固定 TP、KV dtype 或 speculative token 数。该项目的核心是 backend-aware、shape-aware 和 workload-aware 的联合路由。

## 已实现的技术

### gfx1100 算子与后端

- `patch_w7900.py`：对 vLLM 0.23 工作树应用幂等补丁，开放 unified attention tile/launch 参数并检查残留 `gfx1151` 硬编码。
- `vllm_overrides/rdna3_w4a16.py`：整模型 `RDNA3W4A16LinearKernel`，支持非对称 compressed-tensors W4A16、zero-point 语义和 TP 分片。
- `csrc/awq_mmq_gfx1100/`：独立 HIP W4A16 MMQ、Python binding、数值正确性测试与 prefill microbenchmark。
- 普通长 prefill 使用 `tile=16`；DFlash small-query verification 使用 `tile=32, warps=4`。两个 shape 域分别调度，不共享一个全局最优参数。

### DFlash 五路线闭环

1. 恢复 checkpoint 的 `4×SWA(2048) + 1×full` draft attention 语义。
2. 为 `head_dim=128, GQA=4, query<=9` 的非因果 small-query 热路径设置 gfx1100 参数。
3. 在 N=0/4/8 中按真实 prompt token 数和 batch 选择；当前有效决策是 DFlash N=4 与 target-only 的切换。
4. 移植 D-Cut PR #47131，并补齐 V2 logits confidence、keep length 回传和 scheduler 截断。
5. 验证 full draft layer 的 recent-window 上下文压缩。

### 多卡与超长上下文

- BF16 TP=8：不同 100K+ 科研长文的主速度路线。
- 双 TP=4：两个同 NUMA 独立实例处理短请求，提高多租户聚合吞吐。
- Prefix Caching：复用同一长文的 KV/Mamba 对齐状态，避免重复执行 261K prefill。
- torch.compile + HIP Graph：BF16 长文多请求的默认执行方式。
- auto/FP8 KV 双 profile：auto 优先单请求时延，FP8 优先容量。
- `TP=4 × PP=2`：完成 V1 功能闭环和容量测试，但不作为速度 profile。
- 统一启动器记录 TP/PP/DCP/PCP、KV dtype、prefix cache、block size、eager/graph、SP/fused-comms、工作树和完整命令。

### 数据、质量与可复现性

- `longdoc_sanity/`：Nowcast3D 主题科研长文数据集，覆盖事实证据、数字、needle、引用归因和拒答。
- `adaptive_dflash_router.py`：按 tokenizer 后的真实 token 数在 DFlash 与 target-only 服务间路由。
- `scripts/bench_prefix_reuse_stream.py`：测量同一长文连续问题的 TTFT 和 wall time。
- `scripts/bench_prefix_concurrency_stream.py`：预填充共享长前缀，再并发提出不同问题并记录聚合吞吐。
- 每次正式实验保存 manifest、环境版本、实际 prompt/output token、冷/热状态、TTFT、wall time 和服务日志。

## DFlash 实验结果

### 混合 SWA 语义

DFlash checkpoint 包含 4 个 2048-token sliding-window 层和 1 个 full-attention 层。旧路径把五层都按 full attention 执行。恢复训练时语义后：

| Prompt | 5×full | 4×SWA(2K)+1×full | wall 降低 |
|---:|---:|---:|---:|
| 8K | 10.93 s | 9.66 s | 11.6% |
| 16K | 23.73 s | 19.40 s | 18.2% |
| 32K | 59.93 s | 47.35 s | 21.0% |

接受率没有下降。该项同时减少 4 个 draft layer 的长 KV 扫描，并使推理语义重新匹配 checkpoint 训练分布，是 DFlash 五条路线中最稳定的改进。

### Triton 与 ROCM_ATTN

ROCM_ATTN 在 16K target prefill 上可使总 wall 略低约 1.4%，但 DFlash decode 阶段比 Triton 慢约 20%–30%，且冷首请求出现过 0% acceptance、`QA=0/4` 和 83.4 s 异常。两种后端使用不同 KV 物理布局，不能在同一服务中直接混用。

可靠默认路径为全 Triton：

```text
普通长 prefill          -> tile=16
DFlash/verification    -> tile=32, warps=4
```

### 自适应 N 与 14K 交叉点

| Prompt | Target only | DFlash N=4 | DFlash N=8 | 推荐 |
|---:|---:|---:|---:|---|
| 8K | 13.19 s | **9.59 s** | 9.66 s | N=4 |
| 12K | 15.59 s | **14.39 s** | - | N=4 |
| 16K | **18.86 s** | 19.47 s | 19.40 s | N=0 |
| 32K | **36.80 s** | 47.90 s | 47.35 s | N=0 |

N=4 与 N=8 的差异始终小于约 1.2%。增加 N 不能解决长上下文退化，真正重要的是在约 14K 处关闭 DFlash。

### 已验证但默认关闭的路线

| 路线 | 结果 | 结论 |
|---|---|---|
| D-Cut，keep ratio=0.75 | 8K/16K/并发4 分别慢 1.0%/2.1%/1.0% | 功能与质量通过，但同步开销覆盖收益 |
| full draft layer 最近 8K/16K | 16K 慢 17.7%；32K 慢 7.3%/6.7% | 接受率下降导致 target 重算 |
| draft TP=4 | 16K/32K 慢 0.6%/0.5% | 5 层 drafter 太小，collective 开销更高 |

D-Cut 默认关闭；full draft layer 保留完整上下文；`draft_tensor_parallel_size=1`。

## 多卡探索结果

### 261K 共享前缀

Qwen3.6 是 Attention 与 Mamba/GDN 混合模型。启用 prefix cache 后，vLLM 使用实验性的 `mamba_cache_mode=align`。默认 400-token 页使最新版工作树的 261K 首问约为 294 s；历史优化工作树的 V1 runner 可以同时使用 `block_size=784` 与 prefix cache：

| 配置 | 首问 TTFT | 首问 wall | 后续问题 TTFT | 后续问题 wall |
|---|---:|---:|---:|---:|
| 默认 400 页，V1，prefix on | 292.76 s | 294.31 s | 1.60 s | 3.17 s |
| 784 页，V1，prefix on | **150.72 s** | **153.95 s** | **1.43 s** | **4.67 s** |

在 4 个不同问题共享同一 261K 文档的并发实验中：

| 配置 | 4 请求 wall | 平均 TTFT | 聚合输出吞吐 |
|---|---:|---:|---:|
| Prefix cache 关闭 | 616.82 s | 383.28 s | 0.415 tok/s |
| Prefix cache 命中 | **13.09 s** | **6.15 s** | **19.56 tok/s** |

端到端加速 47.1×，平均 TTFT 下降 98.4%。该收益来自复用共同文档前缀，不适用于四篇互不相同的 261K 文档。

### compile + HIP Graph

相同 V1/TP=8/784 页配置下，4 个约 63.5K 请求、每请求输出 64 token：

| 执行方式 | wall | 平均 TTFT | 聚合输出吞吐 |
|---|---:|---:|---:|
| eager | 121.99 s | 74.90 s | 2.10 tok/s |
| compile + HIP Graph | **78.18 s** | **48.31 s** | **3.27 tok/s** |

图执行使 wall 下降 35.9%，吞吐提高 56.0%。BF16 长文服务默认使用编译和图捕获；AWQ4/DFlash 的 eager 配置仅用于自定义算子和 speculative 路径的兼容性调试。

### TP=4 × PP=2

V1 runner 可以运行该配置，并把 KV token 容量从约 205 万提高到 398 万；但 4×约60K wall 为 295.61 s，TP=8/PP=1 基线为 109.71 s，吞吐仅为基线的 37.1%。因此 PP=2 只保留为容量 profile。

V2 runner 在首 token 采样阶段触发：

```text
IndexError: index_fill_(): Expected dtype int64 for index
vllm/v1/worker/gpu/model_states/mamba_hybrid.py
```

### Prefill/Decode 解耦

本项目从源码构建 UCX 1.22 ROCm 与 NIXL 1.4.0，并用 vLLM 原生 `NixlConnector` 打通 Qwen3.6-27B BF16 的 TP=4 Prefill 与 TP=4 Decode。该模型需要同时迁移 Attention KV、Mamba convolution state 和 SSM state；设置 `VLLM_SSM_CONV_STATE_LAYOUT=DS` 后，8K 短请求热态 greedy 输出与直接 Decode 逐字一致。

P/D 相对单个 TP=4 服务体现出阶段隔离：4×64K 的平均请求完成时间由 228.27 s 降到 146.34 s，下降 35.9%，已进入 Decode 的请求不再被后续长 Prefill 阻塞。但该比较使用 8 卡对 4 卡。相同 8 卡资源下，两个独立 TP=4 副本的 batch wall 为 120.06 s，P/D 为 229.93 s；dual TP=4 的聚合吞吐约为 P/D 的 1.91 倍。

UCX/TCP 基线在 W7900 上将 GPU RMA 回退到 `tcp/bond0`：64K 时每个 TP rank 迁移约 1.02 GiB，平均耗时约 3.9 s。为此项目实现了 NIXL 动态 backend `W7900_HIP_IPC`，保留 NIXL scheduler、metadata、TP mapping、KV block lease 和 metrics，只将同节点 GPU payload 改为 `hipIpcGetMemHandle` / `hipIpcOpenMemHandle` / `hipMemcpyAsync`。1 GiB 裸 NIXL plugin 达到 `25.12 GiB/s`；64K vLLM 单请求的有效 payload 带宽约 `25.26 GB/s`。

原生 plugin 的 64K 热态 TTFT 为 `55.633 s`、wall 为 `57.292 s`，相对 UCX/TCP 分别下降约 `7.2%` 和 `6.9%`。并发 4 的 batch wall 下降约 `2.0%`，mean TTFT 下降约 `3.3%`；累计 28 次 rank transfer、30.67 GB payload 无传输或通知失败。该 backend 已成为本项目 P/D 场景的推荐同节点数据面，但 P/D 架构本身仍不是固定并发下的最高聚合吞吐 profile：同资源 dual TP=4 仍能避免阶段解耦带来的排队和容量分割。完整实现与数据见 [HIP IPC transport](hip_ipc_transport/README.md)、[P/D 环境](pd_disaggregation/README.md) 和 [实验报告](results/20260803_w7900_hip_ipc_transport.md)。

在此基础上，非对称启动器将 8 张卡统一组织为 TP2 副本，并支持 `p2_d6`（1 Prefill + 3 Decode）、`p4_d4`（2 + 2）和 `p6_d2`（3 + 1）。64K 输入、32-token 输出、并发 6 时，`p6_d2` 相对 `p2_d6` 将 batch wall 从 642.43 s 降到 218.66 s，说明长输入短输出应优先增加 Prefill 副本。8K 输入、2,048-token 输出、并发 12 时最优点反转，`p2_d6` 的 wall 为 188.31 s，较 `p6_d2` 的 212.49 s 低 11.4%，聚合输出吞吐高 12.8%。因此项目不把物理 4/4 固化为唯一方案，而是依据 Prefill 工作量、Decode token 数和 TTFT/吞吐目标选择资源比例。128K、3 并发在 TP2+FP8 KV、16K chunk 下能够完成，但约 15 分钟 TTFT，仅定义为容量模式。完整矩阵见 [非对称 P/D 实验](results/20260804_asymmetric_pd_matrix.md)。

### 当前上游能力边界

| 技术 | 当前状态 | 证据 |
|---|---|---|
| DCP | 未形成可运行性能路径 | Qwen3.6 TP=8/DCP=2 需要 paged-KV softmax LSE；现有 ROCm FlashAttention/Triton/ROCM_ATTN 不满足接口 |
| PCP | 不适用 | 当前实现仅支持 MLA，Qwen3.6 非 MLA |
| 自定义 All-Reduce | 未激活 | TP=2 移除禁用参数后 engine 仍选择 PYNCCL；PCIe-only TP=4/8 不满足全连接条件 |
| Sequence Parallel + fused comms | 编译失败 | 强制启用后 `AsyncTPPass` 在 ROCm 构建中未定义 |
| Prefill/Decode 解耦 | 同节点数据面已优化 | 原生 `W7900_HIP_IPC` backend 完成混合状态迁移；64K TTFT 较 UCX/TCP 下降约 7.2%，但固定并发聚合吞吐仍低于 dual TP=4 |

这些项目属于已验证的能力缺口，不是负性能结果；README 不将其描述为已实现优化。

## 上游来源与补丁

| 能力 | 来源 |
|---|---|
| unified attention tile/launch 参数 | vLLM PR #45207 / commit `55da232d` |
| DFlash 非因果/per-sequence causal attention | PR #44652 的等价上游能力 |
| 混合 SWA、多 KV group V2 路径 | PR #47914、#48113 |
| D-Cut confidence pruning | PR #47131，并补充本项目 V2 scheduler 路径 |
| P/D 混合状态迁移 | vLLM 原生 `NixlConnector`，NIXL 1.4.0，UCX 1.22 ROCm |
| VRAM memtype hint 实验 | NIXL PR #1536 核心逻辑的 1.4.0 前移植；协议仍回退 TCP |
| 同节点 GPU 数据面 | 本项目 `W7900_HIP_IPC` NIXL backend；HIP IPC payload + Unix datagram notification |
| 固定审查基线 | vLLM main `63e78ce3652f4f94e9f484f40db71ca4cf019f21` |

完整可审查补丁位于 `patches/vllm-main-63e78ce-w7900-dflash-five-routes.patch`，应用方法见 [patches/README.md](patches/README.md)。

## 构建

### 现有 ROCm vLLM 容器

```bash
cp .env.w7900.template .env.w7900
set -a
source .env.w7900
set +a

bash scripts/check_w7900_node.sh
bash scripts/prepare_local_vllm.sh
bash scripts/build_local_vllm.sh
```

容器流程从 `/app/vllm` 复制到 `/workspace/vllm-w7900-023` 后修改，避免污染基础源码。

### 独立 W7900 镜像

使用仓库根目录 `../Dockerfile.w7900` 和本目录 `.env.w7900.docker.template`。该 Dockerfile 继承实测 ROCm 7.14/vLLM 0.23 镜像并固定 `PYTORCH_ROCM_ARCH=gfx1100`；不要使用面向 Strix Halo/`gfx1151` 的根 `../Dockerfile`。

### 单独构建 W4A16 HIP 算子

```bash
cd csrc/awq_mmq_gfx1100
PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
python test_correctness.py
python validate_prefill_numerics_gfx1100.py
python benchmark_prefill_gfx1100.py
```

### Prefill/Decode 解耦环境

Python 3.14 容器不能直接安装要求 Python `<3.14` 的 LMCache 0.3.6。使用 vLLM 原生 NIXL connector 的 UCX/NIXL 源码构建和启动方法见 [pd_disaggregation/README.md](pd_disaggregation/README.md)；同节点 HIP IPC backend 的构建与分层门禁见 [hip_ipc_transport/README.md](hip_ipc_transport/README.md)。

NIXL #2039 的上游复核表明，UCX 1.22 在两张 W7900 间能够直接使用 `rocm_ipc`，1 GiB UCP GET/PUT 分别为 `18.243/11.261 GB/s`；当前 NIXL 路径尚未保留该 lane。UCX fallback 默认采用 `UCX_TLS=sm,rocm,tcp,self` 与 `UCX_RMA_PPLN_ENABLE=y`，上游工作聚焦修复 NIXL UCX backend，而不是新增重复 transport plugin。实验见 [UCX `sm` 上游复核](results/20260805_ucx_sm_upstream_followup.md)。

## 启动与复现

### AWQ4 与 DFlash

通用 AWQ4 服务：

```bash
bash scripts/start_local_vllm.sh
```

双 TP=4 服务分别监听 8061（DFlash N=4）和 8062（target-only）后，启动上下文感知路由器：

```bash
python adaptive_dflash_router.py \
  --tokenizer /models/Qwen3.6-27B-AWQ \
  --dflash-url http://127.0.0.1:8061 \
  --target-url http://127.0.0.1:8062 \
  --threshold-tokens 14000 \
  --port 8060
```

响应头 `X-DFlash-Route` 和 `X-Prompt-Tokens` 保存实际路由与 token 数。

### BF16 TP=8 + 261K Prefix Cache

该 profile 使用 V1 model runner。不要与需要 V2 混合 SWA 的 DFlash 服务合并为同一个进程。

```bash
export VLLM_WORKTREE=/workspace/vllm-w7900-023
export VLLM_USE_V2_MODEL_RUNNER=0
export TP=8
export PP=1
export PREFIX_CACHING=enable
export BLOCK_SIZE=784
export KV_CACHE_DTYPE=auto
export ENFORCE_EAGER=0
export LOG=/workspace/multicard_frontier/bf16_tp8_prefix784.log

bash scripts/start_bf16_multicard_variant.sh
```

连续问题：

```bash
python scripts/bench_prefix_reuse_stream.py \
  --file /workspace/bench_data/combined_papers_for_llm_L.txt \
  --chars 950000 \
  --requests 3 \
  --max-tokens 64
```

建立一次 261K 前缀后并发提出四个不同问题：

```bash
python scripts/bench_prefix_concurrency_stream.py \
  --file /workspace/bench_data/combined_papers_for_llm_L.txt \
  --chars 950000 \
  --concurrency 4 \
  --max-tokens 64 \
  --output /workspace/multicard_frontier/prefix_261k_c4.json
```

### 科研长文质量门禁

```bash
cd longdoc_sanity
python validate_suite.py
python run_longdoc_sanity.py --help
python score_longdoc_sanity.py --help
```

正式实验应记录 tokenizer 后的真实 token 数，先执行 health check 和 shape warmup，再区分冷态与热态结果。源文件字符数只用于稳定构造输入，不能代替模型 token 数。

## 目录结构

```text
w7900_optimization/
├── adaptive_dflash_router.py
├── csrc/awq_mmq_gfx1100/
├── longdoc_sanity/
├── patches/
├── pd_disaggregation/
│   ├── activate_pd_env.sh
│   ├── start_prefill_tp4.sh
│   ├── start_decode_tp4.sh
│   ├── start_proxy.sh
│   ├── start_asymmetric_pd.sh
│   ├── start_pd_worker.sh
│   ├── stop_asymmetric_pd.sh
│   └── benchmark_pd.py
├── results/
│   ├── 20260802_dflash_five_routes.md
│   ├── 20260802_pd_disaggregation.md
│   ├── 20260804_asymmetric_pd_matrix.md
│   ├── 20260802_dflash_data/
│   ├── 20260802_dflash_figures/
│   ├── 20260802_multicard_frontier.md
│   └── 20260802_multicard_frontier_results.jsonl
├── scripts/
│   ├── start_bf16_multicard_variant.sh
│   ├── bench_prefix_reuse_stream.py
│   ├── bench_prefix_concurrency_stream.py
│   └── ...
├── tests/
├── vllm_overrides/
└── patch_w7900.py
```

## 结果与证据

- [凝练实验结果](../docs/EXPERIMENT_RESULTS.md)
- [W7900 完整实验报告](../docs/W7900_FULL_EXPERIMENT_REPORT.md)
- [科研长文质量与 rocprof 边界](../docs/W7900_QUALITY_AND_ROCPROF.md)
- [DFlash 五路线实验总结](results/20260802_dflash_five_routes.md)
- [DFlash 原始聚合数据](results/20260802_dflash_data/aggregated_valid_runs.csv)
- [多卡与 261K 共享前缀实验](results/20260802_multicard_frontier.md)
- [Prefill/Decode 解耦实验](results/20260802_pd_disaggregation.md)
- [非对称 Prefill/Decode 资源矩阵](results/20260804_asymmetric_pd_matrix.md)
- [UCX RMA pipeline READ/WRITE 复核](results/20260804_ucx_rma_ppln.md)
- [多卡统一结果 JSONL](results/20260802_multicard_frontier_results.jsonl)
- [图表](../docs/assets/)

## Profiler 边界

单进程 PID 分片与 `torchrun` TP=2 kernel/RCCL trace 已成功。当前实验容器曾同时存在 `_rocm_sdk_devel` 与 `_rocm_sdk_core` 两套 profiler SDK，其动态库冲突会使真实 vLLM attach 触发 signal 6。

因此现有结论使用 standalone kernel、RCCL microbenchmark、服务日志和端到端 A/B 相互约束；没有将不可靠 attach 结果写成真实请求内 kernel 百分比。

## 验证状态

- DFlash 五路线：67 次有效质量请求、34 个聚合配置点。
- Nowcast3D 质量门禁：关键在线配置均为 `QA=4/4`。
- D-Cut：19 个纯逻辑测试通过；9 个需要离线缺失 `facebook/opt-125m` 的 fixture 未执行。
- Prefix cache：8K 和 261K 连续不同问题均命中；4×261K 并发完成。
- P/D 解耦：8K/64K 混合状态正确性通过；单请求、4 并发和 dual TP=4 公平基线完成。
- 实验结束后停止 vLLM 服务但保留容器，GPU 显存释放。
