# Qwen3.6-27B AWQ4 + DFlash 推理优化比赛技术报告草稿

## 1. 项目背景

本项目面向比赛场景，目标是在 AMD Strix Halo / gfx1151 上稳定运行 Qwen3.6-27B，并在受限 UMA 预算下提升真实推理吞吐。主线为 `Qwen3.6-27B-AWQ4` + vLLM + DFlash speculative decoding。项目不追逐单个 microbenchmark，而是要求服务可启动、结果可复现、优化覆盖真实请求路径。

Strix Halo 与 MI300/CDNA 差异明显，ROCm 默认路径不能直接套用。AITER 部分 kernel 面向 CDNA，HIP graph 在 gfx1151 上存在冻结风险，Triton/ROCm nightly 滚动也曾破坏构建。因此项目固定 Ubuntu 26.04、TheRock ROCm 7.13 nightly、PyTorch 2.13.0a0、Triton 3.7.0 和 vLLM v0.20.0 本地补丁版。

当前默认比赛配置如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` | 控制 vLLM 可使用 UMA/GPU 内存上限 |
| `VLLM_MAX_MODEL_LEN` | `65536` | 比赛阶段稳定长上下文窗口 |
| `VLLM_MAX_NUM_SEQS` | `1` | 单流评测，避免并发噪声 |
| `VLLM_DFLASH_N` | `8` | DFlash speculative token 数 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | scheduler token budget |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` | 保持稳定 AWQ decode 路径 |

该配置不是理论最激进设置，而是在 AWQ4 权重、DFlash drafter、ROCm attention、KV cache 和 UMA 余量之间取得的稳定平衡。

## 2. 系统方案

主项目为 `vllm-awq4-qwen-1.0`，服务模型名为 `Qwen3.6-27B-AWQ4`，默认端口 `8001`。模型使用 AWQ INT4 权重量化，推理框架为 vLLM OpenAI-compatible API，支持 chat、responses、completions、视觉输入和工具调用。语言模型 attention 后端使用 `ROCM_ATTN`，视觉编码器使用 `TRITON_ATTN`。项目禁用 HIP graph，以避免 gfx1151 上的冻结问题；同时禁用 AITER 的 CDNA-only 路径。

DFlash drafter 为 `z-lab/Qwen3.6-27B-DFlash`，默认 `num_speculative_tokens=8`。它通过 draft 预测和 target 验证减少 decode 迭代，但也引入 target/draft/SWA/Mamba/attention hybrid page alignment。实验显示 N=4/6 的 block size 为 816，N=8 为 832，N=10/12 为 848，这会影响 KV cache、max concurrency 和 attention fallback。

README 记录了 gfx1151 AWQ MMQ Q4 HIP kernel。它主要改善 prefill 阶段 AWQ W4A16 GEMM，利用 RDNA INT8 WMMA 路径提升大 M 形状效率。但它不解决长上下文 decode；超过约 8K context 后掉速主要来自 KV cache attention scaling，而非 GEMM。

## 3. 实验方法

为了让比赛结果可复现，我们编写了 `test/bench_competition.py`。该脚本统一调用 OpenAI-compatible API，记录 JSON 和 Markdown，并解析 docker logs 中的 runtime 指标。

| 指标类型 | 指标 | 含义 |
| --- | --- | --- |
| API 延迟 | `wall_seconds` | 端到端请求耗时 |
| API 延迟 | `ttft_seconds` / `payload_ttft_seconds` | 首 SSE event / 首有效 payload 时间 |
| 吞吐 | `prefill_tokens_per_ttft` | prompt tokens / TTFT |
| 吞吐 | `decode_tokens_per_second_stream` | streaming decode 阶段吞吐 |
| 吞吐 | `output_tokens_per_second_e2e` | 端到端输出 token 吞吐 |
| 日志 | `kv_cache_tokens` / `max_concurrency` | KV cache 与理论并发能力 |
| 日志 | `attention_block_size` | DFlash/Qwen hybrid attention page geometry |
| 日志 | `avg_draft_acceptance_rate_pct` | DFlash draft token 接受率 |

workload 包括 short、mid、paper8k、paper32k 和 paper120k。当前报告重点使用 standard workload：short、mid、paper8k、paper32k。

## 4. 默认 AWQ4+DFlash baseline

Stage A 使用默认配置 `gpu=0.60`、`N=8`、`MBT=8192`、`max_num_seqs=1`、`AWQ decode=triton`，运行 standard 3-run。结果如下：

| case | wall s | TTFT s | prompt tok | output tok | decode tok/s | total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 6.329 | 0.249 | 26 | 128 | 21.052 | 24.334 |
| mid 2k | 35.958 | 29.986 | 3765 | 128 | 21.432 | 108.265 |
| paper8k | 17.251 | 11.367 | 1480 | 128 | 21.755 | 93.215 |
| paper32k | 67.079 | 55.140 | 6708 | 128 | 10.721 | 101.910 |

runtime geometry 如下：

| 指标 | 数值 |
| --- | ---: |
| model memory | 40.5 GiB |
| KV available | 26.41 GiB |
| KV cache | 91,520 tokens |
| attention block size | 832 |
| max concurrency | 3.43x |

这组结果说明当前方案的主要优势是 decode throughput。paper32k 只有 128 个输出 token，端到端耗时被 TTFT 主导；若输出更长，AWQ4+DFlash 的 decode 优势会更明显。

## 5. DFlash N 扫描

Stage B 固定 `gpu=0.60`、`MBT=8192`，测试 `N=4,6,8,10,12`。结果如下：

| N | block | KV tokens | max conc | accept % | paper8k wall | paper8k TTFT | paper8k decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 816 | 92,208 | 3.73x | 72.1 | 18.747 | 11.448 | 17.535 |
| 6 | 816 | 92,208 | 3.57x | 54.3 | 17.895 | 11.409 | 19.735 |
| 8 | 832 | 91,520 | 3.43x | 45.7 | 17.314 | 11.441 | 21.794 |
| 10 | 848 | 91,584 | 3.26x | 41.6 | 17.148 | 11.504 | 22.678 |
| 12 | 848 | 91,584 | 3.14x | 30.3 | 16.870 | 11.406 | 23.425 |

N=4 的 acceptance 最高，但整体 wall latency 明显偏慢，说明 draft 浪费少不等于系统吞吐高。N=10/12 在 paper8k 单次 screen 上略快，但代价是 acceptance 下降、attention block size 变为 848、max concurrency 降低。综合稳定性、cache geometry 和性能，默认不切到 N=12，shipping profile 保持 N=8。

## 6. Scheduler token budget 扫描

Stage C 固定 N=8，测试 `max_num_batched_tokens=8192,12288,16384`。理论上更大的 MBT 可能让长 prompt prefill 使用更大 chunk，从而降低 TTFT；但实验结果相反：

| MBT | KV tokens | max conc | engine init s | paper8k wall | paper8k TTFT | paper32k wall | paper32k TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 91,520 | 3.43x | 65.50 | 17.548 | 11.513 | 67.520 | 55.567 |
| 12288 | 88,192 | 3.28x | 97.31 | 18.628 | 12.801 | 67.641 | 55.690 |
| 16384 | 84,032 | 3.13x | 131.25 | 18.784 | 12.930 | 68.123 | 56.165 |

扩大 MBT 没有改善真实长文 TTFT，反而增加 profile/compile 成本并压缩 KV pool。因此默认保持 `MBT=8192`。

## 7. GPU memory utilization 扫描

Stage D 固定 N=8、MBT=8192，测试 `gpu_memory_utilization=0.58,0.60,0.62`。结果如下：

| gpu util | KV GiB | KV tokens | max conc | paper8k wall | paper8k TTFT | paper32k wall | paper32k TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.58 | 24.09 | 84,032 | 3.13x | 18.792 | 12.919 | 67.915 | 55.973 |
| 0.60 | 26.41 | 91,520 | 3.43x | 17.548 | 11.513 | 67.520 | 55.567 |
| 0.62 | 28.73 | 99,840 | 3.73x | 18.733 | 12.907 | 68.451 | 56.513 |

更高显存 cap 没有改善当前单流 64K workload。0.62 增大 KV pool，但 `max_num_seqs=1` 无法有效消费额外 KV；0.58 节省内存但略慢。默认保持 `gpu=0.60`。

## 8. BF16 baseline 对比

为体现优化工作量，我们运行 sibling repo `vllm-qwen-1.0` 的 BF16 baseline。BF16 不启用 DFlash，compose 中没有显式设置 `--gpu-memory-utilization`，因此 KV 数字不能和 AWQ4 的 `gpu=0.60` 当作同 cap 比较。

| 指标 | AWQ4+DFlash | BF16 baseline |
| --- | ---: | ---: |
| model memory | 40.5 GiB | 51.2 GiB |
| KV available | 26.41 GiB | 53.83 GiB |
| KV cache | 91,520 | 220,304 |
| attention block size | 832 | 784 |
| max concurrency | 3.43x | 12.92x |
| engine init | 64.86s | 22.80s |

API 对比如下：

| case | AWQ wall | BF16 wall | AWQ TTFT | BF16 TTFT | AWQ decode | BF16 decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 6.329 | 30.221 | 0.249 | 0.670 | 21.052 | 4.331 |
| mid 2k | 35.958 | 41.596 | 29.986 | 9.467 | 21.432 | 3.984 |
| paper8k | 17.251 | 33.744 | 11.367 | 3.301 | 21.755 | 4.205 |
| paper32k | 67.079 | 51.679 | 55.140 | 17.338 | 10.721 | 3.728 |

BF16 的 prefill/TTFT 很强，尤其 paper32k；但 decode 方向相反，AWQ4+DFlash 在标准 workload 上约有 `2.876x` 到 `5.380x` decode speedup。因此报告中应准确表述：BF16 是强 prefill baseline，适合长输入短输出；AWQ4+DFlash 的价值主要在 decode-heavy 或较长生成场景。

## 9. Long-context decode 瓶颈定位

我们对 long-context decode 问题做过探索，但没有完成 attention kernel 级修复。HIP AWQ kernel 主要改善 prefill/GEMM，不触碰 KV attention。实验中持续出现 `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation`，说明当前长上下文 attention 没有进入理想 native ROCm paged attention 路径。DFlash/Qwen hybrid cache 又产生 816/832/848 这类非标准 block size，进一步增加适配难度。

我们尝试过 drafter cache `block_size=16`，但 vLLM allocator 依赖全局 uniform page size 和单一 BlockPool。后续 `kernel_block_size=16` split prototype 虽能启动，但 A/B 变慢：短 decode `5.547s -> 20.801s`，paper32k `60.656s -> 82.623s`。真正修复需要支持 DFlash hybrid page geometry 的 ROCm paged/flash-attention-style KV-aware kernel。

## 10. 最终配置与结论

综合实验，当前推荐比赛默认配置为：`VLLM_GPU_MEMORY_UTIL=0.60`、`VLLM_MAX_MODEL_LEN=65536`、`VLLM_MAX_NUM_SEQS=1`、`VLLM_DFLASH_N=8`、`VLLM_MAX_NUM_BATCHED_TOKENS=8192`、`AWQ_MMQ_DECODE_BACKEND=triton`。该配置 runtime geometry 为 KV cache `91,520 tokens`，max concurrency `3.43x`，attention block size `832`。它不是所有单点指标的最大值，但在 decode 吞吐、DFlash acceptance、KV cache、启动稳定性和系统 UMA 余量之间最均衡。

本项目完成了五类工作：固定 ROCm/PyTorch/Triton/vLLM 版本；构建 AWQ4+DFlash 服务路径；探索 AWQ MMQ Q4 HIP kernel 与 backend policy；建立 competition benchmark 并完成 N、MBT、gpu util sweep；补充 BF16 baseline。后续重点应转向 attention/KV cache，包括支持 832/848 hybrid block size 的 ROCm paged attention kernel、target/draft KV 共存布局和 long-context decode 调度优化。
