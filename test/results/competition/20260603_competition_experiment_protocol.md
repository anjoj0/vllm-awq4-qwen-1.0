# Competition Experiment Protocol

目标：围绕比赛实际路径优化 vLLM AWQ4 + DFlash 的整体吞吐和稳定性，而不是继续局部追逐单个 HIP kernel 的 microbenchmark。所有结果必须能复现、可对比、可写入技术报告。

## 固定基线

- 模型：`Qwen3.6-27B-AWQ4`
- 服务端口：`http://127.0.0.1:8001`
- 默认 profile：`VLLM_GPU_MEMORY_UTIL=0.60`、`VLLM_MAX_MODEL_LEN=65536`、`VLLM_MAX_NUM_SEQS=1`、`VLLM_DFLASH_N=8`、`VLLM_MAX_NUM_BATCHED_TOKENS=8192`、`AWQ_MMQ_DECODE_BACKEND=triton`
- 注意：默认仍保持 Triton AWQ decode。HIP v7/v8 只作为候选小实验，不纳入主线默认路径。

## Workload 分层

| workload | 目的 | 运行方式 |
| --- | --- | --- |
| `short_decode_128` | 短 prompt decode 延迟、DFlash 小上下文收益 | 每个候选配置都跑 |
| `mid_prefill_2k_decode_128` | 中等 prefill + decode 平衡 | N sweep 和 MBT sweep 都跑 |
| `paper_8kchars_decode_128` | 真实短临降雨论文材料的小长文场景 | 每个候选配置都跑 |
| `paper_32kchars_decode_128` | 真实长文 prefill 场景，观察 chunked prefill/TTFT | MBT/gpu util 候选配置跑 |
| `paper_120kchars_decode_128` | 长上下文压力和稳定性验证 | 只给入围配置跑 |

## 指标口径

API 侧指标由 `test/bench_competition.py` 自动保存：

- `wall_seconds`：端到端请求耗时。
- `ttft_seconds`：首个 SSE event 到达时间，适合衡量 prefill + 调度延迟。
- `payload_ttft_seconds`：首个有效内容/推理 payload 到达时间。
- `prompt_tokens`、`completion_tokens`、`total_tokens`：由 vLLM usage 返回。
- `prefill_tokens_per_ttft`：`prompt_tokens / ttft_seconds`，用于比较长 prompt prefill 改动。
- `decode_tokens_per_second_stream`：streaming decode 阶段吞吐估算。
- `output_tokens_per_second_e2e`：输出 token 端到端吞吐。
- `total_tokens_per_second_e2e`：prefill+decode 总 token 端到端吞吐。

日志侧指标从 docker logs 解析：

- `model_memory_gib`、`model_load_seconds`：模型加载显存与耗时。
- `kv_available_gib`、`kv_cache_tokens`、`max_concurrency`：cache/page policy 与 `gpu_memory_utilization` 的直接结果。
- `engine_init_seconds`：启动成本。
- `rocm_paged_attention_fallback_warnings`：是否落到不可控 fallback。
- `spec_decoding_last.mean_acceptance_length`、`avg_draft_acceptance_rate_pct`、`accepted_tps`、`drafted_tps`：DFlash 参数是否真正提高有效 token 的核心证据。

## 参数扫描顺序

### Stage A：默认基线复现

固定 `gpu=0.60`、`N=8`、`MBT=8192`，跑 `standard` 3 次。目标是得到后续所有改动的可比锚点。

推荐命令：

```bash
python3 test/bench_competition.py --label stageA_baseline_060_n8_mbt8192 --cases standard --runs 3 --mode stream --out-dir test/results/competition
```

### Stage B：DFlash `num_speculative_tokens` sweep

固定 `gpu=0.60`、`MBT=8192`，测试 `N=4,6,8,10,12`。先跑 `short_decode_128 mid_prefill_2k_decode_128 paper_8kchars_decode_128`，根据 acceptance 和 wall latency 决定是否扩大到 `paper_32kchars_decode_128`。

判断规则：如果 `N` 增大后 drafted throughput 提高但 acceptance rate/mean acceptance length 明显下降，并且 wall latency 没有改善，就不进入下一轮。

### Stage C：scheduler token budget sweep

使用 Stage B 最优 `N`，测试 `MBT=8192,12288,16384`。重点看 `paper_8kchars_decode_128` 与 `paper_32kchars_decode_128` 的 TTFT、prefill throughput 和 OOM/重启稳定性。

判断规则：`MBT` 增大应降低长 prompt TTFT；如果短 prompt wall 明显变差或 first request autotune OOM，则回退。

### Stage D：GPU memory utilization sweep

使用 Stage B/C 入围配置，测试 `gpu=0.58,0.60,0.62,0.64`。只给能启动且能完成 `standard` 的配置继续跑 `paper_120kchars_decode_128`。

判断规则：更高 `gpu_memory_utilization` 只有在提高 `kv_cache_tokens` 或长上下文稳定性时才有价值。单客户端 64K profile 下，不应为了空闲 KV 池牺牲系统余量。

### Stage E：报告验证

最终候选配置需要：

- 冷启动 + 首请求 3 次稳定。
- `standard` 3-run 保存 JSON/Markdown。
- `paper_120kchars_decode_128` 至少 1-run 成功。
- 保存对应 docker logs 解析结果。

## 当前建议

在新的 sweep 证明更优之前，比赛默认配置仍为：`gpu=0.60`、`N=8`、`MBT=8192`、`max_num_seqs=1`、`AWQ_MMQ_DECODE_BACKEND=triton`。这是目前通过冷启动、60K 真实 prompt、KV cache 和 DFlash 路径验证的最低风险配置。
