# stageD_gpu058_n8_mbt8192_longscreen (stream)

- timestamp: `20260603-163717`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.58` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `8` |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paper_8kchars_decode_128 | 1/1 | 18.792 | 12.919 | 12.919 | 1480.0 | 128.0 | 21.795 | 6.811 | 85.567 |
| paper_32kchars_decode_128 | 1/1 | 67.915 | 55.973 | 55.973 | 6708.0 | 128.0 | 10.719 | 1.885 | 100.656 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `93.947332` |
| `kv_available_gib` | `24.09` |
| `kv_cache_tokens` | `84032` |
| `attention_block_size` | `832` |
| `mamba_page_padding_pct` | `1.09` |
| `max_concurrency` | `3.13` |
| `engine_init_seconds` | `66.23` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 2.48, "accepted_tps": 4.9, "drafted_tps": 26.4, "accepted_tokens": 49, "drafted_tokens": 264, "avg_draft_acceptance_rate_pct": 18.6}` |
| `spec_decoding_samples` | `4` |
