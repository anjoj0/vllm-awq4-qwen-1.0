# stageA_baseline_060_n8_mbt8192 (stream)

- timestamp: `20260603-104958`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` |
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
| short_decode_128 | 3/3 | 6.329 | 0.249 | 0.249 | 26.0 | 128.0 | 21.052 | 20.225 | 24.334 |
| mid_prefill_2k_decode_128 | 3/3 | 35.958 | 29.986 | 29.986 | 3765.0 | 128.0 | 21.432 | 3.560 | 108.265 |
| paper_8kchars_decode_128 | 3/3 | 17.251 | 11.367 | 11.367 | 1480.0 | 128.0 | 21.755 | 7.420 | 93.215 |
| paper_32kchars_decode_128 | 3/3 | 67.079 | 55.140 | 55.140 | 6708.0 | 128.0 | 10.721 | 1.908 | 101.910 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `98.366093` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `91520` |
| `attention_block_size` | `832` |
| `mamba_page_padding_pct` | `1.09` |
| `max_concurrency` | `3.43` |
| `engine_init_seconds` | `64.86` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 1.84, "accepted_tps": 1.6, "drafted_tps": 15.2, "accepted_tokens": 16, "drafted_tokens": 152, "avg_draft_acceptance_rate_pct": 10.5}` |
| `spec_decoding_samples` | `25` |
