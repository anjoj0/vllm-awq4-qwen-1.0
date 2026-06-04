# baseline_060_n8_mbt8192_sanity (stream)

- timestamp: `20260603-104441`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `` |
| `VLLM_MAX_MODEL_LEN` | `` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `` |
| `VLLM_MAX_NUM_SEQS` | `` |
| `VLLM_DFLASH_N` | `` |
| `AWQ_MMQ_DECODE_BACKEND` | `` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 1/1 | 6.339 | 0.262 | 0.263 | 26.0 | 128.0 | 21.067 | 20.194 | 24.296 |
| paper_8kchars_decode_128 | 1/1 | 17.353 | 11.444 | 11.444 | 1480.0 | 128.0 | 21.664 | 7.376 | 92.665 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `98.366093` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `91520` |
| `max_concurrency` | `3.43` |
| `engine_init_seconds` | `64.86` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 3.71, "accepted_tps": 1.9, "drafted_tps": 5.6, "accepted_tokens": 19, "drafted_tokens": 56, "avg_draft_acceptance_rate_pct": 33.9}` |
| `spec_decoding_samples` | `6` |
