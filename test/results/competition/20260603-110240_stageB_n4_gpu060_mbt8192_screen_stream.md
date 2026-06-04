# stageB_n4_gpu060_mbt8192_screen (stream)

- timestamp: `20260603-110240`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `4` |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 1/1 | 14.649 | 7.554 | 7.554 | 26.0 | 128.0 | 18.042 | 8.738 | 10.513 |
| mid_prefill_2k_decode_128 | 1/1 | 38.269 | 30.289 | 30.290 | 3765.0 | 128.0 | 16.041 | 3.345 | 101.727 |
| paper_8kchars_decode_128 | 1/1 | 18.747 | 11.448 | 11.448 | 1480.0 | 128.0 | 17.535 | 6.828 | 85.771 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `96.430535` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `92208` |
| `attention_block_size` | `816` |
| `mamba_page_padding_pct` | `1.62` |
| `max_concurrency` | `3.73` |
| `engine_init_seconds` | `65.36` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 3.88, "accepted_tps": 7.5, "drafted_tps": 10.4, "accepted_tokens": 75, "drafted_tokens": 104, "avg_draft_acceptance_rate_pct": 72.1}` |
| `spec_decoding_samples` | `6` |
