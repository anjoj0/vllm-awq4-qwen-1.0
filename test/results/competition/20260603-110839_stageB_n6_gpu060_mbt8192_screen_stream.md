# stageB_n6_gpu060_mbt8192_screen (stream)

- timestamp: `20260603-110839`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `6` |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 1/1 | 7.240 | 1.442 | 1.442 | 26.0 | 128.0 | 22.077 | 17.679 | 21.270 |
| mid_prefill_2k_decode_128 | 1/1 | 37.254 | 30.396 | 30.396 | 3765.0 | 128.0 | 18.664 | 3.436 | 104.499 |
| paper_8kchars_decode_128 | 1/1 | 17.895 | 11.409 | 11.409 | 1480.0 | 128.0 | 19.735 | 7.153 | 89.858 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `94.917165` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `92208` |
| `attention_block_size` | `816` |
| `mamba_page_padding_pct` | `0.37` |
| `max_concurrency` | `3.57` |
| `engine_init_seconds` | `65.46` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 4.26, "accepted_tps": 8.8, "drafted_tps": 16.2, "accepted_tokens": 88, "drafted_tokens": 162, "avg_draft_acceptance_rate_pct": 54.3}` |
| `spec_decoding_samples` | `4` |
