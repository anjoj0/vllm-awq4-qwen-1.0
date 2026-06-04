# stageB_n10_gpu060_mbt8192_screen (stream)

- timestamp: `20260603-111830`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `10` |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 1/1 | 13.093 | 7.483 | 7.483 | 26.0 | 128.0 | 22.818 | 9.776 | 11.762 |
| mid_prefill_2k_decode_128 | 1/1 | 36.415 | 30.514 | 30.514 | 3765.0 | 128.0 | 21.694 | 3.515 | 106.907 |
| paper_8kchars_decode_128 | 1/1 | 17.148 | 11.504 | 11.504 | 1480.0 | 128.0 | 22.678 | 7.464 | 93.771 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `94.475994` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `91584` |
| `attention_block_size` | `848` |
| `mamba_page_padding_pct` | `1.8` |
| `max_concurrency` | `3.26` |
| `engine_init_seconds` | `66.28` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 5.16, "accepted_tps": 10.4, "drafted_tps": 25.0, "accepted_tokens": 104, "drafted_tokens": 250, "avg_draft_acceptance_rate_pct": 41.6}` |
| `spec_decoding_samples` | `5` |
