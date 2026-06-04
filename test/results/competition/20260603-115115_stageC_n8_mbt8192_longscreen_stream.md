# stageC_n8_mbt8192_longscreen (stream)

- timestamp: `20260603-115115`
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
| paper_8kchars_decode_128 | 1/1 | 17.548 | 11.513 | 11.513 | 1480.0 | 128.0 | 21.209 | 7.294 | 91.634 |
| paper_32kchars_decode_128 | 1/1 | 67.520 | 55.567 | 55.567 | 6708.0 | 128.0 | 10.709 | 1.896 | 101.244 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `96.14149` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `91520` |
| `attention_block_size` | `832` |
| `mamba_page_padding_pct` | `1.09` |
| `max_concurrency` | `3.43` |
| `engine_init_seconds` | `65.5` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 4.65, "accepted_tps": 8.4, "drafted_tps": 18.4, "accepted_tokens": 84, "drafted_tokens": 184, "avg_draft_acceptance_rate_pct": 45.7}` |
| `spec_decoding_samples` | `4` |
