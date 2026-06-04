# stageD_gpu062_n8_mbt8192_longscreen (stream)

- timestamp: `20260603-164520`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.62` |
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
| paper_8kchars_decode_128 | 1/1 | 18.733 | 12.907 | 12.907 | 1480.0 | 128.0 | 21.969 | 6.833 | 85.837 |
| paper_32kchars_decode_128 | 1/1 | 68.451 | 56.513 | 56.514 | 6708.0 | 128.0 | 10.723 | 1.870 | 99.868 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `95.000095` |
| `kv_available_gib` | `28.73` |
| `kv_cache_tokens` | `99840` |
| `attention_block_size` | `832` |
| `mamba_page_padding_pct` | `1.09` |
| `max_concurrency` | `3.73` |
| `engine_init_seconds` | `65.73` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 4.0, "accepted_tps": 0.3, "drafted_tps": 0.8, "accepted_tokens": 3, "drafted_tokens": 8, "avg_draft_acceptance_rate_pct": 37.5}` |
| `spec_decoding_samples` | `5` |
