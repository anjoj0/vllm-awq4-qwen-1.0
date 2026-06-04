# stageC_n8_mbt12288_longscreen (stream)

- timestamp: `20260603-161017`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `12288` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `8` |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paper_8kchars_decode_128 | 1/1 | 18.628 | 12.801 | 12.801 | 1480.0 | 128.0 | 21.966 | 6.871 | 86.320 |
| paper_32kchars_decode_128 | 1/1 | 67.641 | 55.690 | 55.690 | 6708.0 | 128.0 | 10.710 | 1.892 | 101.064 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `98.213228` |
| `kv_available_gib` | `25.24` |
| `kv_cache_tokens` | `88192` |
| `attention_block_size` | `832` |
| `mamba_page_padding_pct` | `1.09` |
| `max_concurrency` | `3.28` |
| `engine_init_seconds` | `97.31` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 2.43, "accepted_tps": 4.3, "drafted_tps": 24.0, "accepted_tokens": 43, "drafted_tokens": 240, "avg_draft_acceptance_rate_pct": 17.9}` |
| `spec_decoding_samples` | `3` |
