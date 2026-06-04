# stageB_n12_gpu060_mbt8192_screen (stream)

- timestamp: `20260603-113032`
- model: `Qwen3.6-27B-AWQ4`
- host: `http://127.0.0.1:8001`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `0.60` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `12` |
| `AWQ_MMQ_DECODE_BACKEND` | `triton` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 1/1 | 7.206 | 1.552 | 1.552 | 26.0 | 128.0 | 22.642 | 17.764 | 21.372 |
| mid_prefill_2k_decode_128 | 1/1 | 36.322 | 30.371 | 30.371 | 3765.0 | 128.0 | 21.507 | 3.524 | 107.179 |
| paper_8kchars_decode_128 | 1/1 | 16.870 | 11.406 | 11.406 | 1480.0 | 128.0 | 23.425 | 7.587 | 95.314 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `40.5` |
| `model_load_seconds` | `95.417771` |
| `kv_available_gib` | `26.41` |
| `kv_cache_tokens` | `91584` |
| `attention_block_size` | `848` |
| `mamba_page_padding_pct` | `0.59` |
| `max_concurrency` | `3.14` |
| `engine_init_seconds` | `65.52` |
| `rocm_paged_attention_fallback_warnings` | `1` |
| `spec_decoding_last` | `{"mean_acceptance_length": 4.63, "accepted_tps": 6.9, "drafted_tps": 22.8, "accepted_tokens": 69, "drafted_tokens": 228, "avg_draft_acceptance_rate_pct": 30.3}` |
| `spec_decoding_samples` | `4` |
