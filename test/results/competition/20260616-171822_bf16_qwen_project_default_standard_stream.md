# bf16_qwen_project_default_standard (stream)

- timestamp: `20260616-171822`
- model: `Qwen3.6-27B`
- host: `http://127.0.0.1:8000`

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
| short_decode_128 | 1/1 | 30.561 | 0.342 | 0.343 | 26.0 | 128.0 | 4.236 | 4.188 | 5.039 |
| mid_prefill_2k_decode_128 | 1/1 | 42.412 | 9.548 | 9.548 | 3765.0 | 128.0 | 3.895 | 3.018 | 91.790 |
| paper_8kchars_decode_128 | 1/1 | 34.802 | 3.545 | 3.545 | 1538.0 | 128.0 | 4.095 | 3.678 | 47.871 |
| paper_32kchars_decode_128 | 1/1 | 52.696 | 17.586 | 17.586 | 6743.0 | 128.0 | 3.646 | 2.429 | 130.391 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `51.2` |
| `model_load_seconds` | `34.714413` |
| `kv_available_gib` | `53.83` |
| `kv_cache_tokens` | `220304` |
| `attention_block_size` | `784` |
| `mamba_page_padding_pct` | `0.13` |
| `max_concurrency` | `12.92` |
| `engine_init_seconds` | `22.72` |
| `rocm_paged_attention_fallback_warnings` | `2` |
