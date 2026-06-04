# bf16_qwen_project_default_standard (stream)

- timestamp: `20260603-170531`
- model: `Qwen3.6-27B`
- host: `http://127.0.0.1:8000`

## Runtime parameters

| key | value |
| --- | --- |
| `VLLM_GPU_MEMORY_UTIL` | `` |
| `VLLM_MAX_MODEL_LEN` | `65536` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_DFLASH_N` | `` |
| `AWQ_MMQ_DECODE_BACKEND` | `` |
| `AWQ_MMQ_DECODE_POLICY` | `` |
| `AWQ_MMQ_SMALL_M_THRESHOLD` | `` |
| `AWQ_ROCM_PAGED_ATTN_STATS` | `` |

## API metrics

| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 1/1 | 30.221 | 0.670 | 0.670 | 26.0 | 128.0 | 4.331 | 4.235 | 5.096 |
| mid_prefill_2k_decode_128 | 1/1 | 41.596 | 9.467 | 9.467 | 3765.0 | 128.0 | 3.984 | 3.077 | 93.590 |
| paper_8kchars_decode_128 | 1/1 | 33.744 | 3.301 | 3.301 | 1480.0 | 128.0 | 4.205 | 3.793 | 47.653 |
| paper_32kchars_decode_128 | 1/1 | 51.679 | 17.338 | 17.341 | 6708.0 | 128.0 | 3.728 | 2.477 | 132.279 |

## Parsed logs

| metric | value |
| --- | ---: |
| `model_memory_gib` | `51.2` |
| `model_load_seconds` | `39.774458` |
| `kv_available_gib` | `53.83` |
| `kv_cache_tokens` | `220304` |
| `attention_block_size` | `784` |
| `mamba_page_padding_pct` | `0.13` |
| `max_concurrency` | `12.92` |
| `engine_init_seconds` | `22.8` |
| `rocm_paged_attention_fallback_warnings` | `1` |
