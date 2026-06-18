# 128K Context Startup Capacity Test

| Field | Value |
|---|---:|
| max_model_len | 131072 |
| gpu_memory_utilization | 0.90 |
| kv_cache_dtype | fp8 |
| max_num_batched_tokens | 16384 |
| DFlash N | 4 |
| health wait seconds | 256.1 |
| model memory | 40.5 GiB |
| available KV memory | 58.87 GiB |
| GPU KV cache size | 235,936 tokens |
| max concurrency for 131,072 tokens | 2.81x |
| init engine time | 138.34 s |

Conclusion: 128K target context is startup-capable on the fp8 KV / gpu_util=0.90 profile with substantial KV headroom. This is a capacity result, not a full 128K generation latency result.
