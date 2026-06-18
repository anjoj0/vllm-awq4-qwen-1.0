# 256K Context Startup Capacity Test

| Field | Value |
|---|---:|
| max_model_len | 262144 |
| gpu_memory_utilization | 0.90 |
| kv_cache_dtype | fp8 |
| max_num_batched_tokens | 16384 |
| DFlash N | 4 |
| health wait seconds | 215.9 |
| model memory | 40.5 GiB |
| available KV memory | 58.87 GiB |
| GPU KV cache size | 235,936 tokens |
| max concurrency for 262,144 tokens | 1.46x |
| init engine time | 138.29 s |

Conclusion: 256K target context is startup-capable under the high-memory single-user fp8 KV profile. This validates the README-level 256K capacity claim for startup/admission, but it does not imply interactive latency is good. The prepared `combined_papers_for_llm.txt` is about 353,478 tokens, so even 256K truncation does not cover the full file.
