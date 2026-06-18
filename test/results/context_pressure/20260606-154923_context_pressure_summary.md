# 64K / 128K / 256K Context Pressure Summary

Source file: `/home/xqhpc/data/AI_project/combined_papers_for_llm.txt`

Tokenized length: **353,478 tokens**. The file is larger than 256K, so a 256K test is still a truncation test, not a full-file test.

| Context class | Test type | max_model_len | Prompt / KV tokens | Output cap | Result | Main metric | Notes |
|---|---|---:|---:|---:|---|---:|---|
| 64K | real generation, prior run | 65,536 | 60,000 prompt | 128 | pass | 762.486 s wall | Real combined-papers generation succeeded under the 2026-06-02 util=0.60 profile; exact 64K rerun was too slow for interactive sweep and was aborted. |
| 128K | startup capacity | 131,072 | 235,936 KV cache | n/a | pass | 2.81x concurrency | High-memory fp8 profile starts successfully. Full generation deferred. |
| 256K | startup capacity | 262,144 | 235,936 KV cache | n/a | pass | 1.46x concurrency | README-level 256K capacity validates under fp8/gpu_util=0.90, but not interactive latency. |

## Interpretation

The current project is not limited to 32K. The stable daily profile is 64K, and a high-memory single-user profile can start at 128K and 256K with fp8 KV cache. The limiting factor for practical use is latency, not only max_model_len. A real 60K prompt from the precipitation-nowcasting paper pack already took 762.486 seconds for 128 generated tokens, so full 128K/256K generation should be treated as overnight/long-running evaluation rather than an interactive benchmark.

For the prepared literature pack, the best evaluation route is hierarchical summarization: split the 353K-token file into 45K-60K chunks, summarize each chunk, then synthesize the chunk summaries. Direct 256K truncation can test capacity but still discards about 97K tokens.
