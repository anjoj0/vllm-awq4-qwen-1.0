# AWQ4+DFlash vs BF16 Baseline Comparison

目的：把 BF16 `vllm-qwen-1.0` baseline 纳入同一套 competition benchmark，用于技术报告中体现优化路径和工作量。

## Configurations

| item | AWQ4 optimized path | BF16 baseline path |
| --- | --- | --- |
| project | `vllm-awq4-qwen-1.0` | `vllm-qwen-1.0` |
| model | `Qwen3.6-27B-AWQ4` | `Qwen3.6-27B` |
| quant/dtype | AWQ INT4 weights, Triton AWQ decode | BF16/native project default |
| DFlash | enabled, `num_speculative_tokens=8` | not enabled |
| max_model_len | 65536 | 65536 |
| gpu memory cap | `0.60` | project default, no explicit cap in compose |
| benchmark runs | Stage A 3-run mean | 1-run baseline |

## Runtime geometry

| metric | AWQ4 optimized | BF16 baseline |
| --- | ---: | ---: |
| model memory GiB | 40.500 | 51.200 |
| KV available GiB | 26.410 | 53.830 |
| KV cache tokens | 91520 | 220304 |
| attention block size | 832 | 784 |
| max concurrency | 3.430 | 12.920 |
| engine init s | 64.860 | 22.800 |

Note: BF16 shows larger KV cache because its compose path does not set `--gpu-memory-utilization`; AWQ4 competition profile deliberately caps at `0.60` to leave UMA headroom and survive DFlash/AWQ runtime costs. KV cache numbers are therefore runtime context, not a same-cap memory efficiency comparison.

## API metrics

| case | AWQ wall | BF16 wall | AWQ wall speedup | AWQ TTFT | BF16 TTFT | BF16 TTFT advantage | AWQ decode tok/s | BF16 decode tok/s | AWQ decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_decode_128 | 6.329 | 30.221 | 4.775x | 0.249 | 0.670 | 0.371x | 21.052 | 4.331 | 4.860x |
| mid_prefill_2k_decode_128 | 35.958 | 41.596 | 1.157x | 29.986 | 9.467 | 3.168x | 21.432 | 3.984 | 5.380x |
| paper_8kchars_decode_128 | 17.251 | 33.744 | 1.956x | 11.367 | 3.301 | 3.443x | 21.755 | 4.205 | 5.174x |
| paper_32kchars_decode_128 | 67.079 | 51.679 | 0.770x | 55.140 | 17.338 | 3.180x | 10.721 | 3.728 | 2.876x |

## Interpretation

- BF16 baseline has much faster prefill/TTFT on these prompts, especially paper32k, because it is native BF16 without DFlash verification overhead and appears to use a larger memory cap.
- AWQ4+DFlash has much faster decode: about 4.9x to 5.2x on the standard cases. This is the main competition value of the AWQ/Triton/DFlash path.
- End-to-end behavior depends on prompt length. For paper32k with only 128 output tokens, BF16 wins wall time because prefill dominates. For decode-heavy workloads or longer generation, AWQ4+DFlash should pull ahead because BF16 decode is only ~3.7-4.3 tok/s here.
- This comparison makes the project work visible: we optimized AWQ4 under a constrained `gpu=0.60` competition profile, validated DFlash N/MBT/gpu policy, and documented where the optimized path wins and where native BF16 remains a strong prefill baseline.

## Source files

- AWQ4 baseline: `test/results/competition/20260603-104958_stageA_baseline_060_n8_mbt8192_stream.json`
- BF16 baseline: `test/results/competition/20260603-170531_bf16_qwen_project_default_standard_stream.json`
- CSV: `test/results/competition/20260603_awq4_vs_bf16_baseline_comparison.csv`
