# Complete bench_full.py Verification Summary

This file combines the stable three-run text/tool benchmark with the one-run vision补测 using the local replacement images.

## Sources

- Text/tool baseline: `test/results/readme_verification/20260606-200114_bench_full_n8_256k_fp8_runs3.json`
- Vision补测: `test/results/readme_verification/20260606-202654_bench_full_n8_256k_fp8_runs1_with_vision.json`
- Runtime profile: `max_model_len=262144`, `gpu_memory_utilization=0.90`, `kv_cache_dtype=fp8`, `max_num_batched_tokens=16384`, `DFlash N=8`, `max_num_seqs=1`, host `127.0.0.1:8001`
- Vision replacement: `IMAGE_A=test/images/forest.png`, `IMAGE_B=test/images/fly.png`

## Complete Results

| Test | Source | Ok runs | Errors | Median t/s | Mean t/s | Min | Max | Completion tokens | Prompt tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `completions_short` | runs=3 text/tool | 3 | 0 | 6.112 | 4.944 | 2.593 | 6.128 | [8, 8, 8] | [5, 5, 5] |
| `chat_factual` | runs=3 text/tool | 3 | 0 | 22.473 | 21.074 | 17.971 | 22.777 | [213, 213, 213] | [29, 29, 29] |
| `chat_explainer` | runs=3 text/tool | 3 | 0 | 19.960 | 20.014 | 19.946 | 20.136 | [1329, 1469, 1469] | [27, 27, 27] |
| `responses_reasoning` | runs=3 text/tool | 3 | 0 | 28.292 | 28.284 | 28.263 | 28.296 | [910, 910, 910] | [57, 57, 57] |
| `vision_frost` | runs=1 vision补测 | 1 | 0 | 16.217 | 16.217 | 16.217 | 16.217 | [691] | [100] |
| `vision_splash` | runs=1 vision补测 | 1 | 0 | 16.782 | 16.782 | 16.782 | 16.782 | [651] | [101] |
| `tool_chat` | runs=3 text/tool | 3 | 0 | 13.895 | 13.874 | 13.825 | 13.901 | [137, 137, 137] | [318, 318, 318] |
| `tool_responses` | runs=3 text/tool | 3 | 0 | 14.251 | 14.136 | 13.886 | 14.270 | [110, 110, 110] | [336, 336, 336] |

## README Claim Check

| README row | README value | Best reproduced evidence | Verdict |
| --- | ---: | ---: | --- |
| AWQ4 + DFlash N=8 single-stream decode median | 18.5 t/s | 19.960 t/s median over 3 runs; 20.026 t/s in the one-run full pass | reproduced/exceeded |
| AWQ4 + DFlash N=8 single-stream decode peak | 24.8 t/s | 28.296 t/s max over 3 runs; 28.305 t/s in the one-run full pass | reproduced/exceeded |

## Vision Case Details

- `vision_frost`: image `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/images/forest.png`, prompt tokens `100`, completion tokens `691`, wall `42.61s`, decode `16.217 t/s`.
- `vision_splash`: image `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/images/fly.png`, prompt tokens `101`, completion tokens `651`, wall `38.79s`, decode `16.782 t/s`.

## Interpretation

- The README speed row is supported by the three-run text/tool benchmark and again by the one-run full benchmark with working vision inputs.
- The two vision rows now validate multimodal request plumbing under the same AWQ4 + DFlash N=8 runtime profile. Their t/s includes vision input processing and should be reported separately from pure text decode throughput.
- The previous vision failures were input-file availability issues, not model/runtime failures.
