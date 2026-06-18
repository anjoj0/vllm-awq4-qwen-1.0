# bench_full.py Single-Run Verification With Vision Images

## Configuration

- Time: `2026-06-06 20:26:54` to `2026-06-06 20:30:22`
- Host: `http://127.0.0.1:8001`
- Model: `Qwen3.6-27B-AWQ4`
- Runs per test: `1`
- Profile: `VLLM_MAX_MODEL_LEN=262144`, `VLLM_GPU_MEMORY_UTIL=0.90`, `VLLM_KV_CACHE_DTYPE=fp8`, `VLLM_MAX_NUM_BATCHED_TOKENS=16384`, `VLLM_DFLASH_N=8`, `VLLM_MAX_NUM_SEQS=1`
- Image replacement: `IMAGE_A=/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/images/forest.png`, `IMAGE_B=/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/images/fly.png`
- Total wall time: `208.29s`
- Raw JSON: `20260606-202654_bench_full_n8_256k_fp8_runs1_with_vision.json`

## Results

| Case | API / scenario | Prompt tokens | Completion tokens | Wall time (s) | Decode t/s | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `completions_short` | completions short factual | 5 | 8 | 1.12 | 7.163 |  |
| `chat_factual` | chat factual (speed of light) | 29 | 213 | 9.67 | 22.036 |  |
| `chat_explainer` | chat explainer (entanglement) | 27 | 1329 | 66.37 | 20.026 | README median claim reference: 18.5 t/s |
| `responses_reasoning` | responses reasoning (trains) | 57 | 910 | 32.15 | 28.305 | README peak claim reference: 24.8 t/s |
| `vision_frost` | vision (frost_1.png  -  1280x720) | 100 | 691 | 42.61 | 16.217 | IMAGE_A replaced by test/images/forest.png |
| `vision_splash` | vision (splash.png  -  1024x1024) | 101 | 651 | 38.79 | 16.782 | IMAGE_B replaced by test/images/fly.png |
| `tool_chat` | tool calling /v1/chat/completions | 318 | 137 | 9.88 | 13.867 | tool calls: 1 |
| `tool_responses` | tool calling /v1/responses | 336 | 110 | 7.71 | 14.275 | tool calls: 1 |

## README Claim Check

| README item | Claimed value | This run | Verdict |
| --- | ---: | ---: | --- |
| AWQ4 + DFlash N=8 median decode | 18.5 t/s | 20.026 t/s | reproduced/exceeded in `chat_explainer` |
| AWQ4 + DFlash N=8 peak decode | 24.8 t/s | 28.305 t/s | reproduced/exceeded in `responses_reasoning` |

## Vision Outputs

- `vision_frost` image: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/images/forest.png`
  Output excerpt: A vast, mountainous landscape is densely packed with evergreen trees that stretch across rolling hills and valleys. Wisps of white mist drift through the lower elevations, creating a hazy, atmospheric effect that contrasts with the sharp details of the foreground foliage.  **Dominant Color:** Green **Apparent Setting:** A mountain forest or wildern...
- `vision_splash` image: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/images/fly.png`
  Output excerpt: Based on the image, here are the objects I can identify:  *   **Airplane:** A large, white turboprop aircraft with red and blue stripes. *   **Propellers:** Two visible propellers on the left wing. *   **Wings:** The main wings extending from the fuselage. *   **Tail Fin:** The vertical stabilizer at the rear, featuring a logo. *   **Landing Gear:*...

## Notes

- This is a one-run completion of `bench_full.py`; it is useful for filling missing multimodal coverage, not for estimating variance.
- Vision cases validate the image input path and coexistence with AWQ4 + DFlash; their throughput includes multimodal processing and should not be compared directly with pure text decode cases.
