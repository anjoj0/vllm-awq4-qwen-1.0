# W7900 Experiment Matrix

Use this file to record W7900 results. Keep raw logs under `results/` and copy
only the stable summaries into the main report.

## Environment

| Item | Value |
| --- | --- |
| Machine | 8x Radeon PRO W7900 |
| GPU arch | gfx1100 |
| ROCm version | TBD |
| Linux kernel | TBD |
| vLLM image | TBD |
| Model | Qwen3.6-27B-AWQ-INT4 |
| Drafter | z-lab/Qwen3.6-27B-DFlash |

## Bring-Up Checklist

| Check | Status | Notes |
| --- | --- | --- |
| `rocminfo` sees 8 GPUs | TBD |  |
| PyTorch sees 8 GPUs | TBD |  |
| single-GPU tensor allocation | TBD |  |
| all 8 GPUs allocation | TBD |  |
| RCCL/all-reduce sanity | TBD |  |
| AWQ4 no DFlash boots | TBD |  |
| AWQ4 + DFlash boots | TBD |  |
| fp8 KV boots | TBD |  |

## Throughput Sweep

| TP | Visible GPUs | Context | KV dtype | DFlash N | MBT | GPU util | Median t/s | Peak t/s | Acceptance | Notes |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 65536 | fp8 | 8 | 8192 | 0.80 | TBD | TBD | TBD | bring-up |
| 2 | 0,1 | 131072 | fp8 | 8 | 16384 | 0.80 | TBD | TBD | TBD | first production candidate |
| 4 | 0,1,2,3 | 262144 | fp8 | 8 | 16384 | 0.80 | TBD | TBD | TBD | capacity candidate |
| 8 | 0-7 | 262144 | fp8 | 8 | 32768 | 0.80 | TBD | TBD | TBD | communication-risk experiment |

## Decision Log

| Date | Decision | Evidence |
| --- | --- | --- |
| TBD | Start with TP=2 for production exploration | Reduces memory pressure with lower communication risk than TP=8 |
