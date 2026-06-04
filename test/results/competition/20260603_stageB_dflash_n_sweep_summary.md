# Stage B DFlash N Sweep Summary

固定配置：`gpu_memory_utilization=0.60`、`max_num_batched_tokens=8192`、`max_num_seqs=1`、`AWQ_MMQ_DECODE_BACKEND=triton`。每个 N 均重启容器后运行同一组 screen workload，因此包含冷启动后首批请求的热路径成本。

## Main table

| N | block | pad % | KV tokens | max conc | accept % last | mean accept len | short wall | mid wall | paper8k wall | paper8k TTFT | paper8k decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 816 | 1.620 | 92208 | 3.730 | 72.100 | 3.880 | 14.649 | 38.269 | 18.747 | 11.448 | 17.535 |
| 6 | 816 | 0.370 | 92208 | 3.570 | 54.300 | 4.260 | 7.240 | 37.254 | 17.895 | 11.409 | 19.735 |
| 8 | 832 | 1.090 | 91520 | 3.430 | 45.700 | 4.650 | 7.412 | 36.366 | 17.314 | 11.441 | 21.794 |
| 10 | 848 | 1.800 | 91584 | 3.260 | 41.600 | 5.160 | 13.093 | 36.415 | 17.148 | 11.504 | 22.678 |
| 12 | 848 | 0.590 | 91584 | 3.140 | 30.300 | 4.630 | 7.206 | 36.322 | 16.870 | 11.406 | 23.425 |

## Observations

- `N=4` acceptance 最高，但 wall latency 明显偏慢，说明 draft 浪费少不等于整体吞吐高。
- `N=6` 是中间点，但 short/mid/paper8k 均没有明确超过 `N=8`。
- `N=8` 仍是最稳的默认点：block size 832，KV cache 91,520 tokens，max concurrency 3.43x，在 Stage A 3-run 中也通过了 paper32k。
- `N=10/12` 在 paper8k 单次 screen 上略快，尤其 `N=12` paper8k wall 为 16.870s，但 block size 变成 848，max concurrency 降到 3.26x/3.14x，最后 acceptance 也降到 41.6%/30.3%。
- 当前不建议直接把默认改到 `N=12`。更合理的下一步是只让 `N=12` 进入 Stage C/长 prompt 验证候选，与 `N=8` 一起测试 `max_num_batched_tokens=12288/16384`。

## Decision

默认 shipping profile 暂时保持 `N=8`。Stage C 候选保留两个：`N=8` 作为稳定基线，`N=12` 作为中长 prompt 激进候选。

## Source files

- N=4: `test/results/competition/20260603-110240_stageB_n4_gpu060_mbt8192_screen_stream.json`
- N=6: `test/results/competition/20260603-110839_stageB_n6_gpu060_mbt8192_screen_stream.json`
- N=8: `test/results/competition/20260603-114323_stageB_n8_gpu060_mbt8192_screen_stream.json`
- N=10: `test/results/competition/20260603-111830_stageB_n10_gpu060_mbt8192_screen_stream.json`
- N=12: `test/results/competition/20260603-113032_stageB_n12_gpu060_mbt8192_screen_stream.json`
- CSV: `test/results/competition/20260603_stageB_dflash_n_sweep_summary.csv`
