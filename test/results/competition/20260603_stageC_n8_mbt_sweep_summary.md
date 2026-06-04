# Stage C N=8 max_num_batched_tokens Sweep

固定配置：`N=8`、`gpu_memory_utilization=0.60`、`max_num_seqs=1`、`AWQ_MMQ_DECODE_BACKEND=triton`。目标是验证增大 scheduler token budget 是否降低真实论文长 prompt 的 TTFT。

| MBT | KV tokens | max conc | engine init s | paper8k wall | paper8k TTFT | paper32k wall | paper32k TTFT | paper32k prefill tok/s | accept % last |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 91520 | 3.430 | 65.500 | 17.548 | 11.513 | 67.520 | 55.567 | 120.719 | 45.700 |
| 12288 | 88192 | 3.280 | 97.310 | 18.628 | 12.801 | 67.641 | 55.690 | 120.453 | 17.900 |
| 16384 | 84032 | 3.130 | 131.250 | 18.784 | 12.930 | 68.123 | 56.165 | 119.434 | 18.400 |

## Decision

`MBT=8192` should remain the default. `12288` and `16384` both increased paper8k/paper32k latency in this run, reduced available KV cache, and increased startup/profile cost. For this single-stream 64K competition profile, larger scheduler chunks did not improve long-prompt TTFT.

## Source files

- MBT=8192: `test/results/competition/20260603-115115_stageC_n8_mbt8192_longscreen_stream.json`
- MBT=12288: `test/results/competition/20260603-161017_stageC_n8_mbt12288_longscreen_stream.json`
- MBT=16384: `test/results/competition/20260603-161903_stageC_n8_mbt16384_longscreen_stream.json`
- CSV: `test/results/competition/20260603_stageC_n8_mbt_sweep_summary.csv`
