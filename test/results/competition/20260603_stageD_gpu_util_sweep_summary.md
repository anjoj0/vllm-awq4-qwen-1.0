# Stage D gpu_memory_utilization Sweep

固定配置：`N=8`、`MBT=8192`、`max_num_seqs=1`、`AWQ_MMQ_DECODE_BACKEND=triton`。目标是验证更高/更低 KV 池是否改善真实论文长 prompt，或只是改变显存余量。

| gpu util | KV GiB | KV tokens | max conc | engine init s | paper8k wall | paper8k TTFT | paper32k wall | paper32k TTFT | paper32k prefill tok/s | accept % last |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.58 | 24.090 | 84032 | 3.130 | 66.230 | 18.792 | 12.919 | 67.915 | 55.973 | 119.844 | 18.600 |
| 0.60 | 26.410 | 91520 | 3.430 | 65.500 | 17.548 | 11.513 | 67.520 | 55.567 | 120.719 | 45.700 |
| 0.62 | 28.730 | 99840 | 3.730 | 65.730 | 18.733 | 12.907 | 68.451 | 56.513 | 118.697 | 37.500 |

## Decision

`gpu_memory_utilization=0.60` remains the best current competition profile. `0.58` saves KV memory but slows paper8k/paper32k slightly. `0.62` increases KV cache to ~99.8K tokens and max concurrency to 3.73x, but paper32k TTFT worsened in this single-stream 64K workload. Since 0.62 already fails to improve latency, 0.64 was not tested in this pass.

## Source files

- gpu=0.58: `test/results/competition/20260603-163717_stageD_gpu058_n8_mbt8192_longscreen_stream.json`
- gpu=0.6: `test/results/competition/20260603-115115_stageC_n8_mbt8192_longscreen_stream.json`
- gpu=0.62: `test/results/competition/20260603-164520_stageD_gpu062_n8_mbt8192_longscreen_stream.json`
- CSV: `test/results/competition/20260603_stageD_gpu_util_sweep_summary.csv`
