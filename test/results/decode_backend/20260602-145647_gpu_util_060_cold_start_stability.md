# GPU Memory Util 0.60 Cold-Start Stability

Date: 2026-06-02

Runs passed: 3 / 3

| run | API ready | first request | KV tokens | KV GiB | max concurrency | first request s | OOM-like logs |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | True | True | 91,520 | 26.41 | 3.43x | 3.382 | 0 |
| 2 | True | True | 91,520 | 26.41 | 3.43x | 3.380 | 0 |
| 3 | True | True | 91,520 | 26.41 | 3.43x | 3.320 | 0 |

Conclusion: 0.60 passed 3/3 cold-start + first-request probes. This supports promoting it from candidate to primary competition memory profile, pending one near-64K real request validation.

Run files:

- `test/results/decode_backend/20260602-144612_gpu_util_060_cold_start_run1.json`
- `test/results/decode_backend/20260602-144944_gpu_util_060_cold_start_run2.json`
- `test/results/decode_backend/20260602-145315_gpu_util_060_cold_start_run3.json`
