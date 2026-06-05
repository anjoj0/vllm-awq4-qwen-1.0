# FP8 A/B Benchmark: DFlash N=3 and MBT=16384 - 2026-06-05

## Context

Patch 24a/b/c was live-applied to the container runtime, committed as `vllm-awq4-qwen:patches-20-21-24abc-live`, and the compose image tag `vllm-awq4-qwen:patches-20-21` was retagged to that patched image. After this, fp8 KV booted successfully.

Benchmark setup: `bench_matrix.py`, case `normal`, contexts `0/8192/16384/32768`, one run per context, `max_tokens=512`, host port `8001`.

## Decode Throughput

| config | 0 ctx | 8K ctx | 16K ctx | 32K ctx | mean |
|---|---:|---:|---:|---:|---:|
| N4 / MBT8192 / fp8 baseline | 14.57 | 12.04 | 10.56 | 7.33 | 11.12 |
| N3 / MBT8192 / fp8 | 12.49 | 12.57 | 10.40 | 7.72 | 10.80 |
| N4 / MBT16384 / fp8 | 14.52 | 13.51 | 10.63 | 7.42 | 11.52 |

## TTFT / Prefill Notes

| config | 8K TTFT | 16K TTFT | 32K TTFT |
|---|---:|---:|---:|
| N4 / MBT8192 / fp8 baseline | 77.8s | 173.3s | 438.0s |
| N3 / MBT8192 / fp8 | 77.8s | 171.7s | 433.5s |
| N4 / MBT16384 / fp8 | 78.6s | 175.8s | 441.3s |

## Decision

- Best balanced setting in this quick A/B: `N=4, MBT=16384, KV=fp8`. It has the best mean decode throughput and wins 8K/16K while keeping short-context performance equal to baseline.
- Best 32K-only decode: `N=3, MBT=8192, KV=fp8` at `7.72 t/s`, but it regresses short-context decode (`12.49` vs baseline `14.57`) and slightly regresses 16K.
- `MBT=16384` is now a valid candidate after Patch 24; before Patch 24 it failed during KV page-size unification.

## Files

- N4 / MBT8192 / fp8 baseline: `test/results/bench_matrix_patch20_21_retest_20260605.json`
- N3 / MBT8192 / fp8: `test/results/bench_matrix_ab_n3_mbt8192_fp8_20260605.json`
- N4 / MBT16384 / fp8: `test/results/bench_matrix_ab_n4_mbt16384_fp8_20260605.json`
