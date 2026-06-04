# Hybrid backend policy comparison

- Policy: `AWQ_MMQ_DECODE_BACKEND=hybrid`, short/mid small-M decode stays on Triton, long-context `M=9` verify routes to HIP v7.
- Long-context detection: prefill `M >= 4096`, verify `M=9`, HIP version `7`.
- Metric: `avg_total_tps_e2e`, one run per scenario.

| Scenario | Triton current | v7 all-HIP | Hybrid | Hybrid vs Triton current | Hybrid vs v7 all-HIP |
|---|---:|---:|---:|---:|---:|
| short_decode_128 | 38.394 | 30.494 | 38.170 | -0.6% | +25.2% |
| mid_prefill_512_decode_64 | 97.293 | 89.520 | 97.676 | +0.4% | +9.1% |
| long_prefill_2k_decode_32 | 112.824 | 112.615 | 111.232 | -1.4% | -1.2% |

## Route Verification

- `('decode', 'triton', -1)`: `14592` calls
- `('decode', 'hip', 7)`: `4096` calls
- `('prefill', 'hip', 1)`: `1280` calls

Top M routes:

| route/backend/version/M | count |
|---|---:|
| `('decode', 'triton', -1, 9)` | 14336 |
| `('decode', 'hip', 7, 9)` | 4096 |
| `('decode', 'triton', -1, 1)` | 256 |
| `('prefill', 'hip', 1, 289)` | 256 |
| `('prefill', 'hip', 1, 536)` | 256 |
| `('prefill', 'hip', 1, 2193)` | 256 |
| `('prefill', 'hip', 1, 8185)` | 256 |
| `('prefill', 'hip', 1, 8192)` | 256 |

## Readout

- The policy routed as intended: most small-M decode stayed on Triton; long-context `M=9` verify produced `4096` HIP v7 calls.
- Performance did not improve versus the current Triton baseline. Current Triton long is `112.824`, while hybrid long is `111.232`.
- Keep hybrid as an opt-in experiment, not the default. The data now points back to KV/cache or a better M=9-specific HIP kernel if continuing compute-side work.
