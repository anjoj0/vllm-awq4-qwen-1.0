# AWQ MMQ Decode Kernel Version Comparison

| Mode | Model mem GiB | KV tokens | short wall | short total tok/s delta | mid wall | mid delta | long wall | long delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| triton_all | 40.5 | 71552 | 8.614 | 0.0% | 23.074 | 0.0% | 78.405 | 0.0% |
| hip_v1_none | 27.27 | 118144 | 16.575 | -48.5% | 29.207 | -21.0% | 80.874 | -3.1% |
| partial_n_ge_16384 | 36.39 | 85696 | 12.239 | -30.2% | 25.891 | -10.9% | 80.417 | -2.5% |
| partial_n_lt_16384 | 32.09 | 100672 | 13.734 | -38.0% | 26.531 | -13.0% | 78.164 | 0.3% |
| hip_v0_scalar_none | 27.27 | 118144 | 110.563 | -91.2% | 71.352 | -67.7% | 101.026 | -22.4% |

## Conclusion

- HIP scalar version=0 is much slower than both Triton fallback and HIP WMMA version=1.
- version=0 should remain a correctness/reference path only.
- A real small-M decode kernel needs parallel reduction across K, not one scalar thread per output element.
