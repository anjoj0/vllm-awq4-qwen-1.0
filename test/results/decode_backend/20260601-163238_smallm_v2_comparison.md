# AWQ MMQ Small-M Decode v2 Comparison

| Mode | Model mem GiB | KV tokens | short wall | short total tok/s delta | mid wall | mid delta | long wall | long delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| triton_all | 40.5 | 71552 | 8.614 | 0.0% | 23.074 | 0.0% | 78.405 | 0.0% |
| hip_v1_none | 27.27 | 118144 | 16.575 | -48.5% | 29.207 | -21.0% | 80.874 | -3.1% |
| hip_v0_scalar_none | 27.27 | 118144 | 110.563 | -91.2% | 71.352 | -67.7% | 101.026 | -22.4% |
| hip_v2_smallm_none | 27.27 | 118144 | 64.845 | -86.7% | 69.067 | -66.6% | 107.786 | -27.3% |

## Conclusion

- v2 is correct in smoke tests but still too slow for end-to-end decode.
- v2 improves over scalar v0 on short decode but is much slower than the current v1 all-HIP path and Triton fallback.
- The next design should reduce block count and compute more N columns per block or use a wave-level GEMV-style mapping rather than one row x 16 columns per block.
