# AWQ MMQ Partial Decode Fallback Policy Comparison

## Startup / Memory

| Mode | Model mem GiB | Delta vs full Triton | KV GiB | KV tokens | 64K concurrency |
|---|---:|---:|---:|---:|---:|
| triton_all | 40.5 | 0.0 | 20.61 | 71552 | 2.68 |
| hip_none | 27.27 | -13.23 | 33.84 | 118144 | 4.4 |
| triton_n_ge_16384 | 36.39 | -4.11 | 24.72 | 85696 | 3.21 |
| triton_n_lt_16384 | 32.09 | -8.41 | 29.02 | 100672 | 3.77 |

## Request Benchmarks

### short_decode_128

| Mode | Wall s | Total tok/s | Delta total tok/s vs full Triton | Output tok/s |
|---|---:|---:|---:|---:|
| triton_all | 8.614 | 42.721 | 0.0% | 9.171 |
| hip_none | 16.575 | 22.022 | -48.5% | 4.585 |
| triton_n_ge_16384 | 12.239 | 29.823 | -30.2% | 6.21 |
| triton_n_lt_16384 | 13.734 | 26.504 | -38.0% | 5.461 |

### mid_prefill_512_decode_64

| Mode | Wall s | Total tok/s | Delta total tok/s vs full Triton | Output tok/s |
|---|---:|---:|---:|---:|
| triton_all | 23.074 | 97.815 | 0.0% | 2.774 |
| hip_none | 29.207 | 77.275 | -21.0% | 2.191 |
| triton_n_ge_16384 | 25.891 | 87.174 | -10.9% | 2.472 |
| triton_n_lt_16384 | 26.531 | 85.069 | -13.0% | 2.412 |

### long_prefill_2k_decode_32

| Mode | Wall s | Total tok/s | Delta total tok/s vs full Triton | Output tok/s |
|---|---:|---:|---:|---:|
| triton_all | 78.405 | 111.639 | 0.0% | 0.408 |
| hip_none | 80.874 | 108.23 | -3.1% | 0.396 |
| triton_n_ge_16384 | 80.417 | 108.845 | -2.5% | 0.398 |
| triton_n_lt_16384 | 78.164 | 111.982 | 0.3% | 0.409 |

## Current conclusion

- Full Triton fallback remains the fastest short-decode path.
- All-HIP saves the most memory but hurts short decode too much for the default profile.
- Partial fallback policies land between those extremes, but neither beats full Triton on the short and mid cases.
- The data supports building a dedicated small-M HIP decode kernel instead of reusing the prefill-oriented MMQ kernel.
