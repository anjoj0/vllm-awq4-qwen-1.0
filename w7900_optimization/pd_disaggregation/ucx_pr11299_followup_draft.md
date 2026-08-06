# Draft follow-up for NIXL #2039 / ROCm UCX #35

This text is a draft and has not been posted.

@edgargabriel Thank you for pointing out the interaction with openucx/ucx#11299. I tested the current PR head (`4dddf15e46735555405bf678be778a23358ec45f`) on the same W7900 node before proposing a separate change.

I applied only `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` on top of #11299. I did not change its ROCm IPC handle cache, MD/EP implementation, device-initiated PUT path, HIP kernels, or tests.

The combined branch built successfully with ROCm and GTest. The ROCm-specific tests added by #11299 produced:

```text
132 tests: 92 passed, 40 expected skips, 0 failed
```

The skips are the cases explicitly unsupported by the PR (8 warp cases with fewer than 64 threads and 32 grid-level cases).

With NIXL's default `peer` error mode, `UCX_TLS=sm,rocm,tcp,self`, and verified 1 GiB payloads, the combined branch selected `rocm_ipc/rocm_ipc` in all four cases:

| GPU pair | Topology | READ | WRITE |
|---|---|---:|---:|
| 0-1 | same NUMA | 27.377 GB/s | 23.527 GB/s |
| 0-4 | cross NUMA | 27.391 GB/s | 23.522 GB/s |

These results are within 0.12% of the standalone `develop + peer-failure flag` build. I also repeated the four legal cross-NUMA peer-exit scenarios three times each: all 12 runs returned either a verified `DONE` or `NIXL_ERR_REMOTE_DISCONNECT`, with no hang.

This suggests the capability change is mechanically compatible with #11299 and complementary to its device-initiated IPC work. I have intentionally excluded the stale-registration case from this claim because it violates the exported-rkey lifetime contract and intersects the cache behavior that #11299 is changing.

Full build details, structured results, source snapshot, and raw-log hashes:

https://github.com/anjoj0/vllm-awq4-qwen-1.0/blob/main/w7900_optimization/results/20260806_ucx_pr11299_compatibility_report.md

Would AMD prefer to absorb the capability flag and a suitable multi-process peer-exit regression into #11299 (or a follow-up AMD-owned PR), or would a small dependent PR from me be useful? I am happy to provide the W7900 reproducer and maintain the test, while leaving handle-cache changes to the ROCm IPC owners.
