@edgargabriel @brminich @sbates130272 @yafshar I completed the focused reproducer and need to correct my previous worker-address hypothesis. The accelerator iface is not lost during NIXL metadata exchange.

The actual discriminator is endpoint error handling. NIXL's UCX backend defaults to `UCP_ERR_HANDLING_MODE_PEER`, while UCX 1.22 reports `rocm_ipc error handling: none`. UCP therefore excludes the `rocm_ipc` data lane from that endpoint.

With all other settings unchanged (`UCX_TLS=sm,rocm,tcp,self`, RMA pipeline enabled, GPU0 <-> GPU1, verified payloads):

| NIXL operation, 1 GiB | UCX `peer` | UCX `none` |
|---|---:|---:|
| READ | 5.203 GB/s | 27.399 GB/s |
| WRITE | 4.962 GB/s | 23.600 GB/s |

This is a 5.27x READ and 4.76x WRITE difference. A direct `ucx_perftest` A/B reproduces the lane-selection behavior without NIXL: no `-e` selects `rocm_ipc/rocm_ipc`; adding only `-e` removes that lane.

As a causality experiment, I rebuilt UCX with only `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` added to `rocm_ipc`. NIXL kept its default `peer` mode and recovered `rocm_ipc/rocm_ipc` at 27.382 GB/s READ and 23.512 GB/s WRITE, within 0.06%/0.37% of the `none` results.

I also ran explicit failure injection before treating that flag as a proposed fix:

- target exit before transfer: immediate `NIXL_ERR_REMOTE_DISCONNECT`;
- clean target exit before transfer: immediate `NIXL_ERR_REMOTE_DISCONNECT`;
- target exit after posting an 8 GiB READ: `DONE` with verified payload;
- target exit after posting an 8 GiB WRITE: `NIXL_ERR_REMOTE_DISCONNECT` in 0.322 s;
- deliberately invalidating a still-exported registration can hang the ROCm IPC path. The original UCX `none` path has the same behavior, so this is not introduced by the flag, but it remains an error-propagation gap. This case also violates the normal rkey lifetime contract and should probably be tracked separately.

The one-line flag is therefore sufficient for the NIXL performance path and passed the peer-exit cases available on W7900, but I am not claiming it is production-ready across ROCm architectures yet.

Full methods, scripts, build patch, structured results, fault matrix, and raw-log SHA256:
https://github.com/anjoj0/vllm-awq4-qwen-1.0/blob/main/w7900_optimization/results/20260805_nixl_ucx_error_mode_root_cause.md

Would you prefer the next patch to target UCX/ROCm's `rocm_ipc` capability plus a peer-exit regression test, or should I first submit a NIXL configuration/diagnostic change that makes the `peer` versus direct-ROCm-IPC tradeoff explicit?
