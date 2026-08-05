Root cause update: this is not a missing `sm` lane, GPU visibility issue, or lost NIXL worker address. The direct `rocm_ipc` data lane is excluded because of endpoint error-handling capability matching.

NIXL's UCX backend defaults to `UCP_ERR_HANDLING_MODE_PEER`, while UCX 1.22 reports:

```text
rocm_ipc error handling: none
cma      error handling: peer failure, ep_check
```

With `UCX_TLS=sm,rocm,tcp,self` and `UCX_RMA_PPLN_ENABLE=y`, changing only the NIXL UCX error mode gives:

| 1 GiB operation | `peer` | `none` |
|---|---:|---:|
| READ | 5.203 GB/s | 27.399 GB/s |
| WRITE | 4.962 GB/s | 23.600 GB/s |

All payload checks pass. `none` selects `rocm_ipc/rocm_ipc`; `peer` uses the host/CMA pipeline. The same selection behavior reproduces without NIXL using `ucx_perftest`: no `-e` selects ROCm IPC, while adding only `-e` removes that lane.

As a causality experiment, I rebuilt UCX 1.22 with only this change:

```diff
- iface_attr->cap.flags = UCT_IFACE_FLAG_GET_ZCOPY |
+ iface_attr->cap.flags = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
+                         UCT_IFACE_FLAG_GET_ZCOPY |
```

NIXL kept `peer` mode and recovered `rocm_ipc/rocm_ipc` at 27.382 GB/s READ and 23.512 GB/s WRITE, within 0.06%/0.37% of the original `none` results.

I also tested failure behavior on W7900 before proposing the flag as a fix:

- peer exit before transfer returns `NIXL_ERR_REMOTE_DISCONNECT` immediately;
- peer exit after posting an 8 GiB READ completes with a verified payload;
- peer exit after posting an 8 GiB WRITE returns `NIXL_ERR_REMOTE_DISCONNECT` in 0.322 s;
- deliberately invalidating a registration while the exported rkey may still be used can hang. Original UCX `none + rocm_ipc` has the same behavior, so this is an existing stale-rkey/error-propagation gap rather than a regression from the flag. It also violates the normal rkey lifetime contract and should be tracked separately.

The one-line capability change is sufficient for the NIXL RMA performance path and passed the peer-exit cases available on W7900, but I cannot validate other ROCm architectures locally.

Full patch, methods, scripts, structured results, failure matrix, and raw logs:
https://github.com/anjoj0/vllm-awq4-qwen-1.0/blob/main/w7900_optimization/results/20260805_nixl_ucx_error_mode_root_cause.md

Should `rocm_ipc` advertise `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` in UCX, and if so, which existing UCX peer-failure regression fixture would you prefer this transport to satisfy before a PR?
