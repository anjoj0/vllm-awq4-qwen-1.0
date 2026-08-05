The investigation has now converged, and a new NIXL transport plugin is no longer the primary upstream proposal.

The W7900 fallback is caused by an error-mode capability mismatch in the existing UCX path: NIXL defaults to `UCP_ERR_HANDLING_MODE_PEER`, while UCX 1.22 reports `rocm_ipc error handling: none`, so UCP excludes that data lane. An experimental one-line UCX capability change restores direct `rocm_ipc` while retaining NIXL peer mode, improving verified 1 GiB READ/WRITE from 5.203/4.962 GB/s to 27.382/23.512 GB/s.

The UCX-level reproducer, patch, failure-injection results, and open questions are now tracked here:
https://github.com/ROCm/ucx/issues/35#issuecomment-5195173629

Could AMD please route that issue to the ROCm UCX transport owner and, if the design is acceptable, help validate it on an Instinct CI system? I only have W7900/gfx1100 access, so I can provide and maintain the patch/tests but cannot claim cross-architecture coverage locally.
