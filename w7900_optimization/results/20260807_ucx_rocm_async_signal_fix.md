# ROCm IPC async signal error propagation: fix attempt and boundary

## Outcome

The first repair attempt found and fixed a real UCX error-propagation bug, but it
does not by itself recover an exporter-invalidated stale rkey on W7900.

ROCr documents `hsa_amd_memory_async_copy()` completion as follows:

```text
success: completion signal is decremented to 0
async failure: completion signal is set to a negative number
```

UCX `uct_rocm_base_progress()` previously extracted only signals equal to zero and
always completed them with `UCS_OK`. A negative HSA error signal therefore remained
in the queue forever. The experimental patch changes the ready condition to
`signal <= 0`, maps negative values to `UCS_ERR_IO_ERROR`, and retains the existing
detach and signal-pool cleanup.

## Patch

The local OpenUCX branch is:

```text
G:\AI_projects\AMD\upstream\ucx-pr11299-peerflag
47fcc361e ROCM: propagate asynchronous signal failures
a33495929 UCT/ROCM: Advertise peer failure support for IPC
4dddf15e4 OpenUCX #11299 test head
```

The standalone patch is
[ucx_rocm_async_signal_error.patch](../pd_disaggregation/ucx_rocm_async_signal_error.patch).

Core behavior:

```c
signal_value = hsa_signal_load_scacquire(rocm_signal->signal);
completion_status = (signal_value < 0) ? UCS_ERR_IO_ERROR : UCS_OK;
uct_invoke_completion(rocm_signal->comp, completion_status);
```

No timeout is used. Completing a request only because a wall-clock deadline passed
would be unsafe: ROCr provides no cancellation acknowledgement, so the DMA could
still access buffers after UCP has released them or the signal descriptor has been
reused.

## Validation

Environment: 8 x W7900, ROCm 7.14 container, OpenUCX #11299 plus the peer-failure
capability and async-signal patch.

| Test | Result |
|---|---|
| ROCm IPC GTest | 132 total, 92 passed, 40 expected skips, 0 failed |
| NIXL 1 GiB READ, 3 hot reps | 27.386 GB/s, payload verified |
| NIXL 1 GiB pipeline WRITE, 3 hot reps | 23.565 GB/s, payload verified |
| normal 1 GiB READ | `DONE`, payload verified |
| exit before transfer | `NIXL_ERR_REMOTE_DISCONNECT` in 0.172 ms |
| clean exit before transfer | `NIXL_ERR_REMOTE_DISCONNECT` in 0.219 ms |
| exit after posting 8 GiB READ | `DONE`, payload verified, 0.404 s |
| exit after posting 8 GiB WRITE | `NIXL_ERR_REMOTE_DISCONNECT`, 0.328 s |
| stale registration, 1 GiB READ | still reaches the 35 s process timeout |

The normal and legal peer-exit results match the unmodified completion path. The
patch therefore preserves the established W7900 behavior while adding handling for
ROCr-reported negative signals.

## Why stale registration still hangs

In the stale test, the exporter remains alive but deregisters and releases the GPU
allocation after publishing its rkey. `hsa_amd_ipc_memory_attach()` and pointer
inspection still succeed on the initiator, and the async copy call returns success.
The W7900 ROCr runtime then leaves the completion signal pending rather than setting
it to a negative value. UCX has no API to cancel that DMA safely and no notification
that the exporter retired the registration.

This is different from a peer process exit. TCP/CMA keepalive can report an exited
peer, but it cannot infer that a live peer invalidated one particular exported rkey.

## Safe repair architecture

The robust solution has two layers.

### Layer 1: propagate runtime errors

Keep the negative-signal patch. It closes the case where ROCr explicitly reports an
asynchronous copy failure and is independently useful for `rocm_copy` and
`rocm_ipc`.

### Layer 2: registration retirement handshake

Prevent stale keys instead of pretending a possibly active DMA was cancelled:

```text
owner                       consumers
  |-- RETIRE(generation) ------>|
  |                             | stop new transfers
  |                             | wait/release active handles and rkeys
  |<----- RETIRE_ACK -----------|
  | deregister/free allocation  |
  |-- RETIRED(generation) ------>|
```

Required properties:

1. exported metadata carries a registration generation/epoch;
2. the owner marks a generation `RETIRING` before deregistration;
3. consumers reject new posts for retiring generations and drain active requests;
4. the owner deregisters only after all known consumers acknowledge, or after their
   endpoint is definitively disconnected;
5. address reuse always creates a new generation, so an old rkey cannot match a new
   allocation at the same virtual address.

This belongs at the NIXL/application control plane, with UCX remaining responsible
for RMA and runtime-reported errors. A fixed UCX wall-clock timeout is not an
equivalent safe fix.

## Upstream status

The async-signal change is suitable as a small ROCm-only follow-up after adding a
unit test that injects a negative HSA signal. It should not be presented as the
complete stale-registration solution. The retirement protocol requires NIXL API or
connector design agreement because the current UCX backend deregisters local memory
without tracking which remote agents still hold public metadata.

## Raw artifacts

```text
results/ucx_rocm_async_signal_fix_20260807.tgz
SHA256 70F90831053C1CF812FD63EB8F6CF73448264E90CC23EB0DAC793B4AB7F730F5
```

The archive contains build logs, JSON GTest output, normal RMA logs, the full legal
peer-exit matrix, and the stale regression.
