# W7900 HIP IPC NIXL backend: upstreaming notes

## Scope

`W7900_HIP_IPC` is a same-host NIXL backend for ROCm VRAM. It exports the
base allocation behind a registered tensor with `hipIpcGetMemHandle`, imports
it in a peer process with `hipIpcOpenMemHandle`, and submits descriptor copies
with `hipMemcpyAsync` on a request-owned non-blocking stream.

The backend currently declares:

- `VRAM_SEG` only;
- `NIXL_READ` and `NIXL_WRITE`;
- same Linux host only;
- Unix datagram notifications;
- prepared request repost with a notification that may change on every post.

It does not claim network transport, host-memory transport, process migration,
or operation across different Linux IPC namespaces.

## Request lifecycle

Each prepared handle owns one HIP stream and two events. Handles do not share a
backend-global stream, so independent NIXL requests can progress concurrently.

| State | Accepted action | Result |
|---|---|---|
| `Prepared` | `postXfer` | enqueue copies, enter `InProgress` |
| `InProgress` | `checkXfer` | `NIXL_IN_PROG` until the end event completes |
| `InProgress` | `postXfer` | `NIXL_ERR_REPOST_ACTIVE` |
| `DataComplete` | `checkXfer` | retry notification or complete |
| `Complete` | `postXfer` | reset per-post fields and repost the same handle |
| `Failed` | `postXfer` | rebuild stream/events, then repost |

`releaseReqH` synchronizes an active stream before destroying it. HIP does not
provide cancellation for an already submitted `hipMemcpyAsync`; release is
therefore safe but may block until the GPU operation completes.

## Failure propagation

The implementation preserves data-copy completion separately from notification
completion. A transient notification send (`EAGAIN`, `EWOULDBLOCK`, `ENOBUFS`)
returns `NIXL_IN_PROG`, so `checkXfer` can retry it without copying the payload
again. Permanent errors are mapped as follows:

| Condition | NIXL status |
|---|---|
| invalid descriptor, overflow, oversized notification | `NIXL_ERR_INVALID_PARAM` |
| active handle repost | `NIXL_ERR_REPOST_ACTIVE` |
| missing peer socket | `NIXL_ERR_NOT_FOUND` |
| peer socket removed/refused | `NIXL_ERR_REMOTE_DISCONNECT` |
| malformed or truncated datagram | `NIXL_ERR_MISMATCH` |
| HIP or unclassified system error | `NIXL_ERR_BACKEND` |

After a terminal failure, the handle retains its failure status for
`checkXfer`. A later `postXfer` destroys and recreates the request-owned HIP
resources before submitting the next round. Notification text and sent state
are reset for every post.

## Metadata and memory safety

Public metadata contains a magic value, wire version, exporter device,
allocation base, allocation size, and `hipIpcMemHandle_t`. Registration uses
`hipMemGetAddressRange`, so a tensor slice exports its allocator-owned base
rather than an invalid interior pointer.

Remote ranges are checked without unchecked address addition. Imported mappings
are cached by IPC handle through `weak_ptr`; the final remote metadata release
closes the HIP mapping. Every HIP API path explicitly selects the backend
device before operating on a stream, event, or mapping.

## Build and load

The standalone gate builds against a NIXL source checkout and installed ROCm
NIXL prefix:

```bash
NIXL_SRC=/path/to/nixl \
NIXL_PREFIX=/path/to/nixl-install \
INSTALL_DIR=/path/to/nixl-install/lib/x86_64-linux-gnu/plugins \
  bash build_plugin.sh
```

For an upstream NIXL change, move the backend under `src/plugins/hip_ipc`, add
it to the plugin Meson options and install rules, and convert the cross-process
Python gate into a NIXL integration test. The standalone script is retained in
this repository so the implementation can be tested against NIXL 1.4 without
modifying the source tree.

## Validation completed

On eight Radeon PRO W7900 GPUs with ROCm 7.14 and NIXL 1.4:

- 1 GiB READ, one cold post plus two hot reposts: `14.77`, `25.19`, and
  `25.24 GiB/s`; all payload checks passed after rebuilding the final source.
- An immediate second post while the first was active returned
  `NIXL_ERR_REPOST_ACTIVE`; the original post still completed correctly.
- A deliberately oversized notification failed after a correct data copy; the
  same handle then reposted with a valid notification and completed correctly.
- vLLM Qwen3.6-27B TP4 Prefill/TP4 Decode completed two 8K deterministic gates
  with zero `NIXL_ERR` and hot TTFT of `3.52 s`.
- TP2 fan-out from one Prefill replica to three Decode replicas passed
  compatibility, payload, and notification checks for all three peers.

## Upstream review boundaries

The data path uses standard HIP IPC APIs and has no gfx1100 ISA dependency, but
the current validation hardware is W7900. Rename the public backend to
`HIP_IPC` only after adding at least one non-W7900 ROCm platform to CI or a
documented manual gate. Cross-host selection must remain the responsibility of
the NIXL scheduler or a fallback backend such as UCX.

Before submitting to NIXL, retain the following review gates:

1. Build with the oldest and newest supported ROCm/NIXL versions.
2. Run READ and WRITE, multi-descriptor, repost, active-repost, recovery, and
   concurrent-handle tests under ASAN/UBSAN where HIP permits it.
3. Validate fork/exec cleanup and agent disconnect/reconnect.
4. Add a second ROCm architecture and both same-NUMA and cross-NUMA GPU pairs.
5. Confirm plugin wire metadata compatibility policy before changing version 1.
