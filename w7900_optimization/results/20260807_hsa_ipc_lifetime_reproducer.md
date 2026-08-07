# Pure HSA IPC allocation-lifetime reproducer on W7900

## Result

A standalone two-process reproducer was implemented with the ROCr HSA API only.
It does not link UCX, NIXL, HIP, PyTorch, or vLLM.

The experiment separates three lifetimes:

| Mode | Exporter action | Importer result |
|---|---|---|
| `valid` | free after the copy completes | attach succeeds; copy signal reaches `0` |
| `stale` | free after importer attaches, before copy | attach remains valid; copy signal reaches `0` |
| `pre_attach_free` | publish handle, then free before first attach | `hsa_amd_ipc_memory_attach()` blocks beyond 20 s |

`pre_attach_free` was repeated three times. All three importer processes reached
the 20 s process timeout (`exit 124`) inside `hsa_amd_ipc_memory_attach()`; the
call did not return `HSA_STATUS_ERROR_INVALID_ARGUMENT`.

## Why this is an ROCr API issue

The installed `hsa_ext_amd.h` explicitly documents both sides of the contract:

```text
The exporter process MUST keep the allocation alive until all importers have
successfully called hsa_amd_ipc_memory_attach. If the exporter frees the memory
before importers attach, subsequent attach calls will fail with
HSA_STATUS_ERROR_INVALID_ARGUMENT.
```

The first two modes show that normal IPC and release-after-attach work on the
same machine. The third mode violates the required lifetime intentionally, but
the documented failure is not returned: the API blocks indefinitely instead.
This is an error-propagation problem, not a request to make stale memory access
succeed.

## Configuration

| Item | Value |
|---|---|
| GPU | 8 x AMD Radeon Pro W7900D, `gfx1100` |
| path | exporter GPU0 to importer GPU4 |
| allocation | 64 MiB GPU-local HSA memory |
| container | `rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0` |
| HSA runtime | runtime 1.21, extension 1.26 |
| kernel | Ubuntu 6.8.0-79-generic |
| ROCk driver | 6.14.14 |

Build:

```bash
ROCM_DEVEL=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I${ROCM_DEVEL}/include/hsa -I${ROCM_DEVEL}/include \
  hsa_ipc_lifetime_repro.cpp \
  -L${ROCM_DEVEL}/lib -Wl,-rpath,${ROCM_DEVEL}/lib \
  -lhsa-runtime64 -pthread -o hsa_ipc_lifetime_repro
```

Run:

```bash
./run_hsa_ipc_lifetime_repro.sh
RUN_TAG=rep1 ./run_hsa_ipc_lifetime_repro.sh pre_attach_free
RUN_TAG=rep2 ./run_hsa_ipc_lifetime_repro.sh pre_attach_free
RUN_TAG=rep3 ./run_hsa_ipc_lifetime_repro.sh pre_attach_free
```

## Correction to the earlier UCX interpretation

The NIXL `stale_registration` harness publishes descriptors, then deregisters
and frees the target tensor before `initialize_xfer()` / `transfer()` on the
initiator. Therefore the relevant runtime boundary is first attach after
exporter free, not copy after an already successful attach.

The pure HSA result localizes the first non-terminating operation to
`hsa_amd_ipc_memory_attach()`. The OpenUCX negative completion-signal fix remains
valid for asynchronous failures reported after a copy is posted, but it cannot
affect a call blocked earlier in IPC attach.

## Artifacts

```text
results/hsa_ipc_lifetime_repro_20260807.tgz
SHA256 C3961CF40AD9FAF970C57DF29764329EFCD22734ED725F3BBAC3FD220259EB33
```

The archive contains the source, binary, runner, valid/post-attach-free logs,
and three independent pre-attach-free repetitions.
