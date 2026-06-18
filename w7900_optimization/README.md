# W7900 Optimization Workspace

This directory contains the migration plan, experiments, and W7900-specific code
for running the AWQ4 + DFlash Qwen3.6 project on a single-node 8x Radeon PRO
W7900 workstation.

## Working Decision

Do not optimize only for "everything on one GPU" as the final target.

Use a staged strategy:

1. Single W7900 smoke path: prove that the current ROCm/vLLM stack, model load,
   AWQ4 kernels, DFlash verify path, fp8 KV cache, and correctness tests work on
   `gfx1100`.
2. One-GPU performance baseline: measure single-stream decode, long-context
   decode, prefill, DFlash acceptance, and KV memory pressure on one 48 GB card.
3. Multi-GPU production path: use the 8-card node for tensor parallel and/or
   data-parallel serving, depending on the target workload.

The competition/project goal is not merely fitting Qwen3.6-27B AWQ4 into one
card. One W7900 can plausibly host the quantized target plus the drafter for
moderate contexts, but the moment the workload needs 128K/256K context, higher
concurrency, or less fragile memory headroom, the 8-card system should be used.

## Recommended Deployment Policy

### Single GPU

Use one W7900 for:

- bring-up and correctness validation;
- quick A/B testing of AWQ4 without DFlash vs AWQ4 + DFlash;
- kernel compatibility checks for `gfx1100`;
- small batch and short/mid context demos.

Expected benefits:

- simplest failure surface;
- no RCCL/tensor-parallel communication overhead;
- easiest comparison with the current Strix Halo single-device results.

Expected limits:

- only 48 GB VRAM per card, so long context and DFlash dual-model KV pressure can
  become the limiter;
- no UMA spillover safety like Strix Halo, so memory failures are harder;
- long-context decode is still KV-attention bandwidth/parallelism bound.

### Multi GPU

Use multiple W7900 cards for:

- 128K/256K context;
- batch/concurrency;
- stable fp8 KV cache headroom;
- target + drafter coexistence with fewer memory tradeoffs;
- service throughput rather than single-request purity.

Initial candidates:

- `tensor_parallel_size=2`: likely first serious production target. It reduces
  per-card weights and KV pressure without excessive communication complexity.
- `tensor_parallel_size=4`: useful if 256K context or higher concurrency needs
  more memory headroom.
- `tensor_parallel_size=8`: test only after TP=2/4 are understood. It can help
  memory, but RDNA workstation PCIe/RCCL overhead may dominate small-batch decode.
- data parallel replicas: preferred when single-GPU or TP=2 already fits and the
  goal is aggregate QPS across independent requests.

## First Experiments

1. Hardware and ROCm validation:
   - confirm 8 visible GPUs;
   - confirm all report `gfx1100`;
   - run torch allocation and all-reduce sanity.
2. Single-GPU compatibility:
   - boot AWQ4 without DFlash;
   - boot AWQ4 + DFlash N=8;
   - run the existing `bench_full.py` and 5-dataset accuracy smoke.
3. Multi-GPU sweep:
   - TP=2, 4, 8 with DFlash enabled;
   - compare `VLLM_MAX_MODEL_LEN=65536`, `131072`, `262144`;
   - compare `VLLM_KV_CACHE_DTYPE=auto` vs `fp8`;
   - sweep `VLLM_MAX_NUM_BATCHED_TOKENS=8192/16384/32768`;
   - sweep `VLLM_DFLASH_N=4/6/8`.
4. Serving policy:
   - if one GPU fits target latency and context: run multiple single-GPU replicas;
   - if long context or memory headroom dominates: use TP=2/4;
   - avoid TP=8 as default unless profiling shows communication is not the
     bottleneck.

## Files

- `migration_plan.md`: technical reasoning and staged migration plan.
- `build_and_run.md`: clone/build/run instructions for the W7900 image.
- `Dockerfile.w7900`: W7900/gfx1100 source-build image.
- `docker-compose.w7900.build.yml`: W7900 build and runtime compose entrypoint.
- `docker-compose.w7900.tp2.yml`: legacy runtime-only TP=2 overlay for an already
  built image; prefer `docker-compose.w7900.build.yml` for new machines.
- `scripts/`: helper scripts for hardware validation and benchmark sweeps.
