# W7900 Migration Plan

## Answer To The Main Question

For an 8x W7900 single node, the final optimization target should not be
"force everything onto one card at all costs".

The practical target is:

- single card for bring-up, correctness, and short/mid-context latency baseline;
- multiple cards for long context, memory headroom, and aggregate serving
  throughput.

There are two different optimization objectives:

1. Best single-request latency at short context.
   - Prefer one GPU if the model and KV cache fit.
   - Avoid tensor-parallel communication overhead.
   - On an 8-card node, run multiple independent replicas if QPS matters.
2. Best long-context / high-concurrency serving.
   - Prefer TP=2 or TP=4 first.
   - TP reduces per-card weight and KV pressure.
   - More memory headroom helps DFlash target + drafter coexistence and fp8 KV
     cache stability.

TP=8 should be treated as an experiment, not the default. It can reduce memory
pressure further, but small-batch decode can become communication-bound.

## Why This Differs From Strix Halo

The current project was tuned on Strix Halo (`gfx1151`) with a unified memory
architecture. W7900 is a discrete RDNA3 workstation GPU (`gfx1100`) with local
VRAM per card. That changes the optimization shape:

- Strix Halo: memory capacity is flexible, but bandwidth and long-context
  attention parallelism are the main wall.
- W7900 single GPU: local 48 GB VRAM is a hard per-card budget.
- W7900 8 GPU: memory capacity is large in aggregate, but tensor-parallel and
  RCCL communication can dominate decode if overused.

The current Strix Halo improvements remain valuable:

- AWQ4 target reduces model memory and weight bandwidth.
- DFlash increases accepted tokens per target verification.
- Patch 20/21 unified attention split-K path is the important long-context
  attention fix.
- fp8 KV cache reduces KV memory pressure and traffic.

But the low-level assumptions need to be retested:

- HIP custom kernels compiled/tuned for `gfx1151` must be retuned for `gfx1100`.
- `gpu_memory_utilization=0.60` was a Strix stability value; W7900 likely needs a
  new sweep.
- TP/DFlash interaction must be verified before treating multi-GPU as production.

## Recommended Roadmap

### Phase 0: System Validation

Goal: confirm the machine is a usable ROCm 8-GPU node.

Checks:

- Linux kernel and ROCm version;
- `rocminfo` reports 8 W7900 GPUs and `gfx1100`;
- `rocm-smi` sees all cards;
- PyTorch can allocate on all 8 devices;
- small RCCL/torch distributed all-reduce works;
- kernel boot args include the multi-GPU stability settings needed by ROCm,
  especially `iommu=pt` if hangs appear.

### Phase 1: Single-GPU Bring-Up

Goal: isolate architecture compatibility before multi-GPU complexity.

Run:

- AWQ4 no DFlash, `HIP_VISIBLE_DEVICES=0`;
- AWQ4 + DFlash N=4 and N=8;
- `VLLM_KV_CACHE_DTYPE=auto` and `fp8`;
- context lengths 32K, 64K, 128K if memory permits.

Keep:

- throughput;
- first-token latency;
- decode t/s;
- DFlash acceptance length;
- KV capacity log;
- correctness smoke output.

### Phase 2: Multi-GPU Shape Sweep

Goal: find the smallest tensor-parallel size that removes memory pressure
without making decode communication-bound.

Run in this order:

1. TP=2
2. TP=4
3. TP=8

For each TP size:

- `VLLM_MAX_MODEL_LEN=65536,131072,262144`
- `VLLM_KV_CACHE_DTYPE=auto,fp8`
- `VLLM_DFLASH_N=4,6,8`
- `VLLM_MAX_NUM_BATCHED_TOKENS=8192,16384,32768`
- `VLLM_GPU_MEMORY_UTIL=0.60,0.70,0.80,0.90`

Decision criteria:

- If TP=2 supports 128K/256K and keeps decode high, stop at TP=2.
- If TP=2 still OOMs or leaves poor KV headroom, test TP=4.
- Use TP=8 only if TP=4 cannot support the target context/concurrency.

### Phase 3: Serving Layout

If the workload is independent short/mid-context requests:

- run 8 single-GPU replicas;
- put a lightweight load balancer in front;
- this avoids tensor-parallel communication during decode.

If the workload is long-context single-session or low-concurrency but huge
context:

- use TP=2/4;
- keep batch small;
- prioritize fp8 KV cache and split-K attention path.

If the workload mixes both:

- reserve some GPUs for TP long-context service;
- reserve remaining GPUs for single-GPU replicas.

## Initial Hypothesis

The first serious candidate should be TP=2, not TP=8.

Reasoning:

- AWQ4 already shrinks weights enough that one GPU may boot for moderate context.
- TP=2 gives meaningful memory headroom for target + drafter + KV cache.
- TP=2 communication overhead is much lower than TP=8.
- If the bottleneck remains KV-attention scanning, the unified attention split-K
  path matters more than spreading every layer across all 8 GPUs.

For competition-style demos, the safest W7900 story is:

1. show one-card compatibility and accuracy equivalence;
2. show TP=2/4 enabling longer context or more concurrency;
3. show that TP=8 is a scaling experiment rather than a mandatory dependency.

## References To Recheck On The Target Machine

- AMD ROCm Linux system requirements and supported GPU list:
  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- AMD ROCm GPU architecture reference:
  https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
