#!/usr/bin/env python3
"""Launch the pinned upstream ROCm image with an explicit speculative backend."""
import importlib.metadata
import json
import os
import sys

DEFAULT_DRAFTS = {
    "dflash": "z-lab/Qwen3.6-27B-DFlash",
    "dspark": "abstract-extraordinary/Qwen3.6-27B-DSpark",
}


def enabled(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_command(env):
    method = env.get("VLLM_SPEC_METHOD", "dflash").strip().lower()
    if enabled(env.get("VLLM_DISABLE_DFLASH", "0")):
        method = "none"
    if method not in {"none", *DEFAULT_DRAFTS}:
        raise ValueError("VLLM_SPEC_METHOD must be none, dflash, or dspark")
    command = [
        "vllm", "serve",
        env.get("VLLM_MODEL_ID", "cyankiwi/Qwen3.6-27B-AWQ-INT4"),
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", env.get("VLLM_SERVED_MODEL_NAME", "Qwen3.6-27B-AWQ4"),
        "--attention-backend", "TRITON_ATTN",
        "--mm-encoder-attn-backend", "TRITON_ATTN",
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_coder",
        "--enable-auto-tool-choice",
        # Qwen's hybrid attention needs chunked prefill for aligned state caching.
        "--enable-chunked-prefill",
        # Report cache hits and timings per request for prefill comparisons.
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--gpu-memory-utilization", env.get("VLLM_GPU_MEMORY_UTIL", "0.55"),
        "--max-num-seqs", env.get("VLLM_MAX_NUM_SEQS", "1"),
        "--max-num-batched-tokens", env.get("VLLM_MAX_NUM_BATCHED_TOKENS", "8192"),
        "--max-model-len", env.get("VLLM_MAX_MODEL_LEN", "32768"),
        "--kv-cache-dtype", env.get("VLLM_KV_CACHE_DTYPE", "auto"),
        "--skip-mm-profiling",
    ]
    block_size = env.get("VLLM_BLOCK_SIZE", "").strip()
    if block_size:
        if int(block_size) <= 0:
            raise ValueError("VLLM_BLOCK_SIZE must be positive")
        # Hybrid attention may raise this minimum to fit recurrent-state pages.
        command += ["--block-size", str(int(block_size))]
    dtype = env.get("VLLM_DTYPE", "").strip()
    if dtype:
        if dtype not in {"auto", "half", "float16", "bfloat16"}:
            raise ValueError("VLLM_DTYPE must be auto, half, float16, or bfloat16")
        command += ["--dtype", dtype]
    profiler_dir = env.get("VLLM_PROFILER_DIR", "").strip()
    if profiler_dir:
        if not os.path.isabs(profiler_dir):
            raise ValueError("VLLM_PROFILER_DIR must be an absolute container path")
        command += ["--profiler-config", json.dumps({
            "profiler": "torch", "torch_profiler_dir": profiler_dir,
            "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
            "ignore_frontend": True,
        })]
    command.append("--enable-prefix-caching"
                   if enabled(env.get("VLLM_ENABLE_PREFIX_CACHING", "1"))
                   else "--no-enable-prefix-caching")
    if enabled(env.get("VLLM_ENFORCE_EAGER", "1")):
        command.append("--enforce-eager")
    decode_graphs = enabled(env.get("VLLM_DECODE_GRAPHS", "0"))
    if decode_graphs and ("--enforce-eager" in command
                          or int(env.get("VLLM_MAX_NUM_SEQS", "1")) != 1):
        raise ValueError("VLLM_DECODE_GRAPHS requires eager=0 and max_num_seqs=1")
    width = 0
    if method != "none":
        draft = env.get("VLLM_DRAFT_MODEL_ID", "").strip() or DEFAULT_DRAFTS[method]
        width = int(env.get("VLLM_SPEC_TOKENS", "").strip()
                    or env.get("VLLM_DFLASH_N", "8"))
        if width < 1:
            raise ValueError("VLLM_SPEC_TOKENS must be positive")
        spec = {
            "method": method, "model": draft, "num_speculative_tokens": width,
            "draft_tensor_parallel_size": 1,
        }
        topk = env.get("VLLM_DSPARK_DRAFT_TOPK", "").strip()
        if topk:
            if method != "dspark" or int(topk) < 1:
                raise ValueError("VLLM_DSPARK_DRAFT_TOPK requires DSpark and a positive integer")
            spec["dspark_draft_topk"] = int(topk)
        command += ["--speculative-config", json.dumps(spec)]
    if decode_graphs:
        # Bound capture to one request's verification rows. Prefill stays eager;
        # torch.compile is separate and intentionally disabled in this profile.
        command += ["--compilation-config", json.dumps({
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": list(range(1, width + 2)),
            "max_cudagraph_capture_size": width + 1,
        })]
    return command


def check_environment(env):
    version = importlib.metadata.version("vllm")
    expected = env.get("VLLM_EXPECTED_VERSION", "0.32.0")
    if version.split("+", 1)[0] != expected:
        raise RuntimeError(f"Expected vLLM {expected}, found {version}")
    import torch
    if not torch.version.hip:
        raise RuntimeError("This profile requires a ROCm PyTorch image")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable; check /dev/kfd and /dev/dri access")
    device = torch.cuda.get_device_properties(0)
    arch = getattr(device, "gcnArchName", "").split(":", 1)[0]
    if arch != "gfx1151":
        raise RuntimeError(f"This Strix Halo profile expects gfx1151, found {arch!r}")
    info = {
        "vllm": version, "torch": torch.__version__, "hip": torch.version.hip,
        "device": device.name, "arch": arch,
        "device_memory_GiB": round(device.total_memory / 2**30, 2),
        "project_mmq": enabled(env.get("VLLM_ENABLE_PROJECT_MMQ", "0")),
    }
    if info["project_mmq"]:
        sys.path.insert(0, "/opt/amd-kernels/awq_mmq_gfx1151")
        # Import the dispatcher first: its registration block loads the adapter.
        # Reversing this order creates a partial-module circular import.
        from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
        import awq_mmq_gfx1151.vllm_kernel as adapter
        from vllm.platforms import PlatformEnum
        if _POSSIBLE_KERNELS[PlatformEnum.ROCM][0] is not adapter.RocmMmqQ4LinearKernel:
            raise RuntimeError("Project MMQ requested but not first in the ROCm dispatcher")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


def main():
    command = build_command(os.environ)
    if sys.argv[1:] == ["--print-command"]:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    if sys.argv[1:] not in ([], ["--check-env"]):
        raise ValueError("Supported options: --print-command, --check-env")
    check_environment(os.environ)
    if sys.argv[1:] == ["--check-env"]:
        return
    print("Starting: " + json.dumps(command, ensure_ascii=False), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
