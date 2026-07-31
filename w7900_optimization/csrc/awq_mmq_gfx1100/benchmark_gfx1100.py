#!/usr/bin/env python3
"""Compare gfx1100 HIP MMQ variants with vLLM TritonW4A16."""
import json
import statistics
import time

import torch

import awq_mmq_gfx1100 as hip
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)

SHAPES = [
    (5120, 5120),
    (5120, 27648),
    (27648, 5120),
]
M_VALUES = (1, 2, 4, 8, 9, 16, 32, 64, 128)


def elapsed_us(fn, warmup: int, repeats: int, rounds: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    return statistics.median(samples)


def main() -> None:
    torch.manual_seed(1)
    print("device", torch.cuda.get_device_name(0), flush=True)
    rows = []
    for k_dim, n_dim in SHAPES:
        # K-major packed W4 and per-group scales are the common input layout
        # consumed by Triton and HIP v5-v9.
        b_q = torch.randint(
            -(2**31), 2**31 - 1, (k_dim, n_dim // 8),
            dtype=torch.int32, device="cuda"
        )
        scales = (torch.rand(k_dim // 32, n_dim, dtype=torch.float16, device="cuda") * 0.01 + 0.001)
        empty_zeros = torch.empty(0, dtype=torch.int32, device="cuda")
        for m_dim in M_VALUES:
            a = torch.randn(m_dim, k_dim, dtype=torch.float16, device="cuda") * 0.1
            ops = {
                "triton": lambda: triton_w4a16_gemm(a, b_q, scales, None, 32, 8),
                "hip_v5": lambda: hip.mmq_q4_gemm_kmajor_wmma_v5(a, b_q, scales, empty_zeros),
                "hip_v6": lambda: hip.mmq_q4_gemm_kmajor_wmma_v6(a, b_q, scales, empty_zeros),
                "hip_v7": lambda: hip.mmq_q4_gemm_kmajor_wmma_v7(a, b_q, scales, empty_zeros),
                "hip_v8": lambda: hip.mmq_q4_gemm_kmajor_wmma_v8(a, b_q, scales, empty_zeros),
                "hip_v9": lambda: hip.mmq_q4_gemm_kmajor_wmma_v9(a, b_q, scales, empty_zeros),
            }
            timings = {}
            repeats = 30 if m_dim <= 16 else 10
            for name, op in ops.items():
                timings[name] = elapsed_us(op, warmup=5, repeats=repeats)
            best = min(timings, key=timings.get)
            row = {"M": m_dim, "N": n_dim, "K": k_dim, "best": best, **timings}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del a
        del b_q, scales, empty_zeros
        torch.cuda.empty_cache()
    print("RESULT_JSON=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
