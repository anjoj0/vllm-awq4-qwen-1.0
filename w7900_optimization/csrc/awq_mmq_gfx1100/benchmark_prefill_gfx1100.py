#!/usr/bin/env python3
"""Benchmark the old adapter's actual M>=32 HIP v1 route against Triton."""
import json
import os
import statistics

import torch

import awq_mmq_gfx1100 as hip
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)

SHAPES = ((5120, 5120), (5120, 27648), (27648, 5120))
M_VALUES = tuple(
    int(value) for value in os.getenv(
        "MMQ_M_VALUES", "32,64,128,256,512,1024,4096"
    ).split(",")
)


def native_to_kmajor(packed: torch.Tensor) -> torch.Tensor:
    n_dim, k8 = packed.shape
    k_dim = k8 * 8
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(n_dim, k_dim)
    kn = unpacked.t().contiguous()
    result = torch.sum(
        (kn.view(k_dim, n_dim // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()
    del unpacked, kn
    return result


def median_us(fn, repeats: int, rounds: int = 3) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / repeats)
    return statistics.median(samples)


def main() -> None:
    torch.manual_seed(2)
    print("device", torch.cuda.get_device_name(0), flush=True)
    rows = []
    for k_dim, n_dim in SHAPES:
        native = torch.randint(
            -(2**31), 2**31 - 1, (n_dim, k_dim // 8),
            dtype=torch.int32, device="cuda"
        )
        native_scales = torch.rand(
            n_dim, k_dim // 32, dtype=torch.float16, device="cuda"
        ) * 0.01 + 0.001
        kmajor = native_to_kmajor(native)
        kmajor_scales = native_scales.t().contiguous()
        empty = torch.empty(0, dtype=torch.int32, device="cuda")
        for m_dim in M_VALUES:
            a = torch.randn(m_dim, k_dim, dtype=torch.float16, device="cuda") * 0.1
            triton = lambda: triton_w4a16_gemm(a, kmajor, kmajor_scales, None, 32, 8)
            hip_v1 = lambda: hip.mmq_q4_gemm(
                a, native, native_scales, version=1, w_zeros=empty
            )
            repeats = 10 if m_dim <= 256 else (5 if m_dim <= 1024 else 2)
            t_triton = median_us(triton, repeats)
            t_hip = median_us(hip_v1, repeats)
            row = {
                "M": m_dim, "N": n_dim, "K": k_dim,
                "triton_us": t_triton, "hip_v1_us": t_hip,
                "hip_over_triton": t_hip / t_triton,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del a
        del native, native_scales, kmajor, kmajor_scales, empty
        torch.cuda.empty_cache()
    print("RESULT_JSON=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
