#!/usr/bin/env python3
"""Measure HIP v1 W4A8 output error relative to Triton W4A16."""
import json

import torch
import torch.nn.functional as F

import awq_mmq_gfx1100 as hip
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)

SHAPES = ((5120, 5120), (5120, 27648), (27648, 5120))


def native_to_kmajor(packed: torch.Tensor) -> torch.Tensor:
    n_dim, k8 = packed.shape
    k_dim = k8 * 8
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(n_dim, k_dim)
    kn = unpacked.t().contiguous()
    out = torch.sum(
        (kn.view(k_dim, n_dim // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()
    del unpacked, kn
    return out


def main() -> None:
    torch.manual_seed(3)
    rows = []
    print("device", torch.cuda.get_device_name(0), flush=True)
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
        for activation_scale in (0.1, 1.0):
            x = torch.randn(32, k_dim, dtype=torch.float16, device="cuda") * activation_scale
            ref = triton_w4a16_gemm(x, kmajor, kmajor_scales, None, 32, 8)
            test = hip.mmq_q4_gemm(
                x, native, native_scales, version=1, w_zeros=empty
            )
            ref32, test32 = ref.float(), test.float()
            delta = test32 - ref32
            rmse = delta.square().mean().sqrt()
            ref_rms = ref32.square().mean().sqrt()
            row = {
                "M": 32, "N": n_dim, "K": k_dim,
                "activation_scale": activation_scale,
                "max_abs": delta.abs().max().item(),
                "mean_abs": delta.abs().mean().item(),
                "relative_rmse": (rmse / ref_rms).item(),
                "cosine": F.cosine_similarity(
                    test32.flatten(), ref32.flatten(), dim=0
                ).item(),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del x, ref, test, ref32, test32, delta
        del native, native_scales, kmajor, kmajor_scales, empty
        torch.cuda.empty_cache()
    print("RESULT_JSON=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
