#!/usr/bin/env python3
"""Small RCCL all-reduce microbenchmark through PyTorch distributed.

This is intentionally dependency-light so it can run inside the provided ROCm
vLLM container with:

  HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/rccl_allreduce_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def parse_sizes(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", default="1,8,32,128,512")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--output-jsonl", default="")
    args = parser.parse_args()

    backend = "nccl"  # ROCm PyTorch maps this to RCCL.
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = dtype_from_name(args.dtype)
    elem_size = torch.empty((), dtype=dtype).element_size()

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "start",
                    "host": socket.gethostname(),
                    "backend": backend,
                    "world_size": world,
                    "dtype": str(dtype).replace("torch.", ""),
                    "sizes_mb": parse_sizes(args.sizes_mb),
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                    "nccl_debug": os.environ.get("NCCL_DEBUG"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rows: list[dict[str, float | int | str]] = []
    for size_mb in parse_sizes(args.sizes_mb):
        numel = max(1, size_mb * 1024 * 1024 // elem_size)
        tensor = torch.ones(numel, device=device, dtype=dtype)

        for _ in range(args.warmup):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        samples = []
        for _ in range(args.iters):
            dist.barrier()
            started = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            dist.barrier()
            samples.append(time.perf_counter() - started)

        avg_s = statistics.mean(samples)
        p50_s = statistics.median(samples)
        p90_s = sorted(samples)[max(0, int(len(samples) * 0.9) - 1)]
        payload_bytes = numel * elem_size
        # Ring all-reduce bus bandwidth convention.
        bus_bytes = payload_bytes * 2 * (world - 1) / world
        row = {
            "event": "all_reduce",
            "world_size": world,
            "dtype": str(dtype).replace("torch.", ""),
            "size_mb": size_mb,
            "payload_bytes": payload_bytes,
            "avg_ms": avg_s * 1000.0,
            "p50_ms": p50_s * 1000.0,
            "p90_ms": p90_s * 1000.0,
            "alg_bandwidth_gbps": payload_bytes / avg_s / 1e9,
            "bus_bandwidth_gbps": bus_bytes / avg_s / 1e9,
        }
        if rank == 0:
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del tensor
        torch.cuda.empty_cache()

    if rank == 0 and args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
