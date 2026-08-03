#!/usr/bin/env python3
"""Cross-process HIP IPC gate with disjoint HIP visibility masks."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import torch

from w7900_hip_ipc_connector import _NativeHipIpc


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _wait_for(path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def export_buffer(args: argparse.Namespace) -> None:
    ready = Path(args.exchange + ".ready.json")
    done = Path(args.exchange + ".done")
    native = _NativeHipIpc(args.library, 0)
    source = torch.empty(args.bytes, dtype=torch.uint8, device="cuda:0")
    source.fill_(args.pattern)
    torch.cuda.synchronize()
    exported = native.export(source.data_ptr())
    _atomic_json(
        ready,
        {
            "pid": os.getpid(),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "tensor_pointer": source.data_ptr(),
            "bytes": args.bytes,
            "pattern": args.pattern,
            "handle": base64.b64encode(exported["handle"]).decode("ascii"),
            "allocation_base": exported["base"],
            "allocation_size": exported["size"],
        },
    )
    _wait_for(done, args.timeout)
    native.shutdown()


def import_buffer(args: argparse.Namespace) -> None:
    ready = Path(args.exchange + ".ready.json")
    done = Path(args.exchange + ".done")
    result_path = Path(args.exchange + ".result.json")
    _wait_for(ready, args.timeout)
    metadata = json.loads(ready.read_text(encoding="utf-8"))

    native = _NativeHipIpc(args.library, 0)
    mapped_base = native.open(base64.b64decode(metadata["handle"]))
    remote_pointer = (
        mapped_base
        + int(metadata["tensor_pointer"])
        - int(metadata["allocation_base"])
    )
    destination = torch.zeros(metadata["bytes"], dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()

    started = time.perf_counter()
    transfer = native.submit(
        [(destination.data_ptr(), remote_pointer, int(metadata["bytes"]))]
    )
    while not native.query(transfer):
        pass
    elapsed = time.perf_counter() - started
    gpu_elapsed_us = native.elapsed_us(transfer)
    native.release(transfer)

    expected = int(metadata["pattern"])
    valid = bool(torch.all(destination == expected).item())
    gib = int(metadata["bytes"]) / 2**30
    result = {
        "valid": valid,
        "source_visible_devices": metadata["visible_devices"],
        "destination_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "bytes": int(metadata["bytes"]),
        "elapsed_seconds": elapsed,
        "bandwidth_GiB_s": gib / elapsed,
        "gpu_elapsed_us": gpu_elapsed_us,
        "gpu_bandwidth_GiB_s": gib / (gpu_elapsed_us / 1.0e6),
        "source_allocation_size": int(metadata["allocation_size"]),
    }
    _atomic_json(result_path, result)
    done.touch()
    native.close(mapped_base)
    native.shutdown()
    if not valid:
        raise RuntimeError("HIP IPC copy validation failed")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("export", "import"))
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--bytes", type=int, default=64 * 2**20)
    parser.add_argument("--pattern", type=int, default=0x5A)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.mode == "export":
        export_buffer(args)
    else:
        import_buffer(args)


if __name__ == "__main__":
    main()
