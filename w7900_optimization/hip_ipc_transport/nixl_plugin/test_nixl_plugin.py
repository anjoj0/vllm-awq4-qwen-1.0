#!/usr/bin/env python3
"""Cross-process NIXL gate for the W7900 HIP IPC backend."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BACKEND = "W7900_HIP_IPC"


def wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def create_agent(name: str):
    from nixl import nixl_agent

    agent = nixl_agent(name)
    agent.create_backend(BACKEND)
    return agent


def register_tensor(agent, tensor):
    descriptor = agent.get_reg_descs(
        [(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, "")],
        "VRAM",
    )
    agent.register_memory(descriptor, backends=[BACKEND])
    return descriptor


def producer(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    agent = create_agent("w7900_hipipc_producer")
    source = torch.empty(args.bytes, dtype=torch.uint8, device="cuda:0")
    source.fill_(args.pattern)
    torch.cuda.synchronize()
    registration = register_tensor(agent, source)
    write_json(
        args.exchange.with_suffix(".producer.json"),
        {
            "metadata": base64.b64encode(agent.get_agent_metadata()).decode("ascii"),
            "pointer": source.data_ptr(),
            "bytes": args.bytes,
            "pattern": args.pattern,
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
    )

    deadline = time.monotonic() + args.timeout
    notification_seen = False
    while time.monotonic() < deadline and not notification_seen:
        notifications = agent.get_new_notifs(backends=[BACKEND])
        notification_seen = b"read-complete" in notifications.get(
            "w7900_hipipc_consumer", []
        )
        if not notification_seen:
            time.sleep(0.001)
    if not notification_seen:
        raise TimeoutError("producer did not receive the transfer notification")
    wait_for(args.exchange.with_suffix(".done"), args.timeout)
    agent.deregister_memory(registration)


def consumer(args: argparse.Namespace) -> None:
    import torch

    producer_path = args.exchange.with_suffix(".producer.json")
    wait_for(producer_path, args.timeout)
    producer_info = json.loads(producer_path.read_text(encoding="utf-8"))

    torch.cuda.set_device(0)
    agent = create_agent("w7900_hipipc_consumer")
    destination = torch.zeros(args.bytes, dtype=torch.uint8, device="cuda:0")
    registration = register_tensor(agent, destination)
    remote_agent = agent.add_remote_agent(
        base64.b64decode(producer_info["metadata"])
    )

    local = agent.get_xfer_descs(
        [(destination.data_ptr(), args.bytes, 0)], "VRAM"
    )
    remote = agent.get_xfer_descs(
        [(int(producer_info["pointer"]), args.bytes, 0)], "VRAM"
    )
    handle = agent.initialize_xfer(
        "READ",
        local,
        remote,
        remote_agent,
        b"read-complete",
        backends=[BACKEND],
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = agent.transfer(handle)
    while state == "PROC":
        state = agent.check_xfer_state(handle)
    elapsed = time.perf_counter() - started
    if state != "DONE":
        raise RuntimeError(f"NIXL transfer failed with state {state}")
    torch.cuda.synchronize()
    valid = bool(torch.all(destination == int(producer_info["pattern"])).item())
    result = {
        "valid": valid,
        "backend": BACKEND,
        "bytes": args.bytes,
        "elapsed_seconds": elapsed,
        "bandwidth_GiB_s": args.bytes / 2**30 / elapsed,
        "source_visible_devices": producer_info["visible_devices"],
        "destination_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    write_json(args.exchange.with_suffix(".result.json"), result)
    agent.release_xfer_handle(handle)
    agent.remove_remote_agent(remote_agent)
    agent.deregister_memory(registration)
    args.exchange.with_suffix(".done").touch()
    if not valid:
        raise RuntimeError("NIXL HIP IPC payload validation failed")
    print(json.dumps(result, indent=2))


def orchestrate(args: argparse.Namespace) -> None:
    for suffix in (".producer.json", ".result.json", ".done"):
        args.exchange.with_suffix(suffix).unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bytes",
        str(args.bytes),
        "--pattern",
        str(args.pattern),
        "--timeout",
        str(args.timeout),
        "--exchange",
        str(args.exchange),
    ]
    producer_env = os.environ.copy()
    producer_env["HIP_VISIBLE_DEVICES"] = args.source_devices
    consumer_env = os.environ.copy()
    consumer_env["HIP_VISIBLE_DEVICES"] = args.destination_devices
    producer_process = subprocess.Popen(command + ["producer"], env=producer_env)
    consumer_process = subprocess.Popen(command + ["consumer"], env=consumer_env)
    consumer_status = consumer_process.wait(timeout=args.timeout)
    producer_status = producer_process.wait(timeout=args.timeout)
    if producer_status or consumer_status:
        raise RuntimeError(
            f"producer={producer_status}, consumer={consumer_status}"
        )
    print(args.exchange.with_suffix(".result.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", nargs="?", choices=("producer", "consumer"))
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=64 * 2**20)
    parser.add_argument("--pattern", type=int, default=0x5A)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--source-devices", default="0,1,2,3")
    parser.add_argument("--destination-devices", default="4,5,6,7")
    args = parser.parse_args()
    if args.role == "producer":
        producer(args)
    elif args.role == "consumer":
        consumer(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
