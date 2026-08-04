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


def wait_for_notification(agent, peer: str, expected: bytes, backend: str,
                          timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        notifications = agent.get_new_notifs(backends=[backend])
        if expected in notifications.get(peer, []):
            return
        time.sleep(0.001)
    raise TimeoutError(f"did not receive notification {expected!r} from {peer}")


def complete_transfer(agent, handle, notification: bytes) -> tuple[str, float, str | None]:
    started = time.perf_counter()
    try:
        state = agent.transfer(handle, notification)
        while state == "PROC":
            state = agent.check_xfer_state(handle)
            if state == "PROC":
                time.sleep(0.0001)
        return state, time.perf_counter() - started, None
    except Exception as error:  # NIXL maps negative backend states to typed exceptions.
        return "ERR", time.perf_counter() - started, repr(error)


def complete_posted_transfer(agent, handle, state: str) -> tuple[str, str | None]:
    try:
        while state == "PROC":
            state = agent.check_xfer_state(handle)
            if state == "PROC":
                time.sleep(0.0001)
        return state, None
    except Exception as error:
        return "ERR", repr(error)


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

    for repost in range(args.reposts):
        wait_for_notification(
            agent,
            "w7900_hipipc_consumer",
            f"read-complete-{repost}".encode(),
            BACKEND,
            args.timeout,
        )
        if repost + 1 < args.reposts:
            source.fill_((args.pattern + repost + 1) & 0xFF)
            torch.cuda.synchronize()
            args.exchange.with_suffix(f".ready-{repost + 1}").touch()

    if args.exercise_recovery:
        source.fill_((args.pattern + args.reposts) & 0xFF)
        torch.cuda.synchronize()
        args.exchange.with_suffix(".ready-recovery").touch()
        wait_for_notification(
            agent,
            "w7900_hipipc_consumer",
            b"recovery-complete",
            BACKEND,
            args.timeout,
        )
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
        b"",
        backends=[BACKEND],
    )
    torch.cuda.synchronize()
    repost_results = []
    valid = True
    active_repost_result = None
    for repost in range(args.reposts):
        if repost:
            wait_for(args.exchange.with_suffix(f".ready-{repost}"), args.timeout)
        if repost == 0 and args.exercise_active_repost:
            started = time.perf_counter()
            state = agent.transfer(handle, b"read-complete-0")
            try:
                rejected_state = agent.transfer(handle, b"must-not-be-sent")
                rejected_error = None
            except Exception as active_error:
                rejected_state = "ERR"
                rejected_error = repr(active_error)
            if rejected_state != "ERR":
                raise RuntimeError(
                    "active repost was not rejected: "
                    f"state={rejected_state}, error={rejected_error}"
                )
            state, error = complete_posted_transfer(agent, handle, state)
            elapsed = time.perf_counter() - started
            active_repost_result = {
                "rejected_state": rejected_state,
                "rejected_error": rejected_error,
                "original_state": state,
            }
        else:
            state, elapsed, error = complete_transfer(
                agent, handle, f"read-complete-{repost}".encode()
            )
        if state != "DONE":
            raise RuntimeError(
                f"NIXL repost {repost} failed with state {state}: {error}"
            )
        torch.cuda.synchronize()
        expected = (int(producer_info["pattern"]) + repost) & 0xFF
        round_valid = bool(torch.all(destination == expected).item())
        valid = valid and round_valid
        repost_results.append(
            {
                "index": repost,
                "pattern": expected,
                "valid": round_valid,
                "elapsed_seconds": elapsed,
                "bandwidth_GiB_s": args.bytes / 2**30 / elapsed,
            }
        )

    recovery_result = None
    if args.exercise_recovery:
        wait_for(args.exchange.with_suffix(".ready-recovery"), args.timeout)
        failed_state, failed_elapsed, injected_error = complete_transfer(
            agent, handle, b"x" * 65536
        )
        if failed_state != "ERR":
            raise RuntimeError(
                "oversized notification did not fail as expected: "
                f"state={failed_state}"
            )
        torch.cuda.synchronize()
        recovery_pattern = (int(producer_info["pattern"]) + args.reposts) & 0xFF
        failed_copy_valid = bool(torch.all(destination == recovery_pattern).item())
        recovered_state, recovered_elapsed, recovered_error = complete_transfer(
            agent, handle, b"recovery-complete"
        )
        if recovered_state != "DONE":
            raise RuntimeError(
                "repost after notification failure returned "
                f"{recovered_state}: {recovered_error}"
            )
        torch.cuda.synchronize()
        recovered_copy_valid = bool(torch.all(destination == recovery_pattern).item())
        valid = valid and failed_copy_valid and recovered_copy_valid
        recovery_result = {
            "injected_state": failed_state,
            "injected_error": injected_error,
            "injected_copy_valid": failed_copy_valid,
            "injected_elapsed_seconds": failed_elapsed,
            "recovered_state": recovered_state,
            "recovered_copy_valid": recovered_copy_valid,
            "recovered_elapsed_seconds": recovered_elapsed,
        }

    elapsed = sum(item["elapsed_seconds"] for item in repost_results)
    result = {
        "valid": valid,
        "backend": BACKEND,
        "bytes": args.bytes,
        "reposts": args.reposts,
        "elapsed_seconds": elapsed,
        "mean_bandwidth_GiB_s": args.bytes * args.reposts / 2**30 / elapsed,
        "repost_results": repost_results,
        "active_repost": active_repost_result,
        "recovery": recovery_result,
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
    for path in args.exchange.parent.glob(args.exchange.name + ".*"):
        path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bytes",
        str(args.bytes),
        "--pattern",
        str(args.pattern),
        "--timeout",
        str(args.timeout),
        "--reposts",
        str(args.reposts),
        "--exchange",
        str(args.exchange),
    ]
    if args.exercise_recovery:
        command.append("--exercise-recovery")
    if args.exercise_active_repost:
        command.append("--exercise-active-repost")
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
    parser.add_argument("--reposts", type=int, default=1)
    parser.add_argument("--exercise-recovery", action="store_true")
    parser.add_argument("--exercise-active-repost", action="store_true")
    parser.add_argument("--source-devices", default="0,1,2,3")
    parser.add_argument("--destination-devices", default="4,5,6,7")
    args = parser.parse_args()
    if args.reposts < 1:
        parser.error("--reposts must be at least 1")
    if args.role == "producer":
        producer(args)
    elif args.role == "consumer":
        consumer(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
