#!/usr/bin/env python3

import argparse
import json
import time

import torch

from nixl import nixl_agent, nixl_agent_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("target", "initiator"), required=True)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--operation", choices=("READ", "WRITE"), default="READ")
    parser.add_argument(
        "--ucx-error-handling",
        choices=("peer", "none"),
        default="peer",
        help="UCX endpoint error handling mode used by the NIXL backend",
    )
    return parser.parse_args()


def wait_until(predicate, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("NIXL peer operation timed out")
        time.sleep(0.001)


def main() -> None:
    args = parse_args()
    listen_port = args.port if args.role == "target" else 0
    agent = nixl_agent(
        args.role,
        nixl_agent_config(True, True, listen_port, backends=[]),
    )
    agent.create_backend(
        "UCX", {"ucx_error_handling_mode": args.ucx_error_handling}
    )

    if args.operation == "READ":
        fill = 7 if args.role == "target" else 0
    else:
        fill = 0 if args.role == "target" else 7
    tensor = torch.full((args.bytes,), fill, dtype=torch.int8, device="cuda:0")
    registered = agent.register_memory(tensor, backends=["UCX"])
    descriptors = agent.get_xfer_descs(tensor)

    if args.role == "target":
        wait_until(lambda: agent.check_remote_metadata("initiator"))
        agent.send_notif("initiator", agent.get_serialized_descs(descriptors))
        wait_until(
            lambda: b"transfer_done"
            in b"".join(agent.get_new_notifs().get("initiator", [])),
            timeout_s=120.0,
        )
        torch.cuda.synchronize()
        expected = 7 if args.operation == "WRITE" else fill
        if not bool(torch.all(tensor[:4096] == expected).item()) or int(
            tensor[-1]
        ) != expected:
            raise RuntimeError("Target GPU buffer failed verification")
        print(
            json.dumps(
                {
                    "role": "target",
                    "status": "verified",
                    "operation": args.operation,
                }
            )
        )
    else:
        agent.fetch_remote_metadata("target", args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)

        notifications = {}
        wait_until(
            lambda: bool(
                notifications.update(agent.get_new_notifs()) or notifications
            )
        )
        remote_descriptors = agent.deserialize_descs(notifications["target"][0])
        wait_until(lambda: agent.check_remote_metadata("target"))

        elapsed = []
        total = args.warmup + args.iterations
        for index in range(total):
            handle = agent.initialize_xfer(
                args.operation, descriptors, remote_descriptors, "target"
            )
            start = time.perf_counter()
            state = agent.transfer(handle)
            if state == "ERR":
                raise RuntimeError("NIXL failed to post transfer")
            wait_until(lambda: agent.check_xfer_state(handle) in ("DONE", "ERR"))
            state = agent.check_xfer_state(handle)
            if state != "DONE":
                raise RuntimeError(f"NIXL transfer ended in state {state}")
            duration = time.perf_counter() - start
            agent.release_xfer_handle(handle)
            if index >= args.warmup:
                elapsed.append(duration)

        torch.cuda.synchronize()
        if not bool(torch.all(tensor[:4096] == 7).item()) or int(tensor[-1]) != 7:
            raise RuntimeError("Transferred GPU buffer failed verification")

        mean_s = sum(elapsed) / len(elapsed)
        result = {
            "role": "initiator",
            "status": "verified",
            "operation": args.operation,
            "bytes": args.bytes,
            "iterations": args.iterations,
            "ucx_error_handling": args.ucx_error_handling,
            "mean_s": mean_s,
            "min_s": min(elapsed),
            "max_s": max(elapsed),
            "mean_GBps": args.bytes / mean_s / 1e9,
        }
        print(json.dumps(result, sort_keys=True))
        agent.send_notif("target", b"transfer_done")

    agent.deregister_memory(registered, backends=["UCX"])


if __name__ == "__main__":
    main()
