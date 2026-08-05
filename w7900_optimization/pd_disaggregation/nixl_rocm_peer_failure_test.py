#!/usr/bin/env python3

import argparse
import json
import os
import time

import torch

from nixl import nixl_agent, nixl_agent_config


FAULTS = (
    "normal",
    "exit_before_transfer",
    "exit_after_post",
    "stale_registration",
    "clean_exit_before_transfer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("target", "initiator"), required=True)
    parser.add_argument("--fault", choices=FAULTS, required=True)
    parser.add_argument("--operation", choices=("READ", "WRITE"), default="READ")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--bytes", type=int, default=1024**3)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--ucx-error-handling", choices=("peer", "none"), default="peer"
    )
    return parser.parse_args()


def wait_until(predicate, timeout_s: float, message: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        time.sleep(0.001)


def wait_notification(agent, peer: str, marker: bytes, timeout_s: float) -> None:
    received = bytearray()

    def found() -> bool:
        for payload in agent.get_new_notifs().get(peer, []):
            received.extend(payload)
        return marker in received

    wait_until(found, timeout_s, f"timed out waiting for {marker!r}")


def make_agent(args: argparse.Namespace):
    listen_port = args.port if args.role == "target" else 0
    agent = nixl_agent(
        args.role,
        nixl_agent_config(True, True, listen_port, backends=[]),
    )
    agent.create_backend(
        "UCX", {"ucx_error_handling_mode": args.ucx_error_handling}
    )
    return agent


def run_target(args: argparse.Namespace) -> None:
    agent = make_agent(args)
    fill = 7 if args.operation == "READ" else 0
    tensor = torch.full((args.bytes,), fill, dtype=torch.int8, device="cuda:0")
    registered = agent.register_memory(tensor, backends=["UCX"])
    descriptors = agent.get_xfer_descs(tensor)

    wait_until(
        lambda: agent.check_remote_metadata("initiator"),
        args.timeout,
        "initiator metadata was not received",
    )
    agent.send_notif("initiator", agent.get_serialized_descs(descriptors))
    wait_notification(agent, "initiator", b"metadata_ready", args.timeout)

    if args.fault == "exit_before_transfer":
        os._exit(42)

    if args.fault in ("stale_registration", "clean_exit_before_transfer"):
        agent.deregister_memory(registered, backends=["UCX"])
        del tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        marker = (
            b"registration_invalidated"
            if args.fault == "stale_registration"
            else b"clean_exit"
        )
        agent.send_notif("initiator", marker)
        if args.fault == "clean_exit_before_transfer":
            return
        wait_notification(agent, "initiator", b"test_done", args.timeout + 5.0)
        return

    if args.fault == "exit_after_post":
        wait_notification(agent, "initiator", b"xfer_posted", args.timeout)
        os._exit(43)

    wait_notification(agent, "initiator", b"transfer_done", args.timeout)
    torch.cuda.synchronize()
    expected = 7
    if not bool(torch.all(tensor[:4096] == expected).item()) or int(
        tensor[-1]
    ) != expected:
        raise RuntimeError("target payload verification failed")
    agent.deregister_memory(registered, backends=["UCX"])
    print(json.dumps({"role": "target", "fault": args.fault, "status": "ok"}))


def get_remote_descriptors(agent, args: argparse.Namespace):
    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)
    notifications = {}
    wait_until(
        lambda: bool(notifications.update(agent.get_new_notifs()) or notifications),
        args.timeout,
        "target descriptor notification was not received",
    )
    remote_descriptors = agent.deserialize_descs(notifications["target"][0])
    wait_until(
        lambda: agent.check_remote_metadata("target"),
        args.timeout,
        "target metadata was not available",
    )
    agent.send_notif("target", b"metadata_ready")
    return remote_descriptors


def run_initiator(args: argparse.Namespace) -> None:
    agent = make_agent(args)
    fill = 0 if args.operation == "READ" else 7
    tensor = torch.full((args.bytes,), fill, dtype=torch.int8, device="cuda:0")
    registered = agent.register_memory(tensor, backends=["UCX"])
    descriptors = agent.get_xfer_descs(tensor)
    remote_descriptors = get_remote_descriptors(agent, args)

    if args.fault == "exit_before_transfer":
        time.sleep(0.5)
    elif args.fault == "stale_registration":
        wait_notification(
            agent, "target", b"registration_invalidated", args.timeout
        )
    elif args.fault == "clean_exit_before_transfer":
        wait_notification(agent, "target", b"clean_exit", args.timeout)
        time.sleep(0.5)

    handle = agent.initialize_xfer(
        args.operation, descriptors, remote_descriptors, "target"
    )
    start = time.monotonic()
    initial_state = "EXCEPTION"
    final_state = "EXCEPTION"
    error = None
    try:
        initial_state = agent.transfer(handle)
        if args.fault == "exit_after_post":
            agent.send_notif("target", b"xfer_posted")

        if initial_state == "ERR":
            final_state = "ERR"
        else:
            deadline = time.monotonic() + args.timeout
            while True:
                final_state = agent.check_xfer_state(handle)
                if final_state in ("DONE", "ERR"):
                    break
                if time.monotonic() >= deadline:
                    final_state = "TIMEOUT"
                    break
                time.sleep(0.001)
    except Exception as exc:  # The exception type is part of the observation.
        final_state = "EXCEPTION"
        error = f"{type(exc).__name__}: {exc}"

    elapsed_s = time.monotonic() - start
    if final_state in ("DONE", "ERR"):
        agent.release_xfer_handle(handle)
    payload_verified = None
    if final_state == "DONE" and args.operation == "READ":
        try:
            torch.cuda.synchronize()
            payload_verified = bool(torch.all(tensor[:4096] == 7).item()) and int(
                tensor[-1]
            ) == 7
            if not payload_verified:
                final_state = "PAYLOAD_MISMATCH"
        except Exception as exc:
            final_state = "PAYLOAD_EXCEPTION"
            error = f"{type(exc).__name__}: {exc}"

    result = {
        "role": "initiator",
        "fault": args.fault,
        "operation": args.operation,
        "bytes": args.bytes,
        "ucx_error_handling": args.ucx_error_handling,
        "initial_state": initial_state,
        "final_state": final_state,
        "elapsed_s": elapsed_s,
        "payload_verified": payload_verified,
        "error": error,
    }
    print(json.dumps(result, sort_keys=True), flush=True)

    if args.fault == "normal" and final_state == "DONE":
        agent.send_notif("target", b"transfer_done")
    elif args.fault == "stale_registration":
        agent.send_notif("target", b"test_done")

    if final_state == "TIMEOUT":
        os._exit(0)
    agent.deregister_memory(registered, backends=["UCX"])


def main() -> None:
    args = parse_args()
    if args.role == "target":
        run_target(args)
    else:
        run_initiator(args)


if __name__ == "__main__":
    main()
