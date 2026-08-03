"""Same-node HIP IPC data plane for the vLLM NIXL pull connector.

The vLLM scheduler, lease protocol, TP mapping, handshake side channel, and
completion notifications remain unchanged. Only NIXL's prepared GPU READ data
path is replaced with HIP IPC mappings and batched device-to-device copies.
"""

from __future__ import annotations

import ctypes
import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_WIRE_MAGIC = "W7900_HIP_IPC_V1"
_LOCAL_AGENT = "NIXL_INIT_AGENT"


class _CopyDesc(ctypes.Structure):
    _fields_ = [
        ("destination", ctypes.c_uint64),
        ("source", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
    ]


class _NativeHipIpc:
    def __init__(self, library_path: str, device: int):
        self.lib = ctypes.CDLL(library_path)
        self.lib.w7900_hip_ipc_handle_size.restype = ctypes.c_size_t
        self.lib.w7900_hip_ipc_last_error.restype = ctypes.c_char_p
        self.handle_size = int(self.lib.w7900_hip_ipc_handle_size())

        self.lib.w7900_hip_ipc_export.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.w7900_hip_ipc_context_create.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.w7900_hip_ipc_context_destroy.argtypes = [ctypes.c_void_p]
        self.lib.w7900_hip_ipc_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.w7900_hip_ipc_close.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.w7900_hip_ipc_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CopyDesc),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.w7900_hip_ipc_query.argtypes = [ctypes.c_void_p]
        self.lib.w7900_hip_ipc_wait.argtypes = [ctypes.c_void_p]
        self.lib.w7900_hip_ipc_elapsed_us.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.w7900_hip_ipc_release.argtypes = [ctypes.c_void_p]

        self.context = ctypes.c_void_p()
        self._check(
            self.lib.w7900_hip_ipc_context_create(device, ctypes.byref(self.context))
        )
        self.device = device

    def _check(self, status: int) -> None:
        if status < 0:
            message = self.lib.w7900_hip_ipc_last_error()
            raise RuntimeError(message.decode() if message else f"HIP error {status}")

    def export(self, pointer: int) -> dict[str, Any]:
        handle = (ctypes.c_ubyte * self.handle_size)()
        base = ctypes.c_uint64()
        size = ctypes.c_uint64()
        device = ctypes.c_int()
        self._check(
            self.lib.w7900_hip_ipc_export(
                pointer,
                handle,
                ctypes.byref(base),
                ctypes.byref(size),
                ctypes.byref(device),
            )
        )
        return {
            "handle": bytes(handle),
            "base": int(base.value),
            "size": int(size.value),
            "device": int(device.value),
        }

    def open(self, handle: bytes) -> int:
        if len(handle) != self.handle_size:
            raise ValueError(f"invalid HIP IPC handle size {len(handle)}")
        handle_buffer = (ctypes.c_ubyte * self.handle_size).from_buffer_copy(handle)
        mapped = ctypes.c_uint64()
        self._check(
            self.lib.w7900_hip_ipc_open(
                self.context, handle_buffer, ctypes.byref(mapped)
            )
        )
        return int(mapped.value)

    def close(self, mapped_base: int) -> None:
        self._check(self.lib.w7900_hip_ipc_close(self.context, mapped_base))

    def submit(self, copies: list[tuple[int, int, int]]) -> int:
        array_type = _CopyDesc * len(copies)
        array = array_type(*(_CopyDesc(*copy) for copy in copies))
        transfer = ctypes.c_void_p()
        self._check(
            self.lib.w7900_hip_ipc_submit(
                self.context, array, len(copies), ctypes.byref(transfer)
            )
        )
        if transfer.value is None:
            raise RuntimeError("HIP IPC submit returned a null transfer")
        return int(transfer.value)

    def query(self, transfer: int) -> bool:
        status = self.lib.w7900_hip_ipc_query(ctypes.c_void_p(transfer))
        self._check(status)
        return status == 0

    def release(self, transfer: int) -> None:
        self._check(self.lib.w7900_hip_ipc_release(ctypes.c_void_p(transfer)))

    def elapsed_us(self, transfer: int) -> float:
        elapsed = ctypes.c_float()
        self._check(
            self.lib.w7900_hip_ipc_elapsed_us(
                ctypes.c_void_p(transfer), ctypes.byref(elapsed)
            )
        )
        return float(elapsed.value)

    def shutdown(self) -> None:
        if self.context.value is not None:
            self._check(self.lib.w7900_hip_ipc_context_destroy(self.context))
            self.context = ctypes.c_void_p()


@dataclass
class _XferDescs:
    nixl_descs: Any
    raw: np.ndarray


@dataclass
class _Dlist:
    nixl_handle: Any
    raw: np.ndarray
    agent_name: str


@dataclass
class _Transfer:
    copies: list[tuple[int, int, int]]
    remote_agent: str
    notification: bytes
    total_bytes: int
    original_desc_count: int
    native_handle: int | None = None
    submitted_at: float = 0.0
    completed_at: float = 0.0
    gpu_duration_us: float = 0.0
    post_duration_us: float = 0.0
    notification_sent: bool = False


class W7900HipIpcTransport:
    """NIXL-compatible facade with a direct same-node HIP IPC data plane."""

    def __init__(self, nixl_agent: Any, device: int, library_path: str):
        self._nixl = nixl_agent
        self._native = _NativeHipIpc(library_path, device)
        self._device = device
        self._local_allocations: dict[tuple[int, int], dict[str, Any]] = {}
        self._remote_allocations: dict[str, list[dict[str, Any]]] = {}
        self._dlists: dict[int, _Dlist] = {}
        self._transfers: dict[int, _Transfer] = {}
        self._ids = itertools.count(1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._nixl, name)

    def get_reg_descs(self, descs: Any, mem_type: str | None = None) -> Any:
        for address, _length, _device, _metadata in descs:
            exported = self._native.export(int(address))
            key = (exported["base"], exported["size"])
            self._local_allocations.setdefault(key, exported)
        return self._nixl.get_reg_descs(descs, mem_type)

    def register_memory(
        self,
        reg_list: Any,
        mem_type: str | None = None,
        backends: list[str] | None = None,
    ) -> Any:
        return self._nixl.register_memory(
            reg_list, mem_type, backends=[] if backends is None else backends
        )

    def get_agent_metadata(self) -> bytes:
        payload = {
            "magic": _WIRE_MAGIC,
            "nixl": self._nixl.get_agent_metadata(),
            "allocations": list(self._local_allocations.values()),
            "pid": os.getpid(),
        }
        return msgspec.msgpack.encode(payload)

    def add_remote_agent(self, metadata: bytes) -> str:
        payload = msgspec.msgpack.decode(metadata)
        if not isinstance(payload, dict) or payload.get("magic") != _WIRE_MAGIC:
            raise RuntimeError("remote worker does not advertise W7900 HIP IPC")
        agent_name = self._nixl.add_remote_agent(payload["nixl"])
        allocations: list[dict[str, Any]] = []
        try:
            for remote in payload["allocations"]:
                allocation = dict(remote)
                allocation["mapped_base"] = self._native.open(allocation["handle"])
                allocations.append(allocation)
        except Exception:
            for allocation in reversed(allocations):
                self._native.close(allocation["mapped_base"])
            self._nixl.remove_remote_agent(agent_name)
            raise
        self._remote_allocations[agent_name] = allocations
        logger.info(
            "W7900 HIP IPC mapped %d allocations from agent %s (pid=%s)",
            len(allocations),
            agent_name,
            payload.get("pid"),
        )
        return agent_name

    def remove_remote_agent(self, agent_name: str) -> Any:
        for allocation in self._remote_allocations.pop(agent_name, []):
            self._native.close(allocation["mapped_base"])
        return self._nixl.remove_remote_agent(agent_name)

    def get_xfer_descs(self, descs: Any, mem_type: str | None = None) -> _XferDescs:
        raw = np.asarray(descs, dtype=np.uint64).reshape(-1, 3).copy()
        return _XferDescs(self._nixl.get_xfer_descs(descs, mem_type), raw)

    def prep_xfer_dlist(
        self,
        agent_name: str,
        xfer_list: _XferDescs,
        mem_type: str | None = None,
        backends: list[str] | None = None,
    ) -> int:
        if not isinstance(xfer_list, _XferDescs):
            raise TypeError("W7900 HIP IPC expected wrapped transfer descriptors")
        nixl_handle = self._nixl.prep_xfer_dlist(
            agent_name,
            xfer_list.nixl_descs,
            mem_type,
            backends=[] if backends is None else backends,
        )
        handle = next(self._ids)
        self._dlists[handle] = _Dlist(nixl_handle, xfer_list.raw, agent_name)
        return handle

    def release_dlist_handle(self, handle: int) -> None:
        dlist = self._dlists.pop(handle)
        self._nixl.release_dlist_handle(dlist.nixl_handle)

    def _translate_remote(self, agent_name: str, address: int, length: int) -> int:
        for allocation in self._remote_allocations[agent_name]:
            base = int(allocation["base"])
            size = int(allocation["size"])
            if base <= address and address + length <= base + size:
                return int(allocation["mapped_base"]) + address - base
        raise RuntimeError(
            f"remote address 0x{address:x}/0x{length:x} is outside exported allocations"
        )

    @staticmethod
    def _coalesce(copies: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        merged: list[tuple[int, int, int]] = []
        for destination, source, length in copies:
            if merged:
                prev_dst, prev_src, prev_len = merged[-1]
                if prev_dst + prev_len == destination and prev_src + prev_len == source:
                    merged[-1] = (prev_dst, prev_src, prev_len + length)
                    continue
            merged.append((destination, source, length))
        return merged

    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: int,
        local_indices: Any,
        remote_xfer_side: int,
        remote_indices: Any,
        notif_msg: bytes = b"",
        backends: list[str] | None = None,
        skip_desc_merge: bool = False,
    ) -> int:
        del backends, skip_desc_merge
        local = self._dlists[local_xfer_side]
        remote = self._dlists[remote_xfer_side]
        if local.agent_name != _LOCAL_AGENT or remote.agent_name == _LOCAL_AGENT:
            raise RuntimeError("invalid local/remote HIP IPC descriptor ownership")

        local_ids = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        remote_ids = np.asarray(remote_indices, dtype=np.int64).reshape(-1)
        if local_ids.size != remote_ids.size:
            raise RuntimeError("local and remote descriptor counts differ")

        copies: list[tuple[int, int, int]] = []
        for local_id, remote_id in zip(local_ids, remote_ids, strict=True):
            local_desc = local.raw[int(local_id)]
            remote_desc = remote.raw[int(remote_id)]
            local_address, local_length = int(local_desc[0]), int(local_desc[1])
            remote_address, remote_length = int(remote_desc[0]), int(remote_desc[1])
            if local_length != remote_length:
                raise RuntimeError(
                    f"descriptor length mismatch: {local_length} != {remote_length}"
                )
            mapped_remote = self._translate_remote(
                remote.agent_name, remote_address, remote_length
            )
            if operation == "READ":
                copies.append((local_address, mapped_remote, local_length))
            elif operation == "WRITE":
                copies.append((mapped_remote, local_address, local_length))
            else:
                raise ValueError(f"unsupported HIP IPC operation {operation}")

        transfer_id = next(self._ids)
        self._transfers[transfer_id] = _Transfer(
            copies=self._coalesce(copies),
            remote_agent=remote.agent_name,
            notification=notif_msg,
            total_bytes=sum(copy[2] for copy in copies),
            original_desc_count=len(copies),
        )
        return transfer_id

    def transfer(self, handle: int) -> None:
        transfer = self._transfers[handle]
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        transfer.native_handle = self._native.submit(transfer.copies)
        transfer.submitted_at = time.perf_counter()
        transfer.post_duration_us = (transfer.submitted_at - started) * 1.0e6

    def check_xfer_state(self, handle: int) -> str:
        transfer = self._transfers[handle]
        if transfer.native_handle is None:
            return "ERR"
        if not self._native.query(transfer.native_handle):
            return "PROC"
        if transfer.completed_at == 0.0:
            transfer.completed_at = time.perf_counter()
            transfer.gpu_duration_us = self._native.elapsed_us(transfer.native_handle)
            observed_us = (transfer.completed_at - transfer.submitted_at) * 1.0e6
            bandwidth_gib_s = (
                transfer.total_bytes / 2**30 / (transfer.gpu_duration_us / 1.0e6)
            )
            logger.info(
                "W7900 HIP IPC READ complete: bytes=%d, descriptors=%d->%d, "
                "gpu_us=%.1f, observed_us=%.1f, bandwidth=%.2f GiB/s",
                transfer.total_bytes,
                transfer.original_desc_count,
                len(transfer.copies),
                transfer.gpu_duration_us,
                observed_us,
                bandwidth_gib_s,
            )
        if transfer.notification and not transfer.notification_sent:
            self._nixl.send_notif(transfer.remote_agent, transfer.notification)
            transfer.notification_sent = True
        return "DONE"

    def get_xfer_telemetry(self, handle: int) -> Any:
        transfer = self._transfers[handle]
        end = transfer.completed_at or time.perf_counter()
        return SimpleNamespace(
            xferDuration=(
                transfer.gpu_duration_us
                if transfer.gpu_duration_us > 0.0
                else max(0.0, end - transfer.submitted_at) * 1.0e6
            ),
            postDuration=transfer.post_duration_us,
            totalBytes=transfer.total_bytes,
            descCount=len(transfer.copies),
        )

    def release_xfer_handle(self, handle: int) -> None:
        transfer = self._transfers.pop(handle)
        if transfer.native_handle is not None:
            self._native.release(transfer.native_handle)

    def shutdown(self) -> None:
        for handle in list(self._transfers):
            self.release_xfer_handle(handle)
        for handle in list(self._dlists):
            self.release_dlist_handle(handle)
        for agent_name in list(self._remote_allocations):
            self.remove_remote_agent(agent_name)
        self._native.shutdown()


class W7900HipIpcConnectorWorker(NixlPullConnectorWorker):
    def __init__(self, vllm_config: Any, engine_id: str, kv_cache_config: Any):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        default_library = Path(__file__).with_name("libw7900_hip_ipc.so")
        library_path = str(extra.get("hip_ipc_library", default_library))
        self.nixl_wrapper = W7900HipIpcTransport(
            self.nixl_wrapper, torch.cuda.current_device(), library_path
        )
        logger.info(
            "W7900HipIpcTransport enabled on device %d using %s",
            torch.cuda.current_device(),
            library_path,
        )

    def shutdown(self):
        wrapper = self.nixl_wrapper
        try:
            # Upstream releases transfers, dlists, remote agents, then memory.
            super().shutdown()
        finally:
            if isinstance(wrapper, W7900HipIpcTransport):
                wrapper.shutdown()


class W7900HipIpcConnector(NixlPullConnector):
    """vLLM connector retaining NIXL control while replacing GPU payload RMA."""

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any):
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = W7900HipIpcConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
