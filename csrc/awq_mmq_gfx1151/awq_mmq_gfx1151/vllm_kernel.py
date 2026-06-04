"""
vLLM MPLinearKernel adapter for the AWQ-INT4 MMQ HIP custom op (gfx1151).

This module subclasses vllm's MPLinearKernel and routes apply_weights through
torch.ops.awq_mmq_gfx1151.mmq_q4_gemm. Registration into the dispatcher
(_POSSIBLE_KERNELS[ROCM]) is done by the patch_strix.py Patch 16, NOT here.

Tensor contract verified against vllm v0.20.0 compressed_tensors_wNa16:
  weight_packed:     [N, K//8]  int32  (8 uint4b8 per int32, low nibble first)
  weight_scale:      [N, K//G]  fp16
  zero_points:       absent for symmetric uint4b8
  g_idx:             absent (no act reordering supported)

Our kernel uses these tensors AS-IS — no repack at process_weights_after_loading
(unlike TritonW4A16 which transposes both). This is a load-time win.
"""
import json
import logging
import os
from collections import Counter
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

# Tile constants from the HIP kernel — kept in sync manually with awq_mmq_gfx1151_kernel.hip.
MMQ_X = 64   # N tile, must divide MMQ_X for full-block efficiency
MMQ_Y = 64   # M tile (handled with bounds checks for tails)
GROUP_SIZE = 32
logger = logging.getLogger(__name__)

SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric: zero point is implicit (=8)
    scalar_types.uint4,    # asymmetric: explicit per-group zero points
]


class RocmMmqQ4LinearKernel(MPLinearKernel):
    """WMMA i32 16x16x16 iu8 kernel for AWQ-INT4 W4A16 g32 sym on gfx1151."""

    SUPPORTED_QUANT_TYPES = SUPPORTED_QUANT_TYPES

    _shape_stats: Counter[tuple[str, str, int, int, int, int]] = Counter()
    _shape_stats_total = 0

    @staticmethod
    def _shape_stats_enabled() -> bool:
        return os.environ.get("AWQ_MMQ_SHAPE_STATS", "0").strip().lower() in (
            "1", "true", "yes", "on")

    @staticmethod
    def _shape_stats_interval() -> int:
        raw = os.environ.get("AWQ_MMQ_SHAPE_STATS_INTERVAL", "1024").strip()
        try:
            return max(1, int(raw))
        except ValueError:
            return 1024

    @classmethod
    def _record_shape_stat(
        cls,
        route: str,
        backend: str,
        version: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        if not cls._shape_stats_enabled():
            return
        key = (route, backend, int(version), int(M), int(N), int(K))
        cls._shape_stats[key] += 1
        cls._shape_stats_total += 1
        interval = cls._shape_stats_interval()
        if cls._shape_stats_total <= 16 or cls._shape_stats_total % interval == 0:
            cls._emit_shape_stats()

    @classmethod
    def _shape_stats_payload(cls) -> dict:
        rows = [
            {
                "route": route,
                "backend": backend,
                "version": version,
                "M": M,
                "N": N,
                "K": K,
                "count": count,
            }
            for (route, backend, version, M, N, K), count in cls._shape_stats.items()
        ]
        rows.sort(key=lambda r: (-r["count"], r["route"], r["backend"], r["version"], r["M"], r["N"], r["K"]))
        return {
            "total_calls": cls._shape_stats_total,
            "top": rows[:32],
            "all": rows,
        }

    @classmethod
    def _emit_shape_stats(cls) -> None:
        payload = cls._shape_stats_payload()
        logger.info("AWQ_MMQ_SHAPE_STATS %s", json.dumps(payload["top"][:12], separators=(",", ":")))
        path = os.environ.get("AWQ_MMQ_SHAPE_STATS_PATH", "").strip()
        if path:
            try:
                out = Path(path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - best-effort diagnostics only
                logger.warning("failed to write AWQ_MMQ_SHAPE_STATS_PATH=%s: %s", path, exc)

    _weight_stats: list[dict] = []

    @staticmethod
    def _weight_stats_enabled() -> bool:
        return os.environ.get("AWQ_MMQ_WEIGHT_STATS", "0").strip().lower() in (
            "1", "true", "yes", "on")

    @classmethod
    def _record_weight_stat(
        cls,
        *,
        K: int,
        N: int,
        use_triton_decode: bool,
        need_kmajor_decode: bool,
        native_bytes: int,
        kmajor_bytes: int,
        has_zeros: bool,
    ) -> None:
        if not cls._weight_stats_enabled():
            return
        row = {
            "idx": len(cls._weight_stats),
            "K": int(K),
            "N": int(N),
            "use_triton_decode": bool(use_triton_decode),
            "need_kmajor_decode": bool(need_kmajor_decode),
            "has_zeros": bool(has_zeros),
            "native_bytes": int(native_bytes),
            "kmajor_bytes": int(kmajor_bytes),
        }
        cls._weight_stats.append(row)
        cls._emit_weight_stats()

    @classmethod
    def _emit_weight_stats(cls) -> None:
        if not cls._weight_stats_enabled():
            return
        by_shape: dict[str, dict] = {}
        for row in cls._weight_stats:
            key = f"K={row['K']},N={row['N']}"
            item = by_shape.setdefault(key, {
                "K": row["K"],
                "N": row["N"],
                "layers": 0,
                "native_bytes": 0,
                "kmajor_bytes": 0,
                "use_triton_decode_layers": 0,
                "kmajor_layers": 0,
            })
            item["layers"] += 1
            item["native_bytes"] += row["native_bytes"]
            item["kmajor_bytes"] += row["kmajor_bytes"]
            item["use_triton_decode_layers"] += int(row["use_triton_decode"])
            item["kmajor_layers"] += int(row["kmajor_bytes"] > 0)
        summary = {
            "layers": len(cls._weight_stats),
            "native_bytes": sum(r["native_bytes"] for r in cls._weight_stats),
            "kmajor_bytes": sum(r["kmajor_bytes"] for r in cls._weight_stats),
            "by_shape": sorted(by_shape.values(), key=lambda r: (-r["kmajor_bytes"], r["K"], r["N"])),
            "layers_detail": cls._weight_stats,
        }
        path = os.environ.get("AWQ_MMQ_WEIGHT_STATS_PATH", "").strip()
        if path:
            try:
                out = Path(path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - best-effort diagnostics only
                logger.warning("failed to write AWQ_MMQ_WEIGHT_STATS_PATH=%s: %s", path, exc)

    @staticmethod
    def _decode_backend() -> str:
        return os.environ.get("AWQ_MMQ_DECODE_BACKEND", "triton").strip().lower()

    @classmethod
    def _small_m_threshold(cls) -> int:
        raw = os.environ.get("AWQ_MMQ_SMALL_M_THRESHOLD", str(cls.SMALL_M_THRESHOLD))
        try:
            return max(0, int(raw))
        except ValueError:
            return cls.SMALL_M_THRESHOLD

    @classmethod
    def _use_triton_decode(cls) -> bool:
        return cls._decode_backend() != "hip" and cls._small_m_threshold() > 0

    @staticmethod
    def _decode_policy() -> str:
        return os.environ.get("AWQ_MMQ_DECODE_POLICY", "all").strip().lower()

    @classmethod
    def _hybrid_enabled(cls) -> bool:
        return cls._decode_backend() in ("hybrid", "auto")

    @staticmethod
    def _hybrid_long_prefill_threshold() -> int:
        raw = os.environ.get("AWQ_MMQ_HYBRID_LONG_PREFILL_THRESHOLD", "4096").strip()
        try:
            return max(0, int(raw))
        except ValueError:
            return 4096

    @staticmethod
    def _hybrid_verify_m_values() -> set[int]:
        raw = os.environ.get("AWQ_MMQ_HYBRID_VERIFY_M", "9").strip()
        values: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.add(int(part))
            except ValueError:
                continue
        return values or {9}

    @staticmethod
    def _hybrid_hip_version() -> int:
        raw = os.environ.get("AWQ_MMQ_HYBRID_HIP_VERSION", "7").strip()
        try:
            version = int(raw)
        except ValueError:
            return 7
        return version if version in (3, 4, 5, 6, 7, 8, 9) else 7

    _hybrid_long_context_active = False
    _hybrid_decode_seen_since_long = False

    @classmethod
    def _update_hybrid_context(cls, M: int) -> None:
        if not cls._hybrid_enabled():
            return
        if int(M) >= cls._hybrid_long_prefill_threshold():
            cls._hybrid_long_context_active = True
            cls._hybrid_decode_seen_since_long = False
        elif cls._hybrid_long_context_active and not cls._hybrid_decode_seen_since_long:
            # Keep the long-context marker through tail prefill chunks. For the
            # current max_num_seqs=1 setup, the next short/mid request will only
            # arrive after decode has been seen and will clear the marker here.
            return
        else:
            cls._hybrid_long_context_active = False
            cls._hybrid_decode_seen_since_long = False

    @classmethod
    def _hybrid_use_hip_for_decode(cls, M: int) -> bool:
        if not cls._hybrid_enabled():
            return False
        use_hip = cls._hybrid_long_context_active and int(M) in cls._hybrid_verify_m_values()
        if cls._hybrid_long_context_active:
            cls._hybrid_decode_seen_since_long = True
        return use_hip

    @staticmethod
    def _hip_decode_version() -> int:
        raw = os.environ.get("AWQ_MMQ_HIP_DECODE_VERSION", "1").strip()
        try:
            version = int(raw)
        except ValueError:
            return 1
        return version if version in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) else 1

    @staticmethod
    def _parse_shape_policy_values(raw: str) -> set[tuple[int, int]]:
        values: set[tuple[int, int]] = set()
        for item in raw.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if "x" not in item:
                continue
            k_raw, n_raw = item.split("x", 1)
            try:
                values.add((int(k_raw), int(n_raw)))
            except ValueError:
                continue
        return values

    def _use_triton_decode_for_layer(self) -> bool:
        if not self._use_triton_decode():
            return False
        policy = self._decode_policy()
        if policy in ("", "all", "triton", "fallback"):
            return True
        if policy in ("none", "hip"):
            return False

        k_dim = self.config.partition_weight_shape[0]
        n_dim = self.config.partition_weight_shape[1]
        for prefix, keep_matches in (("shape_keep_", True), ("shape_exclude_", False)):
            if policy.startswith(prefix):
                shapes = self._parse_shape_policy_values(policy[len(prefix):])
                if not shapes:
                    return True
                matched = (k_dim, n_dim) in shapes
                return matched if keep_matches else not matched

        for prefix, op in (("n_ge_", ">="), ("n_lt_", "<")):
            if policy.startswith(prefix):
                try:
                    limit = int(policy[len(prefix):])
                except ValueError:
                    return True
                return n_dim >= limit if op == ">=" else n_dim < limit

        return True

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # gfx1151 capability check happens via on_gfx1x() in can_implement

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Verbose debug: log every can_implement call with the full config.
        import logging
        _log = logging.getLogger(__name__)
        _log.debug(
            "RocmMmqQ4.can_implement called: full=%s partition=%s wt=%s act=%s g=%d zp=%s gidx=%s",
            c.full_weight_shape, c.partition_weight_shape, c.weight_type,
            c.act_type, c.group_size, c.zero_points, c.has_g_idx,
        )
        result = cls._can_implement_inner(c)
        _log.debug("RocmMmqQ4.can_implement -> %s", result)
        return result

    @classmethod
    def _can_implement_inner(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RocmMmqQ4 targets ROCm only"

        try:
            from vllm.platforms.rocm import on_gfx1x
        except ImportError:
            return False, "vllm.platforms.rocm.on_gfx1x not available"
        if not on_gfx1x():
            return False, "RocmMmqQ4 targets gfx1151 (gfx1x) only"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"weight_type {c.weight_type} not supported; "
                f"only uint4b8 (symmetric AWQ-INT4)",
            )

        # bf16 accepted via inline cast in apply_weights. Native bf16 in the
        # kernel is a v1.2 task (act_quant kernel currently fp16-only).
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, f"only fp16/bf16 activations supported (got {c.act_type})"

        if c.group_size != GROUP_SIZE:
            return (
                False,
                f"group_size={c.group_size} not supported (only {GROUP_SIZE})",
            )

        # Asymmetric quant (zero_points=True) supported via per-group zp tensor.
        if c.has_g_idx:
            return False, "activation reordering (g_idx) not supported"

        K = c.partition_weight_shape[0]
        N = c.partition_weight_shape[1]
        if K % GROUP_SIZE != 0:
            return False, f"K={K} not divisible by group_size={GROUP_SIZE}"
        if N < MMQ_X:
            return False, f"N={N} smaller than MMQ_X tile ({MMQ_X})"
        # M is variable per call; tail handling in the kernel covers any M >= 1.

        # Verify the .so is importable (not just present on disk).
        try:
            import awq_mmq_gfx1151  # noqa: F401
        except ImportError as e:
            return False, f"awq_mmq_gfx1151 module not importable: {e}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Dual-storage layout (decode/prefill dispatch):
        - Our kernel reads w_q AS-IS [N, K//8] int32, w_s [N, K//G] fp16,
          and w_zp PACKED [N//8, K//G] int32 (kernel does inline unpack).
        - TritonW4A16 fallback (small-M decode) needs transposed format:
          w_q_triton [K, N//8] int32, w_s_triton [K//G, N] fp16, w_zp_triton [K//G, N//8].
          Stored under `_awq_mmq_triton_*` attrs.
        Memory cost: ~1x extra weight (transposed copy of w_q). Original w_q
        retained for our kernel since the layouts differ.
        """
        from vllm.model_executor.layers.quantization.utils import replace_parameter

        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        if not w_q.is_contiguous():
            replace_parameter(layer, self.w_q_name,
                              torch.nn.Parameter(w_q.contiguous(), requires_grad=False))
        # Keep w_s in its native dtype (bf16 for Qwen 3.6) so the Triton fallback
        # path doesn't dtype-mismatch at compile time. Cast to fp16 inline in
        # apply_weights only when routing to our kernel.
        if not w_s.is_contiguous():
            replace_parameter(layer, self.w_s_name,
                              torch.nn.Parameter(w_s.contiguous(), requires_grad=False))
        # w_zp stays in PACKED [N//8, K//G] int32 format (no unpack — done in kernel).
        # Pre-compute fp16 cast of scales for our kernel path. Stored as tensor (not
        # nn.Parameter) so it doesn't pollute the layer's state_dict.
        layer._awq_mmq_w_s_fp16 = (
            getattr(layer, self.w_s_name).data.to(torch.float16).contiguous()
            if w_s.dtype != torch.float16 else getattr(layer, self.w_s_name).data
        )

        w_q_current = getattr(layer, self.w_q_name).data
        w_s_current = getattr(layer, self.w_s_name).data
        native_bytes = w_q_current.numel() * w_q_current.element_size() + w_s_current.numel() * w_s_current.element_size()
        if w_zp is not None:
            native_bytes += w_zp.numel() * w_zp.element_size()

        layer._awq_mmq_use_triton_decode = self._use_triton_decode_for_layer()
        need_kmajor_decode = self._hip_decode_version() in (3, 4, 5, 6, 7, 8, 9)
        if not layer._awq_mmq_use_triton_decode and not need_kmajor_decode:
            self._record_weight_stat(
                K=w_q_current.shape[1] * 8,
                N=w_q_current.shape[0],
                use_triton_decode=layer._awq_mmq_use_triton_decode,
                need_kmajor_decode=need_kmajor_decode,
                native_bytes=native_bytes,
                kmajor_bytes=0,
                has_zeros=w_zp is not None)
            return

        # ---- TritonW4A16 fallback format ----
        w_q_now = getattr(layer, self.w_q_name).data
        N_dim, K8 = w_q_now.shape
        K_dim = K8 * 8
        shifts = torch.arange(8, device=w_q_now.device, dtype=torch.int32) * 4
        w_unpacked = ((w_q_now.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
        w_KN = w_unpacked.t().contiguous()
        N8 = N_dim // 8
        w_repacked = torch.sum((w_KN.view(K_dim, N8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32)
        layer._awq_mmq_triton_w_q = w_repacked.contiguous()
        del w_unpacked, w_KN, w_repacked  # free intermediate buffers

        layer._awq_mmq_triton_w_s = getattr(layer, self.w_s_name).data.t().contiguous()
        layer._awq_mmq_triton_w_s_fp16 = (
            layer._awq_mmq_triton_w_s.to(torch.float16).contiguous()
            if layer._awq_mmq_triton_w_s.dtype != torch.float16
            else layer._awq_mmq_triton_w_s
        )
        layer._awq_mmq_triton_w_zp = w_zp.t().contiguous() if w_zp is not None else None
        kmajor_bytes = (
            layer._awq_mmq_triton_w_q.numel() * layer._awq_mmq_triton_w_q.element_size()
            + layer._awq_mmq_triton_w_s.numel() * layer._awq_mmq_triton_w_s.element_size()
        )
        if layer._awq_mmq_triton_w_zp is not None:
            kmajor_bytes += layer._awq_mmq_triton_w_zp.numel() * layer._awq_mmq_triton_w_zp.element_size()
        self._record_weight_stat(
            K=K_dim,
            N=N_dim,
            use_triton_decode=layer._awq_mmq_use_triton_decode,
            need_kmajor_decode=need_kmajor_decode,
            native_bytes=native_bytes,
            kmajor_bytes=kmajor_bytes,
            has_zeros=w_zp is not None)

    # Below this M threshold, route to TritonW4A16 fallback (decode-shape).
    # Tuned to match the ~17 t/s decode floor: DFlash with N=8 spec tokens
    # gives M=8 typical, plus warmup/probe at M=1.
    SMALL_M_THRESHOLD = 32

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        M = x_2d.size(0)
        small_m_threshold = self._small_m_threshold()
        small_m = M < small_m_threshold
        if not small_m:
            self._update_hybrid_context(M)
        hybrid_hip_decode = small_m and self._hybrid_use_hip_for_decode(M)

        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        if (getattr(layer, "_awq_mmq_use_triton_decode", True) and small_m
                and not hybrid_hip_decode):
            self._record_shape_stat("decode", "triton", -1, M, c.partition_weight_shape[1], x_2d.size(1))
            # Decode shape: route through TritonW4A16's fused dequant+matmul.
            from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
                triton_w4a16_gemm,
            )
            zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
            out = triton_w4a16_gemm(
                a=x_2d,
                b_q=layer._awq_mmq_triton_w_q,
                scales=layer._awq_mmq_triton_w_s,
                qzeros=layer._awq_mmq_triton_w_zp,
                group_size=c.group_size if c.group_size != -1 else c.partition_weight_shape[0],
                zp_bias=zp_bias,
            )
        else:
            # Prefill shape: route through our HIP MMQ Q4 kernel.
            w_q, _w_s_native, w_zp, _ = self._get_weight_params(layer)
            w_s_fp16 = layer._awq_mmq_w_s_fp16  # pre-cast at process_weights time
            orig_dtype = x_2d.dtype
            if x_2d.dtype != torch.float16:
                x_2d = x_2d.to(torch.float16)
            if w_zp is None:
                zp_in = torch.empty(0, dtype=torch.int32, device=x.device)
            else:
                # w_zp is PACKED [N//8, K//G] int32; kernel unpacks inline.
                zp_in = w_zp
            version = (self._hybrid_hip_version() if hybrid_hip_decode
                       else self._hip_decode_version() if small_m else 1)
            self._record_shape_stat(
                "decode" if small_m else "prefill",
                "hip", version, M, c.partition_weight_shape[1], x_2d.size(1))
            if version in (3, 4, 5, 6, 7, 8, 9):
                zeros_t = (
                    layer._awq_mmq_triton_w_zp
                    if layer._awq_mmq_triton_w_zp is not None
                    else torch.empty(0, dtype=torch.int32, device=x.device)
                )
                if version == 9:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v9(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
                elif version == 8:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v8(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
                elif version == 7:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v7(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
                elif version == 6:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v6(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
                elif version == 5:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma_v5(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
                elif version == 4:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor_wmma(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
                else:
                    out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm_kmajor(
                        x_2d, layer._awq_mmq_triton_w_q,
                        layer._awq_mmq_triton_w_s_fp16, zeros_t)
            else:
                out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm(x_2d, w_q, w_s_fp16, zp_in, version)
            if orig_dtype != torch.float16:
                out = out.to(orig_dtype)

        if bias is not None:
            out = out + bias

        return out.reshape(out_shape)
