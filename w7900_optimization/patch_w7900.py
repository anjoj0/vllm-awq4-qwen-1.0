"""Apply audited W7900/vLLM 0.23 compatibility fixes, then validate."""
from pathlib import Path
import sys

PR45207_MARKER = "Backport of merged vLLM PR #45207"
ATTN_TUNABLE_MARKER = "W7900 attention tunables"


def backport_pr45207(root: Path) -> None:
    """Backport merged commit 55da232d when this AMD branch predates it."""
    path = root / "vllm/v1/core/kv_cache_utils.py"
    text = path.read_text(encoding="utf-8")
    if PR45207_MARKER in text or "page_size_padded=max_page_size" in text:
        print("PR #45207 Mamba page-padding fix is already present")
        return
    old = '''        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
'''
    new = '''        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        elif isinstance(layer_spec, MambaSpec):
            # Backport of merged vLLM PR #45207 / commit 55da232d:
            # Mamba page size is determined by state shapes and does not scale
            # with block_size. Pad the physical page while preserving its
            # caching granularity.
            new_spec: KVCacheSpec = replace(
                layer_spec, page_size_padded=max_page_size
            )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
'''
    if old not in text:
        raise SystemExit("Cannot safely locate the pre-#45207 unification branch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Applied merged vLLM PR #45207 Mamba page-padding backport")


def _replace_once(text: str, old: str, new: str, description: str) -> str:
    if old not in text:
        raise SystemExit(f"Cannot safely locate {description}")
    return text.replace(old, new, 1)


def patch_attention_tunables(root: Path) -> None:
    """Expose W7900 Triton unified-attention sweep knobs via env vars."""
    unified = root / "vllm/v1/attention/ops/triton_unified_attention.py"
    backend = root / "vllm/v1/attention/backends/triton_attn.py"
    unified_text = unified.read_text(encoding="utf-8")
    backend_text = backend.read_text(encoding="utf-8")

    if ATTN_TUNABLE_MARKER not in unified_text:
        unified_text = _replace_once(
            unified_text,
            "from typing import Any\n\nimport torch\n",
            "from typing import Any\n\nimport os\n\nimport torch\n",
            "unified-attention import block",
        )
        unified_text = _replace_once(
            unified_text,
            "float8_info = torch.finfo(current_platform.fp8_dtype())\n\n\n",
            '''float8_info = torch.finfo(current_platform.fp8_dtype())


def _w7900_env_int(name: str, default: int) -> int:
    """W7900 attention tunables: parse positive power-of-two env overrides."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    value = int(raw)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")
    return value


''',
            "unified-attention W7900 env helper",
        )
        unified_text = _replace_once(
            unified_text,
            '''def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32
''',
            '''def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size, allowing W7900-specific Triton sweep overrides."""
    override_name = (
        "VLLM_TRITON_ATTN_PREFILL_TILE_SIZE"
        if is_prefill
        else "VLLM_TRITON_ATTN_DECODE_TILE_SIZE"
    )
    override = _w7900_env_int(override_name, 0)
    if override:
        return override

    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32
''',
            "unified-attention tile selector",
        )
        unified.write_text(unified_text, encoding="utf-8")
        print("Applied W7900 Triton unified-attention tile tunables")
    else:
        print("W7900 Triton unified-attention tile tunables already present")

    if ATTN_TUNABLE_MARKER not in backend_text:
        backend_text = _replace_once(
            backend_text,
            "from dataclasses import dataclass\nfrom typing import ClassVar\n\nimport torch\n",
            "from dataclasses import dataclass\nfrom typing import ClassVar\n\nimport os\n\nimport torch\n",
            "triton-attention import block",
        )
        backend_text = _replace_once(
            backend_text,
            '''# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
''',
            '''def _w7900_env_int(name: str, default: int) -> int:
    """W7900 attention tunables: parse positive integer env overrides."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


# constants
MIN_LAUNCH_GRID_SIZE_2D = _w7900_env_int(
    "VLLM_TRITON_ATTN_MIN_2D_GRID", 128
)  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = _w7900_env_int(
    "VLLM_TRITON_ATTN_SOFTMAX_SEGMENTS", 16
)  # Number of parallel tiled softmax segments
''',
            "triton-attention W7900 env constants",
        )
        backend.write_text(backend_text, encoding="utf-8")
        print("Applied W7900 Triton attention 2D/3D threshold tunables")
    else:
        print("W7900 Triton attention 2D/3D threshold tunables already present")


def validate(root: Path) -> None:
    required = [
        root / "vllm/platforms/rocm.py",
        root / "vllm/model_executor/models/qwen3_dflash.py",
        root / "vllm/config/speculative.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing vLLM files:\n" + "\n".join(missing))
    rocm = required[0].read_text(encoding="utf-8")
    dflash = required[1].read_text(encoding="utf-8")
    speculative = required[2].read_text(encoding="utf-8")
    if 'return "gfx1151"' in rocm or 'device_name = "gfx1151"' in rocm:
        raise SystemExit("Refusing a Strix gfx1151-forced vLLM source")
    if "get_tensor_model_parallel_world_size" not in dflash:
        raise SystemExit("Qwen3 DFlash source has no tensor-parallel implementation")
    if "draft_tensor_parallel_size" not in speculative:
        raise SystemExit("Current vLLM lacks draft_tensor_parallel_size")
    print(f"Validated local vLLM source: {root}")


if __name__ == "__main__":
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/vllm")
    backport_pr45207(source_root)
    patch_attention_tunables(source_root)
    validate(source_root)
