"""Install narrowly scoped, opt-in Qwen27B projection tiles on pinned vLLM.

The upstream arithmetic and weight layout stay intact. Large group32
asymmetric FP16 projections use a 256x256 tile on gfx1151 only.
"""
import ast
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

SOURCE_SHA256 = "a58df11e8527ad3e47342c1e25c393180fec713bc4e0adb2b0610904fcd05597"
MARKER = "# AMD_PROJECT_QWEN27B_PREFILL_V3"


def patch_source(path):
    original = path.read_text()
    if MARKER in original:
        return False
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("RDNA source differs from the qualified v0.32.0 ROCm file")
    updated = original.replace(
        "from contextlib import nullcontext\n",
        "from contextlib import nullcontext\nimport os as _amd_tile_os\n"
        + MARKER + "\n"
        '_AMD_QWEN27B_PREFILL = _amd_tile_os.environ.get("VLLM_QWEN27B_PREFILL_TILES", "0") == "1"\n'
        '_AMD_QWEN27B_ROCBLAS = _amd_tile_os.environ.get("VLLM_QWEN27B_PREFILL_ROCBLAS", "0") == "1"\n'
        'if _AMD_QWEN27B_ROCBLAS:\n'
        '    import sys as _amd_tile_sys\n'
        '    _amd_tile_sys.path.insert(0, "/opt/amd-kernels")\n'
        '    from rdna_qwen27b.prefill import dequant_matmul as _amd_dequant_matmul\n',
        1,
    )
    anchor = "        if override is not None:\n"
    if updated.count(anchor) != 1:
        raise RuntimeError("Expected one gfx1151 override branch")
    updated = updated.replace(anchor, '''        if (
            _AMD_QWEN27B_ROCBLAS
            and 1536 <= M <= 8192
            and group_size == 32
            and K == 5120 and N in (34816, 16384, 14336)
            and a.dtype == torch.float16 and has_zp
        ):
            return _amd_dequant_matmul(a, b_q, scales, group_size, zp, out=c)
        if (
            _AMD_QWEN27B_PREFILL
            and 1536 <= M <= 8192
            and group_size == 32
            and (K, N) in ((5120, 34816), (5120, 16384), (5120, 14336), (6144, 5120))
            and a.dtype == torch.float16
            and has_zp
        ):
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 256, 256, 32, 32
        elif override is not None:
''', 1)
    ast.parse(updated)
    path.write_text(updated)
    return True


if __name__ == "__main__":
    installed = version("vllm")
    if installed.split("+", 1)[0] != "0.32.0":
        raise RuntimeError(f"Expected vLLM 0.32.0, found {installed}")
    root = Path(next(iter(find_spec("vllm").submodule_search_locations)))
    path = root / "model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"
    print("Qwen27B prefill tile installed:", patch_source(path))
