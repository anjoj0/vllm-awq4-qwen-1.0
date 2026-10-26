"""Register the optional project kernel against the vLLM 0.32 dispatcher only."""
import ast
import importlib.metadata
from importlib.util import find_spec
from pathlib import Path

MARKER = "# AMD_PROJECT_MMQ_V032"
INJECTION = '''

# AMD_PROJECT_MMQ_V032
import os as _amd_mmq_os
import sys as _amd_mmq_sys
if _amd_mmq_os.environ.get("VLLM_ENABLE_PROJECT_MMQ", "0") == "1":
    _amd_mmq_sys.path.insert(0, "/opt/amd-kernels/awq_mmq_gfx1151")
    from awq_mmq_gfx1151.vllm_kernel import RocmMmqQ4LinearKernel as _AmdMmq
    if _AmdMmq not in _POSSIBLE_KERNELS[PlatformEnum.ROCM]:
        _POSSIBLE_KERNELS[PlatformEnum.ROCM].insert(0, _AmdMmq)
    logger.info("AMD project MMQ registered ahead of upstream ROCm kernels")
'''


def register_source(path):
    original = path.read_text(encoding="utf-8")
    if MARKER in original:
        return False
    for required in ("_POSSIBLE_KERNELS", "PlatformEnum.ROCM"):
        if required not in original:
            raise ValueError(f"Unsupported vLLM dispatcher: missing {required}")
    updated = original.rstrip() + "\n" + INJECTION
    ast.parse(updated)
    path.write_text(updated, encoding="utf-8")
    return True


if __name__ == "__main__":
    version = importlib.metadata.version("vllm")
    if version.split("+", 1)[0] != "0.32.0":
        raise RuntimeError(f"MMQ registration expects vLLM 0.32.0, found {version}")
    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vLLM package")
    root = Path(next(iter(spec.submodule_search_locations)))
    path = root / "model_executor/kernels/linear/__init__.py"
    print("MMQ registration changed:", register_source(path))
