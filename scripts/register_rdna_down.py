"""Register the opt-in down-projection adapter in pinned vLLM 0.32."""
import ast
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

MARKER = "# AMD_PROJECT_QWEN27B_DOWN_V1"
INJECTION = '''

# AMD_PROJECT_QWEN27B_DOWN_V1
import os as _amd_down_os
if _amd_down_os.environ.get("VLLM_QWEN27B_DOWN_KMAJOR", "0") == "1":
    if _amd_down_os.environ.get("VLLM_ENABLE_PROJECT_MMQ", "0") == "1":
        raise RuntimeError("Qwen27B down adapter and project MMQ are separate experimental routes")
    import sys as _amd_down_sys
    _amd_down_sys.path.insert(0, "/opt/amd-kernels")
    from rdna_qwen27b.linear import Qwen27BDownProjectionKernel as _QwenDown
    _POSSIBLE_KERNELS[PlatformEnum.ROCM].insert(0, _QwenDown)
'''

if __name__ == "__main__":
    if version("vllm").split("+", 1)[0] != "0.32.0":
        raise RuntimeError("Down projection adapter requires vLLM 0.32.0")
    root = Path(next(iter(find_spec("vllm").submodule_search_locations)))
    path = root / "model_executor/kernels/linear/__init__.py"
    original = path.read_text()
    if MARKER not in original:
        if "_POSSIBLE_KERNELS" not in original or "PlatformEnum.ROCM" not in original:
            raise RuntimeError("Unsupported vLLM dispatcher")
        updated = original.rstrip() + "\n" + INJECTION
        ast.parse(updated)
        path.write_text(updated)
    print("Qwen27B down-projection adapter installed")
