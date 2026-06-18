"""
W7900 compatibility patch pass for the project vLLM source build.

The main project patch bundle is Strix Halo first and includes a vLLM ROCm
platform override that forces `gfx1151`. That is useful on the original APU
platform but wrong on Radeon PRO W7900 (`gfx1100`). This script runs after
`scripts/patch_strix.py` during the W7900 image build and rewrites the forced
architecture to `gfx1100`.

It intentionally leaves the DFlash, non-causal attention, unified attention, fp8
KV, and other generic ROCm fixes in place.
"""

import re
from pathlib import Path


def patch_rocm_platform() -> None:
    path = Path("vllm/platforms/rocm.py")
    if not path.exists():
        raise SystemExit(f"missing {path}; run from /opt/vllm after cloning vLLM")

    text = path.read_text()
    text = text.replace('return "gfx1151"', 'return "gfx1100"')
    text = re.sub(r'device_name = "gfx1151"', 'device_name = "gfx1100"', text)
    path.write_text(text)
    print(" -> W7900: patched vllm/platforms/rocm.py forced arch to gfx1100")


def remove_strix_custom_mmq_registration() -> None:
    """Remove Patch 16, which registers a gfx1151-tuned HIP MMQ kernel.

    W7900 bring-up should start from the upstream/Triton AWQ path. A dedicated
    gfx1100 HIP MMQ kernel can be added later after the W7900 baseline is known.
    """

    path = Path("vllm/model_executor/kernels/linear/__init__.py")
    if not path.exists():
        return

    text = path.read_text()
    start = "# --- Patch 16: AWQ-INT4 MMQ HIP custom op for gfx1151 (Strix Halo) ---"
    end = "# --- end Patch 16 ---"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n" + after.lstrip("\n")
        path.write_text(text)
        print(" -> W7900: removed gfx1151 HIP MMQ Patch 16 registration")


def main() -> None:
    patch_rocm_platform()
    remove_strix_custom_mmq_registration()


if __name__ == "__main__":
    main()
