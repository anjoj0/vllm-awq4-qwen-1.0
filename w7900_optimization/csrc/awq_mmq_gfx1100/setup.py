"""
Build script for the AWQ-INT4 MMQ HIP custom op targeting gfx1100 (Radeon PRO W7900).

Builds a standalone Python extension module `awq_mmq_gfx1100._C` for kernel
correctness and microbenchmark experiments. It is separate from the vLLM 0.23
`RDNA3W4A16LinearKernel` dispatcher override used by the full model service.

Usage (inside a ROCm container with a gfx1100 toolchain):
    cd /workspace/csrc/awq_mmq_gfx1100
    python setup.py build_ext --inplace
    python test_correctness.py

Building this package does not modify vLLM or register the production
dispatcher. See `../../vllm_overrides/rdna3_w4a16.py` for the E2E path.
"""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")
TARGET_ARCH = "gfx1100"

extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
        "-fPIC",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        f"--offload-arch={TARGET_ARCH}",
        "-Wno-unused-result",
        "-Wno-unused-variable",
    ],
}

setup(
    name="awq_mmq_gfx1100",
    version="0.1.0",
    description=(
        "AWQ-INT4 MMQ kernel for prefill on AMD Radeon PRO W7900 (gfx1100). "
        "Mirrors llama.cpp's MMQ Q4 structure with WMMA iu8 inner loop. "
        "Decode path is unaffected and still routes through vLLM's TritonW4A16."
    ),
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="awq_mmq_gfx1100._C",
            sources=[
                "bindings.cpp",
                "awq_mmq_gfx1100_kernel.hip",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
