#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); W7900_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
"$SCRIPT_DIR/prepare_local_vllm.sh"
ENV_FILE=${ENV_FILE:-"$W7900_DIR/.env"}; if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
ROCM_SDK_ROOT=${ROCM_SDK_ROOT:-$(dirname "$(dirname "$(readlink -f "$(command -v hipcc)")")")}
export VLLM_TARGET_DEVICE=rocm ROCM_HOME="$ROCM_SDK_ROOT" ROCM_PATH="$ROCM_SDK_ROOT" HIP_PATH="$ROCM_SDK_ROOT"
export PYTORCH_ROCM_ARCH=gfx1100 AMDGPU_TARGETS=gfx1100 GPU_TARGETS=gfx1100 HIP_ARCHITECTURES=gfx1100 MAX_JOBS=${MAX_JOBS:-8}
export CMAKE_ARGS="${CMAKE_ARGS:-} -DROCM_PATH=$ROCM_SDK_ROOT -DHIP_PATH=$ROCM_SDK_ROOT -DGPU_TARGETS=gfx1100 -DHIP_ARCHITECTURES=gfx1100 -DFETCHCONTENT_SOURCE_DIR_TRITON_KERNELS=/app/vllm/.deps/triton_kernels-src"
export VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE:-0.23.1.dev1}
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-$(python3 -c 'import vllm; print(vllm.__version__.split("+")[0])')}
rm -rf -- "$WORKTREE/.deps/triton_kernels-subbuild" "$WORKTREE/.deps/triton_kernels-build"
cd "$WORKTREE"; python3 -m pip install --no-build-isolation --no-deps --editable . -v
python3 - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count())
PY
