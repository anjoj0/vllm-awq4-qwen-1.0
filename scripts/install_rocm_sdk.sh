#!/bin/bash
# Install TheRock ROCm SDK for gfx1151 via tarball. OS-agnostic.
# Adapted from kyuz0/amd-strix-halo-vllm-toolboxes/scripts/install_rocm_sdk.sh
set -euo pipefail

ROCM_MAJOR_VER="${ROCM_MAJOR_VER:-7}"
GFX="${GFX:-gfx1151}"

# Pinned tarball date. AMD's rocm.nightlies index rolls forward daily and
# upstream regressions (e.g. PyTorch PR #180485 in May 2026) periodically
# break this build. We pin to a known-working snapshot so docker compose
# up reliably reproduces the verified stack. To refresh:
#
#   1. Set ALLOW_LATEST=1 in the env to discover the newest tarball.
#   2. Run a clean rebuild: docker compose build --pull --no-cache.
#   3. Run test/sanity.py and test/bench_full.py.
#   4. If green, update this constant to the new date and commit.
#
# See .research/pytorch-180485-hip-found-regression/FINDINGS.md for the
# kind of breakage this pin shields against.
PINNED_TARBALL="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}.13.0a20260510.tar.gz"
ALLOW_LATEST="${ALLOW_LATEST:-0}"

echo "=== Installing TheRock ROCm SDK ($GFX, major version $ROCM_MAJOR_VER) ==="

cd /tmp

BASE="https://rocm.nightlies.amd.com/tarball"
PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"

if [ "$ALLOW_LATEST" = "1" ]; then
  echo "ALLOW_LATEST=1: resolving newest tarball from ${BASE} (NOT pinned, may break the build)"
  KEY="$(curl -s "${BASE}/" \
    | grep -oE "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\.[^\"<]*\.tar\.gz" \
    | sort -V | uniq | tail -n1)"
  if [ -z "$KEY" ]; then
    echo "ERROR: no tarball matching ${PREFIX} found at ${BASE}" >&2
    exit 1
  fi
else
  KEY="${PINNED_TARBALL}"
  echo "Using pinned tarball ${KEY}. Set ALLOW_LATEST=1 to override."
fi

echo "Downloading tarball: ${KEY}"
aria2c -x 4 -s 4 -j 4 \
  --file-allocation=none \
  --connect-timeout=30 \
  --timeout=60 \
  --lowest-speed-limit=16K \
  --max-tries=0 \
  --retry-wait=5 \
  "${BASE}/${KEY}" -o therock.tar.gz

mkdir -p /opt/rocm
tar xzf therock.tar.gz -C /opt/rocm --strip-components=1
rm therock.tar.gz

BITCODE_PATH=$(find /opt/rocm -type d -name bitcode -print -quit)

# Drop a profile.d fragment so interactive shells in the container pick up
# the ROCm env automatically. The Dockerfile ALSO sets the same variables
# via ENV so non-interactive RUN layers see them during build.
cat > /etc/profile.d/rocm-sdk.sh <<EOF
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HIP_PATH=/opt/rocm
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export HIP_DEVICE_LIB_PATH=${BITCODE_PATH}
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib:\$LD_LIBRARY_PATH
export ROCBLAS_USE_HIPBLASLT=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export VLLM_TARGET_DEVICE=rocm
export HIP_FORCE_DEV_KERNARG=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
# AWQ-on-gfx1151 hardening (see .research/ for source citations):
# - HSA_NO_SCRATCH_RECLAIM avoids vllm#37151 segfault on AWQ load.
# - MIOPEN_FIND_MODE=FAST avoids ViT conv-stem hangs (vllm#37472).
# - FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE enables the only viable
#   FA path on RDNA 3.5 (Triton via ROCm/flash-attention main_perf).
export HSA_NO_SCRATCH_RECLAIM=1
export MIOPEN_FIND_MODE=FAST
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export AMDGPU_TARGETS=${GFX}
EOF
chmod 0644 /etc/profile.d/rocm-sdk.sh

echo "Bitcode path: ${BITCODE_PATH}"
echo "=== ROCm SDK install complete ==="
