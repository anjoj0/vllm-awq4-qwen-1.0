#!/usr/bin/env bash
set -euo pipefail

NIXL_SRC=${NIXL_SRC:-/workspace/pd_disagg_20260802/src/nixl-main}
NIXL_PREFIX=${NIXL_PREFIX:-/workspace/pd_disagg_20260802/venv}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/build}
HIPCC=${HIPCC:-$(command -v hipcc)}

mkdir -p "$OUTPUT_DIR"

"$HIPCC" -O3 -std=c++20 -fPIC -shared \
  w7900_hip_ipc_backend.cpp w7900_hip_ipc_plugin.cpp \
  -I"$NIXL_SRC/src/api/cpp" \
  -I"$NIXL_SRC/src/utils" \
  -I"$NIXL_SRC/src/infra" \
  -I"$NIXL_SRC/src/core" \
  -I"$NIXL_SRC/src/core/telemetry" \
  -I"$NIXL_SRC/subprojects/abseil-cpp-20250814.1" \
  -pthread \
  -L"$NIXL_PREFIX/lib/x86_64-linux-gnu" \
  -Wl,-rpath,"$NIXL_PREFIX/lib/x86_64-linux-gnu" \
  -lnixl -lamdhip64 \
  -o "$OUTPUT_DIR/libplugin_W7900_HIP_IPC.so"

echo "$OUTPUT_DIR/libplugin_W7900_HIP_IPC.so"

if [[ -n "${INSTALL_DIR:-}" ]]; then
  install -d "$INSTALL_DIR"
  install -m 755 "$OUTPUT_DIR/libplugin_W7900_HIP_IPC.so" "$INSTALL_DIR/"
  echo "$INSTALL_DIR/libplugin_W7900_HIP_IPC.so"
fi
