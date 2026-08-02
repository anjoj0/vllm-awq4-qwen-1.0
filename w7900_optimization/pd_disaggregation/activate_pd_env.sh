#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
UCX_ROOT="${UCX_ROOT:-${PD_ROOT}/deps/ucx-rocm}"
NIXL_VENV="${NIXL_VENV:-${PD_ROOT}/venv}"

source "${NIXL_VENV}/bin/activate"

# The ROCm vLLM image uses /opt/python as a relocatable Python prefix. A venv
# created from that interpreter does not automatically inherit its packages.
export PYTHONPATH="/opt/python/lib/python3.14/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${NIXL_VENV}/lib/x86_64-linux-gnu:${NIXL_VENV}/lib/x86_64-linux-gnu/plugins:${UCX_ROOT}/lib:${UCX_ROOT}/lib/ucx:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="${NIXL_VENV}/lib/x86_64-linux-gnu/plugins"

# Keep the control plane on TCP while allowing same-node GPU payloads to use
# ROCm IPC or ROCm copy transports.
export UCX_TLS="${UCX_TLS:-rocm_ipc,rocm_copy,self,tcp}"
