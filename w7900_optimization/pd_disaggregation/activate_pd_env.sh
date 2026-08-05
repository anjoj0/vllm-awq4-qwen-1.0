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

# Shared-memory lanes are required for same-host ROCm IPC wireup. TCP remains
# available for the control plane and as a cross-host fallback.
export UCX_TLS="${UCX_TLS:-sm,rocm,tcp,self}"

# UCX 1.22 needs this opt-in for the UCP PUT rendezvous path. With NIXL on
# W7900, it also moves host-staged fallback traffic from TCP to CMA.
export UCX_RMA_PPLN_ENABLE="${UCX_RMA_PPLN_ENABLE:-y}"
