#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
VLLM_WORKTREE="${VLLM_WORKTREE:-/workspace/vllm-main-20260801}"
source "${PD_ROOT}/scripts/activate_pd_env.sh"

export PYTHONPATH="${VLLM_WORKTREE}:/opt/python/lib/python3.14/site-packages"

exec python \
  "${VLLM_WORKTREE}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --host "${PROXY_HOST:-127.0.0.1}" \
  --port "${PROXY_PORT:-8192}" \
  --prefiller-hosts "${PREFILL_HOST:-127.0.0.1}" \
  --prefiller-ports "${PREFILL_PORT:-8100}" \
  --decoder-hosts "${DECODE_HOST:-127.0.0.1}" \
  --decoder-ports "${DECODE_PORT:-8200}"
