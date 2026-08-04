#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
VLLM_WORKTREE="${VLLM_WORKTREE:-/workspace/vllm-main-20260801}"
source "${PD_ROOT}/scripts/activate_pd_env.sh"

: "${ROLE:?ROLE must be producer or consumer}"
: "${GPU_DEVICES:?GPU_DEVICES must be a comma-separated physical GPU list}"
: "${HTTP_PORT:?HTTP_PORT is required}"
: "${VLLM_PORT:?VLLM_PORT is required}"
: "${SIDE_CHANNEL_PORT:?SIDE_CHANNEL_PORT is required}"

case "${ROLE}" in
  producer|consumer) ;;
  *) echo "invalid ROLE=${ROLE}" >&2; exit 2 ;;
esac

IFS=',' read -r -a visible_gpus <<< "${GPU_DEVICES}"
TP_SIZE="${TP_SIZE:-${#visible_gpus[@]}}"
if [[ "${TP_SIZE}" -ne "${#visible_gpus[@]}" ]]; then
  echo "TP_SIZE=${TP_SIZE} does not match GPU_DEVICES=${GPU_DEVICES}" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-BF16-PD}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

export PYTHONPATH="${VLLM_WORKTREE}:/opt/python/lib/python3.14/site-packages"
export HIP_VISIBLE_DEVICES="${GPU_DEVICES}"
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
export VLLM_NIXL_SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT}"
export VLLM_PORT
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE="${ATTENTION_TILE:-16}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXECUTE_MODEL_TIMEOUT_SECONDS:-300}"

connector_config=$(printf \
  '{"kv_connector":"NixlConnector","kv_role":"kv_%s","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["%s"]}}' \
  "${ROLE}" "${NIXL_BACKEND:-W7900_HIP_IPC}")

exec python -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${HTTP_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --block-size 128 \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --attention-backend TRITON_ATTN \
  --skip-mm-profiling \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --enforce-eager \
  --kv-transfer-config "${connector_config}"
