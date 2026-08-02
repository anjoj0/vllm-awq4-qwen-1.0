#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
source "${PD_ROOT}/scripts/activate_pd_env.sh"

export PYTHONPATH="/workspace/vllm-main-20260801:/opt/python/lib/python3.14/site-packages"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659
export VLLM_PORT=30000
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16
export UCX_NET_DEVICES=all

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-BF16-PD}"
PORT="${PORT:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"

exec python -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --block-size 128 \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --skip-mm-profiling \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX"]}}'
