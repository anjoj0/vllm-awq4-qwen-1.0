#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${VLLM_WORKTREE:-/opt/vllm-w7900-023}
TARGET=${VLLM_TARGET_MODEL:-/models/Qwen3.6-27B-AWQ-INT4}
DRAFT=${VLLM_DRAFT_MODEL:-/models/Qwen3.6-27B-DFlash}
TP=${VLLM_TENSOR_PARALLEL_SIZE:-2}
DRAFT_TP=${VLLM_DRAFT_TENSOR_PARALLEL_SIZE:-1}

export PYTHONPATH="${WORKTREE}:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES=${W7900_VISIBLE_DEVICES:-0,1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
export VLLM_USE_TRITON_AWQ=${VLLM_USE_TRITON_AWQ:-1}
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}

[[ -d "${WORKTREE}/vllm" ]] || { echo "vLLM worktree missing: ${WORKTREE}" >&2; exit 1; }
[[ -f "${TARGET}/config.json" ]] || { echo "Target model missing: ${TARGET}" >&2; exit 1; }

args=(
  serve "${TARGET}"
  --host 0.0.0.0
  --port "${VLLM_HOST_PORT:-8000}"
  --served-model-name "${VLLM_SERVED_MODEL_NAME:-Qwen3.6-27B-AWQ4}"
  --tensor-parallel-size "${TP}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL:-0.80}"
  --max-model-len "${VLLM_MAX_MODEL_LEN:-131072}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}"
  --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE:-fp8}"
  --attention-backend "${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
  --skip-mm-profiling
)

if [[ "${VLLM_ENFORCE_EAGER:-1}" == 1 ]]; then
  args+=(--enforce-eager)
fi

if [[ "${VLLM_DISABLE_DFLASH:-0}" != 1 ]]; then
  [[ -f "${DRAFT}/config.json" ]] || { echo "Draft model missing: ${DRAFT}" >&2; exit 1; }
  spec=$(DRAFT="${DRAFT}" DRAFT_TP="${DRAFT_TP}" DFLASH_N="${VLLM_DFLASH_N:-8}" python3 - <<'PY'
import json
import os

print(json.dumps({
    "method": "dflash",
    "model": os.environ["DRAFT"],
    "num_speculative_tokens": int(os.environ["DFLASH_N"]),
    "draft_tensor_parallel_size": int(os.environ["DRAFT_TP"]),
}))
PY
  )
  args+=(--speculative-config "${spec}")
fi

echo "W7900 gfx1100: GPUs=${HIP_VISIBLE_DEVICES} TP=${TP} DRAFT_TP=${DRAFT_TP}"
echo "MODEL=${TARGET} KV=${VLLM_KV_CACHE_DTYPE:-fp8} MAX_LEN=${VLLM_MAX_MODEL_LEN:-131072} TILE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE}"
cd "${WORKTREE}"
exec vllm "${args[@]}"
