#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); W7900_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE=${ENV_FILE:-"$W7900_DIR/.env"}; if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
export HIP_VISIBLE_DEVICES=${W7900_VISIBLE_DEVICES:-0,1} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HSA_NO_SCRATCH_RECLAIM=${W7900_HSA_NO_SCRATCH_RECLAIM:-1} MIOPEN_FIND_MODE=FAST VLLM_ROCM_USE_AITER=0 VLLM_USE_TRITON_AWQ=1 AWQ_MMQ_ENABLE=0
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
TARGET=${VLLM_TARGET_MODEL:-/workspace/cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/master}; DRAFT=${VLLM_DRAFT_MODEL:-/workspace/z-lab--Qwen3.6-27B-DFlash/snapshots/master}; TP=${VLLM_TENSOR_PARALLEL_SIZE:-2}; DRAFT_TP=${VLLM_DRAFT_TENSOR_PARALLEL_SIZE:-1}
[[ -f "$TARGET/config.json" ]] || { echo "Target model missing: $TARGET" >&2; exit 1; }
ARGS=(serve "$TARGET" --host 0.0.0.0 --port "${VLLM_HOST_PORT:-8000}" --served-model-name "${VLLM_SERVED_MODEL_NAME:-Qwen3.6-27B-AWQ4}" --tensor-parallel-size "$TP" --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL:-0.80}" --max-model-len "${VLLM_MAX_MODEL_LEN:-131072}" --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}" --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}" --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE:-fp8}" --attention-backend "${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" --skip-mm-profiling)
if [[ "${VLLM_ENFORCE_EAGER:-1}" == 1 ]]; then ARGS+=(--enforce-eager); fi
if [[ "${VLLM_DISABLE_DFLASH:-0}" != 1 ]]; then
  [[ -f "$DRAFT/config.json" ]] || { echo "Draft model missing: $DRAFT" >&2; exit 1; }
  SPEC=$(DRAFT="$DRAFT" DRAFT_TP="$DRAFT_TP" DFLASH_N="${VLLM_DFLASH_N:-8}" python3 - <<'PY'
import json, os
print(json.dumps({"method":"dflash","model":os.environ["DRAFT"],"num_speculative_tokens":int(os.environ["DFLASH_N"]),"draft_tensor_parallel_size":int(os.environ["DRAFT_TP"])}))
PY
)
  ARGS+=(--speculative-config "$SPEC")
fi
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES TP=$TP DRAFT_TP=$DRAFT_TP"
echo "EAGER=${VLLM_ENFORCE_EAGER:-1} MAX_LEN=${VLLM_MAX_MODEL_LEN:-131072} MAX_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384} MAX_SEQS=${VLLM_MAX_NUM_SEQS:-1}"
echo "ATTN_PREFILL_TILE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-default} ATTN_DECODE_TILE=${VLLM_TRITON_ATTN_DECODE_TILE_SIZE:-default} ATTN_MIN_2D_GRID=${VLLM_TRITON_ATTN_MIN_2D_GRID:-default} ATTN_SOFTMAX_SEGMENTS=${VLLM_TRITON_ATTN_SOFTMAX_SEGMENTS:-default}"
cd "$WORKTREE"; exec vllm "${ARGS[@]}"
