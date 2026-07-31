#!/usr/bin/env bash
set -euo pipefail

# Recommended W7900 AWQ4 capacity route.
# Uses all 8 cards and disables DFlash by default because long-prompt tests
# show speculative decoding is negative for 24K+ prompts.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W7900_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

export ENV_FILE=${ENV_FILE:-/dev/null}
export W7900_VISIBLE_DEVICES=${W7900_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
export VLLM_HOST_PORT=${VLLM_HOST_PORT:-8048}
export VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-Qwen3.6-27B-AWQ4}
export VLLM_TARGET_MODEL=${VLLM_TARGET_MODEL:-/workspace/cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/master}
export VLLM_DISABLE_DFLASH=${VLLM_DISABLE_DFLASH:-1}
export VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-8}
export VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-0.80}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
export VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-fp8}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}

exec "$W7900_DIR/scripts/start_local_vllm.sh"
