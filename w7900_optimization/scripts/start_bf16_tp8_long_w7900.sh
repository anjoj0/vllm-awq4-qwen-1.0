#!/usr/bin/env bash
set -euo pipefail

# Recommended W7900 long-context / quality-first route.
# Uses all 8 W7900 cards, full BF16 Qwen3.6-27B, FP8 KV cache, and the
# gfx1100-tuned Triton unified-attention prefill tile.

PORT=${PORT:-8030}
MODEL=${MODEL:-/models/Qwen3.6-27B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.6-27B-BF16}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
LOG=${LOG:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/run_bf16_tp8_long_w7900.log}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}

cd "$WORKTREE"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
  --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}" \
  --skip-mm-profiling \
  2>&1 | tee "$LOG"
