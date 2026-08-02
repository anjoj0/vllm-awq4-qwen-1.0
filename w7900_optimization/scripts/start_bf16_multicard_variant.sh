#!/usr/bin/env bash
set -euo pipefail

# Generic W7900 BF16 launcher for controlled multi-card A/B experiments.
PORT=${PORT:-8080}
MODEL=${MODEL:-/models/Qwen3.6-27B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.6-27B-BF16}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-main-20260801}
LOG=${LOG:-/workspace/multicard_frontier/service.log}

TP=${TP:-8}
PP=${PP:-1}
DCP=${DCP:-1}
PCP=${PCP:-1}
DCP_COMM_BACKEND=${DCP_COMM_BACKEND:-ag_rs}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
PREFIX_CACHING=${PREFIX_CACHING:-disable}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-1}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-}
BLOCK_SIZE=${BLOCK_SIZE:-}
ENABLE_SP=${ENABLE_SP:-0}
FUSE_GEMM_COMMS=${FUSE_GEMM_COMMS:-0}
SP_MIN_TOKEN_NUM=${SP_MIN_TOKEN_NUM:-}

if [[ -z "$COMPILATION_CONFIG" && ( "$ENABLE_SP" == 1 || "$FUSE_GEMM_COMMS" == 1 ) ]]; then
  if [[ "$ENABLE_SP" == 1 ]]; then sp_value=true; else sp_value=false; fi
  if [[ "$FUSE_GEMM_COMMS" == 1 ]]; then fuse_value=true; else fuse_value=false; fi
  sp_threshold=""
  if [[ -n "$SP_MIN_TOKEN_NUM" ]]; then
    [[ "$SP_MIN_TOKEN_NUM" =~ ^[0-9]+$ ]] || { echo "SP_MIN_TOKEN_NUM must be an integer" >&2; exit 2; }
    sp_threshold=",\"sp_min_token_num\":${SP_MIN_TOKEN_NUM}"
  fi
  COMPILATION_CONFIG="{\"pass_config\":{\"enable_sp\":${sp_value},\"fuse_gemm_comms\":${fuse_value}${sp_threshold}}}"
fi

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}

mkdir -p "$(dirname "$LOG")"

cmd=(
  python3 -m vllm.entrypoints.cli.main serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --decode-context-parallel-size "$DCP"
  --prefill-context-parallel-size "$PCP"
  --dcp-comm-backend "$DCP_COMM_BACKEND"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --skip-mm-profiling
)

case "$PREFIX_CACHING" in
  enable) cmd+=(--enable-prefix-caching) ;;
  disable) cmd+=(--no-enable-prefix-caching) ;;
  auto) ;;
  *) echo "PREFIX_CACHING must be enable, disable, or auto" >&2; exit 2 ;;
esac

if [[ "$ENFORCE_EAGER" == 1 ]]; then
  cmd+=(--enforce-eager)
fi
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == 1 ]]; then
  cmd+=(--disable-custom-all-reduce)
fi
if [[ -n "$COMPILATION_CONFIG" ]]; then
  cmd+=(--compilation-config "$COMPILATION_CONFIG")
fi
if [[ -n "$BLOCK_SIZE" ]]; then
  cmd+=(--block-size "$BLOCK_SIZE")
fi

{
  echo "timestamp_utc=$(date -u +%FT%TZ)"
  echo "worktree=$WORKTREE"
  echo "hip_visible_devices=$HIP_VISIBLE_DEVICES"
  printf 'command='; printf '%q ' "${cmd[@]}"; echo
  echo "VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER"
  echo "VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=$VLLM_TRITON_ATTN_PREFILL_TILE_SIZE"
  echo "BLOCK_SIZE=${BLOCK_SIZE:-auto}"
  echo "ENABLE_SP=$ENABLE_SP"
  echo "FUSE_GEMM_COMMS=$FUSE_GEMM_COMMS"
  echo "SP_MIN_TOKEN_NUM=${SP_MIN_TOKEN_NUM:-auto}"
  echo "COMPILATION_CONFIG=${COMPILATION_CONFIG:-default}"
} > "${LOG}.manifest"

cd "$WORKTREE"
exec "${cmd[@]}" 2>&1 | tee "$LOG"
