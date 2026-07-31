#!/usr/bin/env bash
set -euo pipefail

# Recommended W7900 short-context / multi-tenant route.
# Starts two independent BF16 TP=4 servers:
#   - port 8041 on GPU0-3
#   - port 8042 on GPU4-7
#
# This layout reached ~159.7 output tok/s for 4+4 short-context concurrency,
# versus ~129.7 output tok/s for a single BF16 TP=8 service.

WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
MODEL=${MODEL:-/models/Qwen3.6-27B}
RESULTS=${RESULTS:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results}

start_one() {
  local devices=$1
  local port=$2
  local name=$3
  local log=$4

  (
    export HIP_VISIBLE_DEVICES="$devices"
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
    export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
    export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
    export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}
    cd "$WORKTREE"
    exec vllm serve "$MODEL" \
      --host 0.0.0.0 \
      --port "$port" \
      --served-model-name "$name" \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
      --max-model-len "${MAX_MODEL_LEN:-131072}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
      --max-num-seqs "${MAX_NUM_SEQS:-4}" \
      --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
      --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}" \
      --skip-mm-profiling
  ) > "$log" 2>&1 &
  echo "$!"
}

mkdir -p "$RESULTS"
pid_a=$(start_one "0,1,2,3" "${PORT_A:-8041}" "${MODEL_A_NAME:-Qwen3.6-27B-BF16-G0}" "$RESULTS/run_dual_bf16_tp4_g0.log")
pid_b=$(start_one "4,5,6,7" "${PORT_B:-8042}" "${MODEL_B_NAME:-Qwen3.6-27B-BF16-G4}" "$RESULTS/run_dual_bf16_tp4_g4.log")

echo "Started dual TP=4 BF16 services:"
echo "  GPU0-3 PID=$pid_a PORT=${PORT_A:-8041}"
echo "  GPU4-7 PID=$pid_b PORT=${PORT_B:-8042}"
echo "Logs:"
echo "  $RESULTS/run_dual_bf16_tp4_g0.log"
echo "  $RESULTS/run_dual_bf16_tp4_g4.log"
echo
echo "Health checks:"
echo "  curl http://127.0.0.1:${PORT_A:-8041}/health"
echo "  curl http://127.0.0.1:${PORT_B:-8042}/health"

wait "$pid_a" "$pid_b"
