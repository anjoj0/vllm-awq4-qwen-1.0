#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
RESULT_DIR="$ROOT/results"
RUN_ID=${RUN_ID:-20260729_rocm_attn_short_addendum}
RESULT_MD="$RESULT_DIR/${RUN_ID}.md"
JSONL="$RESULT_DIR/${RUN_ID}.jsonl"
SCHED_LOG="$RESULT_DIR/${RUN_ID}.scheduler.log"

mkdir -p "$RESULT_DIR"
cd "$ROOT" || exit 1

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}

log() {
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] $*"
}

md() {
  echo "$*" >> "$RESULT_MD"
}

stop_vllm() {
  log "stopping existing vLLM processes"
  pkill -TERM -f "vllm serve" 2>/dev/null || true
  pkill -TERM -f "vllm.entrypoints.openai" 2>/dev/null || true
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -TERM -f "VLLM::Worker_TP" 2>/dev/null || true
  pkill -TERM -f "multiprocessing.resource_tracker" 2>/dev/null || true
  sleep 15
  if pgrep -f "vllm serve|vllm.entrypoints.openai|VLLM::EngineCore|VLLM::Worker_TP" >/dev/null 2>&1; then
    log "forcing stale vLLM processes to exit"
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "vllm.entrypoints.openai" 2>/dev/null || true
    pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -KILL -f "VLLM::Worker_TP" 2>/dev/null || true
    pkill -KILL -f "multiprocessing.resource_tracker" 2>/dev/null || true
    sleep 10
  fi
}

wait_ready() {
  local port="$1"
  local logfile="$2"
  local timeout_s="${3:-900}"
  local deadline=$((SECONDS + timeout_s))
  local url="http://127.0.0.1:${port}/v1/models"
  while [ "$SECONDS" -lt "$deadline" ]; do
    python3 - "$url" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
urllib.request.urlopen(sys.argv[1], timeout=5).read()
PY
    if [ "$?" -eq 0 ]; then
      log "service on port ${port} is ready"
      return 0
    fi
    if ! pgrep -f "vllm serve" >/dev/null 2>&1 && [ "$SECONDS" -gt 60 ]; then
      log "service on port ${port} appears dead before readiness"
      tail -100 "$logfile" >> "$SCHED_LOG" 2>/dev/null || true
      return 1
    fi
    sleep 10
  done
  log "service on port ${port} readiness timeout"
  tail -120 "$logfile" >> "$SCHED_LOG" 2>/dev/null || true
  return 1
}

bench_single() {
  local label="$1"
  local port="$2"
  local chars="$3"
  local max_tokens="$4"
  local requests="$5"
  local concurrency="$6"
  log "bench ${label}: chars=${chars}, max_tokens=${max_tokens}, requests=${requests}, concurrency=${concurrency}"
  {
    echo
    echo "### ${label}"
    echo
    echo '```json'
  } >> "$RESULT_MD"
  timeout 1200 python3 scripts/bench_concurrency_local.py \
    --url "http://127.0.0.1:${port}/v1/chat/completions" \
    --model Qwen3.6-27B-BF16 \
    --file /workspace/bench_data/combined_papers_for_llm.txt \
    --chars "$chars" \
    --max-tokens "$max_tokens" \
    --requests "$requests" \
    --concurrency "$concurrency" | tee -a "$JSONL" >> "$RESULT_MD"
  local rc=${PIPESTATUS[0]}
  echo '```' >> "$RESULT_MD"
  echo "{\"label\":\"${label}\",\"return_code\":${rc}}" >> "$JSONL"
  return "$rc"
}

start_backend() {
  local backend="$1"
  local port="$2"
  local logf="$3"
  log "starting BF16 TP8 backend=${backend} max_len=32768 port=${port}"
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PORT="$port" \
  MODEL=/models/Qwen3.6-27B \
  SERVED_MODEL_NAME=Qwen3.6-27B-BF16 \
  GPU_MEMORY_UTILIZATION=0.80 \
  MAX_MODEL_LEN=32768 \
  MAX_NUM_BATCHED_TOKENS=8192 \
  MAX_NUM_SEQS=8 \
  KV_CACHE_DTYPE=fp8 \
  ATTENTION_BACKEND="$backend" \
  VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16 \
  LOG="$logf" \
  bash scripts/start_bf16_tp8_long_w7900.sh > "${logf}.outer" 2>&1 &
}

record_header() {
  md "# ROCM_ATTN short/decode addendum"
  md
  md "- 开始时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
  md "- 目的：确认 ROCM_ATTN 是否在短上下文、高并发、decode-heavy 场景优于 TRITON_ATTN tile=16。"
  md "- 配置：BF16 Qwen3.6-27B，TP=8，max_model_len=32768，max_num_batched_tokens=8192，max_num_seqs=8，KV=fp8。"
  md
  md "## AITER 可用性"
  md
  {
    echo '```text'
    python3 - <<'PY'
import importlib.util
for mod in ["aiter", "aiter.ops"]:
    try:
        spec = importlib.util.find_spec(mod)
        print(mod, "FOUND" if spec else "missing", getattr(spec, "origin", "") if spec else "")
    except Exception as exc:
        print(mod, "ERROR", repr(exc))
PY
    python3 -m pip list 2>/dev/null | grep -i aiter || true
    echo '```'
  } >> "$RESULT_MD" 2>&1
}

run_backend() {
  local backend="$1"
  local port="$2"
  local tag="$3"
  local logf="$RESULT_DIR/${tag}.log"
  md
  md "## ${backend}"
  stop_vllm
  start_backend "$backend" "$port" "$logf"
  if wait_ready "$port" "$logf" 1200; then
    {
      echo
      echo "启动日志摘要：${logf}"
      grep -E "Available KV|GPU KV cache size|Maximum concurrency|torch.compile|Graph capturing|init engine|max_seq_len|Using attention backend|attention backend|ROCm custom paged attention|falling back" "$logf" 2>/dev/null | tail -100 || true
    } >> "$RESULT_MD"
    bench_single "${tag}_decode_2k_4c" "$port" 2000 256 4 4 || true
    bench_single "${tag}_short_10k_4c" "$port" 10000 128 4 4 || true
    bench_single "${tag}_prefill_24k_1c" "$port" 100000 128 1 1 || true
  else
    md "- 启动失败或超时：${logf}"
  fi
  stop_vllm
}

main() {
  exec > >(tee -a "$SCHED_LOG") 2>&1
  record_header
  run_backend TRITON_ATTN 8070 bf16_tp8_triton_attn_mlen32k_add
  run_backend ROCM_ATTN 8071 bf16_tp8_rocm_attn_mlen32k_add
  md
  md "## 结束"
  md
  md "- 结束时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
  log "ROCM_ATTN addendum finished"
}

main "$@"
