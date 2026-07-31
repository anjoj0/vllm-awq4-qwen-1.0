#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
RESULT_DIR="$ROOT/results"
RUN_ID=${RUN_ID:-20260729_fp8_262k_retest}
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
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}

log() {
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] $*" | tee -a "$SCHED_LOG"
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
}

wait_ready() {
  local port="$1"
  local logfile="$2"
  local timeout_s="${3:-2400}"
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
    sleep 10
  done
  log "service on port ${port} readiness timeout"
  tail -160 "$logfile" >> "$SCHED_LOG" 2>/dev/null || true
  return 1
}

extract_startup_summary() {
  local logfile="$1"
  {
    echo
    echo "启动日志摘要：${logfile}"
    grep -E "Model loading took|Available KV|GPU KV cache size|Maximum concurrency|torch.compile|Graph capturing|init engine|Padding mamba|Setting attention block size|Triton kernel JIT|Application startup complete" "$logfile" 2>/dev/null | tail -160 || true
  } >> "$RESULT_MD"
}

bench_single() {
  local label="$1"
  local port="$2"
  local chars="$3"
  local max_tokens="$4"
  log "bench ${label}: chars=${chars}, max_tokens=${max_tokens}"
  {
    echo
    echo "### ${label}"
    echo
    echo '```json'
  } >> "$RESULT_MD"
  timeout 3600 python3 scripts/bench_concurrency_local.py \
    --url "http://127.0.0.1:${port}/v1/chat/completions" \
    --model Qwen3.6-27B-BF16 \
    --file /workspace/bench_data/combined_papers_for_llm_L.txt \
    --chars "$chars" \
    --max-tokens "$max_tokens" \
    --requests 1 \
    --concurrency 1 | tee -a "$JSONL" >> "$RESULT_MD"
  local rc=${PIPESTATUS[0]}
  echo '```' >> "$RESULT_MD"
  echo "{\"label\":\"${label}\",\"return_code\":${rc}}" >> "$JSONL"
  return "$rc"
}

summarize_jsonl() {
  md
  md "## 自动提取结果"
  md
  python3 - "$JSONL" >> "$RESULT_MD" <<'PY'
import json, sys
rows = []
pending = None
try:
    for line in open(sys.argv[1], encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "label" in obj:
            if pending is not None:
                pending["label"] = obj["label"]
                pending["return_code"] = obj.get("return_code")
                rows.append(pending)
                pending = None
        else:
            pending = obj
except FileNotFoundError:
    pass
print("| label | chars | prompt_tokens | max_tokens | wall_s | output tok/s | rc |")
print("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    pt = ""
    if r.get("rows"):
        pt = r["rows"][0].get("prompt_tokens", "")
    print(f"| {r.get('label','')} | {r.get('chars','')} | {pt} | {r.get('max_tokens','')} | {r.get('wall_s',0):.4f} | {r.get('aggregate_output_tok_s',0):.4f} | {r.get('return_code','')} |")
PY
}

PORT=8075
LOGF="$RESULT_DIR/bf16_tp8_triton_tile16_fp8_mlen262k_retest.log"

md "# W7900 FP8 KV 262K retest"
md
md "- 开始时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
md "- 配置：Qwen3.6-27B BF16, TP=8, TRITON_ATTN tile=16, kv_cache_dtype=fp8, max_model_len=262144, gpu_memory_utilization=0.85。"
md "- 目的：连续两次 950k chars，确认 07:00 FP8 262K 耗时是否为首请求/JIT 偶然噪声。"

stop_vllm
log "starting BF16 TP8 fp8 max_len=262144 port=${PORT}"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT="$PORT" \
MODEL=/models/Qwen3.6-27B \
SERVED_MODEL_NAME=Qwen3.6-27B-BF16 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_MODEL_LEN=262144 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_NUM_SEQS=8 \
KV_CACHE_DTYPE=fp8 \
ATTENTION_BACKEND=TRITON_ATTN \
VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16 \
LOG="$LOGF" \
bash scripts/start_bf16_tp8_long_w7900.sh > "${LOGF}.outer" 2>&1 &

if wait_ready "$PORT" "$LOGF" 2400; then
  extract_startup_summary "$LOGF"
  bench_single "bf16_tp8_triton_tile16_fp8_mlen262k_retest_950k_first" "$PORT" 950000 64 || true
  bench_single "bf16_tp8_triton_tile16_fp8_mlen262k_retest_950k_second" "$PORT" 950000 64 || true
else
  md
  md "- 服务启动失败或超时。"
  extract_startup_summary "$LOGF"
fi
extract_startup_summary "$LOGF"
stop_vllm
summarize_jsonl
md
md "- 结束时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
log "FP8 262K retest finished"
