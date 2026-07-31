#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
RESULT_DIR="$ROOT/results"
RUN_ID=${RUN_ID:-20260729_kv_warmup_addendum}
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
  local msg="$*"
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] $msg" | tee -a "$SCHED_LOG"
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
  if pgrep -f "vllm serve" >/dev/null 2>&1 || pgrep -f "vllm.entrypoints.openai" >/dev/null 2>&1 || pgrep -f "VLLM::EngineCore" >/dev/null 2>&1 || pgrep -f "VLLM::Worker_TP" >/dev/null 2>&1; then
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
  local timeout_s="${3:-1800}"
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

extract_startup_summary() {
  local logfile="$1"
  {
    echo
    echo "启动日志摘要：${logfile}"
    grep -E "model weights take|model memory|Available KV|GPU KV cache size|Maximum concurrency|torch.compile|Graph capturing|init engine|profile|Padding mamba|Setting attention block size|Using attention backend|attention backend|Cannot use ROCm custom paged attention|Triton kernel JIT" "$logfile" 2>/dev/null | tail -120 || true
  } >> "$RESULT_MD"
}

bench_single() {
  local label="$1"
  local port="$2"
  local model="$3"
  local file="$4"
  local chars="$5"
  local max_tokens="$6"
  local requests="$7"
  local concurrency="$8"
  log "bench ${label}: chars=${chars}, max_tokens=${max_tokens}, requests=${requests}, concurrency=${concurrency}"
  {
    echo
    echo "### ${label}"
    echo
    echo '```json'
  } >> "$RESULT_MD"
  timeout 2400 python3 scripts/bench_concurrency_local.py \
    --url "http://127.0.0.1:${port}/v1/chat/completions" \
    --model "$model" \
    --file "$file" \
    --chars "$chars" \
    --max-tokens "$max_tokens" \
    --requests "$requests" \
    --concurrency "$concurrency" | tee -a "$JSONL" >> "$RESULT_MD"
  local rc=${PIPESTATUS[0]}
  echo '```' >> "$RESULT_MD"
  echo "{\"label\":\"${label}\",\"return_code\":${rc}}" >> "$JSONL"
  return "$rc"
}

start_bf16_tp8() {
  local port="$1"
  local kv_dtype="$2"
  local max_len="$3"
  local util="$4"
  local max_batched="$5"
  local logf="$6"
  log "starting BF16 TP8 kv=${kv_dtype} max_len=${max_len} util=${util} max_batched=${max_batched} port=${port}"
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PORT="$port" \
  MODEL=/models/Qwen3.6-27B \
  SERVED_MODEL_NAME=Qwen3.6-27B-BF16 \
  GPU_MEMORY_UTILIZATION="$util" \
  MAX_MODEL_LEN="$max_len" \
  MAX_NUM_BATCHED_TOKENS="$max_batched" \
  MAX_NUM_SEQS=8 \
  KV_CACHE_DTYPE="$kv_dtype" \
  ATTENTION_BACKEND=TRITON_ATTN \
  VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16 \
  LOG="$logf" \
  bash scripts/start_bf16_tp8_long_w7900.sh > "${logf}.outer" 2>&1 &
}

record_header() {
  md "# W7900 KV / warmup addendum"
  md
  md "- 开始时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
  md "- RUN_ID：${RUN_ID}"
  md "- 目标：验证首请求 JIT/warmup 影响，并比较 BF16 TP8 下 fp8 KV 与 auto KV 的容量/速度。"
  md "- 主配置：Qwen3.6-27B BF16，TP=8，TRITON_ATTN，tile=16。"
  md
}

run_fp8_repeatability() {
  md
  md "## 1. fp8 KV 重复请求 / warmup 稳定性"
  stop_vllm
  local tag="bf16_tp8_triton_tile16_fp8_mlen131k_warmup"
  local port=8072
  local logf="$RESULT_DIR/${tag}.log"
  start_bf16_tp8 "$port" fp8 131072 0.80 16384 "$logf"
  if wait_ready "$port" "$logf" 1800; then
    extract_startup_summary "$logf"
    bench_single "${tag}_10k_4c_first" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 10000 128 4 4 || true
    bench_single "${tag}_10k_4c_second" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 10000 128 4 4 || true
    bench_single "${tag}_24k_first" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
    bench_single "${tag}_24k_second" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
    bench_single "${tag}_103k_first" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 400000 128 1 1 || true
    bench_single "${tag}_103k_second" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 400000 128 1 1 || true
  else
    md
    md "- fp8 KV 服务启动失败或健康检查失败。"
    extract_startup_summary "$logf"
  fi
  extract_startup_summary "$logf"
  stop_vllm
}

run_auto_kv_131k() {
  md
  md "## 2. auto KV 对照：max_model_len=131072"
  stop_vllm
  local tag="bf16_tp8_triton_tile16_auto_mlen131k"
  local port=8073
  local logf="$RESULT_DIR/${tag}.log"
  start_bf16_tp8 "$port" auto 131072 0.80 16384 "$logf"
  if wait_ready "$port" "$logf" 1800; then
    extract_startup_summary "$logf"
    bench_single "${tag}_10k_4c" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 10000 128 4 4 || true
    bench_single "${tag}_24k" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
    bench_single "${tag}_103k" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 400000 128 1 1 || true
  else
    md
    md "- auto KV 131K 服务启动失败或健康检查失败。"
    extract_startup_summary "$logf"
  fi
  extract_startup_summary "$logf"
  stop_vllm
}

run_auto_kv_262k_boundary() {
  md
  md "## 3. auto KV 对照：max_model_len=262144 边界启动与 256K-class"
  stop_vllm
  local tag="bf16_tp8_triton_tile16_auto_mlen262k"
  local port=8074
  local logf="$RESULT_DIR/${tag}.log"
  start_bf16_tp8 "$port" auto 262144 0.85 16384 "$logf"
  if wait_ready "$port" "$logf" 2400; then
    extract_startup_summary "$logf"
    bench_single "${tag}_950kchars" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm_L.txt 950000 64 1 1 || true
  else
    md
    md "- auto KV 262K 服务启动失败或健康检查失败；记录为 FP8 KV 容量必要性的对照。"
    extract_startup_summary "$logf"
  fi
  extract_startup_summary "$logf"
  stop_vllm
}

write_summary() {
  md
  md "## 4. 自动提取结果"
  md
  python3 - "$JSONL" >> "$RESULT_MD" <<'PY'
import json, sys
path = sys.argv[1]
rows = []
try:
    with open(path, encoding="utf-8") as f:
        pending = None
        for line in f:
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

print("| label | chars | requests/concurrency | max_tokens | wall_s | output tok/s | prompt_tokens | rc |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    pts = ""
    if r.get("rows"):
        vals = [x.get("prompt_tokens") for x in r["rows"] if x.get("prompt_tokens") is not None]
        if vals:
            pts = str(vals[0]) if len(set(vals)) == 1 else ",".join(map(str, vals[:4]))
    print(f"| {r.get('label','')} | {r.get('chars','')} | {r.get('requests','')}/{r.get('concurrency','')} | {r.get('max_tokens','')} | {r.get('wall_s',0):.4f} | {r.get('aggregate_output_tok_s',0):.4f} | {pts} | {r.get('return_code','')} |")
PY
  md
  md "- 结束时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
}

record_header
run_fp8_repeatability
run_auto_kv_131k
run_auto_kv_262k_boundary
write_summary
log "KV/warmup addendum finished"
