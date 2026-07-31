#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
RESULT_DIR="$ROOT/results"
RUN_ID=${RUN_ID:-20260730_overnight_w7900_exploration}
TARGET_BJT=${TARGET_BJT:-"2026-07-30 08:00:00"}
PORT_BASE=${PORT_BASE:-8090}

MD="$RESULT_DIR/${RUN_ID}.md"
JSONL="$RESULT_DIR/${RUN_ID}.jsonl"
LOG="$RESULT_DIR/${RUN_ID}.runner.log"
PID_FILE="$RESULT_DIR/${RUN_ID}.pid"

mkdir -p "$RESULT_DIR"
echo "$$" > "$PID_FILE"
cd "$ROOT" || exit 1

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
export VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=${VLLM_TRITON_ATTN_PREFILL_TILE_SIZE:-16}

log() {
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] $*" | tee -a "$LOG"
}

md() {
  echo "$*" >> "$MD"
}

target_epoch() {
  TZ=Asia/Shanghai date -d "$TARGET_BJT" +%s
}

now_epoch() {
  TZ=Asia/Shanghai date +%s
}

time_left_s() {
  echo $(( $(target_epoch) - $(now_epoch) ))
}

have_time_for() {
  local min_s="$1"
  [ "$(time_left_s)" -gt "$min_s" ]
}

stop_vllm() {
  log "stopping vLLM/bench processes"
  pkill -TERM -f "vllm serve" 2>/dev/null || true
  pkill -TERM -f "vllm.entrypoints.openai" 2>/dev/null || true
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -TERM -f "VLLM::Worker_TP" 2>/dev/null || true
  pkill -TERM -f "multiprocessing.resource_tracker" 2>/dev/null || true
  pkill -TERM -f "bench_concurrency_local.py" 2>/dev/null || true
  pkill -TERM -f "torchrun" 2>/dev/null || true
  sleep 12
  if pgrep -f "vllm serve|VLLM::EngineCore|VLLM::Worker_TP" >/dev/null 2>&1; then
    log "forcing stale vLLM processes"
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "vllm.entrypoints.openai" 2>/dev/null || true
    pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -KILL -f "VLLM::Worker_TP" 2>/dev/null || true
    pkill -KILL -f "multiprocessing.resource_tracker" 2>/dev/null || true
    sleep 8
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
    if ! pgrep -f "vllm serve" >/dev/null 2>&1 && [ "$SECONDS" -gt 90 ]; then
      log "service on port ${port} appears dead before readiness"
      tail -120 "$logfile" >> "$LOG" 2>/dev/null || true
      return 1
    fi
    sleep 10
  done
  log "service on port ${port} readiness timeout"
  tail -160 "$logfile" >> "$LOG" 2>/dev/null || true
  return 1
}

start_bf16_tp8() {
  local port="$1"
  local kv_dtype="$2"
  local max_len="$3"
  local util="$4"
  local max_batched="$5"
  local logf="$6"
  log "starting BF16 TP8 kv=${kv_dtype} max_len=${max_len} util=${util} port=${port}"
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

append_startup_summary() {
  local logfile="$1"
  {
    echo
    echo "启动日志摘要：${logfile}"
    grep -E "Directly load|torch.compile took|Model loading took|Available KV|GPU KV cache size|Maximum concurrency|Graph capturing|init engine|Triton kernel JIT|Application startup complete" "$logfile" 2>/dev/null | tail -120 || true
  } >> "$MD"
}

wait_current_longdoc() {
  local pid_file="$RESULT_DIR/longdoc_sanity/bf16_tp8_auto_mlen262k_Lcore.pid"
  local pid=""
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file" 2>/dev/null || true)"
  fi
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    log "waiting current near256K longdoc sanity pid=${pid}"
    while kill -0 "$pid" 2>/dev/null; do
      if ! have_time_for 900; then
        log "time budget nearly exhausted while longdoc still running"
        return 1
      fi
      tail -20 "$RESULT_DIR/longdoc_sanity/bf16_tp8_auto_mlen262k_Lcore.nohup.log" >> "$LOG" 2>/dev/null || true
      sleep 120
    done
    log "current near256K longdoc sanity exited"
  else
    log "no active near256K longdoc pid found"
  fi
  return 0
}

append_latest_longdoc_summary() {
  md
  md "## near256K 科研长文 sanity"
  local latest
  latest=$(find "$RESULT_DIR/longdoc_sanity" -maxdepth 2 -name summary.md -type f 2>/dev/null | sort | tail -n 1)
  if [ -n "$latest" ]; then
    md
    md "- summary: \`$latest\`"
    md
    sed -n '1,220p' "$latest" >> "$MD" 2>/dev/null || true
  else
    md
    md "- 未找到 summary.md；保留原始日志：\`$RESULT_DIR/longdoc_sanity/bf16_tp8_auto_mlen262k_Lcore.nohup.log\`"
  fi
}

run_awq_mmq_large_m() {
  if ! have_time_for 2400; then
    log "skip AWQ MMQ large-M: insufficient time"
    return 0
  fi
  md
  md "## AWQ4 HIP MMQ large-M microbench"
  stop_vllm
  local bench_dir="$ROOT/csrc/awq_mmq_gfx1100"
  local out="$RESULT_DIR/${RUN_ID}.awq_mmq_large_m.log"
  log "running AWQ MMQ large-M benchmark"
  (
    cd "$bench_dir" || exit 1
    export PYTHONPATH="$bench_dir:$WORKTREE:${PYTHONPATH:-}"
    python3 - <<'PY' || MAX_JOBS=8 python3 setup.py build_ext --inplace
import awq_mmq_gfx1100
print("awq_mmq_gfx1100 import ok")
PY
    HIP_VISIBLE_DEVICES=0 MMQ_M_VALUES="${MMQ_M_VALUES:-4096,8192,16384}" \
      timeout 3600 python3 benchmark_prefill_gfx1100.py
  ) > "$out" 2>&1
  local rc=$?
  echo "{\"event\":\"awq_mmq_large_m\",\"return_code\":${rc},\"log\":\"${out}\"}" >> "$JSONL"
  md
  md "- log: \`$out\`"
  md "- return_code: \`$rc\`"
  md
  md '```json'
  grep -E '^\{' "$out" | tail -80 >> "$MD" || true
  md '```'
  return 0
}

run_rccl_curve() {
  if ! have_time_for 2700; then
    log "skip RCCL: insufficient time"
    return 0
  fi
  md
  md "## RCCL All-Reduce microbenchmark"
  stop_vllm
  local out="$RESULT_DIR/${RUN_ID}.rccl.jsonl"
  local logf="$RESULT_DIR/${RUN_ID}.rccl.log"
  : > "$out"
  : > "$logf"
  for n in 2 4 8; do
    if ! have_time_for 900; then
      log "stop RCCL loop before TP=${n}: insufficient time"
      break
    fi
    local devices
    if [ "$n" = "2" ]; then devices="0,1"; fi
    if [ "$n" = "4" ]; then devices="0,1,2,3"; fi
    if [ "$n" = "8" ]; then devices="0,1,2,3,4,5,6,7"; fi
    log "RCCL all-reduce nproc=${n} devices=${devices}"
    HIP_VISIBLE_DEVICES="$devices" NCCL_DEBUG=WARN \
      timeout 900 python3 -m torch.distributed.run --standalone --nproc_per_node="$n" \
        scripts/rccl_allreduce_bench.py \
        --sizes-mb "${RCCL_SIZES_MB:-1,8,32,128,512}" \
        --iters "${RCCL_ITERS:-20}" \
        --warmup "${RCCL_WARMUP:-6}" \
        --dtype bf16 \
        --output-jsonl "$out" >> "$logf" 2>&1
    echo "{\"event\":\"rccl_run\",\"nproc\":${n},\"return_code\":$?,\"log\":\"${logf}\"}" >> "$JSONL"
  done
  md
  md "- jsonl: \`$out\`"
  md "- log: \`$logf\`"
  md
  md '```json'
  tail -80 "$out" >> "$MD" 2>/dev/null || true
  md '```'
}

run_power_token_j() {
  if ! have_time_for 2700; then
    log "skip power/token/J: insufficient time"
    return 0
  fi
  md
  md "## 功耗与 token/J"
  stop_vllm
  local tag="${RUN_ID}.power_bf16_tp8_auto_103k"
  local port=$((PORT_BASE + 1))
  local svc_log="$RESULT_DIR/${tag}.service.log"
  start_bf16_tp8 "$port" auto 131072 0.80 16384 "$svc_log"
  if wait_ready "$port" "$svc_log" 1800; then
    append_startup_summary "$svc_log"
    local sample="$RESULT_DIR/${tag}.power.jsonl"
    local summary="$RESULT_DIR/${tag}.summary.json"
    local child_out="$RESULT_DIR/${tag}.bench.stdout.jsonl"
    local child_err="$RESULT_DIR/${tag}.bench.stderr.log"
    log "running power-sampled 103K request"
    timeout 2400 python3 scripts/power_sample_bench.py \
      --sample-jsonl "$sample" \
      --summary-json "$summary" \
      --child-stdout "$child_out" \
      --child-stderr "$child_err" \
      --interval 1.0 \
      --rocm-smi rocm-smi \
      -- python3 scripts/bench_concurrency_local.py \
        --url "http://127.0.0.1:${port}/v1/chat/completions" \
        --model Qwen3.6-27B-BF16 \
        --file /workspace/bench_data/combined_papers_for_llm.txt \
        --chars 400000 \
        --max-tokens 128 \
        --requests 1 \
        --concurrency 1 >> "$LOG" 2>&1
    local rc=$?
    echo "{\"event\":\"power_token_j\",\"return_code\":${rc},\"summary\":\"${summary}\"}" >> "$JSONL"
    md
    md "- summary: \`$summary\`"
    md
    md '```json'
    cat "$summary" >> "$MD" 2>/dev/null || true
    md '```'
  else
    log "power service failed to start"
    append_startup_summary "$svc_log"
  fi
  stop_vllm
}

run_cache_probe() {
  if ! have_time_for 3000; then
    log "skip cache probe: insufficient time"
    return 0
  fi
  md
  md "## Triton / torch.compile cache 持久化启动探针"
  stop_vllm
  local cache_root="$RESULT_DIR/${RUN_ID}.isolated_cache"
  mkdir -p "$cache_root"
  local port=$((PORT_BASE + 2))
  for pass in cold warm; do
    local tag="${RUN_ID}.cache_${pass}"
    local svc_log="$RESULT_DIR/${tag}.service.log"
    log "cache probe ${pass}"
    XDG_CACHE_HOME="$cache_root" VLLM_CACHE_ROOT="$cache_root/vllm" \
      HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      PORT="$port" \
      MODEL=/models/Qwen3.6-27B \
      SERVED_MODEL_NAME=Qwen3.6-27B-BF16 \
      GPU_MEMORY_UTILIZATION=0.75 \
      MAX_MODEL_LEN=24576 \
      MAX_NUM_BATCHED_TOKENS=8192 \
      MAX_NUM_SEQS=8 \
      KV_CACHE_DTYPE=auto \
      ATTENTION_BACKEND=TRITON_ATTN \
      VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16 \
      LOG="$svc_log" \
      bash scripts/start_bf16_tp8_long_w7900.sh > "${svc_log}.outer" 2>&1 &
    if wait_ready "$port" "$svc_log" 1500; then
      append_startup_summary "$svc_log"
    else
      append_startup_summary "$svc_log"
    fi
    stop_vllm
  done
  echo "{\"event\":\"cache_probe\",\"cache_root\":\"${cache_root}\"}" >> "$JSONL"
}

record_header() {
  : > "$MD"
  : > "$JSONL"
  md "# W7900 overnight exploration"
  md
  md "- RUN_ID: \`$RUN_ID\`"
  md "- start: $(TZ=Asia/Shanghai date '+%F %T BJT')"
  md "- stop target: \`$TARGET_BJT BJT\`"
  md "- root: \`$ROOT\`"
  md "- worktree: \`$WORKTREE\`"
  md
  md "## 环境快照"
  md
  md '```text'
  {
    date
    rocm-smi --showproductname --showdriverversion 2>/dev/null || true
    rocm-smi --showmeminfo vram 2>/dev/null | head -120 || true
    python3 - <<'PY'
import torch, vllm
print("torch", torch.__version__)
print("vllm", getattr(vllm, "__version__", "unknown"))
print("cuda/hip available", torch.cuda.is_available(), torch.version.hip)
PY
  } >> "$MD" 2>&1
  md '```'
}

main() {
  record_header
  log "overnight exploration started; target=${TARGET_BJT} BJT"
  wait_current_longdoc || true
  append_latest_longdoc_summary
  run_awq_mmq_large_m
  run_rccl_curve
  run_power_token_j
  run_cache_probe
  stop_vllm
  md
  md "## 结束"
  md
  md "- end: $(TZ=Asia/Shanghai date '+%F %T BJT')"
  md "- remaining_s_at_end: $(time_left_s)"
  log "overnight exploration finished"
}

main "$@"
