#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization}
WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
RESULT_DIR="$ROOT/results"
RUN_ID=${RUN_ID:-20260729_0700_w7900_experiments}
RESULT_MD="$RESULT_DIR/${RUN_ID}.md"
JSONL="$RESULT_DIR/${RUN_ID}.jsonl"
SCHED_LOG="$RESULT_DIR/${RUN_ID}.scheduler.log"
TARGET_BJT=${TARGET_BJT:-"2026-07-29 07:00:00"}

mkdir -p "$RESULT_DIR"
cd "$ROOT" || exit 1

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}

log() {
  local msg="$*"
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] $msg"
}

md() {
  echo "$*" >> "$RESULT_MD"
}

wait_until_bjt() {
  local now target sleep_s
  now=$(TZ=Asia/Shanghai date +%s)
  target=$(TZ=Asia/Shanghai date -d "$TARGET_BJT" +%s)
  if [ "$now" -lt "$target" ]; then
    sleep_s=$((target - now))
    log "waiting ${sleep_s}s until ${TARGET_BJT} BJT"
    sleep "$sleep_s"
  else
    log "target time already reached; starting immediately"
  fi
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
  python3 - <<'PY' || true
import subprocess, time, re

def max_vram_percent():
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "rocm-smi --showmemuse | grep 'VRAM%' || true"],
            text=True,
        )
    except Exception:
        return None
    vals = []
    for line in out.splitlines():
        match = re.search(r"VRAM%.*:\\s*(\\d+)", line)
        if match:
            vals.append(int(match.group(1)))
    return max(vals) if vals else None

deadline = time.time() + 90
while time.time() < deadline:
    value = max_vram_percent()
    if value is None or value <= 5:
        print(f"VRAM release check: max_vram_percent={value}")
        raise SystemExit(0)
    time.sleep(5)
print(f"VRAM release check timeout: max_vram_percent={max_vram_percent()}")
PY
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
      tail -80 "$logfile" >> "$SCHED_LOG" 2>/dev/null || true
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
    grep -E "model weights take|model memory|Available KV|GPU KV cache size|Maximum concurrency|torch.compile|Graph capturing|init engine|profile|Padding mamba|Setting attention block size|max_seq_len|Using attention backend|attention backend" "$logfile" 2>/dev/null | tail -80 || true
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
  timeout 1800 python3 scripts/bench_concurrency_local.py \
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

record_environment() {
  md "# W7900 07:00 自动实验记录"
  md
  md "- 开始时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
  md "- RUN_ID：${RUN_ID}"
  md "- 目标启动时间：${TARGET_BJT} BJT"
  md "- WORKTREE：${WORKTREE}"
  md
  md "## 环境与模型边界"
  md
  {
    echo '```text'
    date
    rocm-smi --showproductname --showdriverversion 2>/dev/null || true
    rocm-smi --showmeminfo vram 2>/dev/null | head -80 || true
    python3 - <<'PY'
import json, os
for path in ["/models/Qwen3.6-27B/config.json", "/workspace/cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/master/config.json", "/workspace/z-lab--Qwen3.6-27B-DFlash/snapshots/master/config.json"]:
    print("CONFIG", path)
    cfg=json.load(open(path))
    text=cfg.get("text_config", cfg)
    keys=["model_type","num_hidden_layers","hidden_size","head_dim","num_attention_heads","num_key_value_heads","full_attention_interval","max_position_embeddings","rope_theta","rope_scaling","sliding_window","max_window_layers"]
    print({k:text.get(k) for k in keys if k in text})
PY
    python3 scripts/count_prompt_tokens.py --model /models/Qwen3.6-27B --file /workspace/bench_data/combined_papers_for_llm.txt --chars 100000 || true
    python3 scripts/count_prompt_tokens.py --model /models/Qwen3.6-27B --file /workspace/bench_data/combined_papers_for_llm.txt --chars 400000 || true
    python3 scripts/count_prompt_tokens.py --model /models/Qwen3.6-27B --file /workspace/bench_data/combined_papers_for_llm_L.txt --chars 930000 || true
    python3 scripts/count_prompt_tokens.py --model /models/Qwen3.6-27B --file /workspace/bench_data/combined_papers_for_llm_L.txt --chars 950000 || true
    python3 scripts/count_prompt_tokens.py --model /models/Qwen3.6-27B --file /workspace/bench_data/combined_papers_for_llm_L.txt --chars 970000 || true
    python3 scripts/count_prompt_tokens.py --model /models/Qwen3.6-27B --file /workspace/bench_data/combined_papers_for_llm_L.txt --chars 990000 || true
    echo '```'
  } >> "$RESULT_MD" 2>&1
}

run_mmq_microbench() {
  md
  md "## 1. AWQ4 HIP MMQ gfx1100 大 M prefill microbench"
  log "running AWQ4 HIP MMQ microbench"
  local logf="$RESULT_DIR/mmq_gfx1100_prefill_0700.log"
  (
    cd "$ROOT/csrc/awq_mmq_gfx1100" || exit 1
    HIP_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$WORKTREE:${PYTHONPATH:-}" MMQ_M_VALUES=4096,8192,16384,32768 \
      timeout 1200 python3 benchmark_prefill_gfx1100.py
  ) > "$logf" 2>&1
  local rc=$?
  {
    echo
    echo "- return_code: ${rc}"
    echo "- raw_log: ${logf}"
    echo
    echo '```text'
    tail -120 "$logf" 2>/dev/null || true
    echo '```'
  } >> "$RESULT_MD"
  echo "{\"label\":\"mmq_gfx1100_prefill_0700\",\"return_code\":${rc}}" >> "$JSONL"
}

start_bf16_tp8() {
  local port="$1"
  local util="$2"
  local backend="$3"
  local max_len="$4"
  local logf="$5"
  log "starting BF16 TP8 backend=${backend} util=${util} max_len=${max_len} port=${port}"
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PORT="$port" \
  MODEL=/models/Qwen3.6-27B \
  SERVED_MODEL_NAME=Qwen3.6-27B-BF16 \
  GPU_MEMORY_UTILIZATION="$util" \
  MAX_MODEL_LEN="$max_len" \
  MAX_NUM_BATCHED_TOKENS=16384 \
  MAX_NUM_SEQS=8 \
  KV_CACHE_DTYPE=fp8 \
  ATTENTION_BACKEND="$backend" \
  VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16 \
  LOG="$logf" \
  bash scripts/start_bf16_tp8_long_w7900.sh > "${logf}.outer" 2>&1 &
}

run_memory_sweep() {
  md
  md "## 2. BF16 TP=8 离散显存 / KV cache 扫描"
  local idx=0
  for util in 0.70 0.85 0.90; do
    stop_vllm
    local tag="bf16_tp8_tile16_util${util//./}_0700"
    local port=$((8050 + idx))
    idx=$((idx + 1))
    local logf="$RESULT_DIR/${tag}.log"
    md
    md "### gpu_memory_utilization=${util}"
    start_bf16_tp8 "$port" "$util" TRITON_ATTN 131072 "$logf"
    if wait_ready "$port" "$logf" 1800; then
      extract_startup_summary "$logf"
      bench_single "${tag}_short_2c" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 10000 64 2 2 || true
    else
      md
      md "- 启动失败或超时，详见 ${logf}"
      extract_startup_summary "$logf"
    fi
    stop_vllm
  done
}

run_rocm_attn() {
  md
  md "## 3. ROCM_ATTN vs TRITON_ATTN 对照"

  stop_vllm
  local control_tag="bf16_tp8_triton_attn_control_0700"
  local control_port=8053
  local control_logf="$RESULT_DIR/${control_tag}.log"
  md
  md "### TRITON_ATTN tile=16 control"
  start_bf16_tp8 "$control_port" 0.80 TRITON_ATTN 131072 "$control_logf"
  if wait_ready "$control_port" "$control_logf" 1800; then
    extract_startup_summary "$control_logf"
    bench_single "${control_tag}_24k" "$control_port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
    bench_single "${control_tag}_103k" "$control_port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 400000 128 1 1 || true
  else
    md
    md "- TRITON_ATTN control 启动失败或健康检查失败；保留失败日志。"
    extract_startup_summary "$control_logf"
  fi

  stop_vllm
  local tag="bf16_tp8_rocm_attn_0700"
  local port=8051
  local logf="$RESULT_DIR/${tag}.log"
  start_bf16_tp8 "$port" 0.80 ROCM_ATTN 131072 "$logf"
  if wait_ready "$port" "$logf" 1800; then
    extract_startup_summary "$logf"
    bench_single "${tag}_24k" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
    bench_single "${tag}_103k" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm.txt 400000 128 1 1 || true
  else
    md
    md "- ROCM_ATTN 未能在当前 vLLM/Qwen3.6 hybrid 配置下启动或健康检查失败；保留失败日志作为兼容性证据。"
    extract_startup_summary "$logf"
  fi
  stop_vllm
}

run_256k_boundary() {
  md
  md "## 4. BF16 TP=8 256K-class 上下文边界"
  stop_vllm
  local tag="bf16_tp8_tile16_max262k_0700"
  local port=8052
  local logf="$RESULT_DIR/${tag}.log"
  start_bf16_tp8 "$port" 0.85 TRITON_ATTN 262144 "$logf"
  if wait_ready "$port" "$logf" 2400; then
    extract_startup_summary "$logf"
    bench_single "${tag}_950kchars" "$port" Qwen3.6-27B-BF16 /workspace/bench_data/combined_papers_for_llm_L.txt 950000 64 1 1 || true
  else
    md
    md "- 262144 max_model_len 启动失败或超时；记录为 256K 边界限制。"
    extract_startup_summary "$logf"
  fi
  stop_vllm
}

start_awq4_tp4_dflash() {
  local port="$1"
  local n="$2"
  local disable="$3"
  local logf="$4"
  local name="Qwen3.6-27B-AWQ4"
  if [ "$disable" = "1" ]; then
    log "starting AWQ4 TP4 target-only port=${port}"
  else
    log "starting AWQ4 TP4 DFlash N=${n} port=${port}"
  fi
  ENV_FILE=/dev/null \
  W7900_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_HOST_PORT="$port" \
  VLLM_TARGET_MODEL=/workspace/cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/master \
  VLLM_DRAFT_MODEL=/workspace/z-lab--Qwen3.6-27B-DFlash/snapshots/master \
  VLLM_SERVED_MODEL_NAME="$name" \
  VLLM_TENSOR_PARALLEL_SIZE=4 \
  VLLM_DRAFT_TENSOR_PARALLEL_SIZE=1 \
  VLLM_GPU_MEMORY_UTIL=0.80 \
  VLLM_MAX_MODEL_LEN=32768 \
  VLLM_MAX_NUM_BATCHED_TOKENS=16384 \
  VLLM_MAX_NUM_SEQS=1 \
  VLLM_KV_CACHE_DTYPE=fp8 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  VLLM_ENFORCE_EAGER=1 \
  VLLM_DISABLE_DFLASH="$disable" \
  VLLM_DFLASH_N="$n" \
  VLLM_TRITON_ATTN_PREFILL_TILE_SIZE=16 \
  bash scripts/start_local_vllm.sh > "$logf" 2>&1 &
}

run_dflash_sweep() {
  md
  md "## 5. AWQ4 TP=4 DFlash N 敏感性"
  local port=8060

  stop_vllm
  local base_log="$RESULT_DIR/awq4_tp4_target_only_0700.log"
  start_awq4_tp4_dflash "$port" 0 1 "$base_log"
  if wait_ready "$port" "$base_log" 1200; then
    extract_startup_summary "$base_log"
    bench_single "awq4_tp4_target_only_6k" "$port" Qwen3.6-27B-AWQ4 /workspace/bench_data/combined_papers_for_llm.txt 30000 256 1 1 || true
    bench_single "awq4_tp4_target_only_24k" "$port" Qwen3.6-27B-AWQ4 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
  else
    md "- AWQ4 TP4 target-only 启动失败，跳过 DFlash N sweep。"
    stop_vllm
    return
  fi
  stop_vllm

  for n in 4 8 12; do
    local tag="awq4_tp4_dflash_n${n}_0700"
    local logf="$RESULT_DIR/${tag}.log"
    md
    md "### DFlash N=${n}"
    start_awq4_tp4_dflash "$port" "$n" 0 "$logf"
    if wait_ready "$port" "$logf" 1200; then
      extract_startup_summary "$logf"
      bench_single "${tag}_6k" "$port" Qwen3.6-27B-AWQ4 /workspace/bench_data/combined_papers_for_llm.txt 30000 256 1 1 || true
      bench_single "${tag}_24k" "$port" Qwen3.6-27B-AWQ4 /workspace/bench_data/combined_papers_for_llm.txt 100000 128 1 1 || true
      {
        echo
        echo "SpecDecoding metrics tail:"
        echo
        echo '```text'
        grep -E "SpecDecoding metrics|Mean acceptance length|Avg Draft acceptance rate|Per-position acceptance rate" "$logf" 2>/dev/null | tail -30 || true
        echo '```'
      } >> "$RESULT_MD"
    else
      md "- DFlash N=${n} 启动失败或超时，详见 ${logf}"
      extract_startup_summary "$logf"
    fi
    stop_vllm
  done
}

main() {
  exec > >(tee -a "$SCHED_LOG") 2>&1
  wait_until_bjt
  record_environment
  stop_vllm
  run_mmq_microbench
  run_memory_sweep
  run_rocm_attn
  run_256k_boundary
  run_dflash_sweep
  stop_vllm
  md
  md "## 结束"
  md
  md "- 结束时间：$(TZ=Asia/Shanghai date '+%F %T BJT')"
  if [ -f scripts/summarize_w7900_run.py ]; then
    log "generating run summary"
    python3 scripts/summarize_w7900_run.py --result-dir "$RESULT_DIR" --run-id "$RUN_ID" || true
  fi
  log "all scheduled experiments finished"
}

main "$@"
