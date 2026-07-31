#!/usr/bin/env bash
set -u

RUN_ID=${RUN_ID:-20260730_overnight_w7900_exploration}
TARGET_BJT=${TARGET_BJT:-"2026-07-30 08:00:00"}
INTERVAL_S=${INTERVAL_S:-1800}
CONTAINER=${CONTAINER:-xdhpc}
LOG=${LOG:-/root/${RUN_ID}.host_guard.log}
BASELINE=${BASELINE:-/root/${RUN_ID}.who.baseline}

target_epoch() {
  TZ=Asia/Shanghai date -d "$TARGET_BJT" +%s
}

now_epoch() {
  TZ=Asia/Shanghai date +%s
}

log() {
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] $*" | tee -a "$LOG"
}

stop_container_work() {
  local reason="$1"
  log "STOP: ${reason}"
  docker exec "$CONTAINER" bash -lc '
    pkill -TERM -f "run_overnight_w7900_exploration|run_longdoc_sanity|bench_concurrency_local|power_sample_bench|rccl_allreduce_bench|benchmark_prefill_gfx1100|torchrun|vllm serve|vllm.entrypoints.openai|VLLM::EngineCore|VLLM::Worker_TP" 2>/dev/null || true
    sleep 15
    pkill -KILL -f "run_overnight_w7900_exploration|run_longdoc_sanity|bench_concurrency_local|power_sample_bench|rccl_allreduce_bench|benchmark_prefill_gfx1100|torchrun|vllm serve|vllm.entrypoints.openai|VLLM::EngineCore|VLLM::Worker_TP" 2>/dev/null || true
  ' >> "$LOG" 2>&1 || true
}

snapshot() {
  {
    echo
    echo "===== $(date -Is) ====="
    echo "[who -u]"
    who -u || true
    echo "[docker ps]"
    docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}' || true
    echo "[rocm-smi use/mem]"
    /opt/rocm/bin/rocm-smi --showuse --showmemuse 2>/dev/null || true
    echo "[non-root user procs]"
    ps -eo user,pid,stat,etime,cmd --sort=etime | egrep -v '^(root|USER)' | tail -80 || true
  } >> "$LOG" 2>&1
}

new_nonroot_who_lines() {
  local current
  current=$(mktemp)
  who -u | awk '$1 != "root" {print}' > "$current" || true
  if [ ! -f "$BASELINE" ]; then
    cp "$current" "$BASELINE"
    rm -f "$current"
    return 1
  fi
  comm -13 <(sort "$BASELINE") <(sort "$current") | sed '/^$/d'
  rm -f "$current"
}

new_running_containers() {
  docker ps --format '{{.Names}}' | awk -v keep="$CONTAINER" '$0 != keep {print}'
}

main() {
  mkdir -p "$(dirname "$LOG")"
  who -u | awk '$1 != "root" {print}' > "$BASELINE" || true
  log "host guard started; target=${TARGET_BJT} BJT interval=${INTERVAL_S}s"
  snapshot
  while true; do
    if [ "$(now_epoch)" -ge "$(target_epoch)" ]; then
      stop_container_work "target time reached (${TARGET_BJT} BJT)"
      exit 0
    fi

    local new_who
    new_who=$(new_nonroot_who_lines || true)
    if [ -n "$new_who" ]; then
      log "new non-root login detected:"
      echo "$new_who" >> "$LOG"
      stop_container_work "new non-root login"
      exit 0
    fi

    local containers
    containers=$(new_running_containers || true)
    if [ -n "$containers" ]; then
      log "another running container detected:"
      echo "$containers" >> "$LOG"
      stop_container_work "another running container"
      exit 0
    fi

    snapshot
    local remaining sleep_s
    remaining=$(( $(target_epoch) - $(now_epoch) ))
    sleep_s="$INTERVAL_S"
    if [ "$remaining" -lt "$sleep_s" ]; then
      sleep_s="$remaining"
    fi
    if [ "$sleep_s" -gt 0 ]; then
      sleep "$sleep_s"
    fi
  done
}

main "$@"
