#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
RUN_DIR="${1:-${PD_ROOT}/runs/current}"
PID_FILE="${RUN_DIR}/pids"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "no PID file at ${PID_FILE}" >&2
  exit 1
fi

mapfile -t pids < "${PID_FILE}"
for pid in "${pids[@]}"; do
  kill -TERM "${pid}" 2>/dev/null || true
done

for _ in $(seq 1 30); do
  alive=0
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && alive=1
  done
  (( alive == 0 )) && break
  sleep 1
done

for pid in "${pids[@]}"; do
  kill -KILL "${pid}" 2>/dev/null || true
done

echo "stopped services listed in ${PID_FILE}; container remains running"
