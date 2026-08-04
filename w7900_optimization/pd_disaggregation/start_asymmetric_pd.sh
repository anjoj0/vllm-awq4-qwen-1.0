#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
VLLM_WORKTREE="${VLLM_WORKTREE:-/workspace/vllm-main-20260801}"
PROFILE="${1:-p2_d6}"
RUN_ID="${RUN_ID:-${PROFILE}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${PD_ROOT}/runs/${RUN_ID}}"
WORKER_SCRIPT="${PD_ROOT}/scripts/start_pd_worker.sh"
PROXY_SCRIPT="${VLLM_WORKTREE}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
PROXY_PORT="${PROXY_PORT:-8192}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
STARTUP_RETRIES="${STARTUP_RETRIES:-2}"

case "${PROFILE}" in
  p2_d6)
    producer_specs=("0,1:8100:20000:5500")
    consumer_specs=("2,3:8200:30000:5600" "4,5:8201:30010:5610" "6,7:8202:30020:5620")
    ;;
  p4_d4)
    producer_specs=("0,1:8100:20000:5500" "2,3:8101:20010:5510")
    consumer_specs=("4,5:8200:30000:5600" "6,7:8201:30010:5610")
    ;;
  p6_d2)
    producer_specs=("0,1:8100:20000:5500" "2,3:8101:20010:5510" "4,5:8102:20020:5520")
    consumer_specs=("6,7:8200:30000:5600")
    ;;
  *) echo "unknown profile: ${PROFILE} (expected p2_d6, p4_d4, or p6_d2)" >&2; exit 2 ;;
esac

mkdir -p "${RUN_DIR}/logs"
: > "${RUN_DIR}/pids"
: > "${RUN_DIR}/workers.tsv"

cleanup_on_error() {
  local status=$?
  trap - ERR INT TERM
  if [[ -f "${RUN_DIR}/pids" ]]; then
    while read -r pid; do
      kill -TERM "${pid}" 2>/dev/null || true
    done < "${RUN_DIR}/pids"
    sleep 5
    while read -r pid; do
      kill -KILL "${pid}" 2>/dev/null || true
    done < "${RUN_DIR}/pids"
  fi
  exit "${status}"
}
trap cleanup_on_error ERR INT TERM

start_worker() {
  local role="$1" index="$2" spec="$3"
  local devices http_port vllm_port side_port
  IFS=':' read -r devices http_port vllm_port side_port <<< "${spec}"
  local name="${role}_${index}"
  nohup env \
    ROLE="${role}" GPU_DEVICES="${devices}" TP_SIZE=2 \
    HTTP_PORT="${http_port}" VLLM_PORT="${vllm_port}" \
    SIDE_CHANNEL_PORT="${side_port}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN:-65536}}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}" \
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}" \
    NIXL_BACKEND="${NIXL_BACKEND:-W7900_HIP_IPC}" \
    EXECUTE_MODEL_TIMEOUT_SECONDS="${EXECUTE_MODEL_TIMEOUT_SECONDS:-300}" \
    bash "${WORKER_SCRIPT}" > "${RUN_DIR}/logs/${name}.log" 2>&1 &
  local pid=$!
  LAST_PID="${pid}"
  printf '%s\n' "${pid}" >> "${RUN_DIR}/pids"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${devices}" "${http_port}" "${vllm_port}" "${side_port}" "${pid}" \
    >> "${RUN_DIR}/workers.tsv"
}

start_worker_with_retry() {
  local role="$1" index="$2" spec="$3" port="$4"
  local attempt
  for attempt in $(seq 1 "${STARTUP_RETRIES}"); do
    start_worker "${role}" "${index}" "${spec}"
    if wait_healthy "${port}" "${RUN_DIR}/logs/${role}_${index}.log"; then
      return 0
    fi
    echo "${role}_${index} startup attempt ${attempt} failed; retrying" >&2
    kill -TERM "${LAST_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${LAST_PID}" 2>/dev/null || true
  done
  echo "${role}_${index} exhausted ${STARTUP_RETRIES} startup attempts" >&2
  return 1
}

wait_healthy() {
  local port="$1" log="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "service on port ${port} failed to become healthy" >&2
      tail -100 "${log}" >&2 || true
      return 1
    fi
    sleep 2
  done
}

producer_ports=()
consumer_ports=()
for index in "${!producer_specs[@]}"; do
  IFS=':' read -r _ port _ _ <<< "${producer_specs[$index]}"
  start_worker_with_retry producer "${index}" "${producer_specs[$index]}" "${port}"
  producer_ports+=("${port}")
done
for index in "${!consumer_specs[@]}"; do
  IFS=':' read -r _ port _ _ <<< "${consumer_specs[$index]}"
  start_worker_with_retry consumer "${index}" "${consumer_specs[$index]}" "${port}"
  consumer_ports+=("${port}")
done

prefill_hosts=()
for _ in "${producer_ports[@]}"; do prefill_hosts+=(127.0.0.1); done
decode_hosts=()
for _ in "${consumer_ports[@]}"; do decode_hosts+=(127.0.0.1); done

source "${PD_ROOT}/scripts/activate_pd_env.sh"
nohup python "${PROXY_SCRIPT}" \
  --host 127.0.0.1 --port "${PROXY_PORT}" \
  --prefiller-hosts "${prefill_hosts[@]}" \
  --prefiller-ports "${producer_ports[@]}" \
  --decoder-hosts "${decode_hosts[@]}" \
  --decoder-ports "${consumer_ports[@]}" \
  > "${RUN_DIR}/logs/proxy.log" 2>&1 &
proxy_pid=$!
printf '%s\n' "${proxy_pid}" >> "${RUN_DIR}/pids"

deadline=$((SECONDS + 30))
until curl -fsS "http://127.0.0.1:${PROXY_PORT}/healthcheck" > "${RUN_DIR}/health.json"; do
  if (( SECONDS >= deadline )); then
    echo "proxy failed to become healthy" >&2
    exit 1
  fi
  sleep 1
done

cat > "${RUN_DIR}/manifest.env" <<EOF
PROFILE=${PROFILE}
RUN_ID=${RUN_ID}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN:-65536}}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
NIXL_BACKEND=${NIXL_BACKEND:-W7900_HIP_IPC}
EXECUTE_MODEL_TIMEOUT_SECONDS=${EXECUTE_MODEL_TIMEOUT_SECONDS:-300}
PROXY_PORT=${PROXY_PORT}
EOF

ln -sfn "${RUN_DIR}" "${PD_ROOT}/runs/current"
trap - ERR INT TERM
echo "${RUN_DIR}"
cat "${RUN_DIR}/health.json"
