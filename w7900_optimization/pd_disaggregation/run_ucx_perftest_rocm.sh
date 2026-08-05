#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
UCX_ROOT="${UCX_ROOT:-${PD_ROOT}/deps/ucx-rocm}"
PERFTEST="${PERFTEST:-${UCX_ROOT}/bin/ucx_perftest}"
OUT="${OUT:-/workspace/ucx_perftest_rocm_$(date +%Y%m%d_%H%M%S)}"
TLS="${TLS:-sm,rocm,tcp,self}"
SIZE="${SIZE:-67108864}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-1}"

export LD_LIBRARY_PATH="${UCX_ROOT}/lib:${UCX_ROOT}/lib/ucx:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${TLS}"
export UCX_PROTO_INFO=y
export UCX_RMA_PPLN_ENABLE="${UCX_RMA_PPLN_ENABLE:-y}"
mkdir -p "${OUT}"

run_case() {
    local test="$1"
    local port="$2"
    local server_log="${OUT}/${test}_server.log"
    local client_log="${OUT}/${test}_client.log"
    local args=(-t "${test}" -m rocm -s "${SIZE}" -n "${ITERATIONS}" \
                -w "${WARMUP}" -p "${port}" -f -X)

    echo "CASE test=${test} size=${SIZE} iterations=${ITERATIONS} tls=${UCX_TLS}"
    ROCR_VISIBLE_DEVICES=0,1 \
        "${PERFTEST}" "${args[@]}" >"${server_log}" 2>&1 &
    local server_pid=$!
    sleep 2

    set +e
    ROCR_VISIBLE_DEVICES=1,0 \
        "${PERFTEST}" 127.0.0.1 "${args[@]}" \
        >"${client_log}" 2>&1
    local client_rc=$?
    wait "${server_pid}"
    local server_rc=$?
    set -e

    echo "RETURN client=${client_rc} server=${server_rc}"
    grep -E 'rocm_ipc|remote memory (read|write)|Final|Bandwidth|ERROR|failed' \
        "${client_log}" "${server_log}" || true
    if (( client_rc != 0 || server_rc != 0 )); then
        return 1
    fi
}

run_case ucp_get 5580
run_case ucp_put_bw 5581

printf 'RESULT_DIR=%s\n' "${OUT}"
