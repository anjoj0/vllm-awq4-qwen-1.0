#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
BENCH="${BENCH:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/nixl_rocm_two_gpu_bench.py}"
OUT="${OUT:-/workspace/ucx_sm_$(date +%Y%m%d_%H%M%S)}"
BYTES="${BYTES:-1073741824}"
ITERATIONS="${ITERATIONS:-3}"
TLS="${TLS:-sm,rocm,tcp,self}"
CASE_SET="${CASE_SET:-all}"

source /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/activate_pd_env.sh
mkdir -p "${OUT}"

run_case() {
    local label="$1"
    local operation="$2"
    local pipeline="$3"
    local port="$4"
    local target_log="${OUT}/${label}_${operation}_target.log"
    local initiator_log="${OUT}/${label}_${operation}_initiator.log"

    export UCX_TLS="${TLS}"
    export UCX_PROTO_INFO=y
    if [[ "${pipeline}" == "y" ]]; then
        export UCX_RMA_PPLN_ENABLE=y
    else
        unset UCX_RMA_PPLN_ENABLE || true
    fi

    echo "CASE label=${label} operation=${operation} pipeline=${pipeline} bytes=${BYTES} port=${port}"
    HIP_VISIBLE_DEVICES=0 python "${BENCH}" \
        --role target --port "${port}" --bytes "${BYTES}" \
        --warmup 1 --iterations "${ITERATIONS}" --operation "${operation}" \
        >"${target_log}" 2>&1 &
    local target_pid=$!
    sleep 2

    set +e
    HIP_VISIBLE_DEVICES=1 python "${BENCH}" \
        --role initiator --ip 127.0.0.1 --port "${port}" --bytes "${BYTES}" \
        --warmup 1 --iterations "${ITERATIONS}" --operation "${operation}" \
        >"${initiator_log}" 2>&1
    local initiator_rc=$?
    wait "${target_pid}"
    local target_rc=$?
    set -e

    echo "RETURN initiator=${initiator_rc} target=${target_rc}"
    grep -E '^{|rocm_ipc|tcp/|pipeline|remote memory (read|write)' \
        "${initiator_log}" "${target_log}" || true
    if (( initiator_rc != 0 || target_rc != 0 )); then
        return 1
    fi
}

case "${CASE_SET}" in
    all)
        run_case sm_default READ n 5570
        run_case sm_default WRITE n 5571
        run_case sm_pipeline READ y 5572
        run_case sm_pipeline WRITE y 5573
        ;;
    sm_default_READ) run_case sm_default READ n 5570 ;;
    sm_default_WRITE) run_case sm_default WRITE n 5571 ;;
    sm_pipeline_READ) run_case sm_pipeline READ y 5572 ;;
    sm_pipeline_WRITE) run_case sm_pipeline WRITE y 5573 ;;
    *)
        echo "Unknown CASE_SET=${CASE_SET}" >&2
        exit 2
        ;;
esac

printf 'RESULT_DIR=%s\n' "${OUT}"
