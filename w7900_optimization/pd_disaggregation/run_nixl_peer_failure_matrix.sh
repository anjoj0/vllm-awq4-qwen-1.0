#!/usr/bin/env bash

set -euo pipefail

PD_ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
UCX_ROOT="${UCX_ROOT:-${PD_ROOT}/deps/ucx-rocm-peer-flag}"
TEST="${TEST:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/nixl_rocm_peer_failure_test.py}"
OUT="${OUT:-/workspace/nixl_peer_failure_$(date +%Y%m%d_%H%M%S)}"
ERROR_HANDLING="${ERROR_HANDLING:-peer}"
PROCESS_TIMEOUT="${PROCESS_TIMEOUT:-35}"
CASE_SET="${CASE_SET:-all}"
TARGET_DEVICE="${TARGET_DEVICE:-0}"
INITIATOR_DEVICE="${INITIATOR_DEVICE:-1}"

source /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/activate_pd_env.sh
export UCX_TLS="${UCX_TLS:-sm,rocm,tcp,self}"
export UCX_PROTO_INFO=y
export UCX_RMA_PPLN_ENABLE=y
mkdir -p "${OUT}"

run_case() {
    local fault="$1"
    local operation="$2"
    local bytes="$3"
    local port="$4"
    local name="${fault}_${operation}"
    local target_log="${OUT}/${name}_target.log"
    local initiator_log="${OUT}/${name}_initiator.log"

    echo "CASE fault=${fault} operation=${operation} bytes=${bytes} error_handling=${ERROR_HANDLING} target_gpu=${TARGET_DEVICE} initiator_gpu=${INITIATOR_DEVICE}"
    set +e
    timeout "${PROCESS_TIMEOUT}s" env HIP_VISIBLE_DEVICES="${TARGET_DEVICE}" python "${TEST}" \
        --role target --fault "${fault}" --operation "${operation}" \
        --bytes "${bytes}" --port "${port}" \
        --ucx-error-handling "${ERROR_HANDLING}" >"${target_log}" 2>&1 &
    local target_pid=$!
    sleep 2
    timeout "${PROCESS_TIMEOUT}s" env HIP_VISIBLE_DEVICES="${INITIATOR_DEVICE}" python "${TEST}" \
        --role initiator --fault "${fault}" --operation "${operation}" \
        --bytes "${bytes}" --port "${port}" \
        --ucx-error-handling "${ERROR_HANDLING}" >"${initiator_log}" 2>&1
    local initiator_rc=$?
    wait "${target_pid}"
    local target_rc=$?
    set -e

    echo "RETURN target=${target_rc} initiator=${initiator_rc}"
    grep -E '\{"bytes"|\{"role"|UCX  ERROR|failed|fatal|TIMEOUT' \
        "${target_log}" "${initiator_log}" || true
}

gib=$((1024 * 1024 * 1024))
case "${CASE_SET}" in
    all)
        run_case normal READ "${gib}" 5600
        run_case exit_before_transfer READ "${gib}" 5601
        run_case clean_exit_before_transfer READ "${gib}" 5602
        run_case stale_registration READ "${gib}" 5603
        run_case exit_after_post READ "$((8 * gib))" 5604
        run_case exit_after_post WRITE "$((8 * gib))" 5605
        ;;
    normal_READ) run_case normal READ "${gib}" 5600 ;;
    exit_before_transfer_READ)
        run_case exit_before_transfer READ "${gib}" 5601
        ;;
    clean_exit_before_transfer_READ)
        run_case clean_exit_before_transfer READ "${gib}" 5602
        ;;
    stale_registration_READ)
        run_case stale_registration READ "${gib}" 5603
        ;;
    exit_after_post_READ)
        run_case exit_after_post READ "$((8 * gib))" 5604
        ;;
    exit_after_post_WRITE)
        run_case exit_after_post WRITE "$((8 * gib))" 5605
        ;;
    *)
        echo "Unknown CASE_SET=${CASE_SET}" >&2
        exit 2
        ;;
esac

printf 'RESULT_DIR=%s\n' "${OUT}"
