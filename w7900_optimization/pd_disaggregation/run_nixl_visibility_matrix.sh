#!/usr/bin/env bash

set -euo pipefail

BENCH="${BENCH:-/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/nixl_rocm_two_gpu_bench.py}"
OUT="${OUT:-/workspace/nixl_visibility_$(date +%Y%m%d_%H%M%S)}"
BYTES="${BYTES:-67108864}"
ITERATIONS="${ITERATIONS:-3}"
TLS="${TLS:-sm,rocm,tcp,self}"
PIPELINE="${PIPELINE:-y}"
ERROR_HANDLING="${ERROR_HANDLING:-peer}"
CASE_SET="${CASE_SET:-all}"

source /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/activate_pd_env.sh
mkdir -p "${OUT}"

export UCX_TLS="${TLS}"
export UCX_PROTO_INFO=y
if [[ "${PIPELINE}" == "y" ]]; then
    export UCX_RMA_PPLN_ENABLE=y
else
    unset UCX_RMA_PPLN_ENABLE || true
fi

launch_for_role() {
    local mode="$1"
    local role="$2"
    shift 2

    case "${mode}:${role}" in
        hip_single:target)
            env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0 "$@"
            ;;
        hip_single:initiator)
            env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 "$@"
            ;;
        rocr_single:target)
            env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 "$@"
            ;;
        rocr_single:initiator)
            env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 "$@"
            ;;
        rocr_ordered:target)
            env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1 "$@"
            ;;
        rocr_ordered:initiator)
            env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1,0 "$@"
            ;;
        *)
            echo "Unknown visibility mode/role: ${mode}:${role}" >&2
            return 2
            ;;
    esac
}

run_case() {
    local mode="$1"
    local operation="$2"
    local port="$3"
    local target_log="${OUT}/${mode}_${operation}_target.log"
    local initiator_log="${OUT}/${mode}_${operation}_initiator.log"

    echo "CASE mode=${mode} operation=${operation} pipeline=${PIPELINE} error_handling=${ERROR_HANDLING} bytes=${BYTES} port=${port}"
    launch_for_role "${mode}" target python "${BENCH}" \
        --role target --port "${port}" --bytes "${BYTES}" \
        --warmup 1 --iterations "${ITERATIONS}" --operation "${operation}" \
        --ucx-error-handling "${ERROR_HANDLING}" \
        >"${target_log}" 2>&1 &
    local target_pid=$!
    sleep 2

    set +e
    launch_for_role "${mode}" initiator python "${BENCH}" \
        --role initiator --ip 127.0.0.1 --port "${port}" --bytes "${BYTES}" \
        --warmup 1 --iterations "${ITERATIONS}" --operation "${operation}" \
        --ucx-error-handling "${ERROR_HANDLING}" \
        >"${initiator_log}" 2>&1
    local initiator_rc=$?
    wait "${target_pid}"
    local target_rc=$?
    set -e

    echo "RETURN initiator=${initiator_rc} target=${target_rc}"
    grep -E '^{|rocm_ipc|cma/memory|tcp/|pipeline|remote memory (read|write)' \
        "${initiator_log}" "${target_log}" || true
    if (( initiator_rc != 0 || target_rc != 0 )); then
        return 1
    fi
}

case "${CASE_SET}" in
    all)
        run_case hip_single READ 5590
        run_case hip_single WRITE 5591
        run_case rocr_single READ 5592
        run_case rocr_single WRITE 5593
        run_case rocr_ordered READ 5594
        run_case rocr_ordered WRITE 5595
        ;;
    hip_single_READ) run_case hip_single READ 5590 ;;
    hip_single_WRITE) run_case hip_single WRITE 5591 ;;
    rocr_single_READ) run_case rocr_single READ 5592 ;;
    rocr_single_WRITE) run_case rocr_single WRITE 5593 ;;
    rocr_ordered_READ) run_case rocr_ordered READ 5594 ;;
    rocr_ordered_WRITE) run_case rocr_ordered WRITE 5595 ;;
    *)
        echo "Unknown CASE_SET=${CASE_SET}" >&2
        exit 2
        ;;
esac

printf 'RESULT_DIR=%s\n' "${OUT}"
