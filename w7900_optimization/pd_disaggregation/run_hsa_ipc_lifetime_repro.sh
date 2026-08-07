#!/usr/bin/env bash

set -uo pipefail

ROOT=${ROOT:-/workspace/pd_disagg_20260802/hsa_ipc_lifetime_repro}
BIN=${BIN:-${ROOT}/hsa_ipc_lifetime_repro}
EXPORTER_GPU=${EXPORTER_GPU:-0}
IMPORTER_GPU=${IMPORTER_GPU:-4}
SIZE_BYTES=${SIZE_BYTES:-67108864}
WAIT_MS=${WAIT_MS:-5000}
RUN_TAG=${RUN_TAG:-}

run_case() {
    local mode=$1
    local socket_path="/tmp/hsa_ipc_lifetime_${mode}_$$.sock"
    local suffix=${RUN_TAG:+_${RUN_TAG}}
    local exporter_log="${ROOT}/${mode}${suffix}_exporter.log"
    local importer_log="${ROOT}/${mode}${suffix}_importer.log"

    "${BIN}" exporter "${socket_path}" "${mode}" "${EXPORTER_GPU}" \
        "${SIZE_BYTES}" >"${exporter_log}" 2>&1 &
    local exporter_pid=$!
    sleep 0.5

    timeout 20s "${BIN}" importer "${socket_path}" "${IMPORTER_GPU}" \
        "${WAIT_MS}" >"${importer_log}" 2>&1
    local importer_status=$?
    wait "${exporter_pid}"
    local exporter_status=$?

    printf 'CASE mode=%s exporter_status=%d importer_status=%d\n' \
        "${mode}" "${exporter_status}" "${importer_status}"
    printf '%s\n' "${importer_status}" \
        >"${ROOT}/${mode}${suffix}_importer.exit"
    printf '%s\n' "${exporter_status}" \
        >"${ROOT}/${mode}${suffix}_exporter.exit"
}

if (($# == 0)); then
    set -- valid stale pre_attach_free
fi

for mode in "$@"; do
    run_case "${mode}"
done
