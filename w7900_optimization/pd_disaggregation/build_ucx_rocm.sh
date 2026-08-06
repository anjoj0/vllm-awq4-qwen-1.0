#!/usr/bin/env bash

set -euo pipefail

SRC="${SRC:-/workspace/pd_disagg_20260802/src/ucx-develop-peer-flag-20260806-v3}"
PREFIX="${PREFIX:-/workspace/pd_disagg_20260802/deps/ucx-develop-peer-flag-20260806-v3}"
ROCM_DEVEL="${ROCM_DEVEL:-/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel}"
JOBS="${JOBS:-$(nproc)}"
LOG_DIR="${LOG_DIR:-/workspace/pd_disagg_20260802/logs/ucx-build-$(date +%Y%m%d_%H%M%S)}"
ENABLE_GTEST="${ENABLE_GTEST:-y}"

mkdir -p "${LOG_DIR}"
cd "${SRC}"

bash ./autogen.sh >"${LOG_DIR}/autogen.log" 2>&1

configure_args=(
    --disable-logging
    --disable-debug
    --disable-assertions
    --disable-params-check
    --enable-mt
    --prefix="${PREFIX}"
    --enable-shared
    --disable-static
    --disable-doxygen-doc
    --enable-optimizations
    --enable-cma
    --enable-devel-headers
    --with-rocm="${ROCM_DEVEL}"
    --without-cuda
    --with-verbs
    --with-dm
)
if [[ "${ENABLE_GTEST}" == "y" ]]; then
    configure_args+=(--enable-gtest)
fi

bash ./configure "${configure_args[@]}" >"${LOG_DIR}/configure.log" 2>&1
make -j"${JOBS}" >"${LOG_DIR}/build.log" 2>&1
make install >"${LOG_DIR}/install.log" 2>&1

"${PREFIX}/bin/ucx_info" -v >"${LOG_DIR}/ucx_info_version.log" 2>&1
LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}" \
    "${PREFIX}/bin/ucx_info" -d >"${LOG_DIR}/ucx_info_devices.log" 2>&1

printf 'UCX_PREFIX=%s\nLOG_DIR=%s\n' "${PREFIX}" "${LOG_DIR}"
