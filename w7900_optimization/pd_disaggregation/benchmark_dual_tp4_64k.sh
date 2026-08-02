#!/usr/bin/env bash

set -euo pipefail

ROOT="${PD_ROOT:-/workspace/pd_disagg_20260802}"
PYTHON="${ROOT}/venv/bin/python"
BENCH="${ROOT}/scripts/benchmark_pd.py"
SOURCE="${SOURCE:-/workspace/bench_data/combined_papers_for_llm.txt}"
RESULTS="${ROOT}/results"

export PYTHONPATH=/opt/python/lib/python3.14/site-packages

batch_start=$(date +%s.%N)
"${PYTHON}" "${BENCH}" \
  --url http://127.0.0.1:8100/v1/completions \
  --source "${SOURCE}" \
  --prompt-tokens 64000 \
  --max-tokens 32 \
  --concurrency 2 \
  --output "${RESULTS}/dual_tp4_group0_64k_c2.json" &
group0_pid=$!

"${PYTHON}" "${BENCH}" \
  --url http://127.0.0.1:8200/v1/completions \
  --source "${SOURCE}" \
  --prompt-tokens 64000 \
  --max-tokens 32 \
  --concurrency 2 \
  --output "${RESULTS}/dual_tp4_group1_64k_c2.json" &
group1_pid=$!

wait "${group0_pid}"
wait "${group1_pid}"
batch_end=$(date +%s.%N)

"${PYTHON}" -c \
  "print({'dual_tp4_batch_wall_s': float('${batch_end}') - float('${batch_start}')})"
