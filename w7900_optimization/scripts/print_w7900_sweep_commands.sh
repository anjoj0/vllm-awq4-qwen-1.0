#!/usr/bin/env bash
set -euo pipefail

W7900_COMPOSE="${W7900_COMPOSE:-w7900_optimization/docker-compose.w7900.build.yml}"
PORT="${VLLM_HOST_PORT:-8001}"

cat <<EOF
# Run from the repository root.

# 0. Hardware smoke
DOCKER_BIN="sudo docker" HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash w7900_optimization/scripts/check_w7900_node.sh

# 1. Build W7900/gfx1100 image
sudo docker compose -f ${W7900_COMPOSE} build

# 2. Single-GPU baseline, DFlash enabled
W7900_VISIBLE_DEVICES=0 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_HOST_PORT=${PORT} \
VLLM_MAX_MODEL_LEN=65536 \
VLLM_KV_CACHE_DTYPE=fp8 \
sudo docker compose -f ${W7900_COMPOSE} up -d --force-recreate

# 3. TP=2 long-context candidate
W7900_VISIBLE_DEVICES=0,1 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
VLLM_HOST_PORT=${PORT} \
VLLM_MAX_MODEL_LEN=131072 \
VLLM_KV_CACHE_DTYPE=fp8 \
VLLM_MAX_NUM_BATCHED_TOKENS=16384 \
sudo docker compose -f ${W7900_COMPOSE} up -d --force-recreate

# 4. TP=4 capacity candidate
W7900_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TENSOR_PARALLEL_SIZE=4 \
VLLM_HOST_PORT=${PORT} \
VLLM_MAX_MODEL_LEN=262144 \
VLLM_KV_CACHE_DTYPE=fp8 \
VLLM_MAX_NUM_BATCHED_TOKENS=16384 \
sudo docker compose -f ${W7900_COMPOSE} up -d --force-recreate

# 5. TP=8 experiment only
W7900_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_TENSOR_PARALLEL_SIZE=8 \
VLLM_HOST_PORT=${PORT} \
VLLM_MAX_MODEL_LEN=262144 \
VLLM_KV_CACHE_DTYPE=fp8 \
VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
sudo docker compose -f ${W7900_COMPOSE} up -d --force-recreate
EOF
