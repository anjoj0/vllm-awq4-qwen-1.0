#!/usr/bin/env bash
set -uo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
HOST="${VLLM_EVAL_HOST:-http://127.0.0.1:8001}"
MODEL="${VLLM_EVAL_MODEL:-Qwen3.6-27B-AWQ4}"
TOKENIZER="${VLLM_EVAL_TOKENIZER:-/home/xqhpc/data/AI_project/hf-cache/hub/models--cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/c9b937c5466c5c0575fc15edd1f8c516cb1e62fd}"
LMEVAL="$ROOT/.venv-lm-eval-py313/bin/lm_eval"
OUT_ROOT="$ROOT/test/results/accuracy/full_${RUN_ID}"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
# httpx rejects the user default socks:// ALL_PROXY. Force HTTP proxy for
# dataset downloads while NO_PROXY keeps local vLLM traffic on loopback.
export ALL_PROXY="http://127.0.0.1:7890/"
export all_proxy="http://127.0.0.1:7890/"
export HTTP_PROXY="http://127.0.0.1:7890/"
export HTTPS_PROXY="http://127.0.0.1:7890/"
export http_proxy="http://127.0.0.1:7890/"
export https_proxy="http://127.0.0.1:7890/"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.0/8,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.0/8,::1}"

MODEL_ARGS="model=${MODEL},base_url=${HOST}/v1/completions,tokenizer_backend=huggingface,tokenizer=${TOKENIZER},tokenized_requests=False,num_concurrent=1,max_retries=3,timeout=300,max_length=262144"

echo "Run ID: $RUN_ID"
echo "Host: $HOST"
echo "Model: $MODEL"
echo "Tokenizer: $TOKENIZER"
echo "Output: $OUT_ROOT"
echo

python3 -c "import urllib.request; print('health', urllib.request.urlopen('${HOST}/health', timeout=10).status)"

echo "=== GSM8K full, 5-shot, greedy ==="
date
"$LMEVAL" run \
  --model local-completions \
  --model_args "$MODEL_ARGS" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 1 \
  --gen_kwargs temperature=0,max_gen_toks=512 \
  --log_samples \
  --output_path "$OUT_ROOT/gsm8k_dflash_n8_full" \
  2>&1 | tee "$LOG_DIR/gsm8k.log"
echo "GSM8K exit status: ${PIPESTATUS[0]}"
date

echo "=== HellaSwag full, 0-shot, greedy loglikelihood ==="
date
"$LMEVAL" run \
  --model local-completions \
  --model_args "$MODEL_ARGS" \
  --tasks hellaswag \
  --num_fewshot 0 \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUT_ROOT/hellaswag_dflash_n8_full" \
  2>&1 | tee "$LOG_DIR/hellaswag.log"
echo "HellaSwag exit status: ${PIPESTATUS[0]}"
date

echo "=== DONE ==="
echo "$OUT_ROOT"
