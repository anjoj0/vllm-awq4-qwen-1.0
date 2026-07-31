#!/usr/bin/env bash
set -uo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-20260615-dflash-n8-systemd}"
WAIT_FOR_UNIT="${WAIT_FOR_UNIT:-codex-accuracy-full-20260615.service}"
HOST="${VLLM_EVAL_HOST:-http://127.0.0.1:8001}"
MODEL="${VLLM_EVAL_MODEL:-Qwen3.6-27B-AWQ4}"
TOKENIZER="${VLLM_EVAL_TOKENIZER:-cyankiwi/Qwen3.6-27B-AWQ-INT4}"
LMEVAL="$ROOT/.venv-lm-eval-py313/bin/lm_eval"
PYTHON="$ROOT/.venv-lm-eval-py313/bin/python"
OUT_ROOT="$ROOT/test/results/accuracy/full_${RUN_ID}"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
# httpx rejects socks:// proxies. Force an HTTP proxy for dataset downloads
# while NO_PROXY keeps local vLLM API calls on loopback.
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
echo "Wait unit: $WAIT_FOR_UNIT"
echo "Host: $HOST"
echo "Model: $MODEL"
echo "Tokenizer: $TOKENIZER"
echo "Output: $OUT_ROOT"
echo

if command -v systemctl >/dev/null 2>&1; then
  while systemctl --user is-active --quiet "$WAIT_FOR_UNIT"; do
    echo "Waiting for $WAIT_FOR_UNIT to finish before starting remaining tasks..."
    date
    sleep 300
  done
fi

python3 -c "import urllib.request; print('health', urllib.request.urlopen('${HOST}/health', timeout=10).status)"

echo "=== ARC Challenge full, 25-shot, greedy loglikelihood ==="
date
"$LMEVAL" run \
  --model local-completions \
  --model_args "$MODEL_ARGS" \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUT_ROOT/arc_challenge_dflash_n8_full" \
  2>&1 | tee "$LOG_DIR/arc_challenge.log"
echo "ARC Challenge exit status: ${PIPESTATUS[0]}"
date

echo "=== HumanEval full, pass@1, greedy generation ==="
date
"$LMEVAL" run \
  --model local-completions \
  --model_args "$MODEL_ARGS" \
  --tasks humaneval \
  --num_fewshot 0 \
  --batch_size 1 \
  --gen_kwargs temperature=0,max_gen_toks=1024 \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path "$OUT_ROOT/humaneval_dflash_n8_full" \
  2>&1 | tee "$LOG_DIR/humaneval.log"
echo "HumanEval exit status: ${PIPESTATUS[0]}"
date

echo "=== MT-Bench answer generation, greedy chat completions ==="
date
"$PYTHON" test/generate_chat_eval_answers.py \
  --suite mt_bench \
  --host "$HOST" \
  --model "$MODEL" \
  --max-tokens 1024 \
  --out-dir "$OUT_ROOT/mt_bench" \
  2>&1 | tee "$LOG_DIR/mt_bench_generate.log"
echo "MT-Bench answer generation exit status: ${PIPESTATUS[0]}"
date

echo "=== Summarize available accuracy results ==="
"$PYTHON" test/summarize_accuracy_results.py \
  --run-root "$OUT_ROOT" \
  --out "$OUT_ROOT/accuracy_summary.md" \
  2>&1 | tee "$LOG_DIR/summarize.log"

echo "=== REMAINING DONE ==="
echo "$OUT_ROOT"
