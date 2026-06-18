#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-20260616-bf16-systemd}"
WAIT_UNIT="${WAIT_UNIT:-codex-accuracy-bf16-remaining-20260616.service}"
OUT_ROOT="$ROOT/test/results/accuracy/full_${RUN_ID}"
MT_DIR="$OUT_ROOT/mt_bench"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/deepseek_mtbench_judge_waiter.log"

mkdir -p "$LOG_DIR" "$MT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== BF16 MT-Bench DeepSeek judge waiter ==="
date
echo "run_id=$RUN_ID"
echo "wait_unit=$WAIT_UNIT"
echo "mt_dir=$MT_DIR"
echo "log_file=$LOG_FILE"

export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7890/}"
export all_proxy="${all_proxy:-http://127.0.0.1:7890/}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890/}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890/}"
export http_proxy="${http_proxy:-http://127.0.0.1:7890/}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890/}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.0/8,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.0/8,::1}"

find_latest_answers() {
  find "$MT_DIR" -maxdepth 1 -type f -name '*answers.jsonl' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

ANSWERS=""
while true; do
  ANSWERS="$(find_latest_answers)"
  if [[ -n "$ANSWERS" ]]; then
    lines="$(wc -l < "$ANSWERS" || echo 0)"
    echo "found answers candidate: $ANSWERS lines=$lines"
    if [[ "$lines" -ge 80 ]]; then
      break
    fi
  else
    echo "answers not found yet"
  fi

  if systemctl --user is-active --quiet "$WAIT_UNIT"; then
    echo "$WAIT_UNIT is still active; waiting..."
  else
    echo "$WAIT_UNIT is not active but answers are not complete yet; waiting..."
  fi
  date
  sleep 300
done

while [[ -z "${DEEPSEEK_API_KEY:-}" ]]; do
  env_line="$(systemctl --user show-environment | grep '^DEEPSEEK_API_KEY=' || true)"
  if [[ -n "$env_line" ]]; then
    export "$env_line"
    break
  fi
  echo "DEEPSEEK_API_KEY is not in this service or systemd user environment."
  echo "Run this in the terminal where you exported the key:"
  echo "  systemctl --user import-environment DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_JUDGE_MODEL"
  date
  sleep 300
done

if [[ -z "${DEEPSEEK_BASE_URL:-}" ]]; then
  env_line="$(systemctl --user show-environment | grep '^DEEPSEEK_BASE_URL=' || true)"
  [[ -n "$env_line" ]] && export "$env_line"
fi
if [[ -z "${DEEPSEEK_JUDGE_MODEL:-}" ]]; then
  env_line="$(systemctl --user show-environment | grep '^DEEPSEEK_JUDGE_MODEL=' || true)"
  [[ -n "$env_line" ]] && export "$env_line"
fi

echo "DeepSeek env detected:"
echo "  DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
echo "  DEEPSEEK_JUDGE_MODEL=${DEEPSEEK_JUDGE_MODEL:-deepseek-v4-flash}"
echo "  DEEPSEEK_API_KEY=<hidden>"

SMOKE_OUT="$MT_DIR/deepseek_v4_flash_judge_scores_smoke.jsonl"
SMOKE_SUMMARY="$MT_DIR/deepseek_v4_flash_judge_summary_smoke.json"
FULL_OUT="$MT_DIR/deepseek_v4_flash_judge_scores.jsonl"
FULL_SUMMARY="$MT_DIR/deepseek_v4_flash_judge_summary.json"

echo "=== DeepSeek judge smoke test ==="
python3 test/score_mt_bench_deepseek_judge.py \
  --answers "$ANSWERS" \
  --out "$SMOKE_OUT" \
  --summary "$SMOKE_SUMMARY" \
  --limit 1

python3 - "$SMOKE_SUMMARY" <<'PY'
import json
import sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
if data.get("valid_count") != 1:
    print(f"smoke judge failed: {data}", file=sys.stderr)
    raise SystemExit(1)
print("smoke judge ok")
PY

echo "=== DeepSeek judge full MT-Bench ==="
python3 test/score_mt_bench_deepseek_judge.py \
  --answers "$ANSWERS" \
  --out "$FULL_OUT" \
  --summary "$FULL_SUMMARY"

echo "=== DeepSeek judge done ==="
date
echo "answers=$ANSWERS"
echo "scores=$FULL_OUT"
echo "summary=$FULL_SUMMARY"
