#!/usr/bin/env bash
set -euo pipefail

# Run the recommended quality checks against one already-running vLLM service.
# Restart the service with another model/KV/TP profile, change CONFIG_LABEL, and
# run this script again. The comparison tool joins the resulting directories.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8030}
MODEL=${MODEL:-Qwen3.6-27B-BF16}
TOKENIZER=${TOKENIZER:-/models/Qwen3.6-27B}
CORPUS=${CORPUS:-/workspace/bench_data/combined_papers_for_llm_L.txt}
CONFIG_LABEL=${CONFIG_LABEL:?Set CONFIG_LABEL, for example bf16_tp8_auto}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/results}

common=(
  --base-url "$BASE_URL"
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --corpus "$CORPUS"
  --config-label "$CONFIG_LABEL"
  --output-root "$OUTPUT_ROOT"
)

python3 "$SCRIPT_DIR/validate_suite.py" --corpus "$CORPUS"

# Controlled evidence retrieval plus four positional needles.
python3 "$SCRIPT_DIR/run_longdoc_sanity.py" \
  "${common[@]}" --profile 103k --suite core --context-mode evidence

# Full-paper comprehension smoke, without duplicate needle requests.
python3 "$SCRIPT_DIR/run_longdoc_sanity.py" \
  "${common[@]}" --profile 103k --suite smoke --context-mode full-paper --no-needles

# Four positional needles close to the model's native context limit.
python3 "$SCRIPT_DIR/run_longdoc_sanity.py" \
  "${common[@]}" --profile near256k --suite needle --context-mode evidence

echo "Completed quality sanity for $CONFIG_LABEL"
echo "Results: $OUTPUT_ROOT"
