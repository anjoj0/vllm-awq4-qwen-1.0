#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W7900_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE=${ENV_FILE:-"$W7900_DIR/.env"}
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
SOURCE=${VLLM_SOURCE:-/app/vllm}; WORKTREE=${VLLM_WORKTREE:-/workspace/vllm-w7900-023}
[[ -d "$SOURCE/vllm" ]] || { echo "Missing vLLM source: $SOURCE" >&2; exit 1; }
if [[ ! -d "$WORKTREE/vllm" ]]; then mkdir -p "$(dirname "$WORKTREE")"; cp -a "$SOURCE" "$WORKTREE"; fi
python3 "$W7900_DIR/patch_w7900.py" "$WORKTREE"
RDNA3_OVERRIDE="$W7900_DIR/vllm_overrides/rdna3_w4a16.py"
RDNA3_TARGET="$WORKTREE/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py"
[[ -f "$RDNA3_OVERRIDE" ]] || { echo "Missing RDNA3 override: $RDNA3_OVERRIDE" >&2; exit 1; }
[[ -f "$RDNA3_TARGET" ]] || { echo "Base vLLM lacks the RDNA3 W4A16 integration: $RDNA3_TARGET" >&2; exit 1; }
install -m 0644 "$RDNA3_OVERRIDE" "$RDNA3_TARGET"
grep -q 'class RDNA3W4A16LinearKernel' "$RDNA3_TARGET"
echo "VLLM_WORKTREE=$WORKTREE"
