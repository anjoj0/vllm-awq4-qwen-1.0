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
echo "VLLM_WORKTREE=$WORKTREE"
