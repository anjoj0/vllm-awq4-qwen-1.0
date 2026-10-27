#!/usr/bin/env bash
# Independent environment; never changes the vLLM container or its dependencies.
set -euo pipefail
assistant_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$assistant_dir/.." && pwd)"
harness_venv="${QWEN_HARNESS_VENV:-$project_root/.runtime/harness-venv}"
if [[ ! -x "$harness_venv/bin/python" ]]; then
  python3 -m venv "$harness_venv"
fi
if [[ ! -f "$harness_venv/.requirements-installed" ]] || ! cmp -s "$assistant_dir/requirements.txt" "$harness_venv/.requirements-installed"; then
  "$harness_venv/bin/python" -m pip install --index-url https://pypi.org/simple --timeout 20 --retries 1 -r "$assistant_dir/requirements.txt"
  cp "$assistant_dir/requirements.txt" "$harness_venv/.requirements-installed"
fi
exec "$harness_venv/bin/python" "$assistant_dir/server.py" "$@"
