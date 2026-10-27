#!/usr/bin/env bash
# One backend at a time on the shared-memory APU. No host driver/sysctl changes.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
action="${1:-help}"
if [[ "$action" == prepare ]]; then
  shift
  exec python3 scripts/prepare_strix_halo.py "$@"
fi
backend="${2:-dflash}"
case "$backend" in
  target-only|dflash|dspark|dspark-49k|dflash2) ;;
  *) echo "Unknown backend: $backend" >&2; exit 2 ;;
esac
profile="$project_dir/.runtime/strix-halo/$backend.env"
compose=(docker compose --project-directory "$project_dir" --env-file "$profile" -f "$project_dir/docker-compose.yml")

if [[ "$action" == help || "$action" == --help ]]; then
  cat <<'USAGE'
Usage: bash scripts/strix_halo.sh ACTION [BACKEND]
Backends: target-only, dflash (default), dspark, dspark-49k, dflash2 (optional)
  prepare [--cache PATH]   Audit offline weights and write pinned profiles
  config                  Validate the selected Compose configuration
  build                   Build the pinned vLLM image (requires Docker/network)
  check                   Check actual container vLLM/ROCm/GPU architecture
  up                      Start this backend, wait for health, then smoke-test
  smoke                   Test health, model ID, chat, Responses and tool calling
  logs                    Follow this service's logs
  harness                 Start the local multimodal Agent workbench on :8088
Profiles are in .runtime/strix-halo/. Edit them to tune memory/context/port.
BACKEND selects a launch configuration; it is not a hot-switch inside vLLM.
USAGE
  exit 0
fi
if [[ ! -f "$profile" ]]; then
  echo "Missing $profile. Run: bash scripts/strix_halo.sh prepare" >&2
  exit 1
fi

case "$action" in
  config)
    "${compose[@]}" config --quiet
    ;;
  build|check|up)
    if ! docker info --format '{{.ServerVersion}}'; then
      echo "Docker daemon is inaccessible in this execution environment; no container was started." >&2
      exit 1
    fi
    "${compose[@]}" config --quiet
    if [[ "$action" == build ]]; then
      "${compose[@]}" build --pull vllm
    elif [[ "$action" == check ]]; then
      "${compose[@]}" run --rm --no-deps --pull never vllm --check-env
    else
      "${compose[@]}" up -d --no-build --pull never vllm
      bash "$0" smoke "$backend"
    fi
    ;;
  smoke|harness)
    # Read the rendered port/model rather than assuming the profile was unedited.
    runtime_json="$("${compose[@]}" config --format json)"
    host_port="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["vllm"]["ports"][0]["published"])' <<< "$runtime_json")"
    model="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["vllm"]["environment"]["VLLM_SERVED_MODEL_NAME"])' <<< "$runtime_json")"
    upstream="http://127.0.0.1:$host_port"
    if [[ "$action" == smoke ]]; then
      exec python3 test/smoke_strix.py --host "$upstream" --model "$model" \
        --label "$backend" --wait 1800
    else
      exec bash expo_assistant/run.sh --upstream "$upstream" --model "$model" \
        --subtitle "Strix Halo · AMD ROCm"
    fi
    ;;
  logs) exec "${compose[@]}" logs -f --tail=100 vllm ;;
  *) echo "Unknown action: $action. Use --help." >&2; exit 2 ;;
esac
