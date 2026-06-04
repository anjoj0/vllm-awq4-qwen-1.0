#!/usr/bin/env python3
"""A/B benchmark for AWQ MMQ decode backend modes.

Compares the verified TritonW4A16 decode fallback against the experimental
all-HIP path controlled by AWQ_MMQ_DECODE_BACKEND. The script intentionally
uses only stdlib so it can run on the host without project dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "http://127.0.0.1:8001"
DEFAULT_MODEL = "Qwen3.6-27B-AWQ4"
DEFAULT_CONTAINER = "vllm-awq4-qwen"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPT_SEED = (
    "ROCm vLLM AWQ INT4 Strix Halo gfx1151 custom HIP kernel DFlash speculative "
    "decoding benchmark latency throughput memory KV cache Radeon local inference. "
)

CASES = [
    {"name": "short_decode_128", "repeat": 8, "max_tokens": 128},
    {"name": "mid_prefill_512_decode_64", "repeat": 64, "max_tokens": 64},
    {"name": "long_prefill_2k_decode_32", "repeat": 256, "max_tokens": 32},
]


def http_json(method: str, url: str, body: dict | None = None, timeout: int = 600) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(host: str, max_wait: int = 900) -> None:
    deadline = time.time() + max_wait
    last = None
    while time.time() < deadline:
        try:
            http_json("GET", f"{host}/v1/models", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not ready after {max_wait}s: {last}")


def run_cmd(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"ERROR: {type(exc).__name__}: {exc}"


def docker_logs(container: str) -> str:
    return run_cmd(["sg", "docker", "-c", f"docker logs --tail 600 {container}"], timeout=30)


def parse_latest_startup(log: str) -> dict:
    info: dict[str, object] = {}
    matches = re.findall(r"Patch 16: ([^\n]+)", log)
    if matches:
        info["patch16_last"] = matches[-1].strip()
    m = re.findall(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", log)
    if m:
        info["model_memory_gib"] = float(m[-1][0])
        info["model_load_seconds"] = float(m[-1][1])
    m = re.findall(r"Available KV cache memory: ([0-9.]+) GiB", log)
    if m:
        info["kv_available_gib"] = float(m[-1])
    m = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", log)
    if m:
        info["kv_cache_tokens"] = int(m[-1].replace(",", ""))
    m = re.findall(r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", log)
    if m:
        info["max_concurrency"] = float(m[-1])
    m = re.findall(r"init engine .* took ([0-9.]+) s", log)
    if m:
        info["engine_init_seconds"] = float(m[-1])
    info["registered_count"] = log.count("Patch 16: RocmMmqQ4LinearKernel registered")
    info["failed_register_count"] = log.count("Patch 16: failed to register")
    return info


def sysfs_memory() -> dict:
    # Sysfs VRAM/GTT reads can block on some ROCm/iGPU setups under load.
    # Keep the request benchmark deterministic; use docker/vLLM startup logs
    # for memory in this A/B script.
    return {}


def run_case(host: str, model: str, case: dict, run_idx: int) -> dict:
    prompt = (
        "请用中文概括以下 ROCm 推理优化上下文，并保持回答简短。\n\n"
        + PROMPT_SEED * case["repeat"]
    )
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": case["max_tokens"],
        "temperature": 0,
    }
    t0 = time.perf_counter()
    try:
        payload = http_json("POST", f"{host}/v1/completions", body, timeout=900)
        wall = time.perf_counter() - t0
        usage = payload.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        text = (payload.get("choices") or [{}])[0].get("text") or ""
        return {
            "case": case["name"],
            "run": run_idx,
            "ok": True,
            "wall_seconds": round(wall, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens") or prompt_tokens + completion_tokens,
            "output_tokens_per_second_e2e": round(completion_tokens / wall, 3) if wall else 0,
            "total_tokens_per_second_e2e": round((prompt_tokens + completion_tokens) / wall, 3) if wall else 0,
            "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
            "text_preview": text[:160],
        }
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")[:1000]
        return {"case": case["name"], "run": run_idx, "ok": False, "error": f"HTTP {exc.code}: {err}"}
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"case": case["name"], "run": run_idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def summarize(results: list[dict]) -> dict:
    by_case: dict[str, list[dict]] = {}
    for r in results:
        by_case.setdefault(r["case"], []).append(r)
    summary = {}
    for case, rows in by_case.items():
        ok = [r for r in rows if r.get("ok")]
        if not ok:
            summary[case] = {"ok_runs": 0, "errors": [r.get("error") for r in rows]}
            continue
        def avg(key: str) -> float:
            vals = [float(r[key]) for r in ok if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0
        summary[case] = {
            "ok_runs": len(ok),
            "avg_wall_seconds": avg("wall_seconds"),
            "avg_prompt_tokens": avg("prompt_tokens"),
            "avg_completion_tokens": avg("completion_tokens"),
            "avg_output_tps_e2e": avg("output_tokens_per_second_e2e"),
            "avg_total_tps_e2e": avg("total_tokens_per_second_e2e"),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out-dir", default="test/results/decode_backend")
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    if not args.skip_wait:
        wait_ready(args.host)

    print(f"[{args.label}] collecting startup log", flush=True)
    startup_log = docker_logs(args.container)
    started_at = time.strftime("%Y%m%d-%H%M%S")
    output = {
        "label": args.label,
        "host": args.host,
        "model": args.model,
        "timestamp": started_at,
        "env": {
            "AWQ_MMQ_DECODE_BACKEND": os.environ.get("AWQ_MMQ_DECODE_BACKEND"),
            "AWQ_MMQ_HIP_DECODE_VERSION": os.environ.get("AWQ_MMQ_HIP_DECODE_VERSION"),
            "AWQ_MMQ_DECODE_POLICY": os.environ.get("AWQ_MMQ_DECODE_POLICY"),
            "AWQ_MMQ_SMALL_M_THRESHOLD": os.environ.get("AWQ_MMQ_SMALL_M_THRESHOLD"),
            "AWQ_MMQ_HYBRID_LONG_PREFILL_THRESHOLD": os.environ.get("AWQ_MMQ_HYBRID_LONG_PREFILL_THRESHOLD"),
            "AWQ_MMQ_HYBRID_VERIFY_M": os.environ.get("AWQ_MMQ_HYBRID_VERIFY_M"),
            "AWQ_MMQ_HYBRID_HIP_VERSION": os.environ.get("AWQ_MMQ_HYBRID_HIP_VERSION"),
            "AWQ_MMQ_SHAPE_STATS": os.environ.get("AWQ_MMQ_SHAPE_STATS"),
            "AWQ_MMQ_SHAPE_STATS_INTERVAL": os.environ.get("AWQ_MMQ_SHAPE_STATS_INTERVAL"),
            "AWQ_MMQ_SHAPE_STATS_PATH": os.environ.get("AWQ_MMQ_SHAPE_STATS_PATH"),
            "AWQ_MMQ_WEIGHT_STATS": os.environ.get("AWQ_MMQ_WEIGHT_STATS"),
            "AWQ_MMQ_WEIGHT_STATS_PATH": os.environ.get("AWQ_MMQ_WEIGHT_STATS_PATH"),
            "VLLM_GPU_MEMORY_UTIL": os.environ.get("VLLM_GPU_MEMORY_UTIL"),
            "VLLM_MAX_MODEL_LEN": os.environ.get("VLLM_MAX_MODEL_LEN"),
            "VLLM_MAX_NUM_BATCHED_TOKENS": os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS"),
            "VLLM_MAX_NUM_SEQS": os.environ.get("VLLM_MAX_NUM_SEQS"),
            "VLLM_DFLASH_N": os.environ.get("VLLM_DFLASH_N"),
        },
        "startup": parse_latest_startup(startup_log),
        "memory_before": sysfs_memory(),
        "cases": [],
    }

    for case in CASES:
        for i in range(1, args.runs + 1):
            print(f"[{args.label}] {case['name']} run {i}/{args.runs}", flush=True)
            output["cases"].append(run_case(args.host, args.model, case, i))
            time.sleep(1)

    output["memory_after"] = sysfs_memory()
    output["summary"] = summarize(output["cases"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{started_at}_{args.label}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
