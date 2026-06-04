#!/usr/bin/env python3
"""Competition-oriented benchmark harness for vllm-awq4-qwen.

Focuses on report-ready measurements for the stable competition path:
cache/page policy, DFlash speculative decoding parameters, scheduler token
budget, and GPU memory utilization.

The script intentionally uses only Python stdlib. It measures API-visible
latency/tokens and can optionally parse a docker log file captured separately.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:8001"
DEFAULT_MODEL = "Qwen3.6-27B-AWQ4"
DEFAULT_PAPER_PATH = "/home/xqhpc/data/AI_project/combined_papers_for_llm.txt"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SEED = (
    "ROCm vLLM AWQ INT4 Strix Halo gfx1151 Qwen3.6 DFlash speculative decoding "
    "KV cache page policy scheduler token budget long context rainfall diffusion model. "
)

PARAM_ENV_KEYS = [
    "VLLM_GPU_MEMORY_UTIL",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_DFLASH_N",
    "AWQ_MMQ_DECODE_BACKEND",
    "AWQ_MMQ_DECODE_POLICY",
    "AWQ_MMQ_SMALL_M_THRESHOLD",
    "AWQ_ROCM_PAGED_ATTN_STATS",
]


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: int = 900) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(host: str, max_wait: int = 900) -> None:
    deadline = time.time() + max_wait
    last: Exception | None = None
    while time.time() < deadline:
        try:
            http_json("GET", f"{host}/v1/models", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not ready after {max_wait}s: {last}")


def post_stream_chat(host: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_event_t: float | None = None
    first_payload_t: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    event_count = 0
    payload_event_count = 0
    error: str | None = None
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event_count += 1
                if first_event_t is None:
                    first_event_t = time.perf_counter()
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    delta = choices[0].get("delta") or {}
                    if any(delta.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls")):
                        payload_event_count += 1
                        if first_payload_t is None:
                            first_payload_t = time.perf_counter()
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        status = None
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    ttft = None if first_event_t is None else first_event_t - t0
    payload_ttft = None if first_payload_t is None else first_payload_t - t0
    decode_den = wall - (payload_ttft or ttft or 0)
    return {
        "mode": "stream_chat",
        "status": status,
        "ok": error is None and status == 200,
        "error": error,
        "wall_seconds": wall,
        "ttft_seconds": ttft,
        "payload_ttft_seconds": payload_ttft,
        "event_count": event_count,
        "payload_event_count": payload_event_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens") or prompt_tokens + completion_tokens,
        "prefill_tokens_per_ttft": prompt_tokens / ttft if ttft and prompt_tokens else None,
        "decode_tokens_per_second_stream": completion_tokens / decode_den if decode_den > 0 and completion_tokens else None,
        "output_tokens_per_second_e2e": completion_tokens / wall if wall > 0 and completion_tokens else None,
        "total_tokens_per_second_e2e": (prompt_tokens + completion_tokens) / wall if wall > 0 and (prompt_tokens or completion_tokens) else None,
        "finish_reason": finish_reason,
    }


def post_nonstream_chat(host: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> dict[str, Any]:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    t0 = time.perf_counter()
    raw: dict[str, Any] = {}
    status = None
    error = None
    try:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with OPENER.open(req, timeout=timeout) as resp:
            status = resp.status
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    usage = raw.get("usage") or {}
    choice = (raw.get("choices") or [{}])[0] or {}
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return {
        "mode": "nonstream_chat",
        "status": status,
        "ok": error is None and status == 200,
        "error": error,
        "wall_seconds": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens") or prompt_tokens + completion_tokens,
        "output_tokens_per_second_e2e": completion_tokens / wall if wall > 0 and completion_tokens else None,
        "total_tokens_per_second_e2e": (prompt_tokens + completion_tokens) / wall if wall > 0 and (prompt_tokens or completion_tokens) else None,
        "finish_reason": choice.get("finish_reason"),
        "message_has_content": bool(((choice.get("message") or {}).get("content") or "")),
        "message_has_reasoning": bool(((choice.get("message") or {}).get("reasoning_content") or "")),
    }


def make_seed_context(repeat: int) -> str:
    return SEED * repeat


def make_paper_context(path: Path, chars: int) -> str:
    text = path.read_text(errors="ignore")
    return text[:chars]


@dataclass(frozen=True)
class BenchCase:
    name: str
    prompt: str
    max_tokens: int


def build_cases(selected: list[str], paper_path: Path) -> list[BenchCase]:
    all_cases = {
        "short_decode_128": BenchCase(
            "short_decode_128",
            "请用中文简要说明 DFlash speculative decoding 如何影响长上下文推理性能。",
            128,
        ),
        "mid_prefill_2k_decode_128": BenchCase(
            "mid_prefill_2k_decode_128",
            "请总结下面材料中与推理性能有关的要点。\n\n" + make_seed_context(96),
            128,
        ),
        "long_prefill_8k_decode_128": BenchCase(
            "long_prefill_8k_decode_128",
            "请总结下面材料中与推理性能有关的要点。\n\n" + make_seed_context(384),
            128,
        ),
        "paper_8kchars_decode_128": BenchCase(
            "paper_8kchars_decode_128",
            "下面是短临降雨扩散模型预测论文材料，请总结对推理优化最有启发的三点。\n\n" + make_paper_context(paper_path, 8000),
            128,
        ),
        "paper_32kchars_decode_128": BenchCase(
            "paper_32kchars_decode_128",
            "下面是短临降雨扩散模型预测论文材料，请总结对推理优化最有启发的三点。\n\n" + make_paper_context(paper_path, 32000),
            128,
        ),
        "paper_120kchars_decode_128": BenchCase(
            "paper_120kchars_decode_128",
            "下面是短临降雨扩散模型预测论文材料，请总结对推理优化最有启发的三点。\n\n" + make_paper_context(paper_path, 120000),
            128,
        ),
    }
    if selected == ["standard"]:
        selected = [
            "short_decode_128",
            "mid_prefill_2k_decode_128",
            "paper_8kchars_decode_128",
            "paper_32kchars_decode_128",
        ]
    if selected == ["long"]:
        selected = ["paper_32kchars_decode_128", "paper_120kchars_decode_128"]
    return [all_cases[name] for name in selected]


def parse_log_text(text: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    m = re.findall(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", text)
    if m:
        info["model_memory_gib"] = float(m[-1][0])
        info["model_load_seconds"] = float(m[-1][1])
    m = re.findall(r"Available KV cache memory: ([0-9.]+) GiB", text)
    if m:
        info["kv_available_gib"] = float(m[-1])
    m = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", text)
    if m:
        info["kv_cache_tokens"] = int(m[-1].replace(",", ""))
    m = re.findall(r"Setting attention block size to ([0-9]+) tokens", text)
    if m:
        info["attention_block_size"] = int(m[-1])
    m = re.findall(r"Padding mamba page size by ([0-9.]+)%", text)
    if m:
        info["mamba_page_padding_pct"] = float(m[-1])
    m = re.findall(r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", text)
    if m:
        info["max_concurrency"] = float(m[-1])
    m = re.findall(r"init engine .* took ([0-9.]+) s", text)
    if m:
        info["engine_init_seconds"] = float(m[-1])
    info["rocm_paged_attention_fallback_warnings"] = text.count("Cannot use ROCm custom paged attention kernel")
    spec_lines = re.findall(
        r"SpecDecoding metrics: Mean acceptance length: ([0-9.]+), Accepted throughput: ([0-9.]+) tokens/s, Drafted throughput: ([0-9.]+) tokens/s, Accepted: ([0-9]+) tokens, Drafted: ([0-9]+) tokens, .*?Avg Draft acceptance rate: ([0-9.]+)%",
        text,
    )
    if spec_lines:
        info["spec_decoding_last"] = {
            "mean_acceptance_length": float(spec_lines[-1][0]),
            "accepted_tps": float(spec_lines[-1][1]),
            "drafted_tps": float(spec_lines[-1][2]),
            "accepted_tokens": int(spec_lines[-1][3]),
            "drafted_tokens": int(spec_lines[-1][4]),
            "avg_draft_acceptance_rate_pct": float(spec_lines[-1][5]),
        }
        info["spec_decoding_samples"] = len(spec_lines)
    return info


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    result: dict[str, Any] = {}
    metrics = [
        "wall_seconds",
        "ttft_seconds",
        "payload_ttft_seconds",
        "prompt_tokens",
        "completion_tokens",
        "prefill_tokens_per_ttft",
        "decode_tokens_per_second_stream",
        "output_tokens_per_second_e2e",
        "total_tokens_per_second_e2e",
    ]
    for case, items in by_case.items():
        ok = [x for x in items if x.get("ok")]
        summary: dict[str, Any] = {"runs": len(items), "ok_runs": len(ok)}
        for metric in metrics:
            vals = [float(x[metric]) for x in ok if x.get(metric) is not None]
            if vals:
                summary[metric] = {
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
        result[case] = summary
    return result


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(out_path: Path, output: dict[str, Any]) -> Path:
    md_path = out_path.with_suffix(".md")
    lines = [
        f"# {output['label']} ({output['mode']})",
        "",
        f"- timestamp: `{output['timestamp']}`",
        f"- model: `{output['model']}`",
        f"- host: `{output['host']}`",
        "",
        "## Runtime parameters",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key, value in output["env"].items():
        lines.append(f"| `{key}` | `{value if value is not None else ''}` |")

    lines.extend(
        [
            "",
            "## API metrics",
            "",
            "| case | ok/runs | wall s | TTFT s | payload TTFT s | prompt tok | output tok | stream decode tok/s | e2e output tok/s | e2e total tok/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case, summary in output["summary"].items():
        def mean(name: str) -> Any:
            item = summary.get(name)
            return None if not item else item.get("mean")

        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    f"{summary.get('ok_runs', 0)}/{summary.get('runs', 0)}",
                    fmt(mean("wall_seconds")),
                    fmt(mean("ttft_seconds")),
                    fmt(mean("payload_ttft_seconds")),
                    fmt(mean("prompt_tokens"), 1),
                    fmt(mean("completion_tokens"), 1),
                    fmt(mean("decode_tokens_per_second_stream")),
                    fmt(mean("output_tokens_per_second_e2e")),
                    fmt(mean("total_tokens_per_second_e2e")),
                ]
            )
            + " |"
        )

    if output.get("parsed_logs"):
        lines.extend(["", "## Parsed logs", "", "| metric | value |", "| --- | ---: |"])
        for key, value in output["parsed_logs"].items():
            if isinstance(value, dict):
                lines.append(f"| `{key}` | `{json.dumps(value, ensure_ascii=False)}` |")
            else:
                lines.append(f"| `{key}` | `{value}` |")

    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--paper-path", default=DEFAULT_PAPER_PATH)
    ap.add_argument("--cases", nargs="+", default=["standard"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--mode", choices=["stream", "nonstream"], default="stream")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out-dir", default="test/results/competition")
    ap.add_argument("--logs-file")
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    if not args.skip_wait:
        wait_ready(args.host)

    cases = build_cases(args.cases, Path(args.paper_path))
    rows: list[dict[str, Any]] = []
    started = time.strftime("%Y%m%d-%H%M%S")
    for case in cases:
        messages = [{"role": "user", "content": case.prompt}]
        for run_idx in range(1, args.runs + 1):
            print(f"[{args.label}] {case.name} run {run_idx}/{args.runs} mode={args.mode}", flush=True)
            if args.mode == "stream":
                row = post_stream_chat(args.host, args.model, messages, case.max_tokens, args.timeout)
            else:
                row = post_nonstream_chat(args.host, args.model, messages, case.max_tokens, args.timeout)
            row.update({"case": case.name, "run": run_idx, "max_tokens": case.max_tokens})
            rows.append(row)
            time.sleep(1)

    log_info = None
    if args.logs_file:
        log_info = parse_log_text(Path(args.logs_file).read_text(errors="ignore"))

    output = {
        "label": args.label,
        "timestamp": started,
        "host": args.host,
        "model": args.model,
        "mode": args.mode,
        "env": {k: os.environ.get(k) for k in PARAM_ENV_KEYS},
        "cases": rows,
        "summary": summarize(rows),
        "parsed_logs": log_info,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{started}_{args.label}_{args.mode}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    md_path = write_markdown(out_path, output)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")
    print(f"saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
