#!/usr/bin/env python3
"""Two fixed decode workloads for draft selection; not a quality benchmark.

Run after functional smoke/warmup, with no other inference clients. Each case
uses a fresh cache salt, greedy sampling, thinking disabled and a 512-token cap.
Switch backends separately so the target kernel and request shape stay fixed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import uuid

from smoke_strix import Client, ROOT, require
from server import ActiveRequest, App, Config, make_stats, metric_delta

PROMPTS = {
    "code": "Implement a complete Python LRU cache using only the standard library. "
            "Include capacity validation, get, put, delete, clear, __len__, and "
            "thread safety with RLock. Provide type hints, concise docstrings, "
            "an explanation of O(1) operations, and unittest cases for eviction, "
            "updating existing keys, missing keys, zero capacity, and clearing. "
            "Give the full implementation followed by the full tests.",
    "zh_explanation": "请详细解释大语言模型的投机解码：先分别说明主模型、草稿模型、"
            "prefill、decode 的作用，然后用具体的候选 token 序列演示一次全部接受、"
            "一次中途拒绝及回退。再分析接受率、验证成本、草稿成本与实际吞吐的关系，"
            "列出计算每轮平均提交 token 数和 decode token/s 的公式。最后给出一个"
            "适用于本地单用户编程助手的完整测量方案，并说明如何避免前缀缓存和"
            "首轮编译污染测速。请逐项解释，不要只列提纲。",
}
KEYS = ("vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total")


def padded_prompt(client, prompt, input_tokens):
    """Build one deterministic long chat input using actual template tokens."""
    if not input_tokens:
        return prompt
    notes = "".join(f"Note {i}: Cache capacity is a nonnegative integer. Get and put update recency. "
                    "A lock protects compound state changes. Tests must check eviction and missing keys.\n"
                    for i in range(input_tokens // 10 + 1))
    header = "Reference notes for the coding task below:\n"
    tail = "\n\nNow complete this task:\n" + prompt
    low, high = 0, len(notes)
    while low < high:
        mid = (low + high + 1) // 2
        payload = client.chat(header + notes[:mid] + tail)
        tokenized = client.json("/tokenize", {"model": client.model, "messages": payload["messages"],
            "chat_template_kwargs": payload["chat_template_kwargs"], "add_generation_prompt": True})
        if len(tokenized["tokens"]) <= input_tokens:
            low = mid
        else:
            high = mid - 1
    return header + notes[:low] + tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Actual backend/kernel configuration")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-27B-AWQ4")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".runtime/strix-halo")
    parser.add_argument("--runs", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--case", choices=["all", *PROMPTS], default="all")
    parser.add_argument("--input-tokens", type=int, default=0,
                        help="Pad with deterministic reference notes; 0 keeps the original short prompt")
    parser.add_argument("--output-tokens", type=int, choices=(256, 512), default=512)
    args = parser.parse_args()
    require(args.input_tokens == 0 or 1024 <= args.input_tokens <= 30000, "Invalid input length")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.label), "Invalid label")
    path = args.out_dir / ("decode-long-" + args.label + ".json")
    require(not path.exists(), f"Refusing to overwrite {path}; choose another label")
    path.parent.mkdir(parents=True, exist_ok=True)
    client = Client(args.host, args.model, 600)
    app = App(Config(upstream=client.host, model=client.model, metrics_wait=3))
    result = {
        "label": args.label, "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed prompts; greedy, thinking off, fresh salt. "
                 "Decode TPS = (output_tokens-1)/server generation time. "
                 "Length-limited outputs are throughput samples, not complete quality answers.",
        "rows": [],
        "requested_input_tokens": args.input_tokens, "output_cap": args.output_tokens,
    }
    for case, prompt in PROMPTS.items():
        if args.case not in ("all", case):
            continue
        prompt = padded_prompt(client, prompt, args.input_tokens)
        for run in range(args.runs):
            before = app.metrics()
            require(before.ok and not any(before.values.get(k, 0) for k in
                    ("vllm:num_requests_running", "vllm:num_requests_waiting")),
                    "Metrics unavailable or another client is active")
            payload = client.chat(prompt)
            payload.update(max_tokens=args.output_tokens, cache_salt=uuid.uuid4().hex)
            started = time.perf_counter()
            data = client.json("/v1/chat/completions", payload)
            wall = time.perf_counter() - started
            after = app.completed_metrics(before, ActiveRequest(uuid.uuid4().hex))
            usage, metrics = data["usage"], data["metrics"]
            n = usage["completion_tokens"]
            choice = data["choices"][0]
            require(n >= 192 and metrics["generation_time_ms"] > 0,
                    "Output too short for this decode check")
            require(choice["finish_reason"] in ("stop", "length"), "Incomplete/error result")
            require(usage["prompt_tokens_details"]["cached_tokens"] == 0,
                    "Unexpected prefix cache hit")
            stats = make_stats(usage, before, after, wall,
                               metrics["time_to_first_token_ms"] / 1000, choice["finish_reason"])
            require(stats["metric_source"] == "server", stats["note"])
            delta = {key: metric_delta(before, after, key) for key in KEYS}
            rounds, drafted, accepted = (delta[key] for key in KEYS)
            text = choice["message"].get("content") or ""
            row = {
                "case": case, "run": run, "request": payload, "response": data,
                "wall_s": wall, "decode_tps": (n - 1) * 1000 / metrics["generation_time_ms"],
                "ttft_ms": metrics["time_to_first_token_ms"], "spec_counters_delta": delta,
                "mean_acceptance_length": (1 + accepted / rounds
                    if rounds and accepted is not None else None),
                "draft_acceptance_rate": (accepted / drafted
                    if drafted and accepted is not None else None),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "aggregate_metrics_check": stats,
            }
            result["rows"].append(row)
            result["summary"] = {
                name: {"runs": len(rows),
                       "median_decode_tps": statistics.median(r["decode_tps"] for r in rows),
                       "output_tokens": [r["response"]["usage"]["completion_tokens"] for r in rows]}
                for name in PROMPTS
                if (rows := [r for r in result["rows"] if r["case"] == name])
            }
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ("case", "run", "decode_tps", "ttft_ms",
                  "mean_acceptance_length", "draft_acceptance_rate")}, ensure_ascii=False), flush=True)
    print("Evidence:", path, flush=True)


if __name__ == "__main__":
    main()
