#!/usr/bin/env python3
"""Concurrent benchmark for two OpenAI-compatible vLLM endpoints.

This is intended for W7900 multi-instance experiments, e.g. two TP=4
servers pinned to GPU0-3 and GPU4-7.  It reports both per-endpoint and
combined throughput in one JSON line.
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    default=os.getenv("VLLM_LONGTEXT", "/workspace/bench_data/combined_papers_for_llm.txt"),
)
parser.add_argument("--chars", type=int, default=10000)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--requests-per-endpoint", type=int, default=4)
parser.add_argument("--url-a", required=True)
parser.add_argument("--model-a", required=True)
parser.add_argument("--url-b", required=True)
parser.add_argument("--model-b", required=True)
args = parser.parse_args()


with open(args.file, encoding="utf-8") as stream:
    text = stream.read(args.chars)

prompt = (
    "请阅读下面的论文资料，给出结构化摘要，并列出与长上下文推理优化相关的关键技术。\n\n"
    + text
)


def make_body(model: str) -> bytes:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


body_a = make_body(args.model_a)
body_b = make_body(args.model_b)


def one(endpoint: str, url: str, body: bytes) -> dict:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    usage = result.get("usage", {})
    return {
        "endpoint": endpoint,
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
    }


tasks = []
for _ in range(args.requests_per_endpoint):
    tasks.append(("a", args.url_a, body_a))
    tasks.append(("b", args.url_b, body_b))

started = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
    rows = list(pool.map(lambda item: one(*item), tasks))
wall = time.perf_counter() - started

total_completion = sum((row.get("completion_tokens") or 0) for row in rows)
by_endpoint = {}
for row in rows:
    name = row["endpoint"]
    entry = by_endpoint.setdefault(
        name,
        {
            "requests": 0,
            "completion_tokens": 0,
            "max_elapsed_s": 0.0,
            "prompt_tokens": row.get("prompt_tokens"),
        },
    )
    entry["requests"] += 1
    entry["completion_tokens"] += row.get("completion_tokens") or 0
    entry["max_elapsed_s"] = max(entry["max_elapsed_s"], row["elapsed_s"])

for entry in by_endpoint.values():
    elapsed = entry["max_elapsed_s"]
    entry["aggregate_output_tok_s"] = (
        entry["completion_tokens"] / elapsed if elapsed else 0.0
    )

print(
    json.dumps(
        {
            "chars": args.chars,
            "requests_per_endpoint": args.requests_per_endpoint,
            "total_requests": len(rows),
            "max_tokens": args.max_tokens,
            "wall_s": wall,
            "completion_tokens": total_completion,
            "aggregate_output_tok_s": total_completion / wall if wall else 0.0,
            "requests_per_s": len(rows) / wall if wall else 0.0,
            "by_endpoint": by_endpoint,
            "rows": rows,
        },
        ensure_ascii=False,
    )
)
