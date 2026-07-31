#!/usr/bin/env python3
"""Small reproducible OpenAI-compatible latency/throughput benchmark."""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=os.getenv("VLLM_LONGTEXT", "/workspace/bench_data/combined_papers_for_llm.txt"))
parser.add_argument("--chars", type=int, default=10000)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--requests", type=int, default=1)
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
parser.add_argument("--model", default=os.getenv("VLLM_SERVED_MODEL_NAME", "Qwen3.6-27B-AWQ4"))
args = parser.parse_args()

with open(args.file, encoding="utf-8") as stream:
    text = stream.read(args.chars)
payload = {
    "model": args.model,
    "messages": [{"role": "user", "content": "请阅读下面的论文资料，给出结构化摘要，并列出与长上下文推理优化相关的关键技术。\n\n" + text}],
    "temperature": 0.0, "max_tokens": args.max_tokens, "stream": False,
}
body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

def one(_):
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    usage = result.get("usage", {})
    return {"elapsed_s": elapsed, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason")}

started = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    rows = list(pool.map(one, range(args.requests)))
wall = time.perf_counter() - started
completion = sum((r.get("completion_tokens") or 0) for r in rows)
print(json.dumps({"chars": args.chars, "requests": args.requests, "concurrency": args.concurrency, "max_tokens": args.max_tokens, "wall_s": wall, "completion_tokens": completion, "aggregate_output_tok_s": completion / wall if wall else 0.0, "requests_per_s": len(rows) / wall if wall else 0.0, "rows": rows}, ensure_ascii=False))
