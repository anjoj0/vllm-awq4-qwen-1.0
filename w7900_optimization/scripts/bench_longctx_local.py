#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=os.getenv("VLLM_LONGTEXT", "/workspace/bench_data/combined_papers_for_llm.txt"))
parser.add_argument("--chars", type=int, default=400000)
parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
parser.add_argument("--model", default=os.getenv("VLLM_SERVED_MODEL_NAME", "Qwen3.6-27B-AWQ4"))
args = parser.parse_args()
with open(args.file, encoding="utf-8") as stream:
    text = stream.read(args.chars)
payload = {
    "model": args.model,
    "messages": [{"role": "user", "content": "请阅读下面的论文资料，给出结构化摘要，并列出与长上下文推理优化相关的关键技术。\n\n" + text}],
    "temperature": 0.0,
    "max_tokens": 512,
    "stream": False,
}
request = urllib.request.Request(
    args.url,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=1800) as response:
    output = response.read()
print(json.dumps({"input_chars": len(text), "elapsed_s": time.perf_counter() - started,
                  "response": json.loads(output)}, ensure_ascii=False))
