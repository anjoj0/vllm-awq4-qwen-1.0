#!/usr/bin/env python3
"""Two uncached fixed-length requests for target-kernel verification."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import time
import uuid
from smoke_strix import Client, ROOT, require
from server import ActiveRequest, App, Config, iter_sse, make_stats

PASSAGE = (
    "A local inference service runs a quantized target language model and a small draft model. "
    "The target verifies proposed tokens. Prefix caching reuses previously computed attention "
    "keys, values and compatible recurrent state. A new document still requires prefill. "
    "Measure cache hits, time to first token, generation time and memory separately. "
)

def token_ids(client, text):
    response = client.json("/tokenize", {"model": client.model, "prompt": text,
                                         "add_special_tokens": False})
    tokens = response.get("tokens")
    require(isinstance(tokens, list) and tokens and all(isinstance(t, int) for t in tokens),
            "/tokenize did not return token IDs; character estimates are not used")
    return tokens

def run_request(client, observer, prompt, salt, output_tokens):
    before = observer.metrics()
    busy = any(before.values.get(k, 0) > 0 for k in
               ("vllm:num_requests_running", "vllm:num_requests_waiting"))
    require(not busy, "Service is busy; pause other clients before benchmarking")
    payload = {"model": client.model, "prompt": prompt, "add_special_tokens": False,
               "cache_salt": salt, "temperature": 0, "max_tokens": output_tokens,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    first = None
    text, usage, timing, finish, done = [], {}, {}, None, False
    with client.request("/v1/completions", payload) as response:
        require("text/event-stream" in response.headers.get("Content-Type", ""), "Expected SSE")
        for _, raw in iter_sse(response):
            if raw == "[DONE]":
                done = True
                break
            data = json.loads(raw)
            require(not data.get("error"), f"API stream error: {data.get('error')}")
            usage = data.get("usage") or usage
            timing = data.get("metrics") or timing
            for choice in data.get("choices", []):
                if choice.get("text"):
                    if first is None:
                        first = time.perf_counter()
                    text.append(choice["text"])
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - started
    require(done and finish in ("length", "stop") and first is not None,
            "Generation stream ended without a complete nonempty response")
    require(usage.get("prompt_tokens") == len(prompt), "Unexpected prompt token count")
    require(usage.get("completion_tokens") == output_tokens, "Output token budget was not reached")
    after = observer.completed_metrics(before, ActiveRequest(uuid.uuid4().hex))
    stats = make_stats(usage, before, after, elapsed, first - started, finish)
    cached = stats.get("cached_tokens")
    return {
        "usage": usage, "per_request_metrics": timing, "stats": stats,
        "prompt_tokens": len(prompt), "cached_tokens": cached,
        "computed_tokens": len(prompt) - cached if cached is not None else None,
        "client_ttft_s": first - started, "wall_s": elapsed,
        "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
        "output_preview": "".join(text)[:300],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-27B-AWQ4")
    parser.add_argument("--input-tokens", type=int, default=4096)
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.label), "Invalid label")
    require(1 <= args.input_tokens <= 32760, "Input length must leave room for the output")
    path = ROOT / ".runtime/strix-halo" / ("prefill-" + args.label + ".json")
    require(not path.exists(), f"Refusing to overwrite {path}; use another label")
    client = Client(args.host, args.model, 600)
    observer = App(Config(upstream=args.host, model=args.model, metrics_wait=3))
    prompt = token_ids(client, PASSAGE * (args.input_tokens // 30 + 10))[:args.input_tokens]
    require(len(prompt) == args.input_tokens, "Tokenizer returned too few tokens")
    report = {"label": args.label, "started_at": datetime.now(timezone.utc).isoformat(),
              "note": f"Two fresh-salt, uncached {args.input_tokens}-token requests and 8 output tokens. "
                      "Synthetic raw completions, not chat quality or a cache comparison. "
                      "First may include shape JIT; second is the comparison.",
              "prompt_token_ids": prompt, "output_tokens": 8, "rows": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    for phase in ("first_shape_call", "steady"):
        salt = uuid.uuid4().hex
        row = run_request(client, observer, prompt, salt, 8)
        require(row["cached_tokens"] == 0 and row["stats"]["metric_source"] == "server",
                "Unexpected cache hit or ambiguous request metrics")
        row.update(phase=phase, cache_salt=salt)
        report["rows"].append(row)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"phase": phase, "prefill_tps": row["stats"]["prefill_tps"],
                          "ttft_s": row["client_ttft_s"], "cached_tokens": row["cached_tokens"]}), flush=True)
    print("Evidence:", path, flush=True)


if __name__ == "__main__":
    main()
