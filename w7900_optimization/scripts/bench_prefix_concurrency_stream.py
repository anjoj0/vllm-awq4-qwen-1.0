#!/usr/bin/env python3
"""Warm one document prefix, then measure concurrent questions over it."""

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


QUESTIONS = (
    "Give a one-sentence statement of the document's main research topic.",
    "Name two technical methods discussed in the document.",
    "State one limitation or open problem identified by the authors.",
    "Give one numerical finding and explain what it measures.",
    "Compare the main method with one baseline discussed in the document.",
    "Identify one assumption that affects the conclusions.",
    "Summarize one ablation result and its implication.",
    "State one practical application described by the authors.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="/workspace/bench_data/combined_papers_for_llm_L.txt",
    )
    parser.add_argument("--chars", type=int, default=950_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen3.6-27B-BF16")
    parser.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args()


def request_once(
    args: argparse.Namespace,
    document: str,
    question: str,
    max_tokens: int,
) -> dict:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": document + "\n\nQuestion: " + question}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        args.url,
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    usage = {}
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if (choice.get("delta") or {}).get("content") and first_token_at is None:
                    first_token_at = time.perf_counter()
    finished = time.perf_counter()
    return {
        "question": question,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "wall_s": finished - started,
    }


def main() -> None:
    args = parse_args()
    with open(args.file, encoding="utf-8") as stream:
        document = stream.read(args.chars)

    warmup = None
    if not args.skip_warmup:
        warmup = request_once(
            args, document, QUESTIONS[0], args.warmup_max_tokens
        )
    questions = [QUESTIONS[(index + 1) % len(QUESTIONS)] for index in range(args.concurrency)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        rows = list(
            pool.map(
                lambda question: request_once(
                    args, document, question, args.max_tokens
                ),
                questions,
            )
        )
    concurrent_wall = time.perf_counter() - started
    completion_tokens = sum(row["completion_tokens"] or 0 for row in rows)
    result = {
        "chars": args.chars,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "warmup": warmup,
        "concurrent_wall_s": concurrent_wall,
        "mean_ttft_s": statistics.mean(row["ttft_s"] for row in rows),
        "mean_request_wall_s": statistics.mean(row["wall_s"] for row in rows),
        "aggregate_output_tok_s": completion_tokens / concurrent_wall,
        "rows": rows,
    }
    serialized = json.dumps(result, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
