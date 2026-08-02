#!/usr/bin/env python3
"""Measure TTFT and wall time for repeated questions over one long document."""

import argparse
import json
import time
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="/workspace/bench_data/combined_papers_for_llm_L.txt",
    )
    parser.add_argument("--chars", type=int, default=950_000)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen3.6-27B-BF16")
    parser.add_argument("--timeout", type=float, default=3600)
    return parser.parse_args()


QUESTIONS = (
    "Give a one-sentence statement of the document's main research topic.",
    "Name two technical methods discussed in the document.",
    "State one limitation or open problem identified by the authors.",
    "Give one numerical finding and explain what it measures.",
)


def request_once(args: argparse.Namespace, document: str, index: int) -> dict:
    question = QUESTIONS[index % len(QUESTIONS)]
    content = document + "\n\nQuestion: " + question
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        args.url, body, headers={"Content-Type": "application/json"}
    )

    started = time.perf_counter()
    first_token_at = None
    completion_text = []
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
                token = (choice.get("delta") or {}).get("content")
                if token:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    completion_text.append(token)
    finished = time.perf_counter()

    return {
        "request_index": index + 1,
        "question": question,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "wall_s": finished - started,
        "completion_preview": "".join(completion_text)[:160],
    }


def main() -> None:
    args = parse_args()
    with open(args.file, encoding="utf-8") as stream:
        document = stream.read(args.chars)

    rows = []
    for index in range(args.requests):
        row = request_once(args, document, index)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    first = rows[0]
    later = rows[1:]
    summary = {
        "chars": args.chars,
        "requests": len(rows),
        "first_ttft_s": first["ttft_s"],
        "first_wall_s": first["wall_s"],
        "mean_reuse_ttft_s": (
            sum(row["ttft_s"] for row in later if row["ttft_s"] is not None)
            / len(later)
            if later
            else None
        ),
        "mean_reuse_wall_s": (
            sum(row["wall_s"] for row in later) / len(later) if later else None
        ),
        "rows": rows,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
