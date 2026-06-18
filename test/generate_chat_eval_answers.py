#!/usr/bin/env python3
"""Generate public chat-eval answers from the running OpenAI-compatible vLLM API.

This script intentionally only generates model answers. MT-Bench and
AlpacaEval scores require a judge model/API and should be run as a separate
step to avoid confusing "answered" with "scored".
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_mt_bench_questions(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MT_BENCH_URL, timeout=120) as resp:
            cache_path.write_bytes(resp.read())
    questions = []
    for line in cache_path.read_text().splitlines():
        if line.strip():
            questions.append(json.loads(line))
    return questions


def load_alpaca_eval_questions(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    rows = list(ds)
    if limit is not None:
        rows = rows[:limit]
    return [{"id": r.get("instruction_id", i), "turns": [r["instruction"]]} for i, r in enumerate(rows)]


def run_turns(host: str, model: str, turns: list[str], max_tokens: int, timeout: int) -> list[dict]:
    messages = []
    outputs = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        data = post_json(
            f"{host}/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        choice = data["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        outputs.append(
            {
                "prompt": turn,
                "answer": text,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage", {}),
            }
        )
        messages.append({"role": "assistant", "content": text})
    return outputs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.6-27B-AWQ4")
    parser.add_argument("--suite", choices=["mt_bench", "alpaca_eval"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out-dir", default="test/results/accuracy/chat_eval")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)

    if args.suite == "mt_bench":
        questions = load_mt_bench_questions(out_dir / "cache" / "mt_bench_question.jsonl")
        if args.limit is not None:
            questions = questions[: args.limit]
        normalized = [
            {
                "id": q.get("question_id"),
                "category": q.get("category"),
                "turns": q.get("turns", []),
            }
            for q in questions
        ]
    else:
        normalized = load_alpaca_eval_questions(args.limit)

    rows = []
    print(f"suite={args.suite} model={args.model} host={args.host} n={len(normalized)}")
    for idx, q in enumerate(normalized, 1):
        print(f"[{idx}/{len(normalized)}] {q['id']}")
        outputs = run_turns(args.host, args.model, q["turns"], args.max_tokens, args.timeout)
        rows.append(
            {
                "suite": args.suite,
                "model": args.model,
                "id": q["id"],
                "category": q.get("category"),
                "turns": q["turns"],
                "outputs": outputs,
            }
        )

    out_path = out_dir / f"{ts}_{args.suite}_{args.model}_answers.jsonl"
    write_jsonl(out_path, rows)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
