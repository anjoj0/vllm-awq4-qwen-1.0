#!/usr/bin/env python3
"""Score generated MT-Bench answers with a local OpenAI-compatible judge.

This is not the official MT-Bench GPT-4 judge. It is a lightweight local
quality check for competition evidence when an external judge API is not
available. The output is explicitly labeled `local-self-judge`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path


SCORE_RE = re.compile(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_chat(host: str, model: str, messages: list[dict], max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for judge prompt]..."


def build_judge_prompt(answer_row: dict, max_answer_chars: int) -> str:
    turns = answer_row["turns"]
    outputs = answer_row["outputs"]
    parts = [
        "You are grading one model's answer to an MT-Bench style multi-turn question.",
        "Score the answer quality from 1 to 10. Use these criteria: helpfulness, correctness, relevance, completeness, instruction following, clarity, and safety.",
        "Penalize visible chain-of-thought, unfinished answers, repetition, format violations, or ignoring constraints.",
        "Return exactly one line in this format: [[score]]",
        "",
        f"Category: {answer_row.get('category')}",
    ]
    for idx, (question, output) in enumerate(zip(turns, outputs), 1):
        parts.extend(
            [
                "",
                f"Turn {idx} user question:",
                question,
                "",
                f"Turn {idx} model answer:",
                truncate(output.get("answer", ""), max_answer_chars),
                "",
                f"Turn {idx} finish_reason: {output.get('finish_reason')}",
            ]
        )
    return "\n".join(parts)


def parse_score(text: str) -> float | None:
    match = SCORE_RE.search(text or "")
    if not match:
        return None
    score = float(match.group(1))
    if score < 1 or score > 10:
        return None
    return score


def summarize(scores: list[dict]) -> dict:
    valid = [row for row in scores if isinstance(row.get("score"), (int, float)) and row["score"] > 0]
    values = [float(row["score"]) for row in valid]
    by_category: dict[str, list[float]] = {}
    for row in valid:
        by_category.setdefault(row.get("category") or "unknown", []).append(float(row["score"]))
    return {
        "judge": scores[0]["judge"] if scores else "",
        "count": len(scores),
        "valid_count": len(valid),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "by_category": {
            category: {
                "count": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }
            for category, vals in sorted(by_category.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:8001")
    parser.add_argument("--judge-model", default="Qwen3.6-27B-AWQ4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-answer-chars", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    answers = read_jsonl(Path(args.answers))
    if args.limit is not None:
        answers = answers[: args.limit]

    out = Path(args.out)
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    else:
        done = set()

    scores: list[dict] = []
    if out.exists():
        scores.extend(read_jsonl(out))

    system = "You are a strict but fair evaluator. Output only the score line requested by the user."
    judge_name = f"local-self-judge:{args.judge_model}"

    for idx, row in enumerate(answers, 1):
        if row["id"] in done:
            print(f"[{idx}/{len(answers)}] skip id={row['id']}", flush=True)
            continue
        prompt = build_judge_prompt(row, args.max_answer_chars)
        data = post_chat(
            args.host,
            args.judge_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            args.max_tokens,
            args.timeout,
        )
        msg = data["choices"][0]["message"]
        judgment = (msg.get("content") or msg.get("reasoning") or "").strip()
        score = parse_score(judgment)
        score_row = {
            "suite": "mt_bench",
            "id": row["id"],
            "category": row.get("category"),
            "judge": judge_name,
            "score": score if score is not None else -1.0,
            "valid_score": score is not None,
            "judgment": judgment,
            "usage": data.get("usage", {}),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_jsonl(out, score_row)
        scores.append(score_row)
        print(
            f"[{idx}/{len(answers)}] id={row['id']} category={row.get('category')} score={score_row['score']}",
            flush=True,
        )

    summary = summarize(scores)
    summary["answers"] = str(Path(args.answers).resolve())
    summary["scores"] = str(out.resolve())
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
