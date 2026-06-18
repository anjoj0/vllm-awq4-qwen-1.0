#!/usr/bin/env python3
"""Score generated MT-Bench answers with DeepSeek V4 Flash.

This produces a third-party DeepSeek judge score, not the official GPT-4
MT-Bench score. It uses DeepSeek's OpenAI-compatible Chat Completions API,
JSON output, and disabled thinking mode so the result is machine-parseable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for judge prompt]..."


def build_prompt(row: dict[str, Any], max_answer_chars: int) -> str:
    parts = [
        "You are judging one model answer for an MT-Bench style multi-turn question.",
        "Return valid json only. Do not include markdown.",
        "",
        "Scoring rubric:",
        "- score is a number from 1 to 10.",
        "- Judge helpfulness, correctness, relevance, completeness, instruction following, clarity, and safety.",
        "- Penalize visible chain-of-thought/reasoning leakage, unfinished or truncated answers, repetition, format violations, and ignored constraints.",
        "- If the answer is mostly hidden reasoning instead of final user-facing content, give a low score.",
        "",
        "Expected json schema:",
        '{"score": 7.0, "turn1_score": 7.0, "turn2_score": 7.0, "rationale": "short reason"}',
        "",
        f"Category: {row.get('category')}",
        f"Question id: {row.get('id')}",
    ]
    turns = row.get("turns") or []
    outputs = row.get("outputs") or []
    for idx, question in enumerate(turns, 1):
        output = outputs[idx - 1] if idx - 1 < len(outputs) else {}
        parts.extend(
            [
                "",
                f"Turn {idx} user question:",
                str(question),
                "",
                f"Turn {idx} model answer:",
                truncate(str(output.get("answer") or ""), max_answer_chars),
                "",
                f"Turn {idx} finish_reason: {output.get('finish_reason')}",
            ]
        )
    return "\n".join(parts)


def post_deepseek(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict but fair MT-Bench judge. Output valid json only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_judgment(data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    parsed = json.loads(content)
    score = float(parsed["score"])
    if not 1 <= score <= 10:
        raise ValueError(f"score out of range: {score}")
    parsed["score"] = score
    for key in ("turn1_score", "turn2_score"):
        if key in parsed and parsed[key] is not None:
            parsed[key] = float(parsed[key])
    return parsed, content


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid_score")]
    scores = [float(r["score"]) for r in valid]
    by_category: dict[str, list[float]] = {}
    for row in valid:
        by_category.setdefault(row.get("category") or "unknown", []).append(float(row["score"]))
    return {
        "judge": rows[0]["judge"] if rows else "",
        "official_mt_bench": False,
        "count": len(rows),
        "valid_count": len(valid),
        "mean": statistics.mean(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "by_category": {
            category: {
                "count": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }
            for category, vals in sorted(by_category.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--judge-model", default=os.environ.get("DEEPSEEK_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-answer-chars", type=int, default=2500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key: export {args.api_key_env}=...")

    answers = read_jsonl(Path(args.answers))
    if args.limit is not None:
        answers = answers[: args.limit]

    out = Path(args.out)
    done: set[Any] = set()
    rows: list[dict[str, Any]] = []
    if out.exists():
        rows = read_jsonl(out)
        done = {row.get("id") for row in rows}

    judge_name = f"deepseek:{args.judge_model}"
    for idx, row in enumerate(answers, 1):
        qid = row.get("id")
        if qid in done:
            print(f"[{idx}/{len(answers)}] skip id={qid}", flush=True)
            continue

        prompt = build_prompt(row, args.max_answer_chars)
        last_error = None
        record = None
        for attempt in range(1, args.retries + 1):
            try:
                data = post_deepseek(
                    args.base_url,
                    api_key,
                    args.judge_model,
                    prompt,
                    args.timeout,
                    args.max_tokens,
                )
                parsed, raw = parse_judgment(data)
                record = {
                    "suite": "mt_bench",
                    "id": qid,
                    "category": row.get("category"),
                    "judge": judge_name,
                    "score": parsed["score"],
                    "turn1_score": parsed.get("turn1_score"),
                    "turn2_score": parsed.get("turn2_score"),
                    "rationale": parsed.get("rationale", ""),
                    "valid_score": True,
                    "raw_judgment": raw,
                    "usage": data.get("usage", {}),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                break
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        last_error += " " + exc.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                print(f"[{idx}/{len(answers)}] id={qid} attempt={attempt} error={last_error}", flush=True)
                time.sleep(max(args.sleep, 1.0) * attempt)

        if record is None:
            record = {
                "suite": "mt_bench",
                "id": qid,
                "category": row.get("category"),
                "judge": judge_name,
                "score": -1.0,
                "valid_score": False,
                "error": last_error,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        append_jsonl(out, record)
        rows.append(record)
        print(
            f"[{idx}/{len(answers)}] id={qid} category={row.get('category')} score={record['score']} valid={record['valid_score']}",
            flush=True,
        )
        time.sleep(args.sleep)

    summary = summarize(rows)
    summary["answers"] = str(Path(args.answers).resolve())
    summary["scores"] = str(out.resolve())
    summary["base_url"] = args.base_url
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
