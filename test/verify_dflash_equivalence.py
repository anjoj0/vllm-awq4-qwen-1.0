#!/usr/bin/env python3
"""Collect and compare no-spec vs DFlash greedy outputs.

This is a correctness-oriented companion to public benchmark scoring. The
intended flow is:

  1. Start the target model without DFlash and collect baseline outputs.
  2. Restart with DFlash enabled and collect optimized outputs.
  3. Compare prompt-by-prompt byte-level output equality.

The script uses /v1/completions to avoid chat-template and reasoning-parser
differences. It does not prove full distributional equality, but it is a
practical regression test for greedy decoding correctness through the patched
DFlash verify attention path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:8001"
DEFAULT_MODEL = "Qwen3.6-27B-AWQ4"
DEFAULT_OUT_DIR = "test/results/dflash_correctness"

PROMPTS = [
    {"id": "short_math", "prompt": "Compute 19 + 23. Answer with only the number.", "max_tokens": 16},
    {"id": "chinese_summary", "prompt": "用三句话说明投机解码为什么可以提升大模型推理速度。", "max_tokens": 96},
    {
        "id": "code_sort",
        "prompt": "Write a Python function stable_sort_pairs(items) that sorts pairs by the second field while preserving order for ties.",
        "max_tokens": 192,
    },
    {
        "id": "logic",
        "prompt": "Alice is older than Bob. Bob is older than Chen. Who is the youngest? Answer briefly.",
        "max_tokens": 64,
    },
    {
        "id": "format_json",
        "prompt": "Return a compact JSON object with keys city, country, and population_rank for Tokyo.",
        "max_tokens": 96,
    },
    {
        "id": "long_recall",
        "prompt": (
            "Reference text:\n"
            "Strix Halo uses a unified memory architecture. DFlash uses a drafter "
            "model to propose tokens and a target model to verify them. In this "
            "project the main bottleneck past long context is KV-cache attention, "
            "not AWQ GEMM.\n\n"
            "Question: Based only on the reference text, what is the bottleneck "
            "past long context?"
        ),
        "max_tokens": 96,
    },
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.perf_counter() - t0


def health(host: str, timeout: int) -> None:
    with urllib.request.urlopen(f"{host}/health", timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"health returned {resp.status}")


def load_prompts(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return PROMPTS
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect(args: argparse.Namespace) -> Path:
    host = args.host.rstrip("/")
    health(host, args.timeout)
    prompts = load_prompts(args.prompt_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{ts}_{args.label}_outputs.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(prompts, 1):
            prompt_id = item.get("id") or f"prompt_{i}"
            max_tokens = int(item.get("max_tokens", args.max_tokens))
            payload = {
                "model": args.model,
                "prompt": item["prompt"],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            print(f"[{i}/{len(prompts)}] {prompt_id} max_tokens={max_tokens}", flush=True)
            row: dict[str, Any] = {
                "label": args.label,
                "id": prompt_id,
                "prompt": item["prompt"],
                "request": payload,
            }
            try:
                data, wall = post_json(f"{host}/v1/completions", payload, args.timeout)
                choice = (data.get("choices") or [{}])[0]
                text = choice.get("text") or ""
                usage = data.get("usage") or {}
                row.update(
                    {
                        "ok": True,
                        "wall_seconds": wall,
                        "text": text,
                        "text_sha256": sha256_text(text),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": usage,
                        "completion_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                    }
                )
            except urllib.error.HTTPError as exc:
                row.update({"ok": False, "error": exc.read().decode("utf-8", errors="replace")[:1000], "status": exc.code})
            except Exception as exc:  # noqa: BLE001
                row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {out_path}")
    return out_path


def read_jsonl(path: str) -> dict[str, dict[str, Any]]:
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def first_diff(a: str, b: str) -> dict[str, Any] | None:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return {
                "char_index": idx,
                "a_context": a[max(0, idx - 40): idx + 80],
                "b_context": b[max(0, idx - 40): idx + 80],
            }
    if len(a) != len(b):
        return {
            "char_index": limit,
            "a_context": a[max(0, limit - 40): limit + 80],
            "b_context": b[max(0, limit - 40): limit + 80],
        }
    return None


def compare(args: argparse.Namespace) -> Path:
    a = read_jsonl(args.a)
    b = read_jsonl(args.b)
    ids = sorted(set(a) | set(b))
    rows = []
    exact_matches = 0
    ok_pairs = 0
    for prompt_id in ids:
        ar = a.get(prompt_id)
        br = b.get(prompt_id)
        row: dict[str, Any] = {"id": prompt_id, "a_present": ar is not None, "b_present": br is not None}
        if not ar or not br:
            row["exact_text_match"] = False
        else:
            row["a_ok"] = ar.get("ok")
            row["b_ok"] = br.get("ok")
            row["a_sha256"] = ar.get("text_sha256")
            row["b_sha256"] = br.get("text_sha256")
            row["a_completion_tokens"] = ar.get("completion_tokens")
            row["b_completion_tokens"] = br.get("completion_tokens")
            row["a_wall_seconds"] = ar.get("wall_seconds")
            row["b_wall_seconds"] = br.get("wall_seconds")
            row["exact_text_match"] = bool(ar.get("ok") and br.get("ok") and ar.get("text") == br.get("text"))
            if ar.get("ok") and br.get("ok"):
                ok_pairs += 1
            if row["exact_text_match"]:
                exact_matches += 1
            else:
                row["first_diff"] = first_diff(ar.get("text") or "", br.get("text") or "")
        rows.append(row)

    summary = {
        "a": args.a,
        "b": args.b,
        "total_prompts": len(ids),
        "ok_pairs": ok_pairs,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(ids) if ids else None,
        "rows": rows,
    }
    out_path = Path(args.out or Path(args.b).with_suffix(".compare.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    lines = [
        "# DFlash Greedy Equivalence Check",
        "",
        f"- baseline: `{args.a}`",
        f"- candidate: `{args.b}`",
        f"- exact matches: `{exact_matches}/{len(ids)}`",
        f"- exact match rate: `{summary['exact_match_rate']:.4f}`" if ids else "- exact match rate: `n/a`",
        "",
        "| prompt | exact | a tok | b tok | a wall s | b wall s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {id} | {exact} | {atok} | {btok} | {awall} | {bwall} |".format(
                id=row["id"],
                exact="yes" if row.get("exact_text_match") else "no",
                atok=row.get("a_completion_tokens", ""),
                btok=row.get("b_completion_tokens", ""),
                awall=f"{row.get('a_wall_seconds'):.3f}" if isinstance(row.get("a_wall_seconds"), float) else "",
                bwall=f"{row.get('b_wall_seconds'):.3f}" if isinstance(row.get("b_wall_seconds"), float) else "",
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This check compares visible completion text under greedy decoding. "
            "It is designed to catch regressions in the DFlash verify path. "
            "It should be reported alongside acceptance-rate and throughput logs, "
            "not as a replacement for public benchmark accuracy.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"saved {out_path}")
    print(f"saved {md_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--label", required=True)
    p_collect.add_argument("--host", default=DEFAULT_HOST)
    p_collect.add_argument("--model", default=DEFAULT_MODEL)
    p_collect.add_argument("--prompt-file")
    p_collect.add_argument("--max-tokens", type=int, default=128)
    p_collect.add_argument("--timeout", type=int, default=900)
    p_collect.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p_collect.set_defaults(func=collect)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--a", required=True, help="baseline JSONL, normally no-spec")
    p_compare.add_argument("--b", required=True, help="candidate JSONL, normally DFlash")
    p_compare.add_argument("--out")
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
