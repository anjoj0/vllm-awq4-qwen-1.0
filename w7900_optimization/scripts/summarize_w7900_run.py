#!/usr/bin/env python3
"""Summarize a W7900 experiment run into a compact Markdown report.

This script is intentionally conservative:
- It does not assume a rigid log schema.
- It extracts the most useful lines from startup logs.
- It parses benchmark JSONL emitted by bench_concurrency_local.py.

Typical usage:
    python3 scripts/summarize_w7900_run.py \
        --result-dir results \
        --run-id 20260729_0700_w7900_experiments
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


KEYWORDS = (
    "model memory",
    "GPU KV cache size",
    "available KV cache memory",
    "Maximum concurrency",
    "engine init",
    "torch.compile",
    "graph capture",
    "profile/warmup",
    "Starting vLLM server",
    "waiting",
    "starting",
    "stopping",
    "bench",
    "finished",
    "ready",
    "Padding mamba",
    "Setting attention block size",
    "SpecDecoding metrics",
    "acceptance rate",
    "Mean acceptance length",
    "Avg Draft acceptance rate",
    "Traceback",
    "error",
    "failed",
    "OOM",
)


def extract_lines(path: Path, limit: int = 120) -> list[str]:
    if not path.is_file():
        return [f"[missing] {path.name}"]
    keep: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if any(key.lower() in line.lower() for key in KEYWORDS):
                keep.append(line.rstrip())
    if len(keep) > limit:
        keep = keep[:limit] + ["..."]
    return keep or ["[no matching summary lines]"]


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    results: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "wall_s" in obj and "aggregate_output_tok_s" in obj:
                pending = obj
                results.append(obj)
            elif "label" in obj and "return_code" in obj:
                if pending is not None and "label" not in pending:
                    pending["label"] = obj["label"]
                    pending["return_code"] = obj["return_code"]
                else:
                    results.append(obj)
            else:
                results.append(obj)
    return results


def bench_table(rows: list[dict[str, Any]]) -> str:
    bench_rows = [row for row in rows if "wall_s" in row and "aggregate_output_tok_s" in row]
    if not bench_rows:
        return "_No benchmark JSON found._"
    out = []
    out.append("| Label | Chars | Requests | Concurrency | Wall s | Output tok/s | RC |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in bench_rows:
        out.append(
            "| {label} | {chars} | {requests} | {concurrency} | {wall_s:.2f} | {aggregate_output_tok_s:.2f} | {rc} |".format(
                label=row.get("label", "-"),
                chars=row.get("chars", "-"),
                requests=row.get("requests", "-"),
                concurrency=row.get("concurrency", "-"),
                wall_s=float(row.get("wall_s", 0.0)),
                aggregate_output_tok_s=float(row.get("aggregate_output_tok_s", 0.0)),
                rc=row.get("return_code", "-"),
            )
        )
    return "\n".join(out)


def write_summary(result_dir: Path, run_id: str) -> Path:
    scheduler_log = result_dir / f"{run_id}.scheduler.log"
    jsonl = result_dir / f"{run_id}.jsonl"
    summary = result_dir / f"{run_id}.summary.md"

    log_files = sorted(result_dir.glob(f"{run_id}*.log"))
    log_files += sorted(result_dir.glob(f"{run_id}*.outer"))
    # Service logs are named by experiment tag, e.g.
    # bf16_tp8_tile16_util085_0700.log, so include the 0700 suffix too.
    log_files += sorted(result_dir.glob("*_0700.log"))
    log_files += sorted(result_dir.glob("*_0700.log.outer"))
    # Also include explicit run logs that may not follow the run-id prefix.
    log_files += sorted(p for p in result_dir.glob("*.log") if run_id in p.name)
    # Deduplicate while preserving order.
    seen = set()
    unique_logs = []
    for path in [scheduler_log, *log_files]:
        if path in seen:
            continue
        seen.add(path)
        unique_logs.append(path)

    rows = parse_jsonl(jsonl)

    with summary.open("w", encoding="utf-8") as out:
        out.write(f"# W7900 run summary: {run_id}\n\n")
        out.write(f"- result dir: `{result_dir}`\n")
        out.write(f"- scheduler log: `{scheduler_log.name}`\n")
        out.write(f"- jsonl: `{jsonl.name}`\n\n")

        out.write("## Benchmark table\n\n")
        out.write(bench_table(rows))
        out.write("\n\n")

        out.write("## Important log excerpts\n\n")
        for path in unique_logs:
            out.write(f"### {path.name}\n\n")
            for line in extract_lines(path):
                out.write(f"- {line}\n")
            out.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default="20260729_0700_w7900_experiments")
    args = parser.parse_args()
    path = write_summary(args.result_dir, args.run_id)
    print(path)


if __name__ == "__main__":
    main()
