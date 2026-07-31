#!/usr/bin/env python3
"""Run a command while sampling ROCm board power.

The script writes raw power samples and a compact JSON summary. If the child
command prints the existing bench_concurrency_local.py JSON line, token/J is
computed automatically.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


POWER_RE = re.compile(r"GPU\[(\d+)\].*Power \(W\):\s*([0-9.]+)")


def read_power_once(rocm_smi: str) -> dict[int, float]:
    proc = subprocess.run(
        [rocm_smi, "--showpower"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    values: dict[int, float] = {}
    for line in proc.stdout.splitlines():
        match = POWER_RE.search(line)
        if match:
            values[int(match.group(1))] = float(match.group(2))
    return values


def load_child_metrics(path: Path) -> dict:
    metrics: dict = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "wall_s" in obj and "completion_tokens" in obj:
            metrics = obj
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--child-stdout", required=True)
    parser.add_argument("--child-stderr", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--rocm-smi", default="rocm-smi")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise SystemExit("missing child command after --")

    sample_path = Path(args.sample_jsonl)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    stdout_path = Path(args.child_stdout)
    stderr_path = Path(args.child_stderr)

    stop = threading.Event()
    samples: list[dict] = []

    def sampler() -> None:
        with sample_path.open("w", encoding="utf-8") as f:
            while not stop.is_set():
                ts = time.time()
                powers = read_power_once(args.rocm_smi)
                row = {
                    "ts": ts,
                    "powers_w": powers,
                    "total_w": sum(powers.values()) if powers else None,
                }
                samples.append(row)
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                stop.wait(args.interval)

    thread = threading.Thread(target=sampler, daemon=True)
    started = time.perf_counter()
    thread.start()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.Popen(args.cmd, stdout=out, stderr=err, text=True)
        rc = proc.wait()
    stop.set()
    thread.join(timeout=max(2.0, args.interval * 2))
    elapsed_s = time.perf_counter() - started

    totals = [s["total_w"] for s in samples if s.get("total_w") is not None]
    avg_w = sum(totals) / len(totals) if totals else None
    energy_j = avg_w * elapsed_s if avg_w is not None else None
    child = load_child_metrics(stdout_path)
    completion_tokens = child.get("completion_tokens")
    prompt_tokens = sum(
        (row.get("prompt_tokens") or 0) for row in child.get("rows", [])
    ) if child else None
    total_tokens = (
        completion_tokens + prompt_tokens
        if completion_tokens is not None and prompt_tokens is not None
        else None
    )

    summary = {
        "return_code": rc,
        "elapsed_s": elapsed_s,
        "samples": len(samples),
        "avg_total_w": avg_w,
        "min_total_w": min(totals) if totals else None,
        "max_total_w": max(totals) if totals else None,
        "energy_j": energy_j,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "output_token_per_j": (
            completion_tokens / energy_j
            if completion_tokens is not None and energy_j
            else None
        ),
        "total_token_per_j": (
            total_tokens / energy_j if total_tokens is not None and energy_j else None
        ),
        "child_stdout": str(stdout_path),
        "child_stderr": str(stderr_path),
        "sample_jsonl": str(sample_path),
        "child_metrics": child,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    sys.exit(rc)


if __name__ == "__main__":
    main()
