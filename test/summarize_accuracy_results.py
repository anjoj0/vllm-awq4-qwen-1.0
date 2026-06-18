#!/usr/bin/env python3
"""Summarize public accuracy benchmark outputs into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_INFO = {
    "gsm8k": {
        "name": "GSM8K",
        "capability": "math word-problem reasoning",
        "setting": "5-shot, greedy generation",
        "primary": ["exact_match,strict-match", "exact_match,flexible-extract"],
    },
    "hellaswag": {
        "name": "HellaSwag",
        "capability": "commonsense ending selection",
        "setting": "0-shot, multiple-choice loglikelihood",
        "primary": ["acc,none", "acc_norm,none"],
    },
    "arc_challenge": {
        "name": "ARC Challenge",
        "capability": "grade-school science QA",
        "setting": "25-shot, multiple-choice loglikelihood",
        "primary": ["acc,none", "acc_norm,none"],
    },
    "humaneval": {
        "name": "HumanEval",
        "capability": "Python code generation",
        "setting": "0-shot, greedy generation, pass@1",
        "primary": ["pass_at_1,create_test", "pass@1,create_test", "pass_at_1,none"],
    },
}


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def find_result_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/results_*.json"))


def extract_lm_eval_rows(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    results = data.get("results", {})
    sample_info = data.get("n-samples", {})
    nshot_info = data.get("n-shot", {})
    for task, metrics in results.items():
        info = TASK_INFO.get(task, {})
        original = sample_info.get(task, {}).get("original", "")
        effective = sample_info.get(task, {}).get("effective", metrics.get("sample_len", ""))
        nshot = nshot_info.get(task, "")
        primary_keys = [k for k in info.get("primary", []) if k in metrics]
        if not primary_keys:
            primary_keys = [
                k
                for k in metrics
                if k not in {"name", "alias", "sample_len"}
                and not k.endswith("_stderr")
                and "_stderr," not in k
            ]
        metric_text = "<br>".join(f"{k}: {fmt_value(metrics[k])}" for k in primary_keys)
        rows.append(
            {
                "benchmark": info.get("name", task),
                "capability": info.get("capability", ""),
                "setting": info.get("setting", f"{nshot}-shot"),
                "samples": f"{effective}/{original}" if original else str(effective),
                "metrics": metric_text,
                "path": str(path),
                "status": "scored",
            }
        )
    return rows


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def extract_mt_bench_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.glob("**/*mt_bench*_answers.jsonl")):
        rows.append(
            {
                "benchmark": "MT-Bench",
                "capability": "multi-turn chat instruction following",
                "setting": "greedy chat answer generation; judge score not run",
                "samples": str(count_jsonl(path)),
                "metrics": "answers_generated: yes<br>judge_score: pending",
                "path": str(path),
                "status": "answers only",
            }
        )
    return rows


def write_markdown(rows: list[dict[str, str]], out: Path, run_root: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Accuracy Evaluation Summary",
        "",
        f"- Run root: `{run_root}`",
        "- Model endpoint: `Qwen3.6-27B-AWQ4` on `http://127.0.0.1:8001`",
        "- Runtime profile: AWQ4 + DFlash N=8 + fp8 KV cache, greedy decoding where applicable.",
        "- Caveat: MT-Bench entries are generated answers only until scored by a judge model.",
        "- Caveat: HumanEval executes generated Python tests because `--confirm_run_unsafe_code` is required by lm-eval.",
        "",
        "| Benchmark | Capability | Setting | Samples | Metrics | Status | Output |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {capability} | {setting} | {samples} | {metrics} | {status} | `{path}` |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "These public benchmarks cover mathematical reasoning, commonsense reasoning, science QA, code generation, and multi-turn dialogue. "
        "They should be reported as accuracy/quality evidence for the optimized deployment, not as proof of strict mathematical equivalence, "
        "because this runtime uses AWQ4 weights and fp8 KV cache in addition to DFlash speculative decoding."
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out = Path(args.out).resolve() if args.out else run_root / "accuracy_summary.md"

    rows: list[dict[str, str]] = []
    for path in find_result_files(run_root):
        rows.extend(extract_lm_eval_rows(path))
    rows.extend(extract_mt_bench_rows(run_root))

    order = {"GSM8K": 0, "HellaSwag": 1, "ARC Challenge": 2, "HumanEval": 3, "MT-Bench": 4}
    rows.sort(key=lambda r: (order.get(r["benchmark"], 99), r["path"]))

    write_markdown(rows, out, run_root)
    print(f"wrote {out}")
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
