#!/usr/bin/env python3
"""Compare aggregate quality and latency across multiple sanity run directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from longdoc_sanity_lib import load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--baseline-label", default="bf16_tp8_auto")
    parser.add_argument("--output", type=Path, default=Path("longdoc_sanity_comparison.md"))
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        manifest = load_json(run_dir / "manifest.json")
        summary = load_json(run_dir / "summary.json")
        rows.append({"run_dir": str(run_dir), **manifest, **summary})
    baselines = [row for row in rows if row["config_label"] == args.baseline_label]
    if not baselines:
        raise SystemExit(f"No baseline with label {args.baseline_label!r}")
    baseline_by_cohort = {
        (row["profile"], row["suite"], row["context_mode"]): row
        for row in baselines
    }

    lines = [
        "# Nowcast3D Long-Document Sanity Comparison",
        "",
        f"Baseline label: `{args.baseline_label}`",
        "",
        "| Config | Profile | Suite | Mode | QA | Retention | Needle EM | Citation | Evidence | Source | Abstention | JSON | Mean wall (s) | Mean TTFT (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for row in sorted(rows, key=lambda item: (item["profile"], item["config_label"])):
        cohort = (row["profile"], row["suite"], row["context_mode"])
        baseline = baseline_by_cohort.get(cohort)
        retention = None
        if baseline and baseline["qa_score_fraction"]:
            retention = row["qa_score_fraction"] / baseline["qa_score_fraction"]
        lines.append(
            f"| {row['config_label']} | {row['profile']} | {row['suite']} | "
            f"{row['context_mode']} | {row['qa_score_fraction']:.2%} | "
            f"{'n/a' if retention is None else f'{retention:.2%}'} | "
            f"{row['needle_exact_match']:.2%} | {row['citation_validity']:.2%} | "
            f"{row['evidence_support']:.2%} | {row['source_accuracy']:.2%} | "
            f"{row['abstention_accuracy']:.2%} | {row['json_success_rate']:.2%} | "
            f"{row['mean_wall_s']:.3f} | {row['mean_ttft_s']:.3f} |"
        )
        csv_rows.append({
            "config_label": row["config_label"],
            "profile": row["profile"],
            "suite": row["suite"],
            "context_mode": row["context_mode"],
            "qa_score_fraction": row["qa_score_fraction"],
            "quality_retention": retention,
            "needle_exact_match": row["needle_exact_match"],
            "citation_validity": row["citation_validity"],
            "evidence_support": row["evidence_support"],
            "source_accuracy": row["source_accuracy"],
            "numeric_exact_match": row["numeric_exact_match"],
            "abstention_accuracy": row["abstention_accuracy"],
            "json_success_rate": row["json_success_rate"],
            "mean_wall_s": row["mean_wall_s"],
            "mean_ttft_s": row["mean_ttft_s"],
            "run_dir": row["run_dir"],
        })
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(args.output)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
