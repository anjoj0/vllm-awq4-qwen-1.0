#!/usr/bin/env python3
"""Export a compact CSV for manual review of open-ended sanity answers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from longdoc_sanity_lib import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.run_dir / "manual_review.csv"
    rows = []
    for row in load_jsonl(args.run_dir / "results.jsonl"):
        if row.get("status") != "ok":
            continue
        case = row["case"]
        parsed = row.get("response") or {}
        rows.append({
            "case_id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "model_answer": parsed.get("answer", row.get("response_text", "")),
            "source_documents": " | ".join(map(str, parsed.get("source_documents", []))),
            "evidence_quotes": " | ".join(map(str, parsed.get("evidence_quotes", []))),
            "automatic_score": row["score"]["score"],
            "max_score": row["score"]["max_score"],
            "manual_score": "",
            "manual_reason": "",
            "reviewer": "",
        })
    if not rows:
        raise SystemExit("No completed result rows are available for manual review")
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
