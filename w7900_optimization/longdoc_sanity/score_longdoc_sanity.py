#!/usr/bin/env python3
"""Re-score an existing Nowcast3D sanity run without calling the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from longdoc_sanity_lib import (
    load_json,
    load_jsonl,
    score_case,
    summary_markdown,
    summarize_results,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = load_json(args.run_dir / "manifest.json")
    rows = load_jsonl(args.run_dir / "results.jsonl")
    # Exact context citation checks are embedded in each row, so scoring remains
    # reproducible without storing multi-hundred-thousand-token prompts.
    for row in rows:
        if row.get("status") != "ok":
            continue
        row["score"] = score_case(
            row["case"], row.get("response_text", ""), row.get("citation_checks")
        )
    with (args.run_dir / "results.rescored.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_results(rows)
    write_json(args.run_dir / "summary.json", summary)
    (args.run_dir / "summary.md").write_text(
        summary_markdown(manifest, summary, rows), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
