#!/usr/bin/env python3
"""Validate frozen cases and their evidence against the local paper corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from longdoc_sanity_lib import SOURCE_DOCUMENT, extract_source_and_filler, load_json, load_jsonl


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=HERE / "data" / "nowcast3d_cases.jsonl")
    parser.add_argument("--profiles", type=Path, default=HERE / "config" / "profiles.json")
    args = parser.parse_args()

    source, filler, names = extract_source_and_filler(
        args.corpus.read_text(encoding="utf-8")
    )
    cases = load_jsonl(args.cases)
    profiles = load_json(args.profiles)
    errors: list[str] = []
    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if case_id in ids:
            errors.append(f"duplicate case ID: {case_id}")
        ids.add(case_id)
        if case.get("source_document") != SOURCE_DOCUMENT:
            errors.append(f"{case_id}: unexpected source document")
        if case.get("position_bucket") not in {"early", "middle", "late", "very_late"}:
            errors.append(f"{case_id}: invalid position bucket")
        if case.get("answerable") and not case.get("required_answer_groups"):
            errors.append(f"{case_id}: answerable case has no required groups")
        if not case.get("answerable") and case.get("gold_evidence"):
            errors.append(f"{case_id}: unanswerable case must not include answer evidence")
        for evidence in case.get("gold_evidence", []):
            if evidence not in source:
                errors.append(f"{case_id}: evidence is not an exact source substring: {evidence[:80]!r}")
    for name, profile in profiles.items():
        if profile["target_prompt_tokens"] + profile["max_output_tokens"] > profile["max_model_len"]:
            errors.append(f"profile {name}: prompt + output exceeds max_model_len")
    print(f"documents={len(names)} source_chars={len(source)} filler_chars={len(filler)} cases={len(cases)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: all frozen evidence is present verbatim and suite metadata is consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
