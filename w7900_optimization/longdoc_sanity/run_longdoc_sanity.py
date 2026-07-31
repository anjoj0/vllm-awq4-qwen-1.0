#!/usr/bin/env python3
"""Run the fixed Nowcast3D long-document sanity suite against a vLLM API."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from longdoc_sanity_lib import (
    append_jsonl,
    build_prompt,
    check_citations,
    extract_source_and_filler,
    load_json,
    load_jsonl,
    make_needle_cases,
    parse_json_object,
    score_case,
    summary_markdown,
    summarize_results,
    write_json,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "data" / "nowcast3d_cases.jsonl"
DEFAULT_PROFILES = HERE / "config" / "profiles.json"
DEFAULT_CORPUS = Path("/workspace/bench_data/combined_papers_for_llm_L.txt")

SMOKE_IDS = {
    "n3d_fact_001_fields",
    "n3d_numeric_002_blind_eval",
    "n3d_fact_010_metrics",
    "n3d_unanswerable_019_gpu",
}
CORE_IDS = SMOKE_IDS | {
    "n3d_fact_005_helmholtz",
    "n3d_fact_006_two_stages",
    "n3d_synthesis_008_data_regimes",
    "n3d_numeric_013_beijing_event",
    "n3d_synthesis_014_training_protocol",
    "n3d_fact_016_code_availability",
    "n3d_unanswerable_020_energy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evidence-grounded Nowcast3D QA and needle tests against vLLM."
    )
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8030"))
    parser.add_argument("--model", default=os.getenv("VLLM_SERVED_MODEL_NAME", "Qwen3.6-27B-BF16"))
    parser.add_argument("--tokenizer", default=os.getenv("VLLM_TOKENIZER", "/models/Qwen3.6-27B"))
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument(
        "--profile",
        choices=("6k", "8k", "12k", "16k", "24k", "32k", "64k", "103k", "near256k"),
        default="103k",
    )
    parser.add_argument("--context-mode", choices=("evidence", "full-paper"), default="evidence")
    parser.add_argument("--suite", choices=("smoke", "core", "full", "needle"), default="core")
    parser.add_argument("--case-id", action="append", default=[], help="Run only selected case ID(s).")
    parser.add_argument("--no-needles", action="store_true")
    parser.add_argument("--needle-seed", type=int, default=20260729)
    parser.add_argument("--config-label", required=True, help="Stable label such as bf16_tp8_auto.")
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--save-prompts", action="store_true", help="Store full prompts for offline audit.")
    parser.add_argument("--dry-run", action="store_true", help="Build/validate prompts without calling the API.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_cases(all_cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_id:
        wanted = set(args.case_id)
        selected = [case for case in all_cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in selected}
        if missing:
            raise ValueError(f"Unknown case ID(s): {sorted(missing)}")
        return selected
    if args.suite == "needle":
        return []
    if args.suite == "smoke":
        return [case for case in all_cases if case["id"] in SMOKE_IDS]
    if args.suite == "core":
        return [case for case in all_cases if case["id"] in CORE_IDS]
    return all_cases


def load_tokenizer(path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required in the benchmark environment") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> tuple[str, dict[str, Any], dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    ended = time.perf_counter()
    text = "".join(content_parts)
    # A thinking-enabled model should still place the JSON in content. Keep
    # reasoning separately so accidental empty content is visible as a failure.
    timing = {
        "wall_s": ended - started,
        "ttft_s": None if first_content_at is None else first_content_at - started,
    }
    response_meta = {
        "usage": usage,
        "finish_reason": finish_reason,
        "reasoning_content": "".join(reasoning_parts),
    }
    return text, response_meta, timing


def build_payload(args: argparse.Namespace, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }


def main() -> int:
    args = parse_args()
    profiles = load_json(args.profiles)
    profile = profiles[args.profile]
    max_tokens = args.max_output_tokens or profile["max_output_tokens"]
    if profile["target_prompt_tokens"] + max_tokens > profile["max_model_len"]:
        raise SystemExit("Profile prompt and output token budgets exceed max_model_len")

    cases = select_cases(load_jsonl(args.cases), args)
    if not args.no_needles and not args.case_id:
        cases.extend(make_needle_cases(args.profile, args.needle_seed))
    if not cases:
        raise SystemExit("No cases selected")

    corpus = args.corpus.read_text(encoding="utf-8")
    source_text, filler_text, document_names = extract_source_and_filler(corpus)
    tokenizer = load_tokenizer(args.tokenizer)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{args.config_label}_{args.profile}_{args.suite}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "results.jsonl"
    prompts_dir = run_dir / "prompts"

    metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_label": args.config_label,
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "profile": args.profile,
        "profile_config": profile,
        "context_mode": args.context_mode,
        "suite": args.suite,
        "needle_seed": args.needle_seed,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_output_tokens": max_tokens,
        "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "cases": str(args.cases),
        "cases_sha256": sha256_file(args.cases),
        "document_count": len(document_names),
        "case_ids": [case["id"] for case in cases],
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "dry_run": args.dry_run,
    }
    write_json(run_dir / "manifest.json", metadata)

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] building {case['id']}...", flush=True)
        try:
            prompt = build_prompt(
                tokenizer=tokenizer,
                case=case,
                filler_text=filler_text,
                source_text=source_text,
                target_prompt_tokens=profile["target_prompt_tokens"],
                context_mode=args.context_mode,
            )
            if args.save_prompts:
                write_json(prompts_dir / f"{case['id']}.json", {"messages": prompt.messages})
            row: dict[str, Any] = {
                "status": "dry_run" if args.dry_run else "pending",
                "case": case,
                "context": {
                    "profile": args.profile,
                    "mode": args.context_mode,
                    "prompt_tokens": prompt.prompt_tokens,
                    "context_chars": prompt.context_chars,
                    "context_sha256": prompt.context_sha256,
                    "inserted_fraction": prompt.inserted_fraction,
                },
            }
            if args.dry_run:
                rows.append(row)
                append_jsonl(results_path, row)
                continue

            print(
                f"[{index}/{len(cases)}] requesting {case['id']} "
                f"({prompt.prompt_tokens} prompt tokens, source at {prompt.inserted_fraction:.1%})...",
                flush=True,
            )
            response_text, response_meta, timing = post_stream(
                endpoint,
                build_payload(args, prompt.messages, max_tokens),
                args.timeout,
            )
            parsed, _ = parse_json_object(response_text)
            citation_checks = check_citations(parsed, prompt.messages[1]["content"])
            row.update(
                {
                    "status": "ok",
                    "response_text": response_text,
                    "response": parsed,
                    "response_meta": response_meta,
                    "timing": timing,
                    "citation_checks": citation_checks,
                    "score": score_case(case, response_text, citation_checks),
                }
            )
        except (Exception, urllib.error.URLError) as exc:  # Keep the suite auditable after one failed request.
            row = {
                "status": "error",
                "case": case,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"ERROR {case['id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        rows.append(row)
        append_jsonl(results_path, row)

    if args.dry_run:
        write_json(run_dir / "dry_run_summary.json", {
            "built": len(rows),
            "errors": sum(row["status"] == "error" for row in rows),
            "prompt_tokens": [row.get("context", {}).get("prompt_tokens") for row in rows],
        })
        print(f"Dry run complete: {run_dir}")
        return int(any(row["status"] == "error" for row in rows))

    summary = summarize_results(rows)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        summary_markdown(metadata, summary, rows), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {run_dir}")
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
