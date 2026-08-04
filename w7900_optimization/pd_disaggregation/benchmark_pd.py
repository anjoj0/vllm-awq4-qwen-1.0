#!/usr/bin/env python3
"""Benchmark a vLLM completions endpoint with exact token-count prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_prompt_tokens(tokenizer, source: Path, target_tokens: int) -> list[int]:
    text = source.read_text(encoding="utf-8", errors="replace")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError(f"No tokens found in {source}")
    repeats = (target_tokens + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:target_tokens]


async def run_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    request_index: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20260802,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = None
    completion_tokens = None
    text_parts: list[str] = []
    async with client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            item = json.loads(line[6:])
            choices = item.get("choices") or []
            if choices:
                piece = choices[0].get("text") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
            usage = item.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
    ended = time.perf_counter()
    return {
        "request_index": request_index,
        "wall_s": ended - started,
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "completion_tokens": completion_tokens,
        "output_tok_s": (
            None
            if not completion_tokens or ended <= (first_token_at or ended)
            else completion_tokens / (ended - (first_token_at or ended))
        ),
        "text": "".join(text_parts),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen3.6-27B-BF16-PD")
    parser.add_argument("--tokenizer", default="/models/Qwen3.6-27B")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--max-tokens", default=32, type=int)
    parser.add_argument("--concurrency", default=1, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    prompt = build_prompt_tokens(tokenizer, args.source, args.prompt_tokens)
    async with httpx.AsyncClient(timeout=None) as client:
        batch_started = time.perf_counter()
        results = await asyncio.gather(
            *(
                run_one(
                    client,
                    args.url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    index,
                )
                for index in range(args.concurrency)
            )
        )
        batch_wall_s = time.perf_counter() - batch_started

    valid_ttft = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    total_output = sum(r["completion_tokens"] or 0 for r in results)
    report = {
        "url": args.url,
        "prompt_tokens": len(prompt),
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "batch_wall_s": batch_wall_s,
        "aggregate_input_tok_s": len(prompt) * args.concurrency / batch_wall_s,
        "aggregate_output_tok_s": total_output / batch_wall_s,
        "mean_ttft_s": statistics.mean(valid_ttft) if valid_ttft else None,
        "p50_ttft_s": statistics.median(valid_ttft) if valid_ttft else None,
        "p95_ttft_s": percentile(valid_ttft, 0.95),
        "requests": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
