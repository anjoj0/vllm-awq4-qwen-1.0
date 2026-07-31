#!/usr/bin/env python3
"""Token-accurate long-context truncation pressure test.

This script uses the running vLLM server tokenizer to truncate a real document
prompt to a requested token budget, then measures streaming chat latency and
throughput. It is intended for 64K/128K/256K context verification.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HOST = "http://127.0.0.1:8001"
DEFAULT_MODEL = "Qwen3.6-27B-AWQ4"
DEFAULT_SOURCE = os.getenv(
    "VLLM_LONGTEXT", "/workspace/bench_data/combined_papers_for_llm.txt"
)
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "results" / "context_pressure")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


PROMPT_HEADER = """你是一名短临降雨/临近预报方向的研究综述专家。
请严格基于下面给出的论文材料，完成一次长上下文压力测试式综述：

1. 按技术路线总结短临降雨预测方法的演化；
2. 比较 ConvLSTM、TrajGRU、光流、Transformer、扩散模型和基础模型路线；
3. 提炼扩散模型在短临降雨预测中的核心改进点；
4. 总结仍未解决的问题；
5. 给出适合比赛技术报告引用的 8 条结论。

要求：只基于输入材料，不要编造论文；如果材料被截断，请说明结论只覆盖当前截断片段。

下面是论文材料：
"""


def http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: int = 1800) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tokenize(host: str, model: str, text: str) -> list[int]:
    data = http_json(
        "POST",
        f"{host}/tokenize",
        {"model": model, "prompt": text, "add_special_tokens": False},
        timeout=1800,
    )
    return data["tokens"]


def detokenize(host: str, model: str, tokens: list[int]) -> str:
    data = http_json("POST", f"{host}/detokenize", {"model": model, "tokens": tokens}, timeout=1800)
    return data.get("prompt", data.get("text", ""))


def wait_ready(host: str, max_wait: int = 900) -> None:
    deadline = time.time() + max_wait
    last: Exception | None = None
    while time.time() < deadline:
        try:
            http_json("GET", f"{host}/v1/models", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not ready after {max_wait}s: {last}")


def build_truncated_prompt(host: str, model: str, source: Path, prompt_tokens: int) -> dict[str, Any]:
    source_text = source.read_text(errors="ignore")
    full_prompt = PROMPT_HEADER + "\n" + source_text
    full_tokens = tokenize(host, model, full_prompt)
    truncated_tokens = full_tokens[:prompt_tokens]
    truncated_prompt = detokenize(host, model, truncated_tokens)
    actual_tokens = tokenize(host, model, truncated_prompt)
    return {
        "prompt": truncated_prompt,
        "source_chars": len(source_text),
        "source_bytes": len(source_text.encode("utf-8")),
        "full_prompt_tokens": len(full_tokens),
        "requested_prompt_tokens": prompt_tokens,
        "actual_prompt_tokens_before_chat_template": len(actual_tokens),
        "truncated_from_full": len(full_tokens) > len(truncated_tokens),
    }


def stream_chat(host: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_event_t: float | None = None
    first_payload_t: float | None = None
    event_count = 0
    payload_event_count = 0
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    status: int | None = None
    error: str | None = None

    try:
        with OPENER.open(req, timeout=timeout) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event_count += 1
                if first_event_t is None:
                    first_event_t = time.perf_counter()
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text:
                    content_parts.append(text)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if text or reasoning or delta.get("tool_calls"):
                    payload_event_count += 1
                    if first_payload_t is None:
                        first_payload_t = time.perf_counter()
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:4000]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    wall = time.perf_counter() - t0
    ttft = None if first_event_t is None else first_event_t - t0
    payload_ttft = None if first_payload_t is None else first_payload_t - t0
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    decode_denominator = wall - (payload_ttft or ttft or 0)

    return {
        "ok": error is None and status == 200,
        "status": status,
        "error": error,
        "wall_seconds": wall,
        "ttft_seconds": ttft,
        "payload_ttft_seconds": payload_ttft,
        "event_count": event_count,
        "payload_event_count": payload_event_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens") or prompt_tokens + completion_tokens,
        "prefill_tokens_per_ttft": prompt_tokens / ttft if ttft and prompt_tokens else None,
        "decode_tokens_per_second_stream": (
            completion_tokens / decode_denominator if decode_denominator > 0 and completion_tokens else None
        ),
        "output_tokens_per_second_e2e": completion_tokens / wall if wall > 0 and completion_tokens else None,
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
    }


def nonstream_chat(host: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    status: int | None = None
    error: str | None = None
    raw: dict[str, Any] = {}
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            status = resp.status
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:4000]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    wall = time.perf_counter() - t0
    usage = raw.get("usage") or {}
    choice = (raw.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return {
        "ok": error is None and status == 200,
        "status": status,
        "error": error,
        "wall_seconds": wall,
        "ttft_seconds": None,
        "payload_ttft_seconds": None,
        "event_count": None,
        "payload_event_count": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens") or prompt_tokens + completion_tokens,
        "prefill_tokens_per_ttft": None,
        "decode_tokens_per_second_stream": None,
        "output_tokens_per_second_e2e": completion_tokens / wall if wall > 0 and completion_tokens else None,
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
    }


def write_outputs(out_dir: Path, label: str, result: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"{stamp}_{label}.json"
    md_path = out_dir / f"{stamp}_{label}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    run = result["run"]
    gen = result["generation"]
    lines = [
        f"# Context Pressure Test: {label}",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| host | `{run['host']}` |",
        f"| model | `{run['model']}` |",
        f"| requested prompt tokens | {run['requested_prompt_tokens']} |",
        f"| actual prompt tokens before chat template | {run['actual_prompt_tokens_before_chat_template']} |",
        f"| max output tokens | {run['max_tokens']} |",
        f"| request mode | `{run['mode']}` |",
        f"| full prompt tokens | {run['full_prompt_tokens']} |",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| ok | {gen['ok']} |",
        f"| status | {gen['status']} |",
        f"| prompt tokens | {gen['prompt_tokens']} |",
        f"| completion tokens | {gen['completion_tokens']} |",
        f"| TTFT seconds | {gen['ttft_seconds']} |",
        f"| wall seconds | {gen['wall_seconds']} |",
        f"| prefill tokens/s by TTFT | {gen['prefill_tokens_per_ttft']} |",
        f"| stream decode tokens/s | {gen['decode_tokens_per_second_stream']} |",
        f"| e2e output tokens/s | {gen['output_tokens_per_second_e2e']} |",
        f"| finish reason | `{gen['finish_reason']}` |",
    ]
    if gen.get("error"):
        lines.extend(["", "## Error", "", "```text", gen["error"], "```"])
    if gen.get("content"):
        lines.extend(["", "## Output Preview", "", gen["content"][:4000]])
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--mode", choices=["nonstream", "stream"], default="nonstream")
    args = parser.parse_args()

    wait_ready(args.host)
    prompt_info = build_truncated_prompt(args.host, args.model, Path(args.source), args.prompt_tokens)
    if args.mode == "stream":
        generation = stream_chat(args.host, args.model, prompt_info["prompt"], args.max_tokens, args.timeout)
    else:
        generation = nonstream_chat(args.host, args.model, prompt_info["prompt"], args.max_tokens, args.timeout)
    result = {
        "run": {
            "host": args.host,
            "model": args.model,
            "source": args.source,
            "label": args.label,
            "mode": args.mode,
            "requested_prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            **{k: v for k, v in prompt_info.items() if k != "prompt"},
        },
        "generation": generation,
    }
    json_path, md_path = write_outputs(Path(args.out_dir), args.label, result)
    print(json.dumps({
        "label": args.label,
        "ok": generation["ok"],
        "status": generation["status"],
        "prompt_tokens": generation["prompt_tokens"],
        "completion_tokens": generation["completion_tokens"],
        "ttft_seconds": generation["ttft_seconds"],
        "wall_seconds": generation["wall_seconds"],
        "decode_tokens_per_second_stream": generation["decode_tokens_per_second_stream"],
        "json": str(json_path),
        "md": str(md_path),
        "error": generation["error"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
