#!/usr/bin/env python3
"""Functional API acceptance for a local Strix service; no historical TPS gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "expo_assistant"))
from server import iter_sse


class Client:
    def __init__(self, host, model, timeout):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(self.host + path, data=data,
                                         headers={"Content-Type": "application/json"})
        return self.opener.open(request, timeout=timeout or self.timeout)

    def json(self, path, payload=None):
        with self.request(path, payload) as response:
            return json.load(response)

    def chat(self, prompt):
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wait_health(client, seconds):
    deadline = time.monotonic() + seconds
    last_error = ""
    while True:
        try:
            with client.request("/health", timeout=5) as response:
                require(response.status == 200, "health is not 200")
            return {"health": "ready"}
        except (OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, PermissionError):
                raise RuntimeError("This execution environment cannot open the API socket") from exc
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Health timeout: {last_error}")
        print(f"Waiting for model readiness: {last_error}", flush=True)
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def check_models(client):
    data = client.json("/v1/models")
    require(any(m.get("id") == client.model for m in data.get("data", [])),
            "Configured model missing from /v1/models")
    return data


def check_chat(client):
    data = client.json("/v1/chat/completions", client.chat("计算 19 + 23，只输出结果数字。"))
    choice = data["choices"][0]
    message = choice["message"]
    require((message.get("content") or "").strip() == "42", "Arithmetic answer must be 42")
    require(not (message.get("reasoning") or message.get("reasoning_content")),
            "Unexpected reasoning with thinking disabled")
    require(choice.get("finish_reason") == "stop", "Chat did not finish normally")
    require(data.get("usage", {}).get("completion_tokens", 0) > 0, "Missing token usage")
    return data


def check_stream(client, responses=False):
    if responses:
        path = "/v1/responses"
        payload = {"model": client.model, "input": "用一句中文介绍 Strix Halo。",
                   "temperature": 0, "max_output_tokens": 256, "stream": True, "store": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
    else:
        path = "/v1/chat/completions"
        payload = {**client.chat("用一句中文介绍 Strix Halo。"), "stream": True,
                   "stream_options": {"include_usage": True}}
    text, usage, terminal, finished, reasoning = [], {}, False, False, []
    with client.request(path, payload) as response:
        require("text/event-stream" in response.headers.get("Content-Type", ""), "Expected SSE")
        for name, raw in iter_sse(response):
            if raw == "[DONE]":
                if not responses:
                    terminal = True
                break
            data = json.loads(raw)
            require(not data.get("error"), f"Stream error: {data.get('error')}")
            event = name or data.get("type")
            if responses:
                if event == "response.output_text.delta":
                    text.append(data.get("delta", ""))
                elif event in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                    reasoning.append(data.get("delta", ""))
                elif event == "response.completed":
                    usage = data["response"].get("usage", {})
                    terminal = finished = True
                    break
                elif event in ("response.failed", "response.incomplete", "error"):
                    raise ValueError(f"Responses terminated with {event}: {data}")
            else:
                usage = data.get("usage") or usage
                for choice in data.get("choices", []):
                    delta = choice.get("delta") or {}
                    text.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                    if choice.get("finish_reason"):
                        require(choice["finish_reason"] == "stop", "Chat stream was truncated")
                        finished = True
    require(terminal and finished and bool("".join(text).strip()), "Incomplete/empty stream")
    require(not "".join(reasoning).strip(), "Unexpected reasoning with thinking disabled")
    require(usage.get("output_tokens", usage.get("completion_tokens", 0)) > 0, "Missing stream usage")
    return {"text": "".join(text), "usage": usage, "terminal_event_received": terminal}


def check_tool(client):
    payload = client.chat("调用 demo_lookup 查询 topic 为 Strix Halo 的信息。")
    payload["tools"] = [{"type": "function", "function": {
        "name": "demo_lookup", "description": "Look up a demo topic; this test does not execute tools.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}},
                       "required": ["topic"], "additionalProperties": False}}}]
    payload["tool_choice"] = {"type": "function", "function": {"name": "demo_lookup"}}
    data = client.json("/v1/chat/completions", payload)
    choice = data["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    # v0.32 returns "stop" for a named tool choice; auto/required use "tool_calls".
    require(choice.get("finish_reason") in ("stop", "tool_calls") and len(calls) == 1,
            "Missing or incomplete tool call")
    require(calls[0]["function"]["name"] == "demo_lookup", "Wrong function")
    args = json.loads(calls[0]["function"]["arguments"])
    require(args.get("topic") == "Strix Halo", "Incorrect tool arguments")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-27B-AWQ4")
    parser.add_argument("--label", default="manual", help="Requested profile label, not proof of the running backend")
    parser.add_argument("--wait", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    client = Client(args.host, args.model, args.timeout)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "host": args.host,
              "model": args.model, "requested_profile": args.label, "steps": [],
              "scope": "Short text, streaming and forced tool-call smoke only; not a quality/performance benchmark."}
    checks = [("health", lambda: wait_health(client, args.wait)),
              ("models", lambda: check_models(client)),
              ("chat_arithmetic", lambda: check_chat(client)),
              ("chat_stream", lambda: check_stream(client)),
              ("responses_stream", lambda: check_stream(client, responses=True)),
              ("tool_call", lambda: check_tool(client))]
    for name, check in checks:
        started = time.perf_counter()
        row = {"name": name, "passed": False}
        try:
            row["result"] = check()
            row["passed"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError):
                row["upstream_error"] = exc.read(4000).decode(errors="replace")
        row["wall_seconds"] = round(time.perf_counter() - started, 4)
        report["steps"].append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name} ({row['wall_seconds']}s) {row.get('error', '')}", flush=True)
        if not row["passed"] and name in ("health", "models"):
            break
    report["passed"] = len(report["steps"]) == len(checks) and all(r["passed"] for r in report["steps"])
    path = args.out or ROOT / ".runtime/strix-halo" / ("smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Evidence: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
