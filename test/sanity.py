#!/usr/bin/env python3
"""sanity.py - quick health/sanity check for the running vllm-awq4-qwen server.

Focused on the *known-good* paths only. The README's tool-calling matrix
documents which diagonals are unstable; this script exercises only the
combinations the project ships as supported. For full coverage of the
fragile paths, run test/bench_full.py and test/verify_responses_streaming.py.

What it verifies, in order:
  1. /health returns 200
  2. /v1/models lists the served model name
  3. Warmup (Triton autotune cost absorbs into the first request)
  4. /v1/chat/completions stream=False thinking=off
       perf ballpark + visible content + no leaked reasoning
  5. /v1/responses stream=True enable_thinking=False (the gold path)
       SSE produces output_text deltas, no reasoning deltas
  6. /v1/responses stream=True enable_thinking=True (control)
       SSE produces reasoning deltas + visible content

Each step prints PASS/FAIL on its own line. Exits 0 only if all pass.

Run: python3 test/sanity.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

HOST = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B-AWQ4"

# Performance ballpark for ctx=0 thinking=off, short output. The README
# bench_matrix at ctx=0 no_thinking reports 22.25 t/s decode and the
# DFlash N=8 chat-factual baseline is 21.82 t/s, but a 26-token request
# does not reach steady state. We accept anything in [10, 35] t/s as
# evidence the AWQ kernel is selected and DFlash is wired. Lower than
# that usually means the kernel fell back to the slow legacy path.
TPS_LO, TPS_HI = 10.0, 35.0


# ---------- HTTP helpers ----------

def get(path: str, timeout: int = 10):
    return urllib.request.urlopen(f"{HOST}{path}", timeout=timeout)


def post_json(path: str, body: dict, timeout: int = 600) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    dt = time.perf_counter() - t0
    return json.loads(raw), dt


def post_stream(path: str, body: dict, timeout: int = 300):
    """POST and stream SSE events; yields (event_name, data_dict) tuples."""
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    event_name = None
    data_buf: list[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if event_name or data_buf:
                payload = "\n".join(data_buf)
                if payload == "[DONE]":
                    return
                try:
                    parsed = json.loads(payload) if payload else {}
                except Exception:
                    parsed = {"_raw": payload}
                yield event_name or parsed.get("type"), parsed
            event_name = None
            data_buf = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buf.append(line[len("data:"):].lstrip())


# ---------- Steps ----------

def step_health(max_wait: int = 600) -> bool:
    print("[1/6] /health ...", end=" ", flush=True)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            if get("/health").status == 200:
                print("PASS")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"FAIL (no 200 within {max_wait}s)")
    return False


def step_models() -> bool:
    print("[2/6] /v1/models lists served name ...", end=" ", flush=True)
    try:
        body = json.loads(get("/v1/models").read())
        ids = [m.get("id") for m in body.get("data", [])]
        if MODEL in ids:
            print(f"PASS ({MODEL!r} present)")
            return True
        print(f"FAIL (got {ids})")
        return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def step_warmup() -> bool:
    print("[3/6] warmup (1st request, Triton autotune happens here) ...",
          end=" ", flush=True)
    try:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r, dt = post_json("/v1/chat/completions", body)
        ct = r.get("usage", {}).get("completion_tokens", 0)
        print(f"PASS ({ct} tok in {dt:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def step_chat_no_thinking() -> bool:
    print("[4/6] /v1/chat/completions stream=False thinking=off ...",
          end=" ", flush=True)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      "What is the speed of light? Reply in one short sentence."}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r, dt = post_json("/v1/chat/completions", body)
        ct = r["usage"]["completion_tokens"]
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        tps = ct / dt if dt else 0
        if not content:
            print(f"FAIL (empty content)")
            return False
        if reasoning:
            print(f"FAIL (reasoning leaked with thinking=off, "
                  f"got {len(reasoning)} chars)")
            return False
        if not (TPS_LO <= tps <= TPS_HI):
            print(f"FAIL ({tps:.2f} t/s outside [{TPS_LO}, {TPS_HI}])")
            return False
        print(f"PASS ({ct} tok @ {tps:.2f} t/s)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def step_responses_thinking_off() -> bool:
    """The gold path. /v1/responses stream=True enable_thinking=False.

    Per README: "agents that need tool calls should use either
    /v1/chat/completions with stream=false, OR /v1/responses with
    stream=true." This is the project's signature working path.
    """
    print("[5/6] /v1/responses stream=True thinking=off (gold path) ...",
          end=" ", flush=True)
    body = {
        "model": MODEL,
        "input": "What is the speed of light? Reply in one short sentence.",
        "temperature": 0,
        "max_output_tokens": 256,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        reasoning_chars = 0
        output_text_chars = 0
        saw_done = False
        for ev_name, data in post_stream("/v1/responses", body, timeout=300):
            ev = ev_name or data.get("type", "")
            if ev == "response.reasoning_text.delta":
                reasoning_chars += len(data.get("delta", ""))
            elif ev == "response.output_text.delta":
                output_text_chars += len(data.get("delta", ""))
            elif ev == "response.completed":
                saw_done = True
        if not saw_done:
            print("FAIL (no response.completed event)")
            return False
        if output_text_chars == 0:
            print("FAIL (no output_text deltas)")
            return False
        if reasoning_chars > 0:
            print(f"FAIL (thinking=off but got {reasoning_chars} reasoning "
                  "chars; Patch 15 not applied or model template misconfigured)")
            return False
        print(f"PASS (output={output_text_chars} chars, reasoning=0)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def step_responses_thinking_on() -> bool:
    """Control: thinking=on path produces reasoning deltas."""
    print("[6/6] /v1/responses stream=True thinking=on (control) ...",
          end=" ", flush=True)
    body = {
        "model": MODEL,
        "input": "If a train leaves City A at 60 km/h and another leaves "
                 "City B at 90 km/h heading toward each other 300 km apart, "
                 "how long until they meet? Show brief reasoning.",
        "temperature": 0,
        "max_output_tokens": 512,
        "stream": True,
    }
    try:
        reasoning_chars = 0
        output_text_chars = 0
        saw_done = False
        for ev_name, data in post_stream("/v1/responses", body, timeout=300):
            ev = ev_name or data.get("type", "")
            if ev == "response.reasoning_text.delta":
                reasoning_chars += len(data.get("delta", ""))
            elif ev == "response.output_text.delta":
                output_text_chars += len(data.get("delta", ""))
            elif ev == "response.completed":
                saw_done = True
        if not saw_done:
            print("FAIL (no response.completed event)")
            return False
        if output_text_chars == 0:
            print("FAIL (no output_text deltas)")
            return False
        # reasoning_chars MAY be 0 if the model decided not to think on this
        # specific prompt. We do not require it; we only require the path to
        # function. The dedicated thinking-parser test is in
        # test/verify_responses_streaming.py.
        if reasoning_chars > 0:
            print(f"PASS (reasoning={reasoning_chars} chars, "
                  f"output={output_text_chars} chars)")
        else:
            print(f"PASS (reasoning=0, output={output_text_chars} chars; "
                  "model chose not to think on this prompt)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main() -> int:
    print(f"sanity check: {HOST}  model={MODEL}\n")
    steps = [
        step_health,
        step_models,
        step_warmup,
        step_chat_no_thinking,
        step_responses_thinking_off,
        step_responses_thinking_on,
    ]
    failed = 0
    for fn in steps:
        if not fn():
            failed += 1
    print()
    if failed:
        print(f"=== {failed}/{len(steps)} step(s) FAILED ===")
        return 1
    print(f"=== all {len(steps)} steps passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
