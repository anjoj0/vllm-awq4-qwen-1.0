"""HTTP integration checks for the offline exhibit proxy.

Run: python -m unittest discover -s expo_assistant -p test_server.py -v
Only loopback HTTP servers are used; no model or third-party package is needed.
"""
from __future__ import annotations

from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import queue
import socket
import sys
import threading
import time
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "_exhibit_proxy_under_test", Path(__file__).with_name("server.py")
)
assert _SPEC is not None and _SPEC.loader is not None
proxy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = proxy
_SPEC.loader.exec_module(proxy)

MODEL = "Qwen3.6-27B-AWQ4"
PREFILL = "vllm:request_prefill_time_seconds"
DECODE = "vllm:request_decode_time_seconds"
COMPUTED = "vllm:request_prefill_kv_computed_tokens"
BASE = {
    PREFILL + "_count": 10,
    PREFILL + "_sum": 5,
    DECODE + "_count": 10,
    DECODE + "_sum": 20,
    "vllm:request_prompt_tokens_sum": 1200,
    "vllm:request_generation_tokens_sum": 110,
    COMPUTED + "_count": 10,
    COMPUTED + "_sum": 800,
    "vllm:request_success_total": 10,
    "vllm:num_requests_running": 0,
    "vllm:num_requests_waiting": 0,
}
USAGE = {
    "input_tokens": 120,
    "output_tokens": 11,
    "input_tokens_details": {"cached_tokens": 40},
}
ANSWER = "```python\nprint('你好')\n```"


def finished_values(before=None):
    values = dict(BASE if before is None else before)
    for key, amount in {
        PREFILL + "_count": 1, PREFILL + "_sum": 0.5,
        DECODE + "_count": 1, DECODE + "_sum": 2,
        "vllm:request_prompt_tokens_sum": 120,
        "vllm:request_generation_tokens_sum": 11,
        COMPUTED + "_count": 1, COMPUTED + "_sum": 80,
        "vllm:request_success_total": 1,
    }.items():
        values[key] = values.get(key, 0) + amount
    return values


def responses_events(usage=None):
    return [
        ("response.reasoning_text.delta", {"delta": "先分析问题。"}),
        ("response.output_text.delta", {"delta": ANSWER}),
        ("response.completed", {
            "response": {"usage": USAGE if usage is None else usage, "output": []}
        }),
    ]


def chat_events():
    return [
        ("", {"choices": [{"delta": {"reasoning_content": "先分析问题。"}}]}),
        ("", {"choices": [{"delta": {"content": ANSWER}}]}),
        ("", {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ("", {"choices": [], "usage": {
            "prompt_tokens": 120, "completion_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 40},
        }}),
        ("", "[DONE]"),
    ]


class Plan:
    def __init__(self, *, events=None, metrics=None, status=200, hold=False):
        self.events = responses_events() if events is None else events
        self.metrics = metrics
        self.status = status
        self.hold = hold
        self.started = threading.Event()
        self.disconnected = threading.Event()


class FakeVllm:
    def __init__(self, *, before=None, metrics_status=200, other_model=False):
        self.values = dict(BASE if before is None else before)
        self.metrics_status = metrics_status
        self.other_model = other_model
        self.plans = queue.Queue()
        self.captures = []
        self.metrics_reads = 0
        self.pending_metrics = []
        self.guard = threading.Lock()

    def metric_text(self):
        with self.guard:
            self.metrics_reads += 1
            if self.pending_metrics:
                self.values = dict(self.pending_metrics.pop(0))
            values = dict(self.values)
        lines = []
        for name, value in values.items():
            labels = 'model_name="' + MODEL + '",engine="0"'
            if name == "vllm:request_success_total":
                labels += ',finished_reason="stop"'
            lines.append(name + "{" + labels + "} " + str(value))
            if self.other_model:
                lines.append(
                    name + '{model_name="unrelated-model",engine="0"} 99999'
                )
        return "\n".join(lines).encode("utf-8")

    def complete(self, plan):
        with self.guard:
            frames = plan.metrics
            if frames is None:
                frames = [finished_values(self.values)]
            self.pending_metrics = [dict(frame) for frame in frames]


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    @property
    def state(self):
        return self.server.state

    def send_bytes(self, status, raw, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/metrics":
            raw = self.state.metric_text()
            self.send_bytes(self.state.metrics_status, raw, "text/plain")
        elif self.path == "/v1/models":
            raw = json.dumps({"data": [{"id": MODEL}]}).encode()
            self.send_bytes(200, raw, "application/json")
        else:
            self.send_bytes(404, b"missing", "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        plan = self.state.plans.get(timeout=2)
        self.state.captures.append((self.path, body, dict(self.headers)))
        plan.started.set()
        if plan.status != 200:
            self.send_bytes(plan.status, b'{"error":{"message":"model busy"}}',
                            "application/json")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            if not plan.hold:
                self.state.complete(plan)
            for name, obj in plan.events:
                data = obj if isinstance(obj, str) else json.dumps(
                    obj, ensure_ascii=False
                )
                raw = ("event: " + name + "\n" if name else "") + "data: " + data + "\n\n"
                self.wfile.write(raw.encode("utf-8"))
                self.wfile.flush()
            if plan.hold:
                # The proxy has consumed the POST body; EOF proves it closed
                # its upstream socket. This says nothing about GPU completion.
                self.connection.settimeout(4)
                if self.connection.recv(1) == b"":
                    plan.disconnected.set()
        except socket.timeout:
            pass
        except OSError:
            if plan.hold:
                plan.disconnected.set()


@contextmanager
def running_app(*, mode="responses", state=None, demo=False):
    state = FakeVllm() if state is None else state
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    upstream.daemon_threads = True
    upstream.state = state
    config = proxy.Config(
        upstream="http://127.0.0.1:" + str(upstream.server_port),
        model=MODEL, api_mode=mode, timeout=3, metrics_timeout=0.5,
        metrics_wait=0.45, demo=demo,
    )
    app = proxy.create_server(config, host="127.0.0.1", port=0)
    threads = []
    for server in (upstream, app):
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        threads.append(thread)
    try:
        yield app, state
    finally:
        if app.app.active is not None:
            app.app.active.cancel()
        for server in (app, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def post(app, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", app.server_port, timeout=5)
    try:
        conn.request("POST", path, json.dumps(body).encode("utf-8"),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def decode_events(raw):
    result = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        result.append((name, json.loads(data)))
    return result


def chat_body(request_id="turn1", messages=None):
    return {
        "request_id": request_id,
        "messages": messages or [{"role": "user", "content": "给出 Python 示例"}],
        "max_tokens": 128, "temperature": 0, "thinking": True,
    }


class ProxyHttpTests(unittest.TestCase):
    def conversation(self, app, body=None):
        status, raw = post(app, "/api/chat", chat_body() if body is None else body)
        self.assertEqual(status, 200, raw)
        events = decode_events(raw)
        self.assertEqual(events[0][0], "meta")
        self.assertEqual(events[-1][0], "done")
        return events

    def stats(self, events):
        values = [obj for name, obj in events if name == "stats"]
        self.assertEqual(len(values), 1, events)
        self.assertFalse(any(name == "error" for name, _ in events), events)
        return values[0]

    def test_responses_stream_preserves_channels_multiturn_and_server_rates(self):
        state = FakeVllm(other_model=True)
        state.plans.put(Plan())
        messages = [
            {"role": "user", "content": "原问题"},
            {"role": "assistant", "content": "之前的回答"},
            {"role": "user", "content": "继续修改代码"},
        ]
        with running_app(state=state) as (app, state):
            events = self.conversation(app, chat_body(messages=messages))
            stats = self.stats(events)
            deltas = [obj for name, obj in events if name == "delta"]
            self.assertEqual(deltas, [
                {"text": "先分析问题。", "channel": "reasoning"},
                {"text": ANSWER, "channel": "content"},
            ])
            path, payload, headers = state.captures[0]
            self.assertEqual(path, "/v1/responses")
            self.assertEqual(payload["input"], messages)
            self.assertEqual(payload["instructions"], proxy.SYSTEM_PROMPT)
            self.assertTrue(payload["stream"])
            self.assertFalse(payload["store"])
            self.assertEqual(payload["max_output_tokens"], 128)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(headers["X-Request-Id"], "turn1")
            self.assertEqual(stats["input_tokens"], 120)
            self.assertEqual(stats["cached_tokens"], 40)
            self.assertEqual(stats["output_tokens"], 11)
            self.assertEqual(stats["prefill_tps"], 160)
            self.assertEqual(stats["decode_tps"], 5)  # (11 - 1) / 2
            self.assertEqual(stats["metric_source"], "server")
            self.assertFalse(events[-1][1]["cancelled"])

    def test_chat_sse_usage_after_finish_reason_and_multiturn_payload(self):
        state = FakeVllm()
        state.plans.put(Plan(events=chat_events()))
        messages = [
            {"role": "user", "content": "第一轮"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "第二轮"},
        ]
        with running_app(mode="chat", state=state) as (app, state):
            stats = self.stats(self.conversation(app, chat_body(messages=messages)))
            path, payload, _ = state.captures[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(payload["messages"][1:], messages)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["stream_options"], {"include_usage": True})
            self.assertEqual(payload["max_tokens"], 128)
            self.assertEqual(stats["cached_tokens"], 40)
            self.assertEqual(stats["prefill_tps"], 160)
            self.assertEqual(stats["decode_tps"], 5)

    def test_delayed_metrics_require_a_complete_stable_snapshot(self):
        after = finished_values()
        partial = dict(after)
        partial[DECODE + "_count"] = BASE[DECODE + "_count"]
        state = FakeVllm()
        state.plans.put(Plan(metrics=[BASE, partial, after, after]))
        with running_app(state=state) as (app, state):
            stats = self.stats(self.conversation(app))
            self.assertEqual(stats["decode_tps"], 5)
            self.assertEqual(stats["prefill_tps"], 160)
            self.assertGreaterEqual(state.metrics_reads, 5)  # before + four polls

    def test_computed_prefill_histogram_recovers_missing_cached_usage(self):
        usage = {"input_tokens": 120, "output_tokens": 11}
        state = FakeVllm()
        state.plans.put(Plan(events=responses_events(usage)))
        with running_app(state=state) as (app, _):
            stats = self.stats(self.conversation(app))
            self.assertEqual(stats["cached_tokens"], 40)
            self.assertEqual(stats["prefill_tps"], 160)

    def test_usage_cached_fallback_and_unknown_cache_do_not_inflate_prefill(self):
        before = {k: v for k, v in BASE.items() if not k.startswith(COMPUTED)}
        after = {k: v for k, v in finished_values().items() if not k.startswith(COMPUTED)}
        for cached in (40, None):
            with self.subTest(cached=cached):
                usage = {"input_tokens": 120, "output_tokens": 11}
                if cached is not None:
                    usage["input_tokens_details"] = {"cached_tokens": cached}
                state = FakeVllm(before=before)
                state.plans.put(Plan(events=responses_events(usage), metrics=[after]))
                with running_app(state=state) as (app, _):
                    stats = self.stats(self.conversation(app))
                    self.assertEqual(stats["prefill_tps"], 160 if cached == 40 else None)
                    self.assertEqual(stats["decode_tps"], 5)

    def test_unavailable_metrics_never_create_engine_speed(self):
        state = FakeVllm(metrics_status=503)
        state.plans.put(Plan())
        with running_app(state=state) as (app, _):
            stats = self.stats(self.conversation(app))
            self.assertEqual(stats["output_tokens"], 11)
            self.assertIsNone(stats["prefill_tps"])
            self.assertIsNone(stats["decode_tps"])
            self.assertNotEqual(stats["metric_source"], "server")

    def test_shared_incomplete_restart_or_mismatched_metrics_have_no_rates(self):
        complete = finished_values()
        concurrent = dict(complete)
        concurrent[PREFILL + "_count"] += 1
        concurrent[DECODE + "_count"] += 1
        running = dict(complete)
        running["vllm:num_requests_running"] = 1
        incomplete = dict(complete)
        del incomplete[DECODE + "_count"]
        restarted = {key: 0 for key in complete}
        mismatched = dict(complete)
        mismatched["vllm:request_generation_tokens_sum"] += 7
        for label, frame in [
            ("concurrent", concurrent), ("running", running),
            ("missing", incomplete), ("restart", restarted),
            ("usage mismatch", mismatched),
        ]:
            with self.subTest(case=label):
                state = FakeVllm()
                state.plans.put(Plan(metrics=[frame]))
                with running_app(state=state) as (app, _):
                    stats = self.stats(self.conversation(app))
                    self.assertIsNone(stats["prefill_tps"])
                    self.assertIsNone(stats["decode_tps"])
                    self.assertNotEqual(stats["metric_source"], "server")

    def test_one_output_token_has_no_decode_rate(self):
        usage = {**USAGE, "output_tokens": 1}
        after = finished_values()
        after["vllm:request_generation_tokens_sum"] = BASE["vllm:request_generation_tokens_sum"] + 1
        after[DECODE + "_sum"] = BASE[DECODE + "_sum"]
        state = FakeVllm()
        state.plans.put(Plan(events=responses_events(usage), metrics=[after]))
        with running_app(state=state) as (app, _):
            stats = self.stats(self.conversation(app))
            self.assertIsNone(stats["decode_tps"])

    def test_truncated_stream_reports_error_without_success_stats(self):
        for mode, events in [
            ("responses", [("response.output_text.delta", {"delta": "未完成"})]),
            ("chat", [("", {"choices": [{"delta": {"content": "未完成"}}]})]),
        ]:
            with self.subTest(mode=mode):
                state = FakeVllm()
                state.plans.put(Plan(events=events))
                with running_app(mode=mode, state=state) as (app, _):
                    received = self.conversation(app)
                    self.assertTrue(any(name == "error" for name, _ in received))
                    self.assertFalse(any(name == "stats" for name, _ in received))
                    self.assertFalse(received[-1][1]["cancelled"])

    def test_responses_done_sentinel_does_not_replace_terminal_response_event(self):
        state = FakeVllm()
        state.plans.put(Plan(events=[
            ("response.output_text.delta", {"delta": "尚未确认完成"}),
            ("", "[DONE]"),
        ]))
        with running_app(state=state) as (app, _):
            received = self.conversation(app)
            self.assertTrue(any(name == "error" for name, _ in received), received)
            self.assertFalse(any(name == "stats" for name, _ in received), received)

    def test_upstream_http_and_sse_errors_never_emit_success_stats(self):
        plans = [
            Plan(status=503),
            Plan(events=[("error", {"error": {"message": "invalid model"}})]),
        ]
        for plan in plans:
            with self.subTest(status=plan.status):
                state = FakeVllm()
                state.plans.put(plan)
                with running_app(state=state) as (app, _):
                    events = self.conversation(app)
                    errors = [obj["message"] for name, obj in events if name == "error"]
                    self.assertEqual(len(errors), 1)
                    self.assertTrue("model" in errors[0] or "503" in errors[0])
                    self.assertFalse(any(name == "stats" for name, _ in events))

    def test_demo_has_no_upstream_calls_or_performance_figures(self):
        with running_app(demo=True) as (app, state):
            events = self.conversation(app)
            stats = self.stats(events)
            self.assertEqual(stats["source"], "demo")
            for field in ("input_tokens", "output_tokens", "cached_tokens",
                          "prefill_tps", "decode_tps", "delivery_tps", "ttft_s"):
                self.assertIsNone(stats[field], field)
            self.assertEqual(state.captures, [])
            self.assertEqual(state.metrics_reads, 0)

    def test_cancel_interrupts_upstream_socket_and_releases_single_request_slot(self):
        hold = Plan(
            events=[("response.output_text.delta", {"delta": "开始生成"})],
            hold=True,
        )
        state = FakeVllm()
        state.plans.put(hold)
        state.plans.put(Plan())
        with running_app(state=state) as (app, _):
            outcome = queue.Queue()

            def consume():
                try:
                    outcome.put(self.conversation(app, chat_body("cancellable")))
                except BaseException as exc:
                    outcome.put(exc)

            client = threading.Thread(target=consume, daemon=True)
            client.start()
            self.assertTrue(hold.started.wait(2), "upstream never received the request")
            status, raw = post(app, "/api/chat", chat_body("overlap"))
            self.assertEqual(status, 409, raw)
            status, raw = post(app, "/api/cancel", {"request_id": "different"})
            self.assertEqual(status, 200)
            self.assertFalse(json.loads(raw)["cancelled"])
            status, raw = post(app, "/api/cancel", {"request_id": "cancellable"})
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(raw)["cancelled"])
            client.join(timeout=2)
            self.assertFalse(client.is_alive(), "cancel did not unblock streaming read")
            result = outcome.get(timeout=1)
            if isinstance(result, BaseException):
                raise result
            self.assertTrue(result[-1][1]["cancelled"])
            self.assertFalse(any(name in ("stats", "error") for name, _ in result))
            self.assertTrue(hold.disconnected.wait(1), "upstream socket was not closed")
            deadline = time.monotonic() + 1
            while app.app.active is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNone(app.app.active, "cancelled request retained the slot")
            stats = self.stats(self.conversation(app, chat_body("after_cancel")))
            self.assertEqual(stats["decode_tps"], 5)
            self.assertEqual(len(state.captures), 2)  # rejected overlap never forwarded


if __name__ == "__main__":
    unittest.main()
