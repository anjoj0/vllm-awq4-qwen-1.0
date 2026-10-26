#!/usr/bin/env python3
"""Offline chat UI and streaming proxy for an existing vLLM service.

Python 3.9+; standard library only. This process never loads model weights.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import socket
import threading
import time
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "你是一位本地运行的编程问答助手。用中文准确、简洁地回答，必要时提供可复制的代码块，"
    "并注明语言。优先给出结论和可用示例。对不确定的信息说明条件。"
    "你只能根据对话回答，不能实际访问用户文件、执行代码或运行测试；不要声称已经完成这些操作。"
)
SAMPLE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*?)\})?\s+'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)'
    r'(?:\s+\S+)?$'
)
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass
class Config:
    upstream: str = "http://127.0.0.1:8000"
    model: str = "Qwen3.6-27B-AWQ4"
    api_mode: str = "responses"
    subtitle: str = "AMD ROCm · 本地推理"
    max_output_tokens: int = 1024
    timeout: float = 600
    metrics_timeout: float = 1.5
    metrics_wait: float = 1.2
    api_key: str = ""
    demo: bool = False

    def __post_init__(self):
        self.upstream = self.upstream.rstrip("/")
        if self.upstream.endswith("/v1"):
            self.upstream = self.upstream[:-3]
        parsed = urlsplit(self.upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("--upstream 必须是 http/https 服务地址")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("--upstream 不接受内嵌凭据、查询参数或片段")
        if self.api_mode not in ("responses", "chat"):
            raise ValueError("api_mode must be responses or chat")


@dataclass
class Snapshot:
    values: dict[str, float]
    ok: bool


def parse_metrics(text: str, model: str) -> dict[str, float]:
    """Read Prometheus samples, ignoring other models and optional timestamps."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = {}
        for key, value in LABEL.findall(raw_labels or ""):
            try:
                labels[key] = json.loads('"' + value + '"')
            except json.JSONDecodeError:
                labels[key] = value
        if "model_name" in labels and labels["model_name"] != model:
            continue
        value = float(raw_value)
        if math.isfinite(value):
            values[name] = values.get(name, 0.0) + value
    return values


def integer(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def normalize_usage(usage: dict[str, Any]) -> dict[str, Optional[int]]:
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": integer(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": integer(usage.get("output_tokens", usage.get("completion_tokens"))),
        "cached_tokens": integer(details.get("cached_tokens")),
    }


def metric_delta(before: Snapshot, after: Snapshot, key: str) -> Optional[float]:
    if not before.ok or not after.ok or key not in after.values:
        return None
    value = after.values[key] - before.values.get(key, 0.0)
    return value if math.isfinite(value) and value >= 0 else None


def make_stats(
    usage: dict[str, Any], before: Snapshot, after: Snapshot,
    total_s: float, ttft_s: Optional[float], finish_reason: Optional[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        **normalize_usage(usage),
        "prefill_tps": None, "decode_tps": None,
        "ttft_s": ttft_s, "total_s": total_s, "delivery_tps": None,
        "metric_source": "unavailable",
        "note": "等待服务端统计；速度不会按字符数或流式片段数估算。",
        "finish_reason": finish_reason, "source": "live",
    }
    output = result["output_tokens"]
    if output is not None and ttft_s is not None and total_s > ttft_s:
        # Delivery-window throughput is deliberately separate from engine decode.
        result["delivery_tps"] = output / (total_s - ttft_s)
    pre_count = metric_delta(before, after, "vllm:request_prefill_time_seconds_count")
    dec_count = metric_delta(before, after, "vllm:request_decode_time_seconds_count")
    if not before.ok or not after.ok:
        result["metric_source"] = "client" if output is not None else "unavailable"
        result["note"] = "/metrics 不可用；仅显示客户端耗时和 usage，Prefill/Decode 暂不提供。"
        return result
    running = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    busy = any(s.values.get(name, 0) > 0 for s in (before, after) for name in running)
    if busy or any(count is not None and count > 1.00001 for count in (pre_count, dec_count)):
        result["metric_source"] = "shared"
        result["note"] = "检测到同模型并发/其他请求，聚合指标无法归因本轮，速度暂不显示。"
        return result
    if pre_count != 1 or dec_count != 1:
        result["metric_source"] = "client" if output is not None else "unavailable"
        result["note"] = "未获得完整的单请求阶段统计；Prefill/Decode 保留为空。"
        return result
    pre_s = metric_delta(before, after, "vllm:request_prefill_time_seconds_sum")
    dec_s = metric_delta(before, after, "vllm:request_decode_time_seconds_sum")
    prompt_hist = metric_delta(before, after, "vllm:request_prompt_tokens_sum")
    output_hist = metric_delta(before, after, "vllm:request_generation_tokens_sum")
    # Histograms are recorded per finished request, unlike periodic token counters.
    if result["input_tokens"] is None and prompt_hist is not None:
        result["input_tokens"] = int(prompt_hist)
    if result["output_tokens"] is None and output_hist is not None:
        result["output_tokens"] = int(output_hist)
    for actual, hist in ((result["input_tokens"], prompt_hist), (result["output_tokens"], output_hist)):
        if actual is not None and hist is not None and abs(actual - hist) > 0.1:
            result["metric_source"] = "shared"
            result["note"] = "usage 与服务端请求 token 统计不匹配，无法可靠归因阶段速度。"
            return result
    prompt = result["input_tokens"]
    cached = result["cached_tokens"]
    computed = None
    computed_count = metric_delta(before, after, "vllm:request_prefill_kv_computed_tokens_count")
    computed_sum = metric_delta(before, after, "vllm:request_prefill_kv_computed_tokens_sum")
    if computed_count == 1 and computed_sum is not None:
        computed = computed_sum
        if prompt is not None and computed <= prompt and cached is None:
            result["cached_tokens"] = cached = int(prompt - computed)
    elif prompt is not None and cached is not None and cached <= prompt:
        computed = prompt - cached
    if computed is not None and computed > 0 and pre_s and pre_s > 0:
        result["prefill_tps"] = computed / pre_s
    output = result["output_tokens"]
    if output is not None and output > 1 and dec_s and dec_s > 0:
        # vLLM decode duration starts at the FIRST generated token.
        result["decode_tps"] = (output - 1) / dec_s
    result["metric_source"] = "server"
    result["note"] = "服务端阶段统计差分；仅专用单请求时可归因。Decode 排除 prefill 产生的首 token。"
    if cached is None:
        result["note"] += " 上游未报告缓存 token，Prefill 暂不计算。"
    elif prompt is not None and cached == prompt:
        result["note"] += " 输入全部命中缓存，Prefill 暂不计算。"
    if finish_reason in ("length", "max_output_tokens"):
        result["note"] += " 本轮达到输出长度上限，可提高上限或继续追问。"
    return result


class ActiveRequest:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.cancelled = threading.Event()
        self.guard = threading.Lock()
        self.sock: Optional[socket.socket] = None

    def attach(self, sock: Optional[socket.socket]):
        with self.guard:
            self.sock = sock
            if self.cancelled.is_set() and sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def cancel(self):
        self.cancelled.set()
        with self.guard:
            if self.sock:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def iter_sse(response) -> Iterator[tuple[str, str]]:
    name, data = "", []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield name, "\n".join(data)
            name, data = "", []
        elif line.startswith("event:"):
            name = line[6:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield name, "\n".join(data)


class App:
    def __init__(self, config: Config):
        self.config = config
        self.guard = threading.Lock()
        self.active: Optional[ActiveRequest] = None

    def begin(self, request_id: str) -> Optional[ActiveRequest]:
        with self.guard:
            if self.active:
                return None
            self.active = ActiveRequest(request_id)
            return self.active

    def finish(self, active: ActiveRequest):
        with self.guard:
            if self.active is active:
                self.active = None

    def headers(self) -> dict[str, str]:
        result = {"Accept": "application/json"}
        if self.config.api_key:
            result["Authorization"] = "Bearer " + self.config.api_key
        return result

    def connection(self, active: Optional[ActiveRequest] = None):
        parsed = urlsplit(self.config.upstream)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(parsed.hostname, parsed.port, timeout=5)
        conn.connect()
        if conn.sock:
            conn.sock.settimeout(self.config.timeout if active else self.config.metrics_timeout)
        if active:
            active.attach(conn.sock)
            if active.cancelled.is_set():
                conn.close()
                raise InterruptedError("cancelled")
        return conn, parsed.path.rstrip("/")

    def get(self, suffix: str) -> tuple[int, bytes]:
        conn, prefix = self.connection()
        try:
            conn.request("GET", prefix + suffix, headers=self.headers())
            response = conn.getresponse()
            return response.status, response.read(4 * 1024 * 1024)
        finally:
            conn.close()

    def metrics(self) -> Snapshot:
        try:
            status, body = self.get("/metrics")
            return Snapshot(parse_metrics(body.decode("utf-8"), self.config.model), status == 200)
        except (OSError, ValueError, http.client.HTTPException):
            return Snapshot({}, False)

    def completed_metrics(self, before: Snapshot, active: ActiveRequest) -> Snapshot:
        if not before.ok:
            return Snapshot({}, False)
        deadline = time.monotonic() + self.config.metrics_wait
        previous_values = None
        count_keys = (
            "vllm:request_prefill_time_seconds_count",
            "vllm:request_decode_time_seconds_count",
        )
        while True:
            after = self.metrics()
            count = metric_delta(before, after, "vllm:request_decode_time_seconds_count")
            if count is not None and count > 1:
                return after
            if not after.ok or time.monotonic() >= deadline or active.cancelled.is_set():
                return after
            finished = all(metric_delta(before, after, key) == 1 for key in count_keys)
            if finished and previous_values == after.values:
                return after
            previous_values = after.values
            active.cancelled.wait(0.1)

    def status(self) -> dict[str, Any]:
        if self.config.demo:
            return {"ok": True, "model": self.config.model, "models": [self.config.model],
                    "metrics_available": False, "message": "界面预览：模拟文本，不调用模型，不提供测速。"}
        try:
            status, raw = self.get("/v1/models")
            if status != 200:
                raise ValueError("模型列表 HTTP " + str(status))
            models = [item["id"] for item in json.loads(raw).get("data", []) if isinstance(item.get("id"), str)]
            found = self.config.model in models
            return {"ok": found, "model": self.config.model, "models": models,
                    "metrics_available": self.metrics().ok,
                    "message": "服务已连接" if found else "模型名称不匹配，请用 --model 指定 /v1/models 中的 ID。"}
        except (OSError, ValueError, KeyError, http.client.HTTPException) as exc:
            return {"ok": False, "model": self.config.model, "models": [],
                    "metrics_available": False,
                    "message": "暂未连接推理服务：" + str(exc)[:240]}

    def payload(self, body: dict) -> dict:
        common = {"model": self.config.model, "temperature": body["temperature"], "stream": True,
                  "chat_template_kwargs": {"enable_thinking": body["thinking"]}}
        if self.config.api_mode == "responses":
            return {**common, "instructions": SYSTEM_PROMPT, "input": body["messages"],
                    "max_output_tokens": body["max_tokens"], "store": False}
        return {**common, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *body["messages"]],
                "max_tokens": body["max_tokens"], "stream_options": {"include_usage": True}}


def validate_body(body: Any, default_tokens: int) -> dict:
    if not isinstance(body, dict):
        raise ValueError("请求必须是 JSON 对象")
    request_id = body.get("request_id", "")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
        raise ValueError("request_id 格式不正确")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
        raise ValueError("对话消息数量须在 1–40 条以内")
    total = 0
    for item in messages:
        if (not isinstance(item, dict) or item.get("role") not in ("user", "assistant")
                or not isinstance(item.get("content"), str) or not item["content"].strip()):
            raise ValueError("对话只接受非空 user/assistant 文本")
        if set(item) - {"role", "content"}:
            raise ValueError("消息包含不支持的字段")
        total += len(item["content"])
    if messages[-1]["role"] != "user":
        raise ValueError("最后一条消息应为用户问题")
    if total > 100000:
        raise ValueError("对话过长，请新建对话或缩短代码")
    max_tokens = body.get("max_tokens", default_tokens)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 64 <= max_tokens <= 4096:
        raise ValueError("输出上限须为 64–4096 token")
    temperature = body.get("temperature", 0.2)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature 须在 0–2 之间")
    thinking = body.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ValueError("thinking 须为布尔值")
    return {"request_id": request_id, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "thinking": thinking}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> App:
        return self.server.app

    def log_message(self, format, *args):
        # HTTP paths only; never log conversations or API credentials.
        if args and len(args) > 1 and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

    def send_json(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            raise ValueError("请从本页面同源访问")
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            raise ValueError("Content-Type 必须是 application/json")
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > 600000:
            raise ValueError("请求体大小超出范围")
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/config":
            cfg = self.app.config
            self.send_json(200, {"model": cfg.model, "subtitle": cfg.subtitle,
                                "api_mode": cfg.api_mode, "max_output_tokens": cfg.max_output_tokens,
                                "upstream_display": cfg.upstream, "demo": cfg.demo})
            return
        if path == "/api/status":
            self.send_json(200, self.app.status())
            return
        if path.startswith("/api/"):
            self.send_json(404, {"message": "未知接口"})
            return
        relative = unquote(path).lstrip("/") or "index.html"
        static = ROOT / "dist"
        target = (static / relative).resolve()
        if static.resolve() not in target.parents or not target.is_file():
            self.send_json(404, {"message": "文件不存在"})
            return
        raw = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        try:
            body = self.read_json()
        except (ValueError, UnicodeError) as exc:
            self.send_json(400, {"message": str(exc)})
            return
        if self.path == "/api/cancel":
            request_id = body.get("request_id") if isinstance(body, dict) else None
            with self.app.guard:
                active = self.app.active
                found = active is not None and active.request_id == request_id
                if found:
                    active.cancel()
            self.send_json(200, {"cancelled": found})
            return
        if self.path != "/api/chat":
            self.send_json(404, {"message": "未知接口"})
            return
        try:
            body = validate_body(body, self.app.config.max_output_tokens)
        except ValueError as exc:
            self.send_json(400, {"message": str(exc)})
            return
        active = self.app.begin(body["request_id"])
        if not active:
            self.send_json(409, {"message": "上一条请求仍在运行或收尾，请稍后再试。"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            self.event("meta", {"request_id": active.request_id, "model": self.app.config.model,
                                "source": "demo" if self.app.config.demo else "live"})
            if self.app.config.demo:
                self.demo(active)
            else:
                self.live(active, body)
        except (BrokenPipeError, ConnectionResetError):
            active.cancel()
        except Exception as exc:
            if not active.cancelled.is_set():
                try:
                    self.event("error", {"message": "推理请求失败：" + str(exc)[:1000]})
                except (OSError, ValueError):
                    active.cancel()
        finally:
            try:
                self.event("done", {"cancelled": active.cancelled.is_set()})
            except OSError:
                pass
            self.app.finish(active)

    def event(self, name: str, body: dict):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False)
        self.wfile.write(("event: " + name + "\ndata: " + data + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def demo(self, active: ActiveRequest):
        answer = (
            "这是界面预览文本，尚未连接本地模型。正式启动时去掉 --demo 即可。\n\n"
            "下面用一个 Python 生成器展示代码排版：\n\n"
            "```python\ndef batches(items, size=3):\n"
            "    for start in range(0, len(items), size):\n"
            "        yield items[start:start + size]\n\n"
            "print(list(batches([1, 2, 3, 4, 5])))\n```\n\n"
            "测速栏保持为空，连接真实 vLLM 服务后才会显示测量结果。"
        )
        started = time.perf_counter()
        for start in range(0, len(answer), 7):
            if active.cancelled.wait(0.018):
                return
            self.event("delta", {"text": answer[start:start + 7], "channel": "content"})
        self.event("stats", {"input_tokens": None, "output_tokens": None, "cached_tokens": None,
                            "prefill_tps": None, "decode_tps": None, "ttft_s": None,
                            "total_s": time.perf_counter() - started, "delivery_tps": None,
                            "metric_source": "unavailable", "source": "demo",
                            "finish_reason": "stop", "note": "界面预览：模拟文本，不提供性能数字。"})

    def live(self, active: ActiveRequest, body: dict):
        before = self.app.metrics()
        if active.cancelled.is_set():
            return
        conn, prefix = self.app.connection(active)
        started = time.perf_counter()
        first = None
        usage: dict[str, Any] = {}
        finish_reason = None
        content_seen = False
        completed = False

        def emit(text: Any, channel: str):
            nonlocal first, content_seen
            if not isinstance(text, str) or not text:
                return
            if first is None:
                first = time.perf_counter()
            if channel == "content":
                content_seen = True
            self.event("delta", {"text": text, "channel": channel})

        try:
            headers = self.app.headers()
            headers.update({"Content-Type": "application/json", "Accept": "text/event-stream",
                            "X-Request-Id": active.request_id})
            suffix = "/v1/responses" if self.app.config.api_mode == "responses" else "/v1/chat/completions"
            conn.request("POST", prefix + suffix,
                         body=json.dumps(self.app.payload(body), ensure_ascii=False).encode("utf-8"),
                         headers=headers)
            response = conn.getresponse()
            if response.status != 200:
                message = response.read(6000).decode("utf-8", errors="replace")
                raise RuntimeError("上游 HTTP " + str(response.status) + "：" + message[:1600])
            if "text/event-stream" not in response.getheader("Content-Type", ""):
                raise RuntimeError("上游没有返回 SSE；请核对 --api-mode 与服务端流式接口。")
            for name, raw in iter_sse(response):
                if active.cancelled.is_set():
                    return
                if raw == "[DONE]":
                    # Responses requires its typed terminal event. A bare DONE
                    # must not turn a truncated reply into successful history.
                    if self.app.config.api_mode == "chat" and finish_reason is not None:
                        completed = True
                    break
                chunk = json.loads(raw)
                event = name or chunk.get("type", "")
                if chunk.get("error"):
                    error = chunk["error"]
                    raise RuntimeError(error.get("message", str(error)) if isinstance(error, dict) else str(error))
                if self.app.config.api_mode == "responses":
                    if event in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                        emit(chunk.get("delta"), "reasoning")
                    elif event in ("response.output_text.delta", "response.refusal.delta"):
                        emit(chunk.get("delta"), "content")
                    elif event in ("response.completed", "response.incomplete"):
                        result = chunk.get("response") or {}
                        usage = result.get("usage") or {}
                        details = result.get("incomplete_details") or {}
                        finish_reason = details.get("reason") or ("stop" if event == "response.completed" else "length")
                        if not content_seen:
                            for item in result.get("output", []):
                                if item.get("type") == "message":
                                    for part in item.get("content", []):
                                        emit(part.get("text") or part.get("refusal"), "content")
                        completed = True
                        break
                    elif event in ("response.failed", "error"):
                        result = chunk.get("response") or {}
                        error = result.get("error") or {}
                        raise RuntimeError(error.get("message") or chunk.get("message") or "上游生成失败")
                else:
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        emit(delta.get("reasoning_content") or delta.get("reasoning"), "reasoning")
                        emit(delta.get("content"), "content")
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                            completed = True
            ended = time.perf_counter()
        finally:
            active.attach(None)
            conn.close()
        if active.cancelled.is_set():
            return
        if not completed:
            raise RuntimeError("上游连接提前结束，未收到完成事件；本轮不计入成功对话。")
        after = self.app.completed_metrics(before, active)
        if active.cancelled.is_set():
            return
        stats = make_stats(usage, before, after, ended - started,
                           first - started if first is not None else None, finish_reason)
        self.event("stats", stats)


def create_server(config: Config, host: str = "127.0.0.1", port: int = 8088) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.app = App(config)
    return server


def main():
    parser = argparse.ArgumentParser(description="本地 Qwen 编程问答展示页（无第三方 Python 依赖）")
    parser.add_argument("--upstream", default=os.environ.get("QWEN_UPSTREAM", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen3.6-27B-AWQ4"))
    parser.add_argument("--api-mode", choices=("responses", "chat"), default="responses")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--subtitle", default="AMD ROCm · 本地推理")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--demo", action="store_true", help="只预览界面；模拟文字，不生成测速数据")
    args = parser.parse_args()
    if not 64 <= args.max_output_tokens <= 4096 or args.timeout <= 0:
        parser.error("输出上限须为 64–4096，timeout 须大于 0")
    config = Config(upstream=args.upstream, model=args.model, api_mode=args.api_mode,
                    subtitle=args.subtitle, max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout, api_key=os.environ.get("QWEN_API_KEY", ""), demo=args.demo)
    server = create_server(config, args.host, args.port)
    print("本地编程助手：http://" + args.host + ":" + str(server.server_port), flush=True)
    print("界面预览模式：不会调用模型或显示测速。" if config.demo else
          "推理服务：" + config.upstream + "；模型：" + config.model, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.app.active:
            server.app.active.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
