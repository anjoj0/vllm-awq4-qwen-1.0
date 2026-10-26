/* Offline, buildless interface. Model output is always rendered as text nodes. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const ui = {
    form: $("chat-form"), prompt: $("prompt"), send: $("send"), stop: $("stop"),
    newChat: $("new-chat"), exportChat: $("export-chat"), thinking: $("thinking"),
    maxTokens: $("max-tokens"), messages: $("messages"), empty: $("empty-state"),
    scroll: $("chat-scroll"), notice: $("notice"), connection: $("connection"),
    connectionText: $("connection-text"), model: $("model-name"),
    subtitle: $("model-subtitle"), upstream: $("upstream-display"),
    preview: $("preview-banner"), telemetryState: $("telemetry-state"),
    telemetryLabel: $("telemetry-label"), metricNote: $("metric-note"),
    prefill: $("metric-prefill"), decode: $("metric-decode"), ttft: $("metric-ttft"),
    total: $("metric-total"), input: $("metric-input"), output: $("metric-output")
  };
  const state = {
    config: { model: "Qwen3.6-27B", max_output_tokens: 768, demo: false },
    history: [], messages: [], active: null, preview: false,
    resetting: false, checkingStatus: false, renderFrame: null
  };
  const FENCE = String.fromCharCode(96).repeat(3);
  const examples = {
    explain: "请解释下面这段 Python 代码的执行逻辑、时间复杂度和适用场景，并举一个输入输出示例：\n\n" + FENCE + "python\nfrom collections import deque\n\ndef window_max(nums, k):\n    q, result = deque(), []\n    for i, value in enumerate(nums):\n        while q and q[0] <= i - k:\n            q.popleft()\n        while q and nums[q[-1]] <= value:\n            q.pop()\n        q.append(i)\n        if i >= k - 1:\n            result.append(nums[q[0]])\n    return result\n" + FENCE,
    debug: "下面的 Python 函数本想返回有序数组中 target 的下标，但某些输入会一直循环。请定位原因、给出修复代码，并用空数组、单元素和找不到目标的情况验证：\n\n" + FENCE + "python\ndef binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1\n" + FENCE,
    write: "请用 Python 标准库实现一个容量可配置的 LRU 缓存，提供 get(key) 和 put(key, value)。要求：解释核心思路，两个操作的平均时间复杂度为 O(1)，未命中返回 -1，并给出一个简短的测试示例。"
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function notice(message) {
    ui.notice.textContent = message || "";
    ui.notice.hidden = !message;
  }

  function number(value, digits) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0
      ? value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })
      : "—";
  }

  function clearMetrics() {
    [ui.prefill, ui.decode, ui.ttft, ui.total, ui.input, ui.output].forEach((node) => { node.textContent = "—"; });
    ui.metricNote.textContent = state.preview
      ? "界面预览使用模拟文本，不展示推理性能数字。"
      : "Prefill 与 Decode 在请求结束后更新；无独立可靠统计时显示 —。";
    ui.metricNote.title = ui.metricNote.textContent;
  }

  function setPreview(preview) {
    state.preview = Boolean(preview);
    ui.preview.hidden = !state.preview;
    ui.telemetryLabel.textContent = state.preview ? "界面预览" : "本次推理";
    if (state.preview) clearMetrics();
  }

  function applyStats(stats) {
    if (stats.source === "demo" || state.config.demo) setPreview(true);
    if (state.preview) {
      clearMetrics();
      return;
    }
    const independent = stats.metric_source === "server" || stats.metric_source === "client";
    ui.prefill.textContent = independent ? number(stats.prefill_tps, 1) : "—";
    ui.decode.textContent = independent ? number(stats.decode_tps, 1) : "—";
    ui.ttft.textContent = number(stats.ttft_s, 2);
    ui.total.textContent = number(stats.total_s, 2);
    ui.input.textContent = number(stats.input_tokens, 0);
    ui.output.textContent = number(stats.output_tokens, 0);
    let explanation = typeof stats.note === "string" ? stats.note.trim() : "";
    if (stats.metric_source === "shared") {
      explanation = explanation || "服务端指标含其他请求，无法分离本次 Prefill / Decode，速度显示 —。";
    } else if (stats.metric_source === "unavailable") {
      explanation = explanation || "未获得本次请求的独立 Prefill / Decode 指标，速度显示 —。";
    } else if (stats.metric_source === "client") {
      explanation = explanation || "使用客户端可观测统计；缺少服务端细分项时显示 —。";
    } else {
      explanation = explanation || "来自本次请求的服务端实测；未返回的统计项显示 —。";
    }
    if (typeof stats.cached_tokens === "number" && stats.cached_tokens > 0) {
      explanation += " 命中缓存 " + number(stats.cached_tokens, 0) + " tokens。";
    }
    ui.metricNote.textContent = explanation;
    ui.metricNote.title = explanation;
  }

  function updateControls() {
    const busy = Boolean(state.active) || state.resetting;
    ui.send.disabled = busy || !ui.prompt.value.trim();
    ui.stop.hidden = !state.active;
    ui.stop.disabled = !state.active || Boolean(state.active.cancelRequested);
    ui.prompt.disabled = busy;
    ui.thinking.disabled = busy;
    ui.maxTokens.disabled = busy;
    ui.newChat.disabled = state.resetting;
    ui.exportChat.disabled = !state.messages.length;
    document.querySelectorAll(".example").forEach((button) => { button.disabled = busy; });
    document.querySelectorAll(".retry-button").forEach((button) => { button.disabled = busy; });
  }

  function resizePrompt() {
    ui.prompt.style.height = "auto";
    ui.prompt.style.height = Math.min(ui.prompt.scrollHeight, 180) + "px";
  }

  function followingBottom() {
    return ui.scroll.scrollHeight - ui.scroll.scrollTop - ui.scroll.clientHeight < 90;
  }

  function scrollToBottom() {
    ui.scroll.scrollTop = ui.scroll.scrollHeight;
  }

  function appendInline(parent, text) {
    // No HTML parsing, URL rendering, arbitrary attributes or model-generated DOM.
    const pattern = /(\x60[^\x60]+\x60|\*\*[^*]+\*\*)/g;
    let cursor = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      parent.append(document.createTextNode(text.slice(cursor, match.index)));
      if (match[0].charCodeAt(0) === 96) {
        parent.append(element("code", "inline-code", match[0].slice(1, -1)));
      } else {
        parent.append(element("strong", "", match[0].slice(2, -2)));
      }
      cursor = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(cursor)));
  }

  async function copyText(text, button) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const input = element("textarea", "sr-only");
        input.value = text;
        document.body.append(input);
        input.select();
        const copied = document.execCommand("copy");
        input.remove();
        if (!copied) throw new Error("copy unavailable");
      }
      button.textContent = "已复制";
    } catch (_) {
      button.textContent = "请手动选取复制";
    }
    window.setTimeout(() => { if (button.isConnected) button.textContent = "复制代码"; }, 1800);
  }

  function codeBlock(language, source) {
    const wrapper = element("div", "code-block");
    const header = element("div", "code-header");
    header.append(element("span", "", language.slice(0, 36) || "code"));
    const copy = element("button", "copy-code", "复制代码");
    copy.type = "button";
    copy.addEventListener("click", () => copyText(source, copy));
    header.append(copy);
    const pre = element("pre");
    pre.append(element("code", "", source));
    wrapper.append(header, pre);
    return wrapper;
  }

  function markdown(text) {
    const fragment = document.createDocumentFragment();
    const lines = text.replace(/\r\n?/g, "\n").split("\n");
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index += 1; continue; }
      if (line.trimStart().startsWith(FENCE)) {
        const language = line.trimStart().slice(3).trim();
        const code = [];
        index += 1;
        while (index < lines.length && !lines[index].trimStart().startsWith(FENCE)) {
          code.push(lines[index++]);
        }
        if (index < lines.length) index += 1;
        fragment.append(codeBlock(language, code.join("\n")));
        continue;
      }
      const heading = /^(#{1,3})\s+(.+)$/.exec(line);
      if (heading) {
        const node = element("h" + (heading[1].length + 1));
        appendInline(node, heading[2]);
        fragment.append(node);
        index += 1;
        continue;
      }
      const unordered = /^\s*[-*+]\s+(.+)$/.test(line);
      const ordered = /^\s*\d+[.)]\s+(.+)$/.test(line);
      if (unordered || ordered) {
        const list = element(ordered ? "ol" : "ul");
        const pattern = ordered ? /^\s*\d+[.)]\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/;
        while (index < lines.length) {
          const item = pattern.exec(lines[index]);
          if (!item) break;
          const li = element("li");
          appendInline(li, item[1]);
          list.append(li);
          index += 1;
        }
        fragment.append(list);
        continue;
      }
      const paragraph = [line];
      index += 1;
      while (index < lines.length && lines[index].trim()
        && !lines[index].trimStart().startsWith(FENCE)
        && !/^(#{1,3})\s+/.test(lines[index])
        && !/^\s*(?:[-*+]|\d+[.)])\s+/.test(lines[index])) {
        paragraph.push(lines[index++]);
      }
      const p = element("p");
      appendInline(p, paragraph.join("\n"));
      fragment.append(p);
    }
    return fragment;
  }

  function createMessage(role, content) {
    const record = {
      role, content: content || "", reasoning: "", createdAt: new Date(),
      complete: role === "user", cancelled: false, error: "", stats: null, source: "live"
    };
    const article = element("article", "message message-" + role);
    const header = element("div", "message-header");
    header.append(element("span", "message-avatar", role === "user" ? "YOU" : "AI"));
    header.append(element("span", "message-author", role === "user" ? "你" : "本地编程助手"));
    const time = element("time", "message-time", record.createdAt.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }));
    time.dateTime = record.createdAt.toISOString();
    header.append(time);
    const reasoning = element("details", "reasoning");
    reasoning.hidden = true;
    const summary = element("summary", "", "思考过程");
    const reasoningBody = element("div", "reasoning-body");
    reasoning.append(summary, reasoningBody);
    const body = element("div", "message-content");
    const status = element("div", "message-status");
    status.setAttribute("role", "status");
    article.append(header);
    if (role === "assistant") article.append(reasoning);
    article.append(body);
    if (role === "assistant") article.append(status);
    record.dom = { article, body, status, reasoning, reasoningBody };
    state.messages.push(record);
    ui.messages.append(article);
    ui.empty.hidden = true;
    if (role === "user") body.textContent = record.content;
    else {
      const label = element("span", "typing-label");
      label.append(element("span", "typing-dot"), document.createTextNode("正在准备回答…"));
      body.append(label);
    }
    return record;
  }

  function renderAnswer(record) {
    const follow = followingBottom();
    record.dom.reasoning.hidden = !record.reasoning;
    record.dom.reasoningBody.textContent = record.reasoning;
    if (record.content) {
      record.dom.body.replaceChildren(markdown(record.content));
    } else if (record.reasoning && state.active) {
      const label = element("span", "typing-label");
      label.append(element("span", "typing-dot"), document.createTextNode("正在思考，等待正文…"));
      record.dom.body.replaceChildren(label);
    }
    if (follow) scrollToBottom();
  }

  function scheduleRender(record) {
    if (state.renderFrame !== null) return;
    state.renderFrame = window.requestAnimationFrame(() => {
      state.renderFrame = null;
      renderAnswer(record);
    });
  }

  function finalizeMessage(record, prompt) {
    if (state.renderFrame !== null) {
      window.cancelAnimationFrame(state.renderFrame);
      state.renderFrame = null;
    }
    renderAnswer(record);
    const status = record.dom.status;
    status.replaceChildren();
    status.classList.toggle("error", Boolean(record.error));
    if (record.error || record.cancelled) {
      if (!record.content) record.dom.body.replaceChildren();
      const message = record.error
        ? record.error + " 本轮未加入后续上下文。"
        : "已停止 · 本轮未加入后续上下文。";
      status.append(element("span", "", message));
      const retry = element("button", "retry-button", "重试这一轮");
      retry.type = "button";
      retry.addEventListener("click", () => { if (!state.active) sendPrompt(prompt); });
      status.append(retry);
    } else {
      const detail = record.source === "demo" ? "模拟文本 · 界面预览" : "回答完成";
      status.append(element("span", "", detail + (record.stats && record.stats.finish_reason === "length" ? " · 已达到输出上限，可继续追问" : "")));
    }
    updateControls();
  }

  function requestId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    const bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = Array.from(bytes, (v) => v.toString(16).padStart(2, "0")).join("");
    return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join("-");
  }

  async function fetchJSON(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs || 7000);
    try {
      const response = await fetch(url, Object.assign({}, options, { signal: controller.signal, cache: "no-store" }));
      if (!response.ok) throw new Error("HTTP " + response.status);
      return await response.json();
    } finally {
      window.clearTimeout(timer);
    }
  }

  async function refreshStatus() {
    if (state.checkingStatus) return;
    state.checkingStatus = true;
    try {
      const result = await fetchJSON("/api/status");
      ui.connection.dataset.state = result.ok ? "connected" : "error";
      ui.connectionText.textContent = state.config.demo ? "界面预览" : result.ok ? "模型已连接" : "服务未就绪";
      ui.connection.title = typeof result.message === "string" && result.message
        ? result.message + " · 点击重新检查"
        : "点击重新检查模型服务连接";
      if (result.model) ui.model.textContent = String(result.model);
    } catch (_) {
      ui.connection.dataset.state = "error";
      ui.connectionText.textContent = "连接不可用";
      ui.connection.title = "无法连接页面服务，请检查服务是否运行；点击重试。";
    } finally {
      state.checkingStatus = false;
    }
  }

  async function readEvents(response, handler) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventName = "message";
    let data = [];
    const dispatch = () => {
      if (data.length) {
        let payload;
        try { payload = JSON.parse(data.join("\n")); }
        catch (_) { throw new Error("服务返回了无法解析的流式消息。"); }
        handler(eventName, payload);
      }
      eventName = "message";
      data = [];
    };
    const line = (value) => {
      if (value.endsWith("\r")) value = value.slice(0, -1);
      if (!value) { dispatch(); return; }
      if (value.startsWith(":")) return;
      const colon = value.indexOf(":");
      const key = colon < 0 ? value : value.slice(0, colon);
      let content = colon < 0 ? "" : value.slice(colon + 1);
      if (content.startsWith(" ")) content = content.slice(1);
      if (key === "event") eventName = content;
      else if (key === "data") data.push(content);
    };
    try {
      while (true) {
        const result = await reader.read();
        buffer += decoder.decode(result.value, { stream: !result.done });
        let end;
        while ((end = buffer.indexOf("\n")) >= 0) {
          line(buffer.slice(0, end));
          buffer = buffer.slice(end + 1);
        }
        if (result.done) break;
      }
      if (buffer) line(buffer);
      dispatch();
    } finally {
      reader.releaseLock();
    }
  }

  async function sendPrompt(prompt) {
    const text = String(prompt || "").trim();
    if (!text || state.active || state.resetting) return;
    notice("");
    ui.prompt.value = "";
    resizePrompt();
    setPreview(state.config.demo);
    clearMetrics();
    ui.telemetryState.textContent = "生成中 · 结束后更新速度";
    const user = createMessage("user", text);
    const answer = createMessage("assistant", "");
    const controller = new AbortController();
    let resolveFinished;
    const active = {
      id: requestId(), controller, started: performance.now(), cancelRequested: false,
      done: false, stats: null, stopPromise: null,
      finished: new Promise((resolve) => { resolveFinished = resolve; })
    };
    state.active = active;
    updateControls();
    scrollToBottom();
    const elapsedTimer = window.setInterval(() => {
      if (!state.preview && !active.stats) ui.total.textContent = number((performance.now() - active.started) / 1000, 1);
    }, 150);
    try {
      const response = await fetch("/api/chat", {
        method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: active.id,
          messages: state.history.concat([{ role: "user", content: text }]),
          max_tokens: Number(ui.maxTokens.value),
          temperature: 0.2, thinking: ui.thinking.checked
        })
      });
      if (!response.ok) {
        let message = "请求失败（HTTP " + response.status + "）。";
        try {
          const payload = await response.json();
          if (typeof payload.message === "string") message = payload.message;
          else if (typeof payload.error === "string") message = payload.error;
          else if (payload.error && typeof payload.error.message === "string") message = payload.error.message;
        } catch (_) { /* Preserve the concise HTTP error. */ }
        throw new Error(message);
      }
      if (!response.body || !(response.headers.get("content-type") || "").includes("text/event-stream")) {
        throw new Error("服务未返回预期的流式回答。");
      }
      await readEvents(response, (name, data) => {
        if (!data || typeof data !== "object") return;
        if (name === "meta") {
          if (data.model) ui.model.textContent = String(data.model);
          answer.source = data.source === "demo" ? "demo" : "live";
          setPreview(answer.source === "demo" || state.config.demo);
        } else if (name === "delta" && typeof data.text === "string") {
          if (data.channel === "reasoning") answer.reasoning += data.text;
          else if (data.channel === "content") answer.content += data.text;
          scheduleRender(answer);
        } else if (name === "stats") {
          active.stats = data;
          answer.stats = data;
          if (data.source === "demo") { answer.source = "demo"; setPreview(true); }
        } else if (name === "error") {
          answer.error = typeof data.message === "string" ? data.message : "模型服务返回错误。";
        } else if (name === "done") {
          active.done = true;
          answer.cancelled = Boolean(data.cancelled);
        }
      });
      if (!active.done && !active.cancelRequested && !answer.error) throw new Error("连接在回答完成前中断，请重试。");
      answer.cancelled = answer.cancelled || active.cancelRequested;
      if (!answer.error && !answer.cancelled && !answer.content.trim()) {
        answer.error = answer.reasoning ? "本轮没有返回正文，可关闭思考或提高输出上限后重试。" : "模型未返回正文，请重试。";
      }
      answer.complete = active.done && !answer.cancelled && !answer.error;
      if (answer.complete) {
        state.history.push({ role: "user", content: user.content }, { role: "assistant", content: answer.content });
      }
    } catch (error) {
      if (active.cancelRequested) answer.cancelled = true;
      else answer.error = error instanceof Error ? error.message : "请求失败，请重试。";
    } finally {
      window.clearInterval(elapsedTimer);
      if (active.stats) applyStats(active.stats);
      else if (!state.preview) {
        ui.total.textContent = number((performance.now() - active.started) / 1000, 2);
        ui.metricNote.textContent = "本轮未返回完整统计；仅总耗时为页面实测，Prefill / Decode 显示 —。";
        ui.metricNote.title = ui.metricNote.textContent;
      }
      ui.telemetryState.textContent = answer.cancelled ? "已停止" : answer.error ? "本轮未完成" : state.preview ? "模拟文本已完成" : "本轮已完成";
      state.active = null;
      finalizeMessage(answer, text);
      updateControls();
      if (!state.resetting) ui.prompt.focus();
      resolveFinished();
    }
  }

  async function stopActive() {
    const active = state.active;
    if (!active) return;
    if (active.stopPromise) return active.stopPromise;
    active.cancelRequested = true;
    updateControls();
    ui.telemetryState.textContent = "正在停止…";
    active.stopPromise = (async () => {
      try {
        await fetchJSON("/api/cancel", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: active.id })
        }, 2500);
      } catch (_) { /* Always release the local reader, including a disconnected server. */ }
      finally { active.controller.abort(); }
      await active.finished;
    })();
    return active.stopPromise;
  }

  async function newConversation() {
    if (state.resetting) return;
    state.resetting = true;
    updateControls();
    await stopActive();
    state.history = [];
    state.messages = [];
    ui.messages.replaceChildren();
    ui.empty.hidden = false;
    ui.prompt.value = "";
    notice("");
    setPreview(state.config.demo);
    clearMetrics();
    ui.telemetryState.textContent = "等待提问";
    state.resetting = false;
    updateControls();
    resizePrompt();
    ui.prompt.focus();
  }

  function exportConversation() {
    if (!state.messages.length) return;
    const lines = [
      "# 本地编程助手会话", "", "时间：" + new Date().toLocaleString("zh-CN"),
      "模型：" + String(ui.model.textContent), state.preview ? "模式：界面预览 / 模拟文本" : "模式：模型服务", ""
    ];
    for (const record of state.messages) {
      lines.push("## " + (record.role === "user" ? "用户" : "助手"), "", record.content || "（无正文）", "");
      if (record.role === "assistant" && !record.complete) {
        lines.push("状态：" + (record.cancelled ? "已停止" : record.error || "生成中") + "；本轮不进入后续上下文。", "");
      }
      if (record.source === "demo") lines.push("注：此段为模拟文本。", "");
    }
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" }));
    const anchor = element("a");
    anchor.href = url;
    anchor.download = "本地编程会话-" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".md";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  ui.form.addEventListener("submit", (event) => { event.preventDefault(); sendPrompt(ui.prompt.value); });
  ui.prompt.addEventListener("input", () => { resizePrompt(); updateControls(); });
  ui.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      if (!ui.send.disabled) sendPrompt(ui.prompt.value);
    }
  });
  ui.stop.addEventListener("click", stopActive);
  ui.newChat.addEventListener("click", newConversation);
  document.querySelector(".brand").addEventListener("click", (event) => {
    event.preventDefault();
    newConversation();
  });
  ui.exportChat.addEventListener("click", exportConversation);
  ui.connection.addEventListener("click", refreshStatus);
  document.querySelectorAll(".example").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.active || state.resetting) return;
      ui.prompt.value = examples[button.dataset.example] || "";
      resizePrompt();
      updateControls();
      ui.prompt.focus();
    });
  });

  async function initialize() {
    try {
      const config = await fetchJSON("/api/config");
      state.config = Object.assign({}, state.config, config);
      ui.model.textContent = String(state.config.model);
      if (config.subtitle) ui.subtitle.textContent = String(config.subtitle);
      if (config.upstream_display) ui.upstream.textContent = String(config.upstream_display);
      const limit = Number(config.max_output_tokens);
      if (Number.isInteger(limit) && limit > 0) {
        if (!Array.from(ui.maxTokens.options).some((option) => Number(option.value) === limit)) {
          const option = element("option", "", limit.toLocaleString("en-US") + " tokens");
          option.value = String(limit);
          ui.maxTokens.append(option);
        }
        ui.maxTokens.value = String(limit);
      }
      setPreview(config.demo);
    } catch (_) {
      notice("暂时无法读取服务配置。可点击右上角检查连接，服务就绪后再提问。");
    }
    updateControls();
  }

  initialize();
  refreshStatus();
  window.addEventListener("pagehide", () => {
    if (!state.active) return;
    const payload = JSON.stringify({ request_id: state.active.id });
    if (navigator.sendBeacon) {
      navigator.sendBeacon("/api/cancel", new Blob([payload], { type: "application/json" }));
    }
  });
  window.setInterval(() => { if (!state.active) refreshStatus(); }, 30000);
  resizePrompt();
})();
