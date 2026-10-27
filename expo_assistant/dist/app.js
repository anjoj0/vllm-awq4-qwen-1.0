/* Local, buildless UI. Untrusted model/tool content is rendered as text, never HTML. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = {
    session: null,
    sessions: [],
    attachments: [],
    mode: "agent",
    active: null,
    uploading: false,
    loadingCase: false,
    navigating: false,
    deleteTarget: null,
    deleting: false,
    config: {},
    preview: null,
    rendered: [],
    initialized: false,
    example: null,
  };
  const modes = {
    chat: "对话可分段读取和搜索已上传文档，图片由模型直接理解",
    agent: "Agent 可读取和修改当前项目，操作过程实时可见",
    search: "搜索可联网查询与读取项目；此模式不会修改文件",
  };
  const names = {
    list_files: "浏览项目",
    read_file: "读取文件",
    search_files: "搜索项目",
    write_file: "写入文件",
    edit_file: "修改文件",
    read_attachment: "读取附件",
    search_attachment: "搜索附件",
    web_search: "搜索网页",
    fetch_page: "打开网页",
    get_weather: "查询天气",
  };
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const post = (obj) => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj),
  });
  const fmt = (v, digits = 1) =>
    typeof v === "number" && Number.isFinite(v)
      ? v.toLocaleString("en-US", {
          maximumFractionDigits: digits,
          minimumFractionDigits: digits,
        })
      : "—";
  const notice = (text) => {
    $("notice").textContent = text || "";
    $("notice").hidden = !text;
  };
  async function api(url, opts = {}, timeout = 15000) {
    const ctrl = new AbortController(),
      timer = setTimeout(() => ctrl.abort(), timeout);
    try {
      const r = await fetch(url, {
        ...opts,
        signal: ctrl.signal,
        cache: "no-store",
      });
      const data = await r.json();
      if (!r.ok) throw Error(data.message || `HTTP ${r.status}`);
      return data;
    } finally {
      clearTimeout(timer);
    }
  }
  function local(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch (_) {}
    return null;
  }
  function controls() {
    const busy =
      Boolean(state.active) ||
      state.uploading ||
      state.loadingCase ||
      state.navigating ||
      state.deleting;
    $("send").disabled =
      !state.initialized ||
      busy ||
      (!$("prompt").value.trim() && !state.attachments.length);
    $("stop").hidden = !state.active;
    $("send").hidden = Boolean(state.active);
    $("stop").disabled = Boolean(state.active?.cancelled);
    [
      "prompt",
      "thinking",
      "max-tokens",
      "new-chat",
      "attach",
      "file-input",
    ].forEach((id) => {
      $(id).disabled = busy;
    });
    document
      .querySelectorAll(
        "[data-mode],.suggestion,.session-button,.session-delete,.example-button,#case-followup",
      )
      .forEach((n) => (n.disabled = busy));
    $("export-chat").disabled =
      !state.session?.turns?.length && !state.rendered.length;
  }
  function mode(value) {
    state.mode = value;
    document.querySelectorAll("[data-mode]").forEach((b) => {
      const on = b.dataset.mode === value;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", String(on));
    });
    $("mode-hint").textContent = modes[value];
  }
  function resize() {
    $("prompt").style.height = "auto";
    $("prompt").style.height = Math.min($("prompt").scrollHeight, 150) + "px";
  }
  function bottom() {
    const n = $("chat-scroll");
    return n.scrollHeight - n.scrollTop - n.clientHeight < 110;
  }
  function scroll() {
    const n = $("chat-scroll");
    n.scrollTop = n.scrollHeight;
  }
  function safeURL(value) {
    try {
      const u = new URL(value);
      return ["http:", "https:"].includes(u.protocol) ? u.href : null;
    } catch (_) {
      return null;
    }
  }
  function inline(parent, text) {
    const pattern =
      /(\x60[^\x60]+\x60|\*\*[^*]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
    let start = 0,
      m;
    while ((m = pattern.exec(text))) {
      parent.append(document.createTextNode(text.slice(start, m.index)));
      const token = m[0];
      if (token[0] === "`")
        parent.append(el("code", "inline-code", token.slice(1, -1)));
      else if (token.startsWith("**"))
        parent.append(el("strong", "", token.slice(2, -2)));
      else {
        const cut = token.indexOf("]("),
          href = safeURL(token.slice(cut + 2, -1));
        if (href) {
          const a = el("a", "", token.slice(1, cut));
          a.href = href;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          parent.append(a);
        } else parent.append(document.createTextNode(token));
      }
      start = m.index + token.length;
    }
    parent.append(document.createTextNode(text.slice(start)));
  }
  async function copy(text, button) {
    try {
      await navigator.clipboard.writeText(text);
      const old = button.textContent;
      button.textContent = "已复制";
      setTimeout(() => (button.textContent = old), 1500);
    } catch (_) {
      notice("浏览器未允许复制，请手动选取文本。");
    }
  }
  function markdown(text) {
    const out = document.createDocumentFragment(),
      lines = text.replace(/\r\n?/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i++;
        continue;
      }
      if (/^\s*```/.test(line)) {
        const lang = line.trim().slice(3),
          code = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i]))
          code.push(lines[i++]);
        if (i < lines.length) i++;
        const box = el("div", "code-block"),
          head = el("div", "code-header"),
          button = el("button", "copy-code", "复制代码");
        button.type = "button";
        button.onclick = () => copy(code.join("\n"), button);
        head.append(el("span", "", lang || "code"), button);
        const pre = el("pre");
        pre.append(el("code", "", code.join("\n")));
        box.append(head, pre);
        out.append(box);
        continue;
      }
      const h = line.match(/^(#{1,4})\s+(.+)/);
      if (h) {
        const n = el("h" + h[1].length);
        inline(n, h[2]);
        out.append(n);
        i++;
        continue;
      }
      if (
        line.includes("|") &&
        i + 1 < lines.length &&
        /^\s*\|?\s*:?-{3,}:?\s*\|[\s|:\-]+$/.test(lines[i + 1])
      ) {
        const cells = (s) =>
          s
            .trim()
            .replace(/^\|/, "")
            .replace(/\|$/, "")
            .split("|")
            .map((x) => x.trim());
        const wrap = el("div", "table-wrap"),
          table = el("table"),
          head = el("thead"),
          body = el("tbody"),
          row = el("tr");
        cells(line).forEach((x) => {
          const c = el("th");
          inline(c, x);
          row.append(c);
        });
        head.append(row);
        i += 2;
        while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
          const r = el("tr");
          cells(lines[i++]).forEach((x) => {
            const c = el("td");
            inline(c, x);
            r.append(c);
          });
          body.append(r);
        }
        table.append(head, body);
        wrap.append(table);
        out.append(wrap);
        continue;
      }
      if (/^\s*(?:[-*+] |\d+\. )/.test(line)) {
        const ordered = /^\s*\d+\. /.test(line),
          list = el(ordered ? "ol" : "ul");
        while (i < lines.length && /^\s*(?:[-*+] |\d+\. )/.test(lines[i])) {
          const li = el("li");
          inline(li, lines[i++].replace(/^\s*(?:[-*+] |\d+\. )/, ""));
          list.append(li);
        }
        out.append(list);
        continue;
      }
      const p = el("p"),
        paragraph = [line];
      i++;
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^\s*(?:```|#{1,4} |[-*+] |\d+\. )/.test(lines[i])
      )
        paragraph.push(lines[i++]);
      inline(p, paragraph.join("\n"));
      out.append(p);
    }
    return out;
  }
  function attachURL(id) {
    return (
      "/api/attachment?session=" +
      encodeURIComponent(state.session.id) +
      "&id=" +
      encodeURIComponent(id)
    );
  }
  function attachment(id) {
    return (
      state.session?.attachments?.find((x) => x.id === id) ||
      state.attachments.find((x) => x.id === id)
    );
  }
  function chip(item, removable = false) {
    const div = el("div", "attachment-chip");
    if (item.kind === "image") {
      const img = el("img");
      img.src = attachURL(item.id);
      img.alt = item.name;
      div.append(img);
    } else div.append(el("span", "", "▤"));
    const label = el("span", "attachment-name", item.name);
    label.title = item.note || item.name;
    label.tabIndex = 0;
    label.onclick = () => previewAttachment(item);
    label.onkeydown = (e) => {
      if (e.key === "Enter") previewAttachment(item);
    };
    div.append(label);
    if (removable) {
      const b = el("button", "", "×");
      b.type = "button";
      b.setAttribute("aria-label", "移除 " + item.name);
      b.onclick = () => {
        state.attachments = state.attachments.filter((x) => x.id !== item.id);
        tray();
        controls();
      };
      div.append(b);
    }
    return div;
  }
  function tray() {
    $("attachment-tray").replaceChildren(
      ...state.attachments.map((x) => chip(x, true)),
    );
  }
  function clearMetrics() {
    $("telemetry-state").textContent = "等待任务";
    ["prefill", "decode", "ttft", "total", "input", "output"].forEach(
      (k) => ($("metric-" + k).textContent = "—"),
    );
    $("metric-note").textContent =
      "工具执行过程可见；实测速率在每次模型调用完成后更新。";
  }
  function stats(data) {
    if (data.source === "demo") return;
    ["prefill", "decode"].forEach(
      (k) =>
        ($("metric-" + k).textContent =
          data.metric_source === "server" ? fmt(data[k + "_tps"]) : "—"),
    );
    $("metric-ttft").textContent = fmt(data.ttft_s, 2);
    $("metric-total").textContent = fmt(data.total_s, 2);
    ["input", "output"].forEach(
      (k) => ($("metric-" + k).textContent = fmt(data[k + "_tokens"], 0)),
    );
    $("metric-note").textContent = data.note || "";
  }
  function makeTurn(turn) {
    const article = el("article", "turn"),
      user = el("div", "user-message");
    user.append(el("div", "user-text", turn.text));
    if (turn.attachments?.length) {
      const list = el("div", "user-attachments");
      turn.attachments.forEach((id) => {
        const a = attachment(id);
        if (a) list.append(chip(a));
      });
      user.append(list);
    }
    const head = el("div", "assistant-head");
    head.append(
      el("span", "assistant-symbol", "✦"),
      el("span", "", "Qwen"),
      el(
        "span",
        "turn-mode",
        { agent: "AGENT", search: "SEARCH", chat: "CHAT" }[turn.mode] || "CHAT",
      ),
    );
    const content = el("div"),
      status = el("div", "turn-status");
    status.append(el("span", "busy-dot"), el("span", "", "正在理解任务…"));
    article.append(user, head, content, status);
    $("messages").append(article);
    $("empty-state").hidden = true;
    const view = {
      turn,
      article,
      content,
      status,
      rounds: new Map(),
      tools: new Map(),
      answer: "",
      error: false,
    };
    state.rendered.push(view);
    return view;
  }
  function round(view, id) {
    if (!view.rounds.has(id)) {
      const thought = el("details", "reasoning");
      thought.hidden = true;
      thought.append(el("summary", "", "思考过程"));
      const reasoning = el("div", "reasoning-body"),
        content = el("div", "answer-content");
      thought.append(reasoning);
      view.content.append(thought, content);
      view.rounds.set(id, {
        thought,
        reasoning,
        content,
        text: "",
        thinking: "",
        frame: null,
      });
    }
    return view.rounds.get(id);
  }
  function event(view, name, data, replay = false) {
    const follow = bottom();
    if (name === "delta") {
      const r = round(view, data.continuation_of || data.round || 1);
      if (data.channel === "reasoning") {
        r.thinking += data.text;
        r.reasoning.textContent = r.thinking;
        r.thought.hidden = false;
      } else {
        r.text += data.text;
        view.answer += data.text;
        if (replay) r.content.replaceChildren(markdown(r.text));
        else if (!r.frame)
          r.frame = requestAnimationFrame(() => {
            r.content.replaceChildren(markdown(r.text));
            r.frame = null;
            if (follow) scroll();
          });
      }
    } else if (name === "tool") {
      let item = view.tools.get(data.id);
      if (!item) {
        const box = el("details", "tool-card"),
          summary = el("summary"),
          icon = el("span", "tool-status"),
          label = el("span", "", names[data.name] || data.name),
          target = el("span", "tool-target"),
          elapsed = el("span", "tool-duration"),
          detail = el("div", "tool-detail");
        summary.append(icon, label, target, elapsed);
        box.append(summary, detail);
        view.content.append(box);
        item = { box, icon, target, elapsed, detail };
        view.tools.set(data.id, item);
      }
      item.box.className = "tool-card " + data.status;
      item.icon.textContent =
        data.status === "running" ? "◌" : data.status === "error" ? "!" : "✓";
      const args = data.arguments || {};
      item.target.textContent =
        args.path ||
        args.query ||
        args.url ||
        args.location ||
        args.attachment_id ||
        "";
      item.elapsed.textContent =
        data.elapsed_s === undefined ? "进行中" : fmt(data.elapsed_s, 1) + " s";
      item.detail.replaceChildren(
        el(
          "pre",
          "",
          JSON.stringify(data.result || { arguments: args }, null, 2),
        ),
      );
      const result = data.result?.result;
      if (
        result?.name &&
        ["read_attachment", "search_attachment"].includes(data.name)
      ) {
        const pages = result.pages || [];
        const range = pages.length
          ? ` · 第 ${pages[0]}${pages.length > 1 ? "–" + pages[pages.length - 1] : ""} 页`
          : result.offset !== undefined
            ? ` · 字符 ${result.offset}–${result.next_offset}`
            : "";
        item.target.textContent = result.name + range;
        const source = attachment(args.attachment_id);
        if (source) {
          const button = el("button", "quiet-button", "查看附件原文");
          button.type = "button";
          button.onclick = () => previewAttachment(source, result.offset || 0);
          item.detail.append(button);
        }
      }
      if (
        result?.path &&
        ["write_file", "edit_file", "read_file"].includes(data.name)
      ) {
        const b = el("button", "quiet-button", "查看 " + result.path);
        b.type = "button";
        b.onclick = () => previewFile(result.path);
        item.detail.append(b);
      }
      if (data.status === "error") item.box.open = true;
      if (result?.results) {
        const sources = el("div", "sources");
        result.results.forEach((x) => {
          const url = safeURL(x.url);
          if (url) {
            const a = el("a", "source-link", x.title || url);
            a.href = url;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.title = url;
            sources.append(a);
          }
        });
        item.detail.append(sources);
      }
      if (data.name === "get_weather" && result?.current && !item.weather) {
        item.weather = weatherCard(result);
        view.content.append(item.weather);
      }
      view.status.replaceChildren(
        el("span", "busy-dot"),
        el(
          "span",
          "",
          data.status === "running"
            ? (names[data.name] || data.name) + "…"
            : "正在继续处理…",
        ),
      );
    } else if (name === "step") {
      view.status.replaceChildren(
        el("span", "busy-dot"),
        el("span", "", `第 ${data.round} 步 · ${data.message}`),
      );
    } else if (name === "stats") {
      view.stats = data;
      stats(data);
    } else if (name === "round_stats") {
      if (!replay) stats(data);
    } else if (name === "context") {
      if (!replay) notice(data.message);
    } else if (name === "error") {
      view.error = true;
      view.errorMessage = data.message;
      view.status.classList.add("error");
      view.status.textContent = data.message;
    }
    if (follow) scroll();
  }
  function weatherCard(data) {
    const box = el("section", "weather-card"),
      head = el("div", "weather-head"),
      left = el("div");
    left.append(
      el("div", "weather-city", data.location),
      el(
        "div",
        "weather-time",
        (data.current.time || "").replace("T", " ") + " · " + data.timezone,
      ),
    );
    const temp = el(
      "div",
      "weather-temp",
      fmt(data.current.temperature_2m, 1) + "°",
    );
    head.append(left, temp);
    box.append(head);
    box.append(
      el(
        "div",
        "weather-description",
        data.current.description +
          " · 体感 " +
          fmt(data.current.apparent_temperature, 1) +
          "°C · 风速 " +
          fmt(data.current.wind_speed_10m, 1) +
          " km/h",
      ),
    );
    const days = el("div", "weather-days");
    data.forecast.forEach((day) => {
      const card = el("div", "weather-day");
      card.append(
        el("span", "", day.date.slice(5)),
        el("strong", "", day.description),
        el("span", "", fmt(day.min_c, 0) + "° / " + fmt(day.max_c, 0) + "°"),
        el("small", "", "降水 " + fmt(day.rain_probability_percent, 0) + "%"),
      );
      days.append(card);
    });
    box.append(days);
    const source = el("a", "weather-source", "Open-Meteo · 数值天气预报");
    source.href = "https://open-meteo.com/";
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    box.append(source);
    return box;
  }
  function finish(view, status) {
    for (const r of view.rounds.values()) {
      if (r.frame) cancelAnimationFrame(r.frame);
      r.content.replaceChildren(markdown(r.text));
    }
    view.status.replaceChildren();
    const labels = {
      complete: "已完成",
      cancelled: "已停止 · 此前成功的文件操作保留",
      error: "任务中断 · 已执行的操作可在上方查看",
      limited: "达到步骤上限",
      incomplete: "达到输出上限 · 回答尚未结束",
    };
    view.status.append(
      el(
        "span",
        "",
        view.errorMessage || labels[status] || "任务已中断，可以继续提问",
      ),
    );
    view.status.classList.toggle("error", status === "error");
    const actions = el("span", "turn-actions");
    if (view.answer) {
      const b = el("button", "", "复制回答");
      b.onclick = () => copy(view.answer, b);
      actions.append(b);
    }
    if (["error", "cancelled", "incomplete"].includes(status)) {
      const b = el(
        "button",
        "",
        status === "incomplete" ? "继续回答" : "继续此任务",
      );
      b.onclick = () => {
        if (!state.active) {
          $("prompt").value =
            status === "incomplete"
              ? "请从中断处继续完成上一轮回答，避免重复已经写出的内容。"
              : "请根据已上传附件、当前文件和已完成的操作继续：" +
                view.turn.text;
          resize();
          controls();
          $("prompt").focus();
        }
      };
      actions.append(b);
    }
    view.status.append(actions);
  }
  async function sessions() {
    const data = await api("/api/sessions");
    state.sessions = data.sessions;
    $("session-list").replaceChildren();
    if (!data.sessions.length)
      $("session-list").append(el("p", "muted small", "你的思路，从这里继续"));
    data.sessions.forEach((s) => {
      const row = el("div", "session-row");
      row.dataset.sessionId = s.id;
      const b = el(
        "button",
        "session-button" + (s.id === state.session?.id ? " active" : ""),
        s.title,
      );
      b.type = "button";
      b.title = s.title;
      b.onclick = () => loadSession(s.id).catch((e) => notice(e.message));
      const remove = el("button", "icon-button session-delete");
      remove.type = "button";
      remove.title = "删除会话";
      remove.setAttribute("aria-label", "删除会话：" + s.title);
      const icon = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "svg",
      );
      icon.setAttribute("viewBox", "0 0 24 24");
      icon.setAttribute("width", "16");
      icon.setAttribute("height", "16");
      icon.setAttribute("aria-hidden", "true");
      const path = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path",
      );
      path.setAttribute(
        "d",
        "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7m4-7v7",
      );
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "currentColor");
      path.setAttribute("stroke-width", "1.6");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
      icon.append(path);
      remove.append(icon);
      remove.onclick = () => {
        if (
          state.active ||
          state.uploading ||
          state.navigating ||
          state.deleting
        )
          return;
        state.deleteTarget = s;
        $("delete-session-name").textContent = s.title;
        $("delete-session-error").textContent = "";
        $("delete-session-dialog").showModal();
      };
      row.append(b, remove);
      $("session-list").append(row);
    });
    controls();
  }
  async function loadSession(id) {
    if (state.active || state.uploading || state.navigating) return;
    state.navigating = true;
    controls();
    try {
      state.session = await api("/api/session?id=" + id);
      clearMetrics();
      local("qwen-session", id);
      state.attachments = [];
      state.example = null;
      $("case-guide").hidden = true;
      tray();
      state.rendered = [];
      $("messages").replaceChildren();
      $("empty-state").hidden = Boolean(state.session.turns.length);
      $("conversation-title").textContent = state.session.title;
      for (const turn of state.session.turns) {
        const view = makeTurn(turn);
        (turn.events || []).forEach((e) => event(view, e.name, e.data, true));
        finish(view, turn.status === "running" ? "error" : turn.status);
      }
      await sessions();
      scroll();
      $("sidebar").classList.remove("open");
    } finally {
      state.navigating = false;
      controls();
    }
  }
  async function newSession() {
    if (state.active || state.uploading || state.navigating) return;
    state.navigating = true;
    controls();
    let obj;
    try {
      obj = await api("/api/sessions", post({}));
    } finally {
      state.navigating = false;
      controls();
    }
    await loadSession(obj.id);
    clearMetrics();
    notice("");
    $("prompt").value = "";
    resize();
    $("prompt").focus();
  }
  async function deleteSession() {
    const selected = state.deleteTarget;
    if (
      !selected ||
      state.deleting ||
      state.active ||
      state.uploading ||
      state.navigating
    )
      return;
    state.deleting = true;
    $("delete-session-confirm").disabled = true;
    $("delete-session-cancel").disabled = true;
    $("delete-session-confirm").textContent = "删除中…";
    controls();
    let deleted = false;
    try {
      const result = await api(
        "/api/session/delete",
        post({ session_id: selected.id }),
      );
      deleted = true;
      $("delete-session-dialog").close();
      if (local("qwen-session") === selected.id) local("qwen-session", null);
      if (state.session?.id === selected.id) {
        state.session = null;
        state.attachments = [];
        state.rendered = [];
        state.example = null;
        $("messages").replaceChildren();
        $("empty-state").hidden = false;
        $("case-guide").hidden = true;
        $("conversation-title").textContent = "新对话";
        $("prompt").value = "";
        tray();
        clearMetrics();
        resize();
        await sessions();
        if (state.sessions.length) await loadSession(state.sessions[0].id);
        else await newSession();
      } else await sessions();
      notice(result.warning || "会话已删除。");
    } catch (e) {
      if (deleted) notice("会话已删除，列表刷新失败，请刷新页面：" + e.message);
      else $("delete-session-error").textContent = e.message;
    } finally {
      state.deleting = false;
      $("delete-session-confirm").disabled = false;
      $("delete-session-cancel").disabled = false;
      $("delete-session-confirm").textContent = "删除会话";
      controls();
      if (deleted)
        (
          document.querySelector(".session-button.active") || $("new-chat")
        ).focus();
    }
  }
  $("delete-session-confirm").onclick = deleteSession;
  $("delete-session-cancel").onclick = () => $("delete-session-dialog").close();
  $("delete-session-dialog").addEventListener("cancel", (e) => {
    if (state.deleting) e.preventDefault();
  });
  $("delete-session-dialog").addEventListener("close", () => {
    state.deleteTarget = null;
    const target =
      document.querySelector(".session-button.active") || $("new-chat");
    target.focus();
  });
  async function loadExample(id) {
    if (
      state.active ||
      state.uploading ||
      state.loadingCase ||
      state.navigating
    )
      return;
    state.loadingCase = true;
    controls();
    try {
      await newSession();
      const data = await api(
        "/api/example",
        post({ session_id: state.session.id, case_id: id }),
        30000,
      );
      state.example = data;
      state.attachments = data.attachments;
      state.session.attachments.push(...data.attachments);
      state.session.title = data.title;
      $("conversation-title").textContent = data.title;
      mode(data.mode);
      $("prompt").value = data.prompt;
      $("case-check").textContent = data.check;
      $("case-guide").hidden = false;
      tray();
      resize();
      await sessions();
      notice("案例已载入，点击发送即可开始。");
      $("sidebar").classList.remove("open");
    } catch (e) {
      notice(e.message);
    } finally {
      state.loadingCase = false;
      controls();
      $("prompt").focus();
    }
  }
  function openPreview(title, kind) {
    $("preview-title").textContent = title;
    $("preview-kind").textContent = kind;
    $("preview-content").replaceChildren();
    $("preview-notice").textContent = "";
    ["preview-save", "preview-edit", "preview-download"].forEach(
      (k) => ($(k).hidden = true),
    );
    if (!$("preview-dialog").open) $("preview-dialog").showModal();
  }
  async function previewFile(path) {
    try {
      const data = await api("/api/file?path=" + encodeURIComponent(path));
      state.preview = { ...data, kind: "file" };
      openPreview(path, "项目文件");
      $("preview-content").append(el("pre", "", data.content));
      $("preview-download").href =
        "/api/download?path=" + encodeURIComponent(path);
      $("preview-download").hidden = false;
      $("preview-edit").hidden = data.truncated || Boolean(state.active);
      $("preview-notice").textContent = data.truncated
        ? "预览仅显示前 10 万字符；下载可获取完整内容。"
        : "修改会保存差异和原文件备份。";
    } catch (e) {
      notice(e.message);
    }
  }
  async function previewAttachment(item, offset = 0) {
    try {
      state.preview = { ...item, kind: "attachment" };
      openPreview(
        item.name,
        item.kind === "image" ? "图片 · 视觉理解" : "文档 · 提取文字",
      );
      if (item.kind === "image") {
        const img = el("img");
        img.src = attachURL(item.id);
        img.alt = item.name;
        $("preview-content").append(img);
        $("preview-notice").textContent = item.note || "";
      } else {
        const data = await api(attachURL(item.id) + "&offset=" + offset);
        $("preview-content").append(el("pre", "", data.text));
        $("preview-notice").textContent =
          `字符 ${data.offset}–${data.next_offset} / ${data.total_chars} · ${data.note || ""}`;
        if (data.truncated) {
          const b = el("button", "quiet-button", "下一段");
          b.onclick = () => previewAttachment(item, data.next_offset);
          $("preview-content").append(b);
        }
      }
    } catch (e) {
      notice(e.message);
    }
  }
  async function upload(list) {
    if (
      state.active ||
      state.uploading ||
      state.deleting ||
      $("delete-session-dialog").open
    )
      return;
    if (state.navigating || state.loadingCase) {
      notice("正在切换会话，请稍后上传。");
      return;
    }
    if (!state.session) await newSession();
    const chosen = Array.from(list);
    if (chosen.length + state.attachments.length > 4) {
      notice("每次最多发送 4 个附件，其中最多 2 张图片。");
      return;
    }
    state.uploading = true;
    controls();
    try {
      for (const f of chosen) {
        if (f.size > 12 * 1024 * 1024)
          throw Error("单个附件最多 12 MiB：" + f.name);
        if (
          state.attachments.filter((a) => a.kind === "image").length >= 2 &&
          f.type.startsWith("image/")
        )
          throw Error("每次最多 2 张图片");
        notice("正在解析 " + f.name + "…");
        const encoded = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result.split(",")[1]);
          reader.onerror = reject;
          reader.readAsDataURL(f);
        });
        const item = await api(
          "/api/upload",
          post({ session_id: state.session.id, name: f.name, data: encoded }),
          60000,
        );
        state.attachments.push(item);
        state.session.attachments.push(item);
        tray();
      }
      notice("附件已就绪。可以提问，也可以直接发送让模型总结。");
    } catch (e) {
      notice(e.message || "附件读取失败");
    } finally {
      state.uploading = false;
      $("file-input").value = "";
      controls();
    }
  }
  async function readSSE(response, handler) {
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "",
      name = "",
      data = [];
    function dispatch() {
      if (data.length) handler(name, JSON.parse(data.join("\n")));
      name = "";
      data = [];
    }
    function line(s) {
      if (s.endsWith("\r")) s = s.slice(0, -1);
      if (!s) dispatch();
      else if (s.startsWith("event:")) name = s.slice(6).trim();
      else if (s.startsWith("data:")) data.push(s.slice(5).trimStart());
    }
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
  async function send() {
    if (
      state.active ||
      state.uploading ||
      state.navigating ||
      state.loadingCase ||
      !state.initialized
    )
      return;
    let text = $("prompt").value.trim();
    if (!text && !state.attachments.length) return;
    if (!state.session) await newSession();
    text = text || "请理解并总结这些附件。";
    notice("");
    clearMetrics();
    const turn = {
      id: crypto.randomUUID(),
      text,
      attachments: state.attachments.map((x) => x.id),
      mode: state.mode,
      events: [],
      status: "running",
    };
    const view = makeTurn(turn);
    const active = {
      id: turn.id,
      ctrl: new AbortController(),
      cancelled: false,
      done: false,
    };
    state.active = active;
    state.attachments = [];
    tray();
    $("prompt").value = "";
    resize();
    controls();
    scroll();
    $("telemetry-state").textContent = "任务进行中";
    const start = performance.now();
    const timer = setInterval(() => {
      $("metric-total").textContent = fmt(
        (performance.now() - start) / 1000,
        1,
      );
    }, 250);
    try {
      const r = await fetch("/api/agent", {
        ...post({
          request_id: turn.id,
          session_id: state.session.id,
          text,
          attachments: turn.attachments,
          mode: turn.mode,
          max_tokens: Number($("max-tokens").value),
          thinking: $("thinking").checked,
        }),
        signal: active.ctrl.signal,
      });
      if (!r.ok) {
        const data = await r.json();
        throw Error(data.message || `HTTP ${r.status}`);
      }
      if (!r.headers.get("content-type")?.includes("text/event-stream"))
        throw Error("服务未返回流式响应");
      await readSSE(r, (name, data) => {
        turn.events.push({ name, data });
        if (name === "done") {
          active.done = true;
          active.cancelled = active.cancelled || data.cancelled;
        } else event(view, name, data);
      });
      if (!active.done && !active.cancelled) throw Error("流式连接中断");
      turn.status = active.cancelled
        ? "cancelled"
        : view.error
          ? "error"
          : view.stats?.task_status || "complete";
    } catch (e) {
      turn.status = active.cancelled ? "cancelled" : "error";
      if (!active.cancelled) view.errorMessage = e.message;
    } finally {
      clearInterval(timer);
      if (view.stats) stats(view.stats);
      else
        $("metric-total").textContent = fmt(
          (performance.now() - start) / 1000,
          2,
        );
      finish(view, turn.status);
      state.active = null;
      state.session.turns.push(turn);
      $("telemetry-state").textContent =
        turn.status === "complete"
          ? "任务完成"
          : turn.status === "cancelled"
            ? "已停止"
            : turn.status === "incomplete"
              ? "回答尚未结束"
              : "任务中断";
      controls();
      sessions().catch(() => {});
      $("prompt").focus();
    }
  }
  async function stop() {
    const a = state.active;
    if (!a) return;
    a.cancelled = true;
    controls();
    try {
      await api("/api/cancel", post({ request_id: a.id }), 4000);
    } catch (_) {
    } finally {
      a.ctrl.abort();
    }
  }
  async function status() {
    try {
      const data = await api("/api/status");
      $("connection").dataset.state = data.ok ? "connected" : "error";
      $("connection-text").textContent = state.config.demo
        ? "界面预览"
        : data.ok
          ? "模型已连接"
          : "服务未就绪";
      $("connection").title = data.message || "";
    } catch (_) {
      $("connection").dataset.state = "error";
      $("connection-text").textContent = "连接不可用";
    }
  }
  $("chat-form").onsubmit = (e) => {
    e.preventDefault();
    send();
  };
  $("prompt").oninput = () => {
    resize();
    controls();
  };
  $("prompt").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  };
  document
    .querySelectorAll("[data-mode]")
    .forEach((b) => (b.onclick = () => mode(b.dataset.mode)));
  $("stop").onclick = stop;
  $("new-chat").onclick = () => newSession().catch((e) => notice(e.message));
  $("connection").onclick = status;
  $("metrics-toggle").onclick = () => {
    const on = $("telemetry").hidden;
    $("telemetry").hidden = !on;
    $("metrics-toggle").setAttribute("aria-expanded", String(on));
  };
  $("menu-toggle").onclick = () => $("sidebar").classList.toggle("open");
  $("attach").onclick = () => $("file-input").click();
  $("file-input").onchange = () => upload($("file-input").files);
  document
    .querySelectorAll("[data-example]")
    .forEach(
      (b) =>
        (b.onclick = () =>
          loadExample(
            b.dataset.example === "upload" ? "multimodal" : b.dataset.example,
          )),
    );
  document
    .querySelectorAll("[data-case]")
    .forEach((b) => (b.onclick = () => loadExample(b.dataset.case)));
  $("case-followup").onclick = () => {
    if (state.example && !state.active) {
      $("prompt").value = state.example.followup;
      resize();
      controls();
      $("prompt").focus();
    }
  };
  $("preview-close").onclick = () => $("preview-dialog").close();
  $("preview-dialog").onclick = (e) => {
    if (e.target === $("preview-dialog")) {
      const r = e.target.getBoundingClientRect();
      if (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      )
        e.target.close();
    }
  };
  $("preview-ask").onclick = () => {
    const p = state.preview;
    if (!p || state.active) return;
    if (p.kind === "file") {
      mode("agent");
      $("prompt").value =
        "请读取项目文件 " + p.path + "，解释它的作用和关键实现。";
    } else {
      if (!state.attachments.some((x) => x.id === p.id)) {
        if (state.attachments.length >= 4) {
          $("preview-notice").textContent = "本轮附件已满，请先移除一个。";
          return;
        }
        state.attachments.push(attachment(p.id));
        tray();
      }
      $("prompt").value = "请帮我理解 " + p.name + " 中的内容。";
    }
    $("preview-dialog").close();
    resize();
    controls();
    $("prompt").focus();
  };
  $("preview-edit").onclick = () => {
    if (state.active) return;
    const area = el("textarea");
    area.id = "file-editor";
    area.value = state.preview.content;
    area.setAttribute("aria-label", "编辑文件内容");
    $("preview-content").replaceChildren(area);
    $("preview-edit").hidden = true;
    $("preview-save").hidden = false;
    area.focus();
  };
  $("preview-save").onclick = async () => {
    if (state.active) return;
    const p = state.preview,
      b = $("preview-save");
    b.disabled = true;
    try {
      await api(
        "/api/file",
        post({
          path: p.path,
          content: $("file-editor").value,
          expected_sha256: p.sha256,
        }),
      );
      await previewFile(p.path);
      $("preview-notice").textContent =
        "已保存。修改前内容与差异已保存在本地备份目录。";
    } catch (e) {
      $("preview-notice").textContent = e.message;
    } finally {
      b.disabled = false;
    }
  };
  $("export-chat").onclick = () => {
    const turns = state.session?.turns || [];
    const blocks = ["# Qwen 本地工作台\n"];
    turns.forEach((t) => {
      blocks.push("## 你\n\n" + t.text + "\n");
      let text = "";
      const flush = () => {
        if (text) blocks.push(text);
        text = "";
      };
      t.events?.forEach((e) => {
        if (e.name === "delta" && e.data.channel === "content")
          text += e.data.text;
        if (e.name === "tool" && e.data.status !== "running") {
          flush();
          blocks.push(
            "\n工具 " +
              e.data.name +
              "：" +
              JSON.stringify(e.data.result, null, 2) +
              "\n",
          );
        }
      });
      flush();
      blocks.push("\n");
    });
    const url = URL.createObjectURL(
        new Blob([blocks.join("\n")], { type: "text/markdown;charset=utf-8" }),
      ),
      a = el("a");
    a.href = url;
    a.download = "qwen-conversation.md";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  let dragging = 0;
  window.addEventListener("dragenter", (e) => {
    if (e.dataTransfer?.types.includes("Files")) {
      e.preventDefault();
      dragging++;
      $("drop-overlay").hidden = false;
    }
  });
  window.addEventListener("dragleave", () => {
    if (--dragging <= 0) {
      dragging = 0;
      $("drop-overlay").hidden = true;
    }
  });
  window.addEventListener("dragover", (e) => {
    if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragging = 0;
    $("drop-overlay").hidden = true;
    if (e.dataTransfer?.files.length) upload(e.dataTransfer.files);
  });
  $("prompt").addEventListener("paste", (e) => {
    const imgs = Array.from(e.clipboardData?.files || []).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (imgs.length) {
      e.preventDefault();
      upload(imgs);
    }
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      if (!$("delete-session-dialog").open && !state.deleting)
        newSession().catch((x) => notice(x.message));
    }
    if (e.key === "Escape") $("sidebar").classList.remove("open");
  });
  window.addEventListener("pagehide", () => {
    if (state.active)
      navigator.sendBeacon(
        "/api/cancel",
        new Blob([JSON.stringify({ request_id: state.active.id })], {
          type: "application/json",
        }),
      );
  });
  async function init() {
    try {
      state.config = await api("/api/config");
      $("model-name").textContent = state.config.model;
      $("model-subtitle").textContent = state.config.subtitle;
      $("preview-banner").hidden = !state.config.demo;
      $("max-tokens").value = String(state.config.max_output_tokens);
      await sessions();
      const saved = local("qwen-session");
      if (saved && state.sessions.some((s) => s.id === saved))
        await loadSession(saved);
      else if (state.sessions.length) await loadSession(state.sessions[0].id);
      else await newSession();
      state.initialized = true;
      await status();
    } catch (e) {
      notice("工作台初始化失败：" + e.message);
    } finally {
      controls();
      resize();
    }
  }
  init();
  setInterval(() => {
    if (!state.active) status();
  }, 30000);
})();
