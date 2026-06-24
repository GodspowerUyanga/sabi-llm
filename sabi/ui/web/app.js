/* SABI web UI - vanilla JS, no build step, fully offline. */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const state = { cid: null, conversations: [], sending: false, mode: "auto" };

  const els = {
    list: $("#conv-list"),
    messages: $("#messages"),
    empty: $("#empty"),
    input: $("#input"),
    send: $("#send"),
    newChat: $("#new-chat"),
    title: $("#conv-title"),
    mode: $("#mode"),
    agentWarn: $("#agent-warning"),
    modelDot: $("#model-dot"),
    modelLabel: $("#model-label"),
  };

  /* ---------------- API ---------------- */
  const api = {
    async get(u) { const r = await fetch(u); return r.json(); },
    async post(u, b) {
      const r = await fetch(u, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(b || {}),
      });
      return r.json();
    },
    async del(u) { const r = await fetch(u, { method: "DELETE" }); return r.json(); },
  };

  /* ---------------- Markdown (minimal, safe) ---------------- */
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function renderMarkdown(text) {
    const blocks = [];
    // fenced code blocks -> placeholders
    text = text.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
      const i = blocks.length;
      blocks.push('<pre><code>' + escapeHtml(code.replace(/\n$/, "")) + '</code></pre>');
      return "\u0000B" + i + "\u0000";
    });
    let html = escapeHtml(text);
    // inline code
    html = html.replace(/`([^`]+)`/g, (_, c) => '<code class="inline">' + c + "</code>");
    // bold / italic
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    // headings
    html = html.replace(/^###\s+(.*)$/gm, "<h3>$1</h3>");
    html = html.replace(/^##\s+(.*)$/gm, "<h2>$1</h2>");
    html = html.replace(/^#\s+(.*)$/gm, "<h1>$1</h1>");
    // links
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // lists
    html = html.replace(/(?:^|\n)((?:[-*]\s+.*(?:\n|$))+)/g, (m, grp) => {
      const items = grp.trim().split(/\n/).map((l) =>
        "<li>" + l.replace(/^[-*]\s+/, "") + "</li>").join("");
      return "\n<ul>" + items + "</ul>";
    });
    html = html.replace(/(?:^|\n)((?:\d+\.\s+.*(?:\n|$))+)/g, (m, grp) => {
      const items = grp.trim().split(/\n/).map((l) =>
        "<li>" + l.replace(/^\d+\.\s+/, "") + "</li>").join("");
      return "\n<ol>" + items + "</ol>";
    });
    // paragraphs / line breaks
    html = html.split(/\n{2,}/).map((p) => {
      if (/^\s*<(h\d|ul|ol|pre)/.test(p)) return p;
      return "<p>" + p.replace(/\n/g, "<br>") + "</p>";
    }).join("");
    // restore code blocks
    html = html.replace(/\u0000B(\d+)\u0000/g, (_, i) => blocks[+i]);
    return html;
  }

  /* ---------------- Rendering ---------------- */
  function sidebar() {
    els.list.innerHTML = "";
    state.conversations.forEach((c) => {
      const div = document.createElement("div");
      div.className = "conv-item" + (c.id === state.cid ? " active" : "");
      div.innerHTML = '<span class="title"></span><button class="del" title="Delete">✕</button>';
      div.querySelector(".title").textContent = c.title || "New chat";
      div.querySelector(".title").onclick = () => openConv(c.id);
      div.onclick = (e) => { if (!e.target.classList.contains("del")) openConv(c.id); };
      div.querySelector(".del").onclick = async (e) => {
        e.stopPropagation();
        await api.del("/api/conversations/" + c.id);
        if (state.cid === c.id) { state.cid = null; clearMessages(true); }
        await loadConversations();
      };
      els.list.appendChild(div);
    });
  }

  function clearMessages(showEmpty) {
    els.messages.querySelectorAll(".row, .thinking-row").forEach((n) => n.remove());
    els.empty.classList.toggle("hidden", !showEmpty);
  }

  function addMessage(role, content, meta) {
    els.empty.classList.add("hidden");
    const row = document.createElement("div");
    row.className = "row " + (role === "user" ? "user" : "sabi");
    const av = role === "user" ? '<div class="avatar me">You</div>'
                               : '<div class="avatar sabi">S</div>';
    let body = role === "user"
      ? "<div>" + escapeHtml(content).replace(/\n/g, "<br>") + "</div>"
      : renderMarkdown(content || "");
    let extra = "";
    if (meta && meta.actions && meta.actions.length) {
      extra = '<div class="actions-box">' +
        meta.actions.map((a) => "<div>" + escapeHtml(a) + "</div>").join("") + "</div>";
    }
    let metaLine = "";
    if (meta && meta.intent) {
      metaLine = '<div class="meta">' + meta.intent +
        (meta.tps ? " · " + meta.tps + " tok/s" : "") + "</div>";
    }
    row.innerHTML = '<div class="bubble-wrap">' + av +
      '<div class="bubble">' + body + extra + metaLine + "</div></div>";
    els.messages.appendChild(row);
    scrollDown();
  }

  function showThinking() {
    const row = document.createElement("div");
    row.className = "row sabi thinking-row";
    row.innerHTML = '<div class="bubble-wrap"><div class="avatar sabi">S</div>' +
      '<div class="bubble"><span class="thinking">SABI is thinking' +
      '<span class="d"></span><span class="d"></span><span class="d"></span></span></div></div>';
    els.messages.appendChild(row);
    scrollDown();
    return row;
  }

  function scrollDown() { els.messages.scrollTop = els.messages.scrollHeight; }

  /* ---------------- Actions ---------------- */
  async function loadConversations() {
    state.conversations = await api.get("/api/conversations");
    sidebar();
  }

  async function openConv(cid) {
    state.cid = cid;
    sidebar();
    clearMessages(false);
    const conv = await api.get("/api/conversations/" + cid);
    els.title.textContent = conv.title || "New chat";
    (conv.messages || []).forEach((m) => addMessage(m.role, m.content, m.meta));
    if (!conv.messages || !conv.messages.length) els.empty.classList.remove("hidden");
  }

  async function newChat() {
    const conv = await api.post("/api/conversations", {});
    await loadConversations();
    await openConv(conv.id);
  }

  async function send(text) {
    if (state.sending || !text.trim()) return;
    state.sending = true; els.send.disabled = true;
    addMessage("user", text);
    const thinking = showThinking();
    try {
      const res = await api.post("/api/chat", {
        conversation_id: state.cid, message: text, mode: state.mode,
      });
      thinking.remove();
      if (res.conversation_id && res.conversation_id !== state.cid) {
        state.cid = res.conversation_id;
      }
      addMessage("assistant", res.answer || res.error || "(no response)", {
        intent: res.intent, tps: res.tps, actions: res.actions,
      });
      await loadConversations();
      const conv = state.conversations.find((c) => c.id === state.cid);
      if (conv) els.title.textContent = conv.title;
    } catch (e) {
      thinking.remove();
      addMessage("assistant", "⚠ Error contacting SABI: " + e);
    } finally {
      state.sending = false; els.send.disabled = false; els.input.focus();
    }
  }

  async function status() {
    try {
      const s = await api.get("/api/status");
      els.modelLabel.textContent = s.model_ready
        ? s.model_label + " · ready"
        : "model not loaded";
      els.modelDot.className = "dot " + (s.model_ready ? "ready" : "down");
    } catch (e) { els.modelLabel.textContent = "offline"; }
  }

  /* ---------------- Wiring ---------------- */
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
  }
  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const t = els.input.value; els.input.value = ""; autosize(); send(t);
    }
  });
  els.send.onclick = () => { const t = els.input.value; els.input.value = ""; autosize(); send(t); };
  els.newChat.onclick = newChat;
  els.mode.onchange = () => {
    state.mode = els.mode.value;
    els.agentWarn.classList.toggle("hidden", state.mode !== "agent");
  };
  document.querySelectorAll(".sugg").forEach((b) => {
    b.onclick = () => { send(b.textContent); };
  });

  /* ---------------- Init ---------------- */
  (async function init() {
    await status();
    await loadConversations();
    if (state.conversations.length) await openConv(state.conversations[0].id);
    els.input.focus();
  })();
})();
