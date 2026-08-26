"""Web server for `sabi serve`.

Serves a professional chat UI (ChatGPT/Claude-style) with persistent history,
backed by the SABI runtime. Flask is an optional dependency installed via
``pip install "sabi-llm[serve]"``.

sabi serve is a Coding Assistant: code generation, debugging, and programming
tutoring. It never creates, writes, edits, or moves files on the user's
machine — unlike `sabi run`/`sabi agent`/the TUI, every agent-mode request
here runs AgentLoop(restricted=True) (see agent.py), so code the assistant
produces is returned in the reply as text, not saved to disk.

Modes per message:
  * auto   - the default and the main way to use sabi serve: unambiguous
             small talk gets a plain conversational reply, anything else
             gets the restricted agent automatically (read/search files and
             run commands to help debug, but never create/write/edit/move
             one) — no separate mode selection needed. Auto-approves actions
             in the browser, so the UI shows a warning and lists what was
             done.
  * think  - planning / analysis engine, text-only (no filesystem access)
  * code   - code generation engine, text-only (no filesystem access)
  * agent  - explicitly force the (restricted) agent loop even for small
             talk (rarely needed now that auto does this automatically; kept
             for power users who want to be certain a message is acted on)
"""

from __future__ import annotations

import queue
import webbrowser
from pathlib import Path
from threading import Thread, Timer
from typing import Any, Dict, List, Optional

from .config import Config, load_config
from .runtime import Runtime
from .conversations import ConversationStore
from .permissions import PermissionManager
from .agent import Reporter
from .filereader import read_any
from .router import is_smalltalk

WEB_DIR = Path(__file__).resolve().parent / "ui" / "web"

# In-memory map of conversation_id -> list of {name, text} uploaded files.
_UPLOADS: Dict[str, List[dict]] = {}


def _file_context(cid: Optional[str], budget: int = 3000) -> str:
    """Build a context block from files the user uploaded in this conversation."""
    files = _UPLOADS.get(cid or "", [])
    if not files:
        return ""
    parts = []
    for f in files[-3:]:  # last few files
        parts.append(f"--- FILE: {f['name']} ---\n{f['text'][:budget]}")
    return "Attached files the user uploaded:\n" + "\n\n".join(parts)


def _history_for_agent(store: Optional[ConversationStore], cid: Optional[str]) -> list:
    """Prior turns of this conversation, in AgentLoop's role/content shape.

    A fresh AgentLoop is built per HTTP request (unlike the TUI/terminal chat,
    which keep one alive for the whole session and so track this on their
    own) — without this, every request is amnesiac: a follow-up like "list
    them" right after "list my Desktop folders" has no idea what "them"
    refers to, and previous live testing showed the agent not even re-running
    list_dir on the follow-up, just guessing. The conversation store already
    persists every turn for the UI's history sidebar; this just also feeds
    it back to the model. Excludes the just-added current user message
    (that's passed separately as the turn's own request) and caps to the
    same last-8-messages window AgentLoop itself keeps once warm.
    """
    if not store or not cid:
        return []
    conv = store.get(cid)
    if not conv or not conv.get("messages"):
        return []
    prior = conv["messages"][:-1]  # drop the current turn's just-stored user message
    return [{"role": m["role"], "content": m["content"]} for m in prior[-8:]]


class _StreamReporter(Reporter):
    """Feeds an agent turn's tool-call progress into a queue as it happens,
    so /api/chat/stream can show it live instead of blocking silently until
    the whole multi-step turn is done — the actual source of "slow" for a
    request that needs a few tool calls before it can answer.
    """

    def __init__(self, q: "queue.Queue"):
        self.q = q
        self.started = False
        # True the moment any real answer text has been streamed live via
        # answer_delta — the caller uses this to skip re-sending the whole
        # answer again once the turn is done (see generate_agent() below).
        self.answer_started = False

    def _line(self, text: str) -> None:
        prefix = "**Actions taken:**\n" if not self.started else ""
        self.started = True
        self.q.put(prefix + text)

    def proposing(self, tool: str, desc: str) -> None:
        self._line(f"- {desc}\n")

    def ran(self, ok: bool, output: str) -> None:
        preview = (output or "").strip().splitlines()[0][:200] if output else ""
        mark = "✅" if ok else "❌"
        self.q.put(f"  {mark}" + (f" {preview}" if preview else "") + "\n")

    def denied(self, desc: str) -> None:
        self._line(f"- DENIED: {desc}\n")

    def answer_delta(self, text: str) -> None:
        prefix = "\n\n" if not self.answer_started and self.started else ""
        self.answer_started = True
        self.q.put(prefix + text)


def _answer(runtime: Runtime, message: str, mode: str, cid: Optional[str] = None,
            yoruba: bool = False, store: Optional[ConversationStore] = None) -> dict:
    """Produce an assistant reply for a message in the given mode.

    ``yoruba`` is the explicit UI toggle (sabi serve's Yoruba switch) —
    forces the sabi-yoruba-tts translation layer regardless of whether the
    typed message itself looks like Yoruba. THINK/CODE bypass Runtime and
    call the engines directly (existing behaviour, unrelated to this
    change), so the toggle only applies to auto/chat and agent mode.
    """
    ctx = _file_context(cid)
    msg = (message + ("\n\n" + ctx if ctx else ""))
    try:
        if mode == "think":
            gen = runtime.think.run(msg)
            return {"answer": gen.text, "intent": "THINK",
                    "tps": round(gen.tokens_per_second, 2), "actions": []}
        if mode == "code":
            gen = runtime.code.run(msg)
            return {"answer": gen.text, "intent": "CODE",
                    "tps": round(gen.tokens_per_second, 2), "actions": []}
        if mode == "agent":
            perms = PermissionManager(auto_approve=True)  # web auto-approves
            res = runtime.agent(msg, permissions=perms, reporter=Reporter(), force_yoruba=yoruba,
                                history=_history_for_agent(store, cid), restricted=True)
            return {"answer": res.get("answer", ""), "intent": "AGENT",
                    "tps": 0, "actions": res.get("actions", [])}
        # auto: unambiguous small talk stays a plain, tool-free chat reply;
        # anything else gets the restricted agent (Coding Assistant: code
        # generation, debugging, programming tutoring — read/search files and
        # run commands to help debug, but never create/write/edit/move one)
        # automatically — no separate "Agent" mode selection needed for SABI
        # to actually act. Small talk is still hard-gated away from the tool
        # loop for the same reason as the TUI/chat CLI: a bare "hello"
        # reaching the agent loop has, on this size of model, resulted in an
        # invented/executed destructive tool call.
        if is_smalltalk(msg):
            res = runtime.handle(msg, force_yoruba=yoruba)
            if res.get("ok"):
                return {"answer": res.get("text", ""), "intent": res.get("intent", "CHAT"),
                        "tps": res.get("tps", 0), "actions": []}
            return {"answer": "", "error": res.get("error", "request failed"),
                    "intent": res.get("intent", "CHAT"), "actions": []}
        perms = PermissionManager(auto_approve=True)  # web auto-approves
        res = runtime.agent(msg, permissions=perms, reporter=Reporter(), force_yoruba=yoruba,
                            history=_history_for_agent(store, cid), restricted=True)
        return {"answer": res.get("answer", ""), "intent": "AGENT",
                "tps": 0, "actions": res.get("actions", [])}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "error": str(exc), "intent": mode.upper(), "actions": []}


def create_app(runtime: Runtime, store: ConversationStore):
    from flask import Flask, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)

    # ---- static frontend ----
    # SABI is actively developed and re-run locally (edit app.js, restart
    # `sabi serve`, refresh the browser) — a browser that cached the JS/CSS
    # from a previous run would keep executing stale frontend code (e.g. an
    # older build without the Yoruba toggle, or with a bug already fixed
    # since) even after a normal refresh, with no visible sign anything is
    # wrong. no-store forces every load to fetch the current file.
    @app.get("/")
    def index():
        resp = send_from_directory(WEB_DIR, "index.html")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/static/<path:fname>")
    def static_files(fname):
        resp = send_from_directory(WEB_DIR, fname)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---- status ----
    @app.get("/api/status")
    def status():
        m = runtime.model
        return jsonify({
            "version": __import__("sabi").__version__,
            "model_label": runtime.config.abs_model_path().stem,
            "model_ready": bool(m and m.is_available()),
            "model_status": m.status() if m else "n/a",
            "ram_ceiling_gb": runtime.config.ram_ceiling_gb,
            "yoruba_enabled": runtime.config.yoruba_enabled,
            "yoruba_available": runtime.yoruba_available(),
        })

    # ---- conversations ----
    @app.get("/api/conversations")
    def list_convs():
        return jsonify(store.list())

    @app.post("/api/conversations")
    def new_conv():
        return jsonify(store.create())

    @app.get("/api/conversations/<cid>")
    def get_conv(cid):
        conv = store.get(cid)
        return (jsonify(conv), 200) if conv else (jsonify({"error": "not found"}), 404)

    @app.delete("/api/conversations/<cid>")
    def del_conv(cid):
        return jsonify({"ok": store.delete(cid)})

    @app.post("/api/conversations/<cid>/rename")
    def rename_conv(cid):
        title = (request.json or {}).get("title", "").strip() or "Untitled"
        return jsonify({"ok": store.rename(cid, title)})

    # ---- chat (non-streaming; used for agent mode) ----
    @app.post("/api/chat")
    def chat():
        body = request.json or {}
        cid = body.get("conversation_id")
        message = (body.get("message") or "").strip()
        mode = body.get("mode", "auto")
        yoruba = bool(body.get("yoruba"))
        if not message:
            return jsonify({"error": "empty message"}), 400
        if not cid or not store.get(cid):
            cid = store.create()["id"]

        store.add_message(cid, "user", message)
        result = _answer(runtime, message, mode, cid=cid, yoruba=yoruba, store=store)
        reply = result.get("answer") or result.get("error") or "(no response)"
        store.add_message(cid, "assistant", reply, meta={
            "intent": result.get("intent"), "tps": result.get("tps"),
            "actions": result.get("actions", []), "error": result.get("error"),
        })
        return jsonify({"conversation_id": cid, **result})

    # ---- chat (one endpoint, several internal paths, all streamed live) ----
    # Every path here streams real, incremental output as soon as it exists —
    # nothing waits for a fully-finished answer before sending the first
    # byte. THINK/CODE and small-talk CHAT stream token-by-token straight
    # from the model. An agent turn (explicit "agent" mode, or a real
    # request auto-routed to it) runs in a background thread: a Reporter
    # pushes each tool call onto a queue the instant it happens, so progress
    # ("- read a file: ...") appears live while the loop is still running —
    # and even the final answer itself streams token-by-token as it's
    # generated (AgentLoop._chat_step/_JSONFinalStreamer decode it live out
    # of the {"final": "..."} JSON wrapper json_mode requires), not just
    # flushed once at the end. The one case that still can't stream live is
    # a Yoruba reply — translation needs the complete English text first —
    # so that runs once via _answer() and sends the finished result back in
    # a few chunks.
    @app.post("/api/chat/stream")
    def chat_stream():
        from flask import Response, stream_with_context
        body = request.json or {}
        cid = body.get("conversation_id")
        message = (body.get("message") or "").strip()
        mode = body.get("mode", "auto")
        yoruba = bool(body.get("yoruba"))
        if not message:
            return jsonify({"error": "empty message"}), 400
        if not cid or not store.get(cid):
            cid = store.create()["id"]
        store.add_message(cid, "user", message)

        def _stream(gen):
            resp = Response(stream_with_context(gen), mimetype="text/plain")
            resp.headers["X-Conversation-Id"] = cid
            resp.headers["X-Accel-Buffering"] = "no"
            resp.headers["Cache-Control"] = "no-cache"
            return resp

        ctx = _file_context(cid)
        msg = message + ("\n\n" + ctx if ctx else "")

        # THINK/CODE: plain text-only engines, no filesystem/agent involved —
        # always stream for real (yoruba isn't wired up for these modes, same
        # as the existing non-streaming behaviour in _answer()).
        if mode in ("think", "code"):
            engine = runtime.think if mode == "think" else runtime.code

            def generate_engine():
                buf = ""
                try:
                    streamed = False
                    for delta in engine.stream(msg):
                        streamed = True
                        buf += delta
                        yield delta
                    if not streamed:
                        buf = engine.run(msg).text
                        yield buf
                except Exception as exc:  # noqa: BLE001 (includes ModelUnavailable)
                    err = f"\n\n⚠ {exc}"
                    buf += err
                    yield err
                store.add_message(cid, "assistant", buf, meta={"intent": mode.upper()})
            return _stream(generate_engine())

        fast_chat = mode == "auto" and not yoruba and is_smalltalk(message)
        if fast_chat:
            system = runtime.prompts.get("system", "") or None
            messages = ([{"role": "system", "content": system}] if system else []) + \
                [{"role": "user", "content": msg}]

            def generate_fast():
                buf = ""
                try:
                    streamed = False
                    for delta in runtime.model.chat_stream(messages):
                        streamed = True
                        buf += delta
                        yield delta
                    if not streamed:
                        buf = runtime.model.generate(msg, system=system).text
                        yield buf
                except Exception as exc:  # noqa: BLE001 (includes ModelUnavailable)
                    err = f"\n\n⚠ {exc}"
                    buf += err
                    yield err
                store.add_message(cid, "assistant", buf, meta={"intent": "CHAT"})
            return _stream(generate_fast())

        needs_agent = mode == "agent" or (mode == "auto" and not is_smalltalk(message))
        if needs_agent and not yoruba:
            def generate_agent():
                q: "queue.Queue[Optional[str]]" = queue.Queue()
                reporter = _StreamReporter(q)
                result: Dict[str, Any] = {}

                def worker():
                    try:
                        perms = PermissionManager(auto_approve=True)  # web auto-approves
                        result["res"] = runtime.agent(
                            msg, permissions=perms, reporter=reporter,
                            history=_history_for_agent(store, cid), restricted=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        result["error"] = str(exc)
                    finally:
                        q.put(None)  # sentinel: worker is done

                Thread(target=worker, daemon=True).start()
                while True:
                    item = q.get()
                    if item is None:
                        break
                    yield item

                if "error" in result:
                    err = result["error"]
                    yield f"\n\n⚠ {err}"
                    store.add_message(cid, "assistant", "", meta={"intent": "AGENT", "error": err})
                    return
                res = result.get("res", {})
                answer = res.get("answer") or res.get("error") or "(no response)"
                # Usually already streamed live, word by word, via
                # reporter.answer_delta as the model generated it — only
                # send it now as a fallback (e.g. max-steps/repeat-call
                # termination) when nothing was streamed for this turn.
                if not reporter.answer_started:
                    yield ("\n\n" if reporter.started else "") + answer
                store.add_message(cid, "assistant", answer, meta={
                    "intent": "AGENT", "tps": 0, "actions": res.get("actions", []),
                    "error": res.get("error"),
                })
            return _stream(generate_agent())

        def generate_full():
            result = _answer(runtime, message, mode, cid=cid, yoruba=yoruba, store=store)
            answer = result.get("answer") or result.get("error") or "(no response)"
            actions = result.get("actions") or []
            text = ("**Actions taken:**\n" + "\n".join(f"- {a}" for a in actions) + "\n\n" + answer) \
                if actions else answer
            chunk = max(1, len(text) // 12)
            for i in range(0, len(text), chunk):
                yield text[i:i + chunk]
            store.add_message(cid, "assistant", answer, meta={
                "intent": result.get("intent"), "tps": result.get("tps"),
                "actions": actions, "error": result.get("error"),
            })
        return _stream(generate_full())

    # ---- file upload (any format) ----
    @app.post("/api/upload")
    def upload():
        if "file" not in request.files:
            return jsonify({"error": "no file"}), 400
        f = request.files["file"]
        cid = request.form.get("conversation_id") or ""
        if not cid or not store.get(cid):
            cid = store.create()["id"]
        updir = runtime.config.abs_workspace() / ".sabi" / "uploads" / cid
        updir.mkdir(parents=True, exist_ok=True)
        dest = updir / Path(f.filename).name
        f.save(str(dest))
        text = read_any(dest, max_chars=8000)
        _UPLOADS.setdefault(cid, []).append({"name": dest.name, "text": text})
        preview = text[:500] + ("…" if len(text) > 500 else "")
        return jsonify({"conversation_id": cid, "name": dest.name,
                        "chars": len(text), "preview": preview})

    return app


def serve(config: Optional[Config] = None, host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True) -> int:
    try:
        import flask  # noqa: F401
    except Exception:
        print("Flask is not installed. Install the web extra with:\n"
              '    pip install "sabi-llm[serve]"\n'
              "    # or:  pip install flask")
        return 1

    config = config or load_config()
    if not config.abs_model_path().exists():
        from . import downloader
        print("No model found locally — downloading it now (~2 GB)…")
        try:
            downloader.download_model(config)
            print("Model ready.\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Model download failed: {exc}\nStarting anyway — retry with `sabi download`.\n")

    runtime = Runtime(config).start()
    store = ConversationStore(config.abs_workspace() / ".sabi" / "conversations.json")
    app = create_app(runtime, store)

    # Warm the project-wide memory in the background so agent-mode requests
    # can recall the codebase without blocking server startup on the walk.
    Thread(target=lambda: runtime.index_codebase(cwd=str(Path.cwd())), daemon=True).start()

    url = f"http://{host}:{port}"
    print(f"\n  SABI web UI running at  {url}")
    print("  Press Ctrl+C to stop.\n")
    if open_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)
    return 0
