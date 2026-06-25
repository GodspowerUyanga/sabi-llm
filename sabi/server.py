"""Web server for `sabi serve`.

Serves a professional chat UI (ChatGPT/Claude-style) with persistent history,
backed by the SABI runtime. Flask is an optional dependency installed via
``pip install "sabi-llm[serve]"``.

Modes per message:
  * auto   - router decides THINK / CODE / CHAT (default; conversational)
  * think  - planning / analysis engine
  * code   - code generation engine (returns code as text)
  * agent  - acting agent (can create files / run commands); web runs it in
             auto-approve mode, so the UI shows a clear warning and lists the
             actions taken.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from threading import Timer
from typing import Dict, List, Optional

from .config import Config, load_config
from .runtime import Runtime
from .conversations import ConversationStore
from .permissions import PermissionManager
from .agent import Reporter
from .filereader import read_any

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


def _answer(runtime: Runtime, message: str, mode: str, cid: Optional[str] = None) -> dict:
    """Produce an assistant reply for a message in the given mode."""
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
            res = runtime.agent(msg, permissions=perms, reporter=Reporter())
            return {"answer": res.get("answer", ""), "intent": "AGENT",
                    "tps": 0, "actions": res.get("actions", [])}
        # auto / chat
        res = runtime.handle(msg)
        if res.get("ok"):
            return {"answer": res.get("text", ""), "intent": res.get("intent", "CHAT"),
                    "tps": res.get("tps", 0), "actions": []}
        return {"answer": "", "error": res.get("error", "request failed"),
                "intent": res.get("intent", "CHAT"), "actions": []}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "error": str(exc), "intent": mode.upper(), "actions": []}


def create_app(runtime: Runtime, store: ConversationStore):
    from flask import Flask, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)

    # ---- static frontend ----
    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/static/<path:fname>")
    def static_files(fname):
        return send_from_directory(WEB_DIR, fname)

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
        if not message:
            return jsonify({"error": "empty message"}), 400
        if not cid or not store.get(cid):
            cid = store.create()["id"]

        store.add_message(cid, "user", message)
        result = _answer(runtime, message, mode, cid=cid)
        reply = result.get("answer") or result.get("error") or "(no response)"
        store.add_message(cid, "assistant", reply, meta={
            "intent": result.get("intent"), "tps": result.get("tps"),
            "actions": result.get("actions", []), "error": result.get("error"),
        })
        return jsonify({"conversation_id": cid, **result})

    # ---- chat (streaming, token-by-token) ----
    @app.post("/api/chat/stream")
    def chat_stream():
        from flask import Response, stream_with_context
        body = request.json or {}
        cid = body.get("conversation_id")
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        if not cid or not store.get(cid):
            cid = store.create()["id"]
        store.add_message(cid, "user", message)

        ctx = _file_context(cid)
        system = runtime.prompts.get("system", "") or None
        user = message + ("\n\n" + ctx if ctx else "")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        def generate():
            buf = ""
            try:
                streamed = False
                for delta in runtime.model.chat_stream(messages):
                    streamed = True
                    buf += delta
                    yield delta
                if not streamed:
                    buf = runtime.model.generate(user, system=system).text
                    yield buf
            except Exception as exc:  # noqa: BLE001 (includes ModelUnavailable)
                err = f"\n\n⚠ {exc}"
                buf += err
                yield err
            store.add_message(cid, "assistant", buf, meta={"intent": "CHAT"})

        resp = Response(stream_with_context(generate()), mimetype="text/plain")
        resp.headers["X-Conversation-Id"] = cid
        resp.headers["X-Accel-Buffering"] = "no"
        resp.headers["Cache-Control"] = "no-cache"
        return resp

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
    runtime = Runtime(config).start()
    store = ConversationStore(config.abs_workspace() / ".sabi" / "conversations.json")
    app = create_app(runtime, store)

    url = f"http://{host}:{port}"
    print(f"\n  SABI web UI running at  {url}")
    print("  Press Ctrl+C to stop.\n")
    if open_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)
    return 0
