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
from typing import Optional

from .config import Config, load_config
from .runtime import Runtime
from .conversations import ConversationStore
from .permissions import PermissionManager
from .agent import Reporter

WEB_DIR = Path(__file__).resolve().parent / "ui" / "web"


def _answer(runtime: Runtime, message: str, mode: str) -> dict:
    """Produce an assistant reply for a message in the given mode."""
    try:
        if mode == "think":
            gen = runtime.think.run(message)
            return {"answer": gen.text, "intent": "THINK",
                    "tps": round(gen.tokens_per_second, 2), "actions": []}
        if mode == "code":
            gen = runtime.code.run(message)
            return {"answer": gen.text, "intent": "CODE",
                    "tps": round(gen.tokens_per_second, 2), "actions": []}
        if mode == "agent":
            perms = PermissionManager(auto_approve=True)  # web auto-approves
            res = runtime.agent(message, permissions=perms, reporter=Reporter())
            return {"answer": res.get("answer", ""), "intent": "AGENT",
                    "tps": 0, "actions": res.get("actions", [])}
        # auto / chat
        res = runtime.handle(message)
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

    # ---- chat ----
    @app.post("/api/chat")
    def chat():
        body = request.json or {}
        cid = body.get("conversation_id")
        message = (body.get("message") or "").strip()
        mode = body.get("mode", "auto")
        if not message:
            return jsonify({"error": "empty message"}), 400
        if not cid or not store.get(cid):
            conv = store.create()
            cid = conv["id"]

        store.add_message(cid, "user", message)
        result = _answer(runtime, message, mode)
        reply = result.get("answer") or result.get("error") or "(no response)"
        store.add_message(cid, "assistant", reply, meta={
            "intent": result.get("intent"),
            "tps": result.get("tps"),
            "actions": result.get("actions", []),
            "error": result.get("error"),
        })
        return jsonify({"conversation_id": cid, **result})

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
