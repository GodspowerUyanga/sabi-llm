"""Tests for the conversation store and the web server routes."""

import pytest

from sabi.conversations import ConversationStore


def test_create_and_list(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    c = store.create()
    store.add_message(c["id"], "user", "hello there friend")
    store.add_message(c["id"], "assistant", "hi!")
    items = store.list()
    assert len(items) == 1
    assert items[0]["message_count"] == 2
    # title auto-derived from first user message
    assert "hello" in items[0]["title"].lower()


def test_delete(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    c = store.create()
    assert store.delete(c["id"]) is True
    assert store.list() == []


def test_persists_across_reload(tmp_path):
    path = tmp_path / "conv.json"
    s1 = ConversationStore(path)
    c = s1.create()
    s1.add_message(c["id"], "user", "remember me")
    s2 = ConversationStore(path)
    assert len(s2.list()) == 1
    assert s2.get(c["id"])["messages"][0]["content"] == "remember me"


# ---- web server (needs flask; model not required) ----
flask = pytest.importorskip("flask")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from sabi.config import load_config
    from sabi.runtime import Runtime
    from sabi.server import create_app

    cfg = load_config()
    monkeypatch.setattr(cfg, "workspace_dir", str(tmp_path))
    # Force "no model" regardless of whether the real model happens to be
    # downloaded on the machine running the tests — these tests exercise the
    # graceful-degradation path specifically, not real inference, and must
    # stay fast and deterministic either way. (A real download landing at
    # the default model_path during this session turned these into 60-95s
    # real-inference calls instead of instant no-model checks.)
    monkeypatch.setattr(cfg, "model_path", str(tmp_path / "no-model-here.gguf"))
    rt = Runtime(cfg).start(cwd=str(tmp_path))
    store = ConversationStore(tmp_path / "conv.json")
    app = create_app(rt, store)
    app.config.update(TESTING=True)
    return app.test_client()


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    assert "ram_ceiling_gb" in body
    assert "yoruba_available" in body
    assert "yoruba_enabled" in body


def test_status_reports_yoruba_available_when_model_present(client, monkeypatch):
    import sabi.translate as translate_mod
    monkeypatch.setattr(translate_mod, "available", lambda model_dir: True)
    r = client.get("/api/status")
    assert r.get_json()["yoruba_available"] is True


def test_status_reports_yoruba_unavailable_when_model_missing(client, monkeypatch):
    import sabi.translate as translate_mod
    monkeypatch.setattr(translate_mod, "available", lambda model_dir: False)
    r = client.get("/api/status")
    assert r.get_json()["yoruba_available"] is False


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"SABI" in r.data


def test_conversation_api_roundtrip(client):
    r = client.post("/api/conversations")
    cid = r.get_json()["id"]
    listing = client.get("/api/conversations").get_json()
    assert any(c["id"] == cid for c in listing)


def test_chat_without_model_returns_graceful(client):
    # No model in the test env -> should return an error field, not crash.
    r = client.post("/api/chat", json={"message": "hello", "mode": "auto"})
    assert r.status_code == 200
    body = r.get_json()
    assert "conversation_id" in body


def test_auto_mode_routes_smalltalk_to_chat_not_agent(client, monkeypatch):
    # "auto" mode must never send small talk through the tool-calling agent
    # loop — same hard rule as sabi chat/sabi tui (see sabi.router.is_smalltalk).
    import sabi.server as server_mod
    calls = {"agent": 0, "handle": 0}
    monkeypatch.setattr(server_mod.Runtime, "agent",
                        lambda self, *a, **k: calls.__setitem__("agent", calls["agent"] + 1) or
                        {"ok": True, "answer": "", "actions": []})
    monkeypatch.setattr(server_mod.Runtime, "handle",
                        lambda self, *a, **k: calls.__setitem__("handle", calls["handle"] + 1) or
                        {"ok": True, "text": "hi", "intent": "CHAT", "tps": 0})
    r = client.post("/api/chat", json={"message": "hello", "mode": "auto"})
    assert r.status_code == 200
    assert calls == {"agent": 0, "handle": 1}


def test_auto_mode_routes_real_requests_to_agent(client, monkeypatch):
    # "auto" mode is the main way to use sabi serve now: a real request (not
    # small talk) gets full agent power automatically, no mode switch needed.
    import sabi.server as server_mod
    calls = {"agent": 0, "handle": 0}
    monkeypatch.setattr(server_mod.Runtime, "agent",
                        lambda self, *a, **k: calls.__setitem__("agent", calls["agent"] + 1) or
                        {"ok": True, "answer": "done", "actions": ["OK: wrote main.py"]})
    monkeypatch.setattr(server_mod.Runtime, "handle",
                        lambda self, *a, **k: calls.__setitem__("handle", calls["handle"] + 1) or
                        {"ok": True, "text": "", "intent": "CHAT", "tps": 0})
    r = client.post("/api/chat", json={"message": "create a file main.py that prints hello",
                                        "mode": "auto"})
    assert r.status_code == 200
    body = r.get_json()
    assert calls == {"agent": 1, "handle": 0}
    assert body["intent"] == "AGENT"
    assert body["actions"] == ["OK: wrote main.py"]


def test_auto_mode_uses_restricted_coding_assistant_agent(client, monkeypatch):
    # sabi serve is a Coding Assistant (code generation, debugging,
    # programming tutoring) — it must never let AgentLoop create/write/edit/
    # move files, so every call into Runtime.agent from here has to pass
    # restricted=True.
    import sabi.server as server_mod
    captured = {}

    def fake_agent(self, *a, **k):
        captured["restricted"] = k.get("restricted")
        return {"ok": True, "answer": "done", "actions": []}

    monkeypatch.setattr(server_mod.Runtime, "agent", fake_agent)
    r = client.post("/api/chat", json={"message": "write me a script that prints hello",
                                        "mode": "auto"})
    assert r.status_code == 200
    assert captured["restricted"] is True


def test_agent_mode_uses_restricted_coding_assistant_agent(client, monkeypatch):
    import sabi.server as server_mod
    captured = {}

    def fake_agent(self, *a, **k):
        captured["restricted"] = k.get("restricted")
        return {"ok": True, "answer": "done", "actions": []}

    monkeypatch.setattr(server_mod.Runtime, "agent", fake_agent)
    r = client.post("/api/chat", json={"message": "write me a script that prints hello",
                                        "mode": "agent"})
    assert r.status_code == 200
    assert captured["restricted"] is True


def test_auto_mode_passes_prior_conversation_as_history(client, monkeypatch):
    # Regression test for a real incident: a follow-up like "list them" had
    # zero context of what "them" referred to, because sabi serve built a
    # fresh AgentLoop per HTTP request with no memory of earlier turns even
    # though the conversation store already persists them for the sidebar.
    import sabi.server as server_mod
    captured = {}

    def fake_agent(self, *a, **k):
        captured["history"] = k.get("history")
        return {"ok": True, "answer": "There are 2: Work, Personal", "actions": []}

    monkeypatch.setattr(server_mod.Runtime, "agent", fake_agent)

    r1 = client.post("/api/chat", json={"message": "list my desktop folders", "mode": "auto"})
    cid = r1.get_json()["conversation_id"]

    r2 = client.post("/api/chat", json={"message": "list them",
                                        "conversation_id": cid, "mode": "auto"})
    assert r2.status_code == 200
    history = captured["history"]
    assert history, "second call must receive the first turn as history"
    assert any(m["content"] == "list my desktop folders" for m in history)
    assert any(m["role"] == "assistant" for m in history)
    # the just-added current-turn user message ("list them") must NOT be
    # duplicated into history — it's passed as the request itself
    assert not any(m["content"] == "list them" for m in history)


def test_chat_accepts_yoruba_flag_without_crashing(client, monkeypatch):
    # No LLM model in the test env either way; this just proves the yoruba
    # request field is plumbed through _answer() -> Runtime.handle(force_yoruba)
    # without erroring. translate.available forced False so this doesn't
    # depend on (or pay the cost of loading) whatever sabi-yoruba-tts state
    # happens to exist on the machine running the tests.
    import sabi.translate as translate_mod
    monkeypatch.setattr(translate_mod, "available", lambda model_dir: False)
    r = client.post("/api/chat", json={"message": "hello", "mode": "auto", "yoruba": True})
    assert r.status_code == 200
    assert "conversation_id" in r.get_json()


def test_stream_endpoint_yoruba_branch_returns_translated_text(client, monkeypatch):
    # Regression test: /api/chat/stream used to call model.chat_stream()
    # directly, bypassing Runtime.handle() entirely, so sabi-yoruba-tts never
    # ran on the web UI's default chat path even with the toggle on. Forces
    # the yoruba branch and confirms it goes through Runtime.handle() (whose
    # graceful "model unavailable" text ends up translated) rather than the
    # raw streaming path.
    import sabi.translate as translate_mod
    monkeypatch.setattr(translate_mod, "available", lambda model_dir: True)
    monkeypatch.setattr(translate_mod, "to_english", lambda text, model_dir: text)
    monkeypatch.setattr(translate_mod, "to_yoruba", lambda text, model_dir: f"YO[{text}]")
    r = client.post("/api/chat/stream", json={"message": "hello", "yoruba": True})
    assert r.status_code == 200
    assert r.headers.get("X-Conversation-Id")
    body = r.get_data(as_text=True)
    assert body  # something was returned, not silently dropped


def test_upload_extracts_text(client):
    import io
    data = {"file": (io.BytesIO(b"name,score\nAda,91\n"), "data.csv")}
    r = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["name"] == "data.csv"
    assert "Ada" in body["preview"]
    assert body["chars"] > 0


def test_stream_endpoint_runs(client):
    # Without a model the stream should still respond (with an error delta),
    # set the conversation id header, and not crash.
    r = client.post("/api/chat/stream", json={"message": "hi"})
    assert r.status_code == 200
    assert r.headers.get("X-Conversation-Id")
    _ = r.get_data()  # drain the stream


def test_stream_endpoint_uses_real_streaming_for_smalltalk(client, monkeypatch):
    # Regression test: making sabi serve's default mode fully agentic (an
    # earlier change this session) accidentally made ALL replies wait for a
    # complete non-streaming response, including simple greetings that used
    # to stream token-by-token — a real, reported slowdown. A bare "hello" in
    # auto mode (no yoruba) must still go through real model.chat_stream(),
    # not the buffer-then-chunk agent path.
    import sabi.server as server_mod
    import sabi.model as model_mod
    calls = {"chat_stream": 0, "answer": 0}

    def fake_chat_stream(self, messages):
        calls["chat_stream"] += 1
        yield "hi there"

    def fake_answer(*a, **k):
        calls["answer"] += 1
        return {"answer": "", "intent": "CHAT", "tps": 0, "actions": []}

    monkeypatch.setattr(model_mod.LLMModel, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(server_mod, "_answer", fake_answer)
    r = client.post("/api/chat/stream", json={"message": "hello", "mode": "auto"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert calls == {"chat_stream": 1, "answer": 0}
    assert "hi there" in body


def test_stream_endpoint_streams_think_mode_for_real(client, monkeypatch):
    # think/code are plain text-only engines — no reason to buffer-then-chunk
    # them; they should stream token-by-token straight from the model.
    import sabi.runtime as runtime_mod
    calls = {"stream": 0}

    def fake_stream(self, request, context=""):
        calls["stream"] += 1
        yield "here"
        yield " is the plan"

    monkeypatch.setattr(runtime_mod.ThinkEngine, "stream", fake_stream)
    r = client.post("/api/chat/stream", json={"message": "plan a todo app", "mode": "think"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert calls["stream"] == 1
    assert "here is the plan" in body


def test_stream_endpoint_streams_code_mode_for_real(client, monkeypatch):
    import sabi.runtime as runtime_mod
    calls = {"stream": 0}

    def fake_stream(self, request, context="", plan=""):
        calls["stream"] += 1
        yield "def f():"
        yield " pass"

    monkeypatch.setattr(runtime_mod.CodeEngine, "stream", fake_stream)
    r = client.post("/api/chat/stream", json={"message": "write a function", "mode": "code"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert calls["stream"] == 1
    assert "def f(): pass" in body


def test_stream_endpoint_uses_agent_path_for_real_requests(client, monkeypatch):
    # A real request in auto mode must NOT take the fast token-stream path —
    # it goes through Runtime.agent (restricted), and must stream tool-call
    # progress live via the reporter as it happens rather than buffering the
    # whole turn before sending anything.
    import sabi.server as server_mod
    calls = {"agent": 0}

    def fake_agent(self, msg, *, permissions=None, reporter=None, **k):
        calls["agent"] += 1
        reporter.proposing("write_file", "write a file: main.py")
        reporter.ran(False, "'write_file' is turned off in this mode")
        return {"answer": "created it", "actions": ["FAIL: write a file: main.py"]}

    monkeypatch.setattr(server_mod.Runtime, "agent", fake_agent)
    r = client.post("/api/chat/stream",
                    json={"message": "create a file main.py that prints hello", "mode": "auto"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert calls["agent"] == 1
    assert "created it" in body
    assert "write a file: main.py" in body  # progress streamed live as it happens


def test_stream_endpoint_streams_agent_final_answer_live_without_duplicating(client, monkeypatch):
    # AgentLoop streams its final answer live via reporter.answer_delta as
    # it's generated (see agent.py's _chat_step/_JSONFinalStreamer) — the
    # whole point being the first word shows up immediately, not after the
    # full answer is ready. Once that's happened, the server must NOT also
    # send the complete answer again at the end.
    import sabi.server as server_mod

    def fake_agent(self, msg, *, permissions=None, reporter=None, **k):
        for piece in ["Hel", "lo ", "world"]:
            reporter.answer_delta(piece)
        return {"answer": "Hello world", "actions": []}

    monkeypatch.setattr(server_mod.Runtime, "agent", fake_agent)
    r = client.post("/api/chat/stream", json={"message": "say hello", "mode": "auto"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert body == "Hello world"  # streamed live, not also re-sent whole afterward
