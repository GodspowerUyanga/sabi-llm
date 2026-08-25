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
