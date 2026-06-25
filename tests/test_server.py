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
    rt = Runtime(cfg).start()
    store = ConversationStore(tmp_path / "conv.json")
    app = create_app(rt, store)
    app.config.update(TESTING=True)
    return app.test_client()


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "ram_ceiling_gb" in r.get_json()


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
