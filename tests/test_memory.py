"""Tests for the JSON memory store."""

from sabi.memory import MemoryStore


def test_add_and_recall(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.add_turn("user", "hello", "CHAT")
    store.add_turn("assistant", "hi", "CHAT")
    assert store.stats()["turns"] == 2
    assert "hello" in store.history_text()


def test_tasks_persist(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.add_task("build feature", "done", "CODE")
    # reload from disk
    store2 = MemoryStore(path)
    assert store2.stats()["tasks"] == 1


def test_clear(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.add_turn("user", "x")
    store.clear()
    assert store.stats()["turns"] == 0


def test_corrupt_file_is_safe(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = MemoryStore(path)  # should not raise
    assert store.stats()["turns"] == 0
