"""Tests for Runtime.index_codebase — the startup codebase-memory warm-up."""

from sabi.config import load_config
from sabi.runtime import Runtime


def _rt(tmp_path):
    # Isolate memory/vector-store/prompts under tmp_path so tests never touch
    # the real project's own .sabi folder or install-root workspace.
    cfg = load_config(root=tmp_path)
    return Runtime(cfg).start(cwd=str(tmp_path))


def test_index_codebase_indexes_source_files_of_any_language(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "main.py").write_text("def add(a, b):\n    return a + b\n")
    (src / "app.js").write_text("function add(a, b) { return a + b; }\n")
    (src / "Main.java").write_text("class Main { }\n")
    rt = _rt(tmp_path)
    added = rt.index_codebase(cwd=str(src))
    assert added == 3
    sources = {rec["source"] for rec in rt.retriever.store.records}
    assert str(src / "main.py") in sources
    assert str(src / "app.js") in sources
    assert str(src / "Main.java") in sources


def test_index_codebase_skips_junk_dirs_and_non_text_files(tmp_path):
    src = tmp_path / "proj"
    (src / "node_modules").mkdir(parents=True)
    (src / "node_modules" / "dep.js").write_text("noise\n")
    (src / "main.py").write_text("print('hi')\n")
    (src / "binary.bin").write_bytes(b"\x00\x01\x02")
    rt = _rt(tmp_path)
    added = rt.index_codebase(cwd=str(src))
    assert added == 1
    sources = {rec["source"] for rec in rt.retriever.store.records}
    assert str(src / "main.py") in sources
    assert not any("node_modules" in s for s in sources)


def test_index_codebase_does_not_reindex_already_indexed_files(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "main.py").write_text("print('hi')\n")
    rt = _rt(tmp_path)
    first = rt.index_codebase(cwd=str(src))
    second = rt.index_codebase(cwd=str(src))
    assert first == 1
    assert second == 0  # already indexed -> no duplicate records
    assert len([r for r in rt.retriever.store.records
                if r["source"] == str(src / "main.py")]) == 1


def test_index_codebase_respects_max_files_cap(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    for i in range(5):
        (src / f"f{i}.py").write_text(f"x = {i}\n")
    rt = _rt(tmp_path)
    added = rt.index_codebase(cwd=str(src), max_files=2)
    assert added == 2
