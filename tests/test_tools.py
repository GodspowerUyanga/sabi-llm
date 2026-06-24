"""Tests for the local tool layer."""

from sabi.tools import default_registry


def test_write_and_read(tmp_path):
    reg = default_registry(tmp_path)
    w = reg.get("write_file").run(path="hello.txt", content="hi there")
    assert w.ok
    r = reg.get("read_file").run(path="hello.txt")
    assert r.ok and "hi there" in r.output


def test_sandbox_blocks_traversal(tmp_path):
    reg = default_registry(tmp_path)
    res = reg.get("write_file").run(path="../escape.txt", content="x")
    assert not res.ok


def test_shell_blocks_dangerous(tmp_path):
    reg = default_registry(tmp_path)
    res = reg.get("shell").run(command="rm -rf /")
    assert not res.ok


def test_scaffold_python(tmp_path):
    reg = default_registry(tmp_path)
    res = reg.get("scaffold").run(name="demo", kind="python")
    assert res.ok
    assert (tmp_path / "demo" / "main.py").exists()


def test_list_dir(tmp_path):
    reg = default_registry(tmp_path)
    reg.get("write_file").run(path="a.txt", content="1")
    res = reg.get("list_dir").run(path=".")
    assert res.ok and "a.txt" in res.output
