"""Tests for the project scanner."""

from sabi import project_scanner


def test_detects_python_project(tmp_path):
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    info = project_scanner.scan(tmp_path)
    assert "python" in info.languages


def test_detects_git_and_node(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "package.json").write_text('{"dependencies":{"next":"14"}}', encoding="utf-8")
    info = project_scanner.scan(tmp_path)
    assert info.is_git
    assert "javascript" in info.languages
    assert "next" in info.frameworks


def test_detects_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    info = project_scanner.scan(tmp_path)
    assert info.has_venv


def test_summary_is_string(tmp_path):
    info = project_scanner.scan(tmp_path)
    assert isinstance(info.summary(), str)
