"""Config + CLI smoke tests (no model required)."""

from sabi.config import load_config
from sabi import cli


def test_config_defaults():
    cfg = load_config()
    assert cfg.ram_ceiling_gb == 7.0
    assert cfg.abs_model_path().name.endswith(".gguf")


def test_repeat_penalty_set_to_prevent_degenerate_looping():
    # Regression test for a real incident: a small model, with no repeat
    # penalty configured at all, generated a token ("ẹtọ") repeated hundreds
    # of times in a row instead of a real answer. > 1.0 actively discourages
    # repeating recent tokens (1.0 = no penalty, off).
    cfg = load_config()
    assert cfg.repeat_penalty > 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("SABI_TEMPERATURE", "0.1")
    cfg = load_config()
    assert cfg.temperature == 0.1


def test_model_path_prefers_existing_repo_relative_file(tmp_path, monkeypatch):
    # The ADTC 2026 submission contract (metadata.json, download_model.sh)
    # hardcodes "models/<file>" relative to the repo — if that file is
    # already there (a real checkout that ran download_model.sh), SABI must
    # keep using it exactly, not silently look elsewhere.
    (tmp_path / "models").mkdir()
    legacy = tmp_path / "models" / "sabi-v1.Q4_K_M.gguf"
    legacy.write_bytes(b"fake")
    cfg = load_config(root=tmp_path)
    assert cfg.abs_model_path() == legacy


def test_model_path_falls_back_to_home_when_no_repo_copy(tmp_path, monkeypatch):
    # A real `pip install sabi-llm` with no repo checkout at all: nothing at
    # the repo-relative path, so it must resolve under the user's home dir
    # instead of a (likely unwritable, and wrong-scoped) install directory.
    import sabi.config as config_mod
    fake_home = tmp_path / "home"
    monkeypatch.setattr(config_mod, "USER_DATA_ROOT", fake_home / ".sabi")
    cfg = load_config(root=tmp_path / "empty_install_dir")
    assert cfg.abs_model_path() == fake_home / ".sabi" / "models" / "sabi-v1.Q4_K_M.gguf"


def test_cli_version(capsys):
    rc = cli.main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SABI" in out


def test_cli_doctor_runs():
    # doctor returns 0 or 1 but must not raise
    rc = cli.main(["doctor"])
    assert rc in (0, 1)
