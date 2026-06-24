"""Config + CLI smoke tests (no model required)."""

from sabi.config import load_config
from sabi import cli


def test_config_defaults():
    cfg = load_config()
    assert cfg.ram_ceiling_gb == 7.0
    assert cfg.abs_model_path().name.endswith(".gguf")


def test_env_override(monkeypatch):
    monkeypatch.setenv("SABI_TEMPERATURE", "0.1")
    cfg = load_config()
    assert cfg.temperature == 0.1


def test_cli_version(capsys):
    rc = cli.main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SABI" in out


def test_cli_doctor_runs():
    # doctor returns 0 or 1 but must not raise
    rc = cli.main(["doctor"])
    assert rc in (0, 1)
