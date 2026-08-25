"""Tests for the sabi-yoruba-tts wiring in Runtime (no real model needed).

Exercises Runtime._yoruba_status / _to_english / _to_yoruba directly, since
those only depend on config + the translate module, not a loaded LLM.
"""

from sabi.config import load_config
from sabi.runtime import Runtime
from sabi import translate


def _runtime(tmp_path, yoruba_enabled=True):
    rt = Runtime.__new__(Runtime)  # skip start(): these helpers don't need a loaded model
    rt.config = load_config()
    rt.config.yoruba_enabled = yoruba_enabled
    rt.config.yoruba_model_path = str(tmp_path / "sabi-yoruba-tts")
    return rt


def test_status_off_for_english(tmp_path):
    rt = _runtime(tmp_path)
    assert rt._yoruba_status("fix this bug in my code") == "off"


def test_status_unavailable_when_model_missing(tmp_path):
    rt = _runtime(tmp_path)
    assert rt._yoruba_status("Ṣe o le ràn mí lọ́wọ́?") == "unavailable"


def test_status_off_when_disabled_even_with_yoruba_text(tmp_path):
    rt = _runtime(tmp_path, yoruba_enabled=False)
    assert rt._yoruba_status("Ṣe o le ràn mí lọ́wọ́?") == "off"


def test_status_active_when_model_present(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    monkeypatch.setattr(translate, "available", lambda model_dir: True)
    assert rt._yoruba_status("Ṣe o le ràn mí lọ́wọ́?") == "active"


def test_to_english_and_to_yoruba_round_trip(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    monkeypatch.setattr(translate, "to_english", lambda text, model_dir: f"EN[{text}]")
    monkeypatch.setattr(translate, "to_yoruba", lambda text, model_dir: f"YO[{text}]")
    assert rt._to_english("Bawo ni") == "EN[Bawo ni]"
    assert rt._to_yoruba("hello") == "YO[hello]"


def test_to_yoruba_falls_back_gracefully_on_error(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)

    def boom(text, model_dir):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(translate, "to_yoruba", boom)
    out = rt._to_yoruba("hello")
    assert "hello" in out and "unavailable" in out.lower()


# ------------------------------------------------------- force (sabi serve toggle)
def test_force_activates_yoruba_for_english_text_when_model_present(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    monkeypatch.setattr(translate, "available", lambda model_dir: True)
    # Plain English text — would normally be "off" — but the explicit UI
    # toggle (sabi serve's Yorùbá button) should force it regardless.
    assert rt._yoruba_status("what is our total revenue?") == "off"
    assert rt._yoruba_status("what is our total revenue?", force=True) == "active"


def test_force_reports_unavailable_not_active_when_model_missing(tmp_path):
    rt = _runtime(tmp_path)  # no model at this path
    assert rt._yoruba_status("hello", force=True) == "unavailable"


def test_force_off_when_yoruba_disabled_in_config(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, yoruba_enabled=False)
    monkeypatch.setattr(translate, "available", lambda model_dir: True)
    assert rt._yoruba_status("hello", force=True) == "off"


def test_yoruba_available_reflects_translate_available(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    monkeypatch.setattr(translate, "available", lambda model_dir: False)
    assert rt.yoruba_available() is False
    monkeypatch.setattr(translate, "available", lambda model_dir: True)
    assert rt.yoruba_available() is True
