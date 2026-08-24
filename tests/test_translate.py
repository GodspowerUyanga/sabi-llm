"""Tests for sabi-yoruba-tts (no model download needed — translation calls are mocked)."""

from sabi import translate


def test_looks_like_yoruba_diacritics():
    assert translate.looks_like_yoruba("Ṣe o le ràn mí lọ́wọ́?")
    assert not translate.looks_like_yoruba("fix this bug in my python code")


def test_looks_like_yoruba_marker_words():
    assert translate.looks_like_yoruba("bawo ni, how far")
    assert translate.looks_like_yoruba("jowo help me debug this")


def test_available_false_when_model_missing(tmp_path):
    assert translate.available(str(tmp_path / "does-not-exist")) is False


def test_translate_markdown_preserves_code_fences(monkeypatch):
    def fake_batch(lines, model_dir, source_lang, target_lang):
        return [f"YO[{l}]" for l in lines]

    monkeypatch.setattr(translate, "_translate_batch", fake_batch)
    text = "Here is a loop:\n\n```python\nfor i in range(3):\n    print(i)\n```\n\nThat prints 0 to 2."
    out = translate.translate_markdown(text, "models/sabi-yoruba-tts")
    assert "```python\nfor i in range(3):\n    print(i)\n```" in out  # code untouched
    assert "YO[Here is a loop:]" in out
    assert "YO[That prints 0 to 2.]" in out


def test_translate_markdown_preserves_list_prefixes(monkeypatch):
    def fake_batch(lines, model_dir, source_lang, target_lang):
        return [f"YO[{l}]" for l in lines]

    monkeypatch.setattr(translate, "_translate_batch", fake_batch)
    out = translate.translate_markdown("- first item\n- second item", "models/sabi-yoruba-tts")
    assert out == "- YO[first item]\n- YO[second item]"


def test_to_english_and_to_yoruba_use_correct_directions(monkeypatch):
    calls = []

    def fake_batch(lines, model_dir, source_lang, target_lang):
        calls.append((source_lang, target_lang))
        return [f"[{source_lang}->{target_lang}] {l}" for l in lines]

    monkeypatch.setattr(translate, "_translate_batch", fake_batch)
    translate.to_english("Bawo ni", "models/sabi-yoruba-tts")
    translate.to_yoruba("hello", "models/sabi-yoruba-tts")
    assert calls == [
        (translate.TGT_LANG_YO, translate.SRC_LANG_EN),
        (translate.SRC_LANG_EN, translate.TGT_LANG_YO),
    ]
