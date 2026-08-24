"""sabi-yoruba-tts — English<->Yoruba translation for SABI.

SABI's base model (Qwen2.5-Coder-3B-Instruct) is fluent in English but was
never trained to write Yoruba. Rather than force the coder model to speak a
language it doesn't know, Yoruba is handled as a translation layer around it:

  Yoruba question -> (this module) -> English -> SABI reasons/answers/
  writes code in English -> (this module) -> Yoruba reply.

Backend: NLLB-200-distilled-600M, converted to int8 CTranslate2 (see
``scripts/download_yoruba_model.py``). CTranslate2 runs CPU-only and does not
need ``torch`` at inference time, keeping the runtime dependency footprint
small. The model is lazy-loaded on the first Yoruba turn and then kept warm —
even resident, it adds well under 1 GB, leaving real headroom under the 7 GB
ADTC ceiling alongside the ~2 GB / ~3.5-4.5 GB runtime Qwen2.5-Coder-3B model.

License note: the NLLB-200-distilled-600M checkpoint is CC-BY-NC-4.0
(non-commercial), unlike the rest of SABI (MIT). This is disclosed in
REPORT.md; it is appropriate for a non-commercial hackathon submission but
would need a different translation model before any commercial use.

Code correctness matters more than translation fluency here: fenced code
blocks (```...```) are never sent through the translator, so code explained
in Yoruba still contains real, runnable code.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

SRC_LANG_EN = "eng_Latn"
TGT_LANG_YO = "yor_Latn"

_CODE_FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_LINK_LINE_RE = re.compile(r"^\s*\[[^\]]*\]\([^)]*\)\s*$")
_LIST_PREFIX_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)]|#{1,6})\s+)")

# Yoruba-specific diacritics/marks that never appear in English —
# a cheap, dependency-free signal that inbound text is written in Yoruba.
_YORUBA_CHARS = set("ẹẸọỌṣṢàÀèÈìÌòÒùÙńŃ")
_YORUBA_MARKER_WORDS = {"bawo", "wa", "pele", "jowo", "abeg", "e kaaro",
                          "e kaasan", "e kaale", "se daadaa"}

_lock = threading.Lock()
_translator = None
_tokenizer = None
_loaded_dir: str | None = None


def looks_like_yoruba(text: str) -> bool:
    """Cheap heuristic: does *text* contain Yoruba tonal/diacritic marks or
    common romanized marker words (typed without diacritics, as is common)?"""
    if not text:
        return False
    if any(ch in _YORUBA_CHARS for ch in text):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _YORUBA_MARKER_WORDS)


def available(model_dir: str) -> bool:
    """True if a converted CTranslate2 model is present at *model_dir*."""
    d = Path(model_dir)
    return d.is_dir() and (d / "model.bin").exists()


def release() -> None:
    """Drop the loaded model so it stops counting against resident RAM."""
    global _translator, _tokenizer, _loaded_dir
    with _lock:
        _translator = None
        _tokenizer = None
        _loaded_dir = None


def _load(model_dir: str):
    global _translator, _tokenizer, _loaded_dir
    with _lock:
        if _translator is None or _loaded_dir != model_dir:
            import ctranslate2
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(model_dir)
            _translator = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8")
            _loaded_dir = model_dir
    return _translator, _tokenizer


def _translate_batch(lines: list[str], model_dir: str, source_lang: str, target_lang: str) -> list[str]:
    translator, tokenizer = _load(model_dir)
    tokenizer.src_lang = source_lang
    batch = [tokenizer.convert_ids_to_tokens(tokenizer.encode(line)) for line in lines]
    results = translator.translate_batch(
        batch, target_prefix=[[target_lang]] * len(batch), beam_size=4,
    )
    out = []
    for r in results:
        tokens = r.hypotheses[0][1:]  # drop the forced target-language token
        ids = tokenizer.convert_tokens_to_ids(tokens)
        out.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return out


def translate(text: str, model_dir: str, source_lang: str, target_lang: str) -> str:
    """Translate a short, plain (non-markdown) string of text."""
    text = (text or "").strip()
    if not text:
        return text
    return _translate_batch([text], model_dir, source_lang, target_lang)[0]


def translate_markdown(text: str, model_dir: str,
                        source_lang: str = SRC_LANG_EN, target_lang: str = TGT_LANG_YO) -> str:
    """Translate SABI's reply, preserving fenced code blocks and markdown structure.

    Fenced code blocks are left untouched (a translated code sample is a
    broken code sample). Blank lines, table-separator rows and bare markdown
    links are passed through unchanged; list/heading markers are stripped
    before translation and reattached after.
    """
    if not text or not text.strip():
        return text
    parts = _CODE_FENCE_RE.split(text)
    out = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            out.append(part)
        else:
            out.append(_translate_prose(part, model_dir, source_lang, target_lang))
    return "".join(out)


def _translate_prose(text: str, model_dir: str, source_lang: str, target_lang: str) -> str:
    lines = text.split("\n")
    prefixes: dict[int, str] = {}
    bodies: dict[int, str] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _TABLE_SEP_RE.match(line) or _LINK_LINE_RE.match(line):
            continue
        m = _LIST_PREFIX_RE.match(line)
        prefix = m.group(1) if m else ""
        body = line[len(prefix):]
        if not body.strip():
            continue
        prefixes[i] = prefix
        bodies[i] = body

    if not bodies:
        return text

    order = list(bodies.keys())
    translated = _translate_batch([bodies[i] for i in order], model_dir, source_lang, target_lang)

    out_lines = list(lines)
    for i, t in zip(order, translated):
        out_lines[i] = prefixes[i] + t
    return "\n".join(out_lines)


def to_english(text: str, model_dir: str) -> str:
    """Translate a Yoruba user turn to English before routing/generation."""
    return translate(text, model_dir, TGT_LANG_YO, SRC_LANG_EN)


def to_yoruba(text: str, model_dir: str) -> str:
    """Translate SABI's English reply (markdown) to Yoruba, code-safe."""
    return translate_markdown(text, model_dir, SRC_LANG_EN, TGT_LANG_YO)
