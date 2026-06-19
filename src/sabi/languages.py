"""
Language support for Sabi — English (primary) and Nigerian Pidgin.

Scope decision: the ADTC FAQ states English is the primary language of
evaluation and a local language is a *bonus*, not a requirement. Rather than
spread a small model thinly across many African languages it cannot speak
fluently, Sabi focuses on two it does well: clear English and natural Nigerian
Pidgin (an African language spoken by ~100M people). This keeps answers
accurate — which is 50% of the score — instead of producing broken text in
languages the model fails at.

Detection is a dependency-free keyword heuristic (no extra RAM) just enough to
route between English and Pidgin for short prompts.
"""
from __future__ import annotations

LANG_NAMES = {"en": "English", "pcm": "Nigerian Pidgin"}

# High-signal Pidgin markers.
_PCM_MARKERS = {
    "abeg", "wetin", "dey", "wahala", "oga", "una", "comot", "wey", "abi",
    "biko", "shey", "fit", "dem", "go fit", "how far", "how you dey", "make i",
    "make we", "no be", "na so", "sabi", "gist", "small small", "wahalla", "wan",
}


def detect_language(text: str, supported: list[str] | None = None) -> str:
    """Best-guess language code ('en' or 'pcm'); defaults to English."""
    allowed = set(supported or list(LANG_NAMES))
    if "pcm" not in allowed or not text or not text.strip():
        return "en"
    lowered = f" {text.lower()} "
    hits = sum(1 for m in _PCM_MARKERS if f" {m} " in lowered or m in lowered)
    strong = any(f" {m} " in lowered or m in lowered
                 for m in ("dey", "wetin", "abeg", "wey", "na so", "how far", "how you dey", "wahala"))
    short = len(text.split()) <= 6
    if hits >= 2 or (hits >= 1 and (strong or short)):
        return "pcm"
    return "en"


def language_directive(code: str) -> str:
    """System-prompt fragment instructing Sabi which language to reply in."""
    if code != "pcm":
        return ""
    return (
        "\nLANGUAGE INSTRUCTION — reply in NATURAL NIGERIAN PIDGIN. Talk like a "
        "friendly Lagos business person. Study these examples and match the style:\n"
        "  User: How you dey?\n"
        "  Sabi: I dey fine o, thank you! Wetin you wan make I help you do today?\n"
        "  User: Wetin be my total sales?\n"
        "  Sabi: Your total sales na ₦1,250,000. You wan make I break am down by region?\n"
        "  User: Who dey owe me money?\n"
        "  Sabi: Na two people dey owe you: Mary Okafor (₦320,000) and Blessing Eze (₦140,000).\n"
        "  User: Abeg summarise this document.\n"
        "  Sabi: No wahala. Na so the document talk: ...\n"
        "Use words like 'na', 'dey', 'wetin', 'abeg', 'make', 'fit', 'no be', 'o', 'wan'. "
        "Keep am short and sweet. Keep all names, numbers and ₦ amounts EXACTLY as they are — "
        "accuracy pass grammar. Never mix in big English sentences; stay for Pidgin."
    )
