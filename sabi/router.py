"""Intent Router.

Classifies an incoming request into one of three intents:

  * THINK  - business analysis, planning, PRD/SOP, requirements, architecture
  * CODE   - code generation, debugging, scaffolding, file/automation work
  * CHAT   - general question / conversation

A fast keyword heuristic runs first (works fully offline, no model needed).
When the model is available and the heuristic is low-confidence, the router
asks the model to arbitrate using the router prompt template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .model import LLMModel, ModelUnavailable

THINK = "THINK"
CODE = "CODE"
CHAT = "CHAT"

# Deliberately an ALLOWLIST of unambiguous small talk, not a keyword search for
# "no action words present" — the latter is exactly the heuristic that used to
# gate the agent loop and got removed for misclassifying natural action
# phrasing ("in the app folder you made, add a file" has no obvious verb
# either). An allowlist only ever pulls out messages nobody would phrase a
# real request as, so it can't reintroduce that false-negative problem: it
# has zero effect on "create a file", "fix this bug", etc. This exists
# because a bare greeting like "hello" was observed reaching the tool-calling
# agent loop and, on a small model, resulting in an invented, executed tool
# call that corrupted real repository files — a plain-text CHAT reply carries
# no such risk, so this check is a hard gate in code, not just a prompt rule.
_SMALLTALK = {
    "hi", "hello", "hey", "hiya", "yo", "sup", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "how's it going", "hows it going",
    "what's up", "whats up",
    "thanks", "thank you", "thx", "ty", "cheers",
    "ok", "okay", "cool", "nice", "great", "awesome", "got it",
    "bye", "goodbye", "see you", "later",
}


def is_smalltalk(text: str) -> bool:
    """True only for unambiguous greetings/pleasantries with no task content.

    Exact-match against a fixed allowlist (after trimming punctuation/case) —
    not a substring or keyword match — so "hi, can you create a file" does
    NOT match: the moment a message carries any other content, this returns
    False and the normal agent/router path handles it.
    """
    t = re.sub(r"[!.?]+$", "", (text or "").strip().lower())
    return t in _SMALLTALK

_CODE_HINTS = [
    "code", "function", "class", "bug", "debug", "error", "traceback",
    "implement", "refactor", "compile", "script", "api", "endpoint",
    "scaffold", "build a", "write a program", "unit test", "pytest",
    "python", "javascript", "typescript", "react", "next.js", "node",
    "fix", "stack trace", "exception", "regex", "sql", "dockerfile",
    "create file", "generate code", "git", "repository", "package",
]

_THINK_HINTS = [
    "plan", "strategy", "prd", "sop", "requirement", "architecture",
    "design", "analyse", "analyze", "business", "roadmap", "spec",
    "specification", "outline", "proposal", "decision", "trade-off",
    "tradeoff", "estimate", "scope", "milestone", "stakeholder",
    "workflow", "process", "documentation", "user story",
]


@dataclass
class Routing:
    intent: str
    confidence: float
    reason: str


class Router:
    def __init__(self, model: Optional[LLMModel] = None):
        self.model = model

    # ------------------------------------------------------------- heuristic
    def _heuristic(self, text: str) -> Routing:
        t = text.lower()
        code_score = sum(1 for h in _CODE_HINTS if h in t)
        think_score = sum(1 for h in _THINK_HINTS if h in t)

        # Code fences or obvious code patterns strongly imply CODE.
        if "```" in text or re.search(r"\bdef\s+\w+\(|=>|;\s*$", text):
            code_score += 3

        if code_score == 0 and think_score == 0:
            return Routing(CHAT, 0.5, "no strong keyword signal")

        if code_score >= think_score:
            total = code_score + think_score
            conf = min(0.95, 0.55 + 0.1 * (code_score - think_score))
            return Routing(CODE, conf, f"code signal {code_score} vs think {think_score}")
        total = code_score + think_score
        conf = min(0.95, 0.55 + 0.1 * (think_score - code_score))
        return Routing(THINK, conf, f"think signal {think_score} vs code {code_score}")

    # ---------------------------------------------------------- model assist
    def _model_arbitrate(self, text: str, template: str) -> Optional[Routing]:
        if not self.model or not self.model.is_available():
            return None
        try:
            prompt = template.replace("{request}", text)
            gen = self.model.generate(prompt, max_tokens=8, temperature=0.0)
        except ModelUnavailable:
            return None
        label = gen.text.strip().upper()
        for intent in (THINK, CODE, CHAT):
            if intent in label:
                return Routing(intent, 0.9, "model classification")
        return None

    # ---------------------------------------------------------------- route
    def route(self, text: str, router_template: str = "") -> Routing:
        heur = self._heuristic(text)
        # Only spend a model call when the heuristic is unsure.
        if heur.confidence < 0.7 and router_template:
            arb = self._model_arbitrate(text, router_template)
            if arb is not None:
                return arb
        return heur
