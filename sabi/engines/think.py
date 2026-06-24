"""SABI THINK - business reasoning and planning engine.

Handles business analysis, PRD/SOP generation, requirements gathering,
planning, and architecture design. Converts a raw idea into a structured,
actionable specification.
"""

from __future__ import annotations

from typing import Optional

from ..model import LLMModel, Generation, ModelUnavailable


class ThinkEngine:
    def __init__(self, model: LLMModel, system_prompt: str = "", think_prompt: str = ""):
        self.model = model
        self.system_prompt = system_prompt
        self.think_prompt = think_prompt

    def run(self, request: str, context: str = "") -> Generation:
        template = self.think_prompt or (
            "You are SABI THINK, a senior business analyst and solutions architect. "
            "Produce a clear, structured plan for the user's request. Use concise "
            "sections (Goal, Requirements, Plan, Risks, Next steps).\n\n"
            "{context}\nRequest: {request}\n"
        )
        prompt = template.replace("{request}", request).replace(
            "{context}", f"Relevant context:\n{context}\n" if context else ""
        )
        return self.model.generate(prompt, system=self.system_prompt or None)
