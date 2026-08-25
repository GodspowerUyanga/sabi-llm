"""SABI CODE - software generation and debugging engine.

Handles code generation, debugging, project scaffolding, automation scripts,
file generation and validation. Turns the plan produced by THINK into working
software.
"""

from __future__ import annotations

from ..model import LLMModel, Generation


class CodeEngine:
    def __init__(self, model: LLMModel, system_prompt: str = "", code_prompt: str = ""):
        self.model = model
        self.system_prompt = system_prompt
        self.code_prompt = code_prompt

    def _messages(self, request: str, context: str = "", plan: str = "") -> list:
        template = self.code_prompt or (
            "You are SABI CODE, an expert software engineer. Write correct, "
            "runnable code for the user's request. Prefer the simplest working "
            "solution. Return code in fenced blocks and explain briefly.\n\n"
            "{plan}{context}Request: {request}\n"
        )
        prompt = (
            template
            .replace("{request}", request)
            .replace("{plan}", f"Plan to follow:\n{plan}\n\n" if plan else "")
            .replace("{context}", f"Relevant context:\n{context}\n\n" if context else "")
        )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def run(self, request: str, context: str = "", plan: str = "") -> Generation:
        return self.model.chat(self._messages(request, context, plan))

    def stream(self, request: str, context: str = "", plan: str = ""):
        """Yield text deltas as they're generated, for a live, fast feel."""
        yield from self.model.chat_stream(self._messages(request, context, plan))
