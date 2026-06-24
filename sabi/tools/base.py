"""Base abstractions for SABI tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class Tool:
    """A local capability the agent can invoke.

    Subclasses set ``name`` / ``description`` and implement :meth:`run`.
    The ``workspace`` is the sandbox root; tools must not escape it.
    """

    name: str = "tool"
    description: str = ""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def _safe_path(self, relative: str) -> Path:
        """Resolve ``relative`` inside the workspace, blocking traversal."""
        target = (self.workspace / relative).resolve()
        if self.workspace not in target.parents and target != self.workspace:
            raise ValueError(f"path escapes workspace sandbox: {relative}")
        return target

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self):
        return list(self._tools.keys())

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
