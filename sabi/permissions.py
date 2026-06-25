"""Permission system for agent actions (opencode-style).

When the agent wants to touch something outside the current project, or run a
shell command, the user is asked:

    Allow once  /  Allow always  /  Reject

Choosing "Allow always" confirms once and then *sticks* for the rest of the
session (keyed by directory or by "shell"), so the user is not asked again — the
same behaviour as opencode / Claude Code.

The prompter/confirmer callables are injected, so the same logic drives the
terminal modal, the simple REPL, and tests.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Set


class Decision(str, Enum):
    ALLOW_ONCE = "once"
    ALLOW_ALWAYS = "always"
    DENY = "deny"


Prompter = Callable[[str, str], Decision]   # (key, description) -> Decision
Confirmer = Callable[[str], bool]            # (description) -> proceed?


class PermissionManager:
    def __init__(
        self,
        prompter: Optional[Prompter] = None,
        confirmer: Optional[Confirmer] = None,
        auto_approve: bool = False,
        prompt_all: bool = False,
    ):
        self.prompter = prompter
        self.confirmer = confirmer
        self.auto_approve = auto_approve
        # prompt_all=True  -> ask for every action (the simple REPL).
        # prompt_all=False -> ask only for external paths / shell (the TUI).
        self.prompt_all = prompt_all
        self.trusted: Set[str] = set()

    def should_prompt(self, external: bool, is_shell: bool) -> bool:
        if self.auto_approve:
            return False
        if self.prompt_all:
            return True
        return external or is_shell

    def is_trusted(self, key: str) -> bool:
        return key in self.trusted

    def request(self, key: str, action_desc: str) -> bool:
        """Return True if the action may proceed."""
        if self.auto_approve:
            return True
        if key in self.trusted:
            return True
        decision = self._prompt(key, action_desc)
        if decision == Decision.DENY:
            return False
        if decision == Decision.ALLOW_ONCE:
            return True
        if decision == Decision.ALLOW_ALWAYS:
            if self._confirm(action_desc):
                self.trusted.add(key)   # sticks for the rest of the session
                return True
            return False
        return False

    # injected I/O (defaults: deny / proceed)
    def _prompt(self, key: str, action_desc: str) -> Decision:
        return self.prompter(key, action_desc) if self.prompter else Decision.DENY

    def _confirm(self, action_desc: str) -> bool:
        return self.confirmer(action_desc) if self.confirmer else True
