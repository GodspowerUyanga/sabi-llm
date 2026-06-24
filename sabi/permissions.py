"""Permission system for agent actions.

Implements an approval flow modelled on Claude Code / opencode:

  1. When the agent wants to take an action (create a folder, write a file,
     run a command), the user is shown exactly what will happen.
  2. They choose: Allow once / Allow always / Deny.
  3. If they chose "Allow always", the action category is trusted for the rest
     of the session and they are asked to Confirm / Cancel before it runs.
  4. "Allow once" runs the single action. "Deny" cancels it.

The prompter/confirmer callables are injected so the same logic works in the
interactive terminal, in a future web UI, and in tests (auto-approve).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Set


class Decision(str, Enum):
    ALLOW_ONCE = "once"
    ALLOW_ALWAYS = "always"
    DENY = "deny"


# A prompter takes (tool_name, action_description) and returns a Decision.
Prompter = Callable[[str, str], Decision]
# A confirmer takes (action_description) and returns True to proceed.
Confirmer = Callable[[str], bool]


class PermissionManager:
    def __init__(
        self,
        prompter: Optional[Prompter] = None,
        confirmer: Optional[Confirmer] = None,
        auto_approve: bool = False,
    ):
        self.prompter = prompter
        self.confirmer = confirmer
        self.auto_approve = auto_approve
        self.trusted: Set[str] = set()   # tools the user chose "Allow always"

    def is_trusted(self, tool_name: str) -> bool:
        return tool_name in self.trusted

    def request(self, tool_name: str, action_desc: str) -> bool:
        """Return True if the action should proceed."""
        if self.auto_approve:
            return True

        # Already trusted -> just confirm before running.
        if tool_name in self.trusted:
            return self._confirm(action_desc)

        decision = self._prompt(tool_name, action_desc)
        if decision == Decision.DENY:
            return False
        if decision == Decision.ALLOW_ONCE:
            return True
        if decision == Decision.ALLOW_ALWAYS:
            self.trusted.add(tool_name)
            return self._confirm(action_desc)
        return False

    # -- injected I/O (default: deny / proceed) --
    def _prompt(self, tool_name: str, action_desc: str) -> Decision:
        if self.prompter:
            return self.prompter(tool_name, action_desc)
        return Decision.DENY

    def _confirm(self, action_desc: str) -> bool:
        if self.confirmer:
            return self.confirmer(action_desc)
        return True
