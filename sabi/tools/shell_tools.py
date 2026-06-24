"""Shell execution tool.

Runs commands inside the workspace with a timeout and a conservative
deny-list. SABI is offline-first, so the deny-list also blocks obvious
network/destructive operations by default.
"""

from __future__ import annotations

import subprocess

from .base import Tool, ToolResult

_DENY = (
    "rm -rf /", ":(){", "mkfs", "dd if=", "shutdown", "reboot",
    "curl ", "wget ", "ssh ", "scp ", "nc ", "telnet ",
    "> /dev/sd", "chmod -r 777 /",
)


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command inside the workspace (sandboxed, with timeout)."

    def run(self, command: str, timeout: int = 60, **_) -> ToolResult:
        lowered = command.lower()
        for bad in _DENY:
            if bad in lowered:
                return ToolResult(False, error=f"blocked by safety policy: '{bad.strip()}'")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return ToolResult(
                proc.returncode == 0,
                output=out.strip(),
                error="" if proc.returncode == 0 else f"exit code {proc.returncode}",
                data={"returncode": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, error=f"command timed out after {timeout}s")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, error=str(exc))
