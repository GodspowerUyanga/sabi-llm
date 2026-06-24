"""Agentic tool-calling loop.

This is what makes SABI *act* instead of only printing code. The model is given
a small set of real tools and a JSON protocol. On each turn it either:

  * emits a JSON tool call  -> SABI asks permission, runs it, feeds back the
    result, and loops; or
  * replies in plain prose  -> that is the final answer and the loop stops.

Every action passes through the PermissionManager (Allow once / Allow always ->
Confirm / Cancel) and is announced through a Reporter ("SABI is thinking...",
"SABI wants to create a directory..."). Tools operate on the real filesystem
relative to the working directory SABI was launched in, with ~ expansion -- so
"create a folder on my Desktop" really creates it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model import LLMModel, ModelUnavailable
from .permissions import PermissionManager

MAX_STEPS = 8

# Shell commands that are never allowed, even with approval.
_SHELL_DENY = (
    "rm -rf /", ":(){", "mkfs", "dd if=", "shutdown", "reboot",
    "> /dev/sd", "chmod -r 777 /", "mkfs.",
)


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


# --------------------------------------------------------------------- tools
class ToolExecutor:
    """Runs the agent's actions on the real filesystem (relative to ``cwd``)."""

    def __init__(self, cwd: Optional[Path] = None):
        self.cwd = Path(cwd or os.getcwd())

    def _resolve(self, path: str) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(str(path))))
        return p if p.is_absolute() else (self.cwd / p)

    def describe(self, tool: str, args: Dict[str, Any]) -> str:
        if tool == "create_dir":
            return f"create a directory:  {self._resolve(args.get('path', '')).as_posix()}"
        if tool == "write_file":
            p = self._resolve(args.get("path", ""))
            n = len(args.get("content", "") or "")
            return f"write a file:  {p.as_posix()}  ({n} chars)"
        if tool == "read_file":
            return f"read a file:  {self._resolve(args.get('path', '')).as_posix()}"
        if tool == "list_dir":
            return f"list a directory:  {self._resolve(args.get('path', '.')).as_posix()}"
        if tool == "run_shell":
            return f"run a shell command:  {args.get('command', '')}"
        return f"{tool}  {args}"

    def execute(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if tool == "create_dir":
                p = self._resolve(args["path"])
                p.mkdir(parents=True, exist_ok=True)
                return True, f"Created directory {p.as_posix()}"
            if tool == "write_file":
                p = self._resolve(args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args.get("content", ""), encoding="utf-8")
                return True, f"Wrote {len(args.get('content', ''))} chars to {p.as_posix()}"
            if tool == "read_file":
                p = self._resolve(args["path"])
                if not p.exists():
                    return False, f"File not found: {p.as_posix()}"
                return True, p.read_text(encoding="utf-8", errors="replace")[:4000]
            if tool == "list_dir":
                p = self._resolve(args.get("path", "."))
                if not p.exists():
                    return False, f"Path not found: {p.as_posix()}"
                items = sorted(("d " if c.is_dir() else "f ") + c.name for c in p.iterdir())
                return True, "\n".join(items) or "(empty)"
            if tool == "run_shell":
                cmd = args["command"]
                low = cmd.lower()
                for bad in _SHELL_DENY:
                    if bad in low:
                        return False, f"Blocked by safety policy: '{bad.strip()}'"
                proc = subprocess.run(cmd, shell=True, cwd=str(self.cwd),
                                      capture_output=True, text=True, timeout=120)
                out = ((proc.stdout or "") + (proc.stderr or "")).strip()
                return proc.returncode == 0, out or f"(exit {proc.returncode})"
            return False, f"Unknown tool: {tool}"
        except KeyError as exc:
            return False, f"Missing argument {exc} for tool {tool}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------- reporter
class Reporter:
    """Status callbacks. Default is silent; the UI injects a chatty version."""

    def thinking(self) -> None: ...
    def proposing(self, tool: str, desc: str) -> None: ...
    def ran(self, ok: bool, output: str) -> None: ...
    def denied(self, desc: str) -> None: ...
    def final(self, text: str) -> None: ...


# ------------------------------------------------------------------- parser
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a {"tool": ..., "args": {...}} object from the model's reply.

    Returns None if the reply is plain prose (i.e. a final answer).
    """
    candidates: List[str] = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    # also scan for the first balanced { ... } block
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
        break

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("args", {})
            if isinstance(obj["args"], dict):
                return {"tool": str(obj["tool"]), "args": obj["args"]}
    return None


# -------------------------------------------------------------------- result
@dataclass
class AgentResult:
    ok: bool
    answer: str = ""
    actions: List[str] = field(default_factory=list)
    error: str = ""


# --------------------------------------------------------------------- loop
class AgentLoop:
    def __init__(
        self,
        model: LLMModel,
        permissions: PermissionManager,
        system_prompt: str = "",
        cwd: Optional[Path] = None,
        reporter: Optional[Reporter] = None,
        max_steps: int = MAX_STEPS,
    ):
        self.model = model
        self.permissions = permissions
        self.system_prompt = system_prompt or DEFAULT_AGENT_PROMPT
        self.executor = ToolExecutor(cwd)
        self.reporter = reporter or Reporter()
        self.max_steps = max_steps

    def run(self, request: str, context: str = "") -> AgentResult:
        sys_prompt = self.system_prompt.replace("{cwd}", str(self.executor.cwd))
        if context:
            sys_prompt += f"\n\nRelevant context:\n{context}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": request},
        ]
        result = AgentResult(ok=False)

        for _ in range(self.max_steps):
            self.reporter.thinking()
            try:
                gen = self.model.chat(messages)
            except ModelUnavailable as exc:
                result.error = str(exc)
                return result

            call = parse_tool_call(gen.text)
            if not call:
                result.ok = True
                result.answer = gen.text
                self.reporter.final(gen.text)
                return result

            tool, args = call["tool"], call["args"]
            desc = self.executor.describe(tool, args)
            self.reporter.proposing(tool, desc)

            messages.append({"role": "assistant", "content": gen.text})

            if not self.permissions.request(tool, desc):
                self.reporter.denied(desc)
                result.actions.append(f"DENIED: {desc}")
                messages.append({"role": "user", "content":
                                 f"TOOL RESULT: The user denied permission to {desc}. "
                                 "Do not retry; either continue without it or finish."})
                continue

            ok, output = self.executor.execute(tool, args)
            self.reporter.ran(ok, output)
            result.actions.append(("OK: " if ok else "FAIL: ") + desc)
            messages.append({"role": "user", "content":
                             f"TOOL RESULT ({'success' if ok else 'error'}):\n{output}\n"
                             "Call another tool if needed, or give your final answer in plain text."})

        result.ok = True
        result.answer = "Reached the step limit. Here is what I completed:\n" + \
                        "\n".join(result.actions)
        return result


DEFAULT_AGENT_PROMPT = """You are SABI, an offline AI coworker that can take REAL actions on the user's \
computer by calling tools. You are not just a chat assistant -- you can create \
folders and files and run commands.

Available tools:
- create_dir(path)            create a folder
- write_file(path, content)   create or overwrite a file
- read_file(path)             read a file
- list_dir(path)              list a folder
- run_shell(command)          run a shell command

To use a tool, reply with ONLY a single JSON object and nothing else:
{"tool": "create_dir", "args": {"path": "~/Desktop/appfolder"}}

After the tool runs you will receive its result, then you may call another tool \
or finish. When the task is done (or no action is needed), reply in PLAIN TEXT \
with a short confirmation -- no JSON.

Rules:
- To actually create something, you MUST call a tool. Never say you cannot access \
the filesystem, and never just print shell commands or code for the user to run \
themselves -- call the tool instead.
- Use the exact path the user asks for. "~" is the user's home directory.
- Do one tool call per reply. Keep going until the task is complete.

Current working directory: {cwd}
"""
