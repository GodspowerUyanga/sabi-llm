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

    def is_external(self, path: str) -> bool:
        """True if the resolved path is outside the working directory."""
        try:
            r = self._resolve(path)
        except Exception:
            return True
        return self.cwd != r and self.cwd not in r.parents

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
                from .filereader import read_any
                return True, read_any(p, max_chars=4000)
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

# Phrases that signal the user actually wants a filesystem / shell ACTION.
# Anything that doesn't match these is treated as conversation (greet, explain,
# plan, show code) and never touches the filesystem.
_ACTION_PATTERNS = (
    "create a folder", "create folder", "create a file", "create file",
    "create a directory", "create a project", "create an app", "create a script",
    "make a folder", "make folder", "make a file", "make a directory",
    "make a project", "make an app", "build a project", "build an app",
    "build a website", "build an api", "build me", "new folder", "new file",
    "scaffold", "set up a project", "setup a project", "generate a file",
    "generate the", "save it to", "save to", "save as", "save this",
    "write a file", "write to a file", "write to file", "write the code",
    "write code to a", "append to", "delete", "remove the", "remove file", "rm ",
    "run the", "run a command", "run this", "run it", "run my", "execute ",
    "mkdir", "touch ", "list the files", "list files", "show me the files",
    "show files", "scan the", "scan this", "scan my", "scan it", "read the file",
    "read file", "open the file", "fix the", "fix my", "fix this", "refactor",
    "edit the", "edit my", "modify the", "update the", "rename", "move the file",
    "install ", "npm ", "pip install", "implement the", "implement a",
    "add a file", "add a function to", "complete the code", "complete this",
)


_DOC_EXTS = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".tsv", ".pptx",
             ".ppt", ".txt", ".md", ".json", ".html", ".htm", ".py", ".js",
             ".java", ".png", ".jpg", ".jpeg")


def wants_action(text: str) -> bool:
    """True if the message asks SABI to act on files / run commands.

    Greetings and bare 'write a function …' (code as text) return False, so they
    are answered conversationally. Anything naming a real file, or asking to
    read / open / summarize a document, routes to the acting agent.
    """
    t = " " + text.lower().strip() + " "
    if any(p in t for p in _ACTION_PATTERNS):
        return True
    # any mention of a real document/file extension -> act on it
    if any((ext + " ") in t or t.rstrip().endswith(ext) for ext in _DOC_EXTS):
        return True
    # summarize / explain / read a file or document
    if any(v in t for v in ("summarize", "summarise", "read ", "open ")) and \
       any(g in t for g in ("file", "document", "doc ", "pdf", "spreadsheet", "sheet")):
        return True
    # navigation / inspection: "go into X folder", "open the X folder", etc.
    nav = ("go into", "go to the", "open the", "open ", "navigate", "cd into",
           "look inside", "look in the", "explore", "inspect", "what's in",
           "what is in", "what is inside", "tell me what", "go inside")
    targets = ("folder", "directory", " dir ", "repo", "project", "file")
    if any(n in t for n in nav) and any(g in t for g in targets):
        return True
    return False


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
    tokens: int = 0
    elapsed_s: float = 0.0
    steps_taken: int = 0


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

    def _perm_key(self, tool: str, args: Dict[str, Any]) -> str:
        if tool == "run_shell":
            return "shell"
        p = args.get("path")
        if p:
            r = self.executor._resolve(p)
            return str(r if tool == "create_dir" else r.parent)
        return "action"

# --------------------------------------------------------------------- loop
_LOCATION_WORDS = ("desktop", "documents", "downloads", "home directory", "home folder")


class AgentLoop:
    def __init__(
        self,
        model: LLMModel,
        permissions: PermissionManager,
        system_prompt: str = "",
        cwd: Optional[Path] = None,
        reporter: Optional[Reporter] = None,
        max_steps: int = MAX_STEPS,
        keep_history: bool = True,
    ):
        self.model = model
        self.permissions = permissions
        self.system_prompt = system_prompt or DEFAULT_AGENT_PROMPT
        self.executor = ToolExecutor(cwd)
        self.reporter = reporter or Reporter()
        self.max_steps = max_steps
        self.keep_history = keep_history
        self.history: List[dict] = []   # compact memory across turns

    # -- known locations so the model/agent place things correctly --
    def _locations(self) -> Dict[str, Path]:
        home = Path.home()
        locs = {"home": home}
        for name in ("Desktop", "Documents", "Downloads"):
            p = home / name
            if p.exists():
                locs[name.lower()] = p
        return locs

    def _system(self, context: str) -> str:
        home = Path.home()
        prompt = (self.system_prompt
                  .replace("{cwd}", str(self.executor.cwd))
                  .replace("{home}", str(home)))
        locs = self._locations()
        loc_lines = "\n".join(f"  {k}: {v}" for k, v in locs.items())
        prompt += ("\n\nKnown locations on this machine:\n" + loc_lines +
                   "\nWhen the user names a location like 'on Desktop', build the "
                   "absolute path from the list above (e.g. " +
                   f"{locs.get('desktop', home / 'Desktop')}/<name>). Always pass an "
                   "ABSOLUTE path. If no location is given, use the working directory.")
        if context:
            prompt += f"\n\nRelevant context:\n{context}"
        return prompt

    def _maybe_locate(self, tool: str, args: Dict[str, Any], request: str) -> Dict[str, Any]:
        """Safety net: if the user named a location (e.g. Desktop) but the model
        produced a bare relative name, place it under that location."""
        if tool not in ("create_dir", "write_file"):
            return args
        p = str(args.get("path", ""))
        if not p or p.startswith("/") or p.startswith("~") or "/" in p or "\\" in p:
            return args
        low = request.lower()
        locs = self._locations()
        for word in ("desktop", "documents", "downloads"):
            if word in low and word in locs:
                args = dict(args)
                args["path"] = str(locs[word] / p)
                break
        return args

    def _perm_key(self, tool: str, args: Dict[str, Any]) -> str:
        if tool == "run_shell":
            return "shell"
        p = args.get("path")
        if p:
            r = self.executor._resolve(p)
            return str(r if tool == "create_dir" else r.parent)
        return "action"

    def _remember(self, request: str, result: "AgentResult") -> None:
        if not self.keep_history:
            return
        self.history.append({"role": "user", "content": request[:500]})
        note = ""
        if result.actions:
            note = "\n[done: " + "; ".join(result.actions[:6]) + "]"
        self.history.append({"role": "assistant", "content": (result.answer or "")[:600] + note})
        # keep the last 8 messages (4 turns) to stay within the context window
        self.history = self.history[-8:]

    def run(self, request: str, context: str = "") -> AgentResult:
        messages = [{"role": "system", "content": self._system(context)}]
        messages += self.history
        messages.append({"role": "user", "content": request})
        result = AgentResult(ok=False)

        for step in range(self.max_steps):
            self.reporter.thinking()
            try:
                gen = self.model.chat(messages)
            except ModelUnavailable as exc:
                result.error = str(exc)
                return result

            result.tokens += gen.prompt_tokens + gen.completion_tokens
            result.elapsed_s += gen.elapsed_s
            result.steps_taken = step + 1

            call = parse_tool_call(gen.text)
            if not call:
                result.ok = True
                result.answer = gen.text
                self.reporter.final(gen.text)
                self._remember(request, result)
                return result

            tool, args = call["tool"], call["args"]
            args = self._maybe_locate(tool, args, request)
            desc = self.executor.describe(tool, args)
            self.reporter.proposing(tool, desc)

            messages.append({"role": "assistant", "content": gen.text})

            path = args.get("path")
            external = self.executor.is_external(path) if path else False
            is_shell = tool == "run_shell"
            allowed = True
            if self.permissions.should_prompt(external, is_shell):
                allowed = self.permissions.request(self._perm_key(tool, args), desc)

            if not allowed:
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
        result.answer = "Here is what I completed:\n" + "\n".join(result.actions)
        self._remember(request, result)
        return result


DEFAULT_AGENT_PROMPT = """You are SABI, an offline AI coding coworker. You CAN read, write and edit files \
in ANY programming language, create folders, run shell commands, and build whole \
projects on this machine. You are a capable agent, not just a chat bot.

Available tools (to use one, reply with ONLY a single JSON object, nothing else):
- create_dir(path)            create a folder
- write_file(path, content)   create or overwrite a file (write complete, runnable code)
- read_file(path)             read ANY file (PDF, Word, Excel, PowerPoint, CSV, HTML, JSON, images, code, text) and get its text
- list_dir(path)              list a folder
- run_shell(command)          run a shell command

After a tool runs you receive its result, then call another tool or finish with a \
short plain-text summary (no JSON).

Path rules (IMPORTANT):
- Always pass an ABSOLUTE path. "~" or {home} is the home directory.
- If the user names a location, build the absolute path: "on Desktop" -> {home}/Desktop/<name>.
- If the user refers to something you created earlier in this conversation, reuse \
that exact absolute path (you remember what you created).
- Only use the working directory ({cwd}) when the user gives no location.

Examples:
- "create a folder app on my Desktop"
  {"tool": "create_dir", "args": {"path": "{home}/Desktop/app"}}
- "in the app folder you made, create main.py that prints hello"
  {"tool": "write_file", "args": {"path": "{home}/Desktop/app/main.py", "content": "print('hello')"}}
- "what's in my Documents folder?"
  {"tool": "list_dir", "args": {"path": "{home}/Documents"}}

Rules:
- For greetings, questions, or explanations, reply in plain text — do NOT call a tool.
- NEVER say you cannot access files or folders. You CAN, via the tools above.
- When asked to create / edit / read / open / go into / build something, DO it with tools.
- Write complete, correct, runnable code. In prose, wrap code in fenced blocks with the \
language, e.g. ```python ... ``` so it is highlighted.
- One tool per reply. Keep going until the task is done.

Current working directory: {cwd}
Home directory: {home}
"""
