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

MAX_STEPS = 24  # multi-file project scaffolds (several write_file calls) plus a
# compile/run -> read error -> edit_file -> re-run debug loop need more room
# than a single search -> read -> edit -> verify chain (16 was tuned for that
# narrower case).

# Shell commands that are never allowed, even with approval.
_SHELL_DENY = (
    "rm -rf /", ":(){", "mkfs", "dd if=", "shutdown", "reboot",
    "> /dev/sd", "chmod -r 777 /", "mkfs.",
)

# Deletion is deliberately not a supported agent action — there is no
# delete_file/delete_dir tool by design, and this blocks the obvious
# workaround (routing a delete through run_shell) at the code level too, not
# just via the system prompt. Learned from a real incident: the model was
# once told via the prompt that it *could* delete through run_shell "if
# needed", and used that unprompted to delete a file it had just created
# after finishing an unrelated task. \b...\b keeps this from matching inside
# unrelated words ("confirm", "warm", "term").
_DELETE_CMD_RE = re.compile(r"\b(rm|rmdir|del|unlink|shred)\b", re.IGNORECASE)

# Directories skipped when walking a codebase for search/tree — build
# artifacts, VCS internals and dependency trees are noise and can be huge.
_WALK_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", "target", ".egg-info",
}
_MAX_SEARCH_FILE_BYTES = 1_000_000  # skip huge/binary-ish files when grepping


def _walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Hidden dirs (.git, .venv, vendored ".xyz" checkouts, .next, …) are
        # never what a codebase search means to hit, in any language.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and d not in _WALK_SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            yield Path(dirpath) / name


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def _similar_entry(parent: Path, name: str) -> Optional[str]:
    """If `name` doesn't exist under `parent` but something close does (wrong
    case, small typo — e.g. asked for "MSc Artificial Intelligence" when the
    real folder is "MSC ARTIFICIAL INTELLIGENCE"), return the real name.

    Without this, a model that gets a case/spelling detail wrong has no way
    to notice: create_dir would just silently make a second, empty,
    near-duplicate folder next to the one the user actually meant.
    """
    try:
        siblings = [c.name for c in parent.iterdir()]
    except Exception:
        return None
    for s in siblings:
        if s.lower() == name.lower() and s != name:
            return s
    import difflib
    close = difflib.get_close_matches(name, siblings, n=1, cutoff=0.75)
    return close[0] if close else None


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
            p = self._resolve(args.get("path", "")).as_posix()
            offset = args.get("offset")
            if offset:
                return f"read a file:  {p}  (from line {offset}, {args.get('limit', 200)} lines)"
            return f"read a file:  {p}"
        if tool == "list_dir":
            return f"list a directory:  {self._resolve(args.get('path', '.')).as_posix()}"
        if tool == "search_files":
            where = self._resolve(args.get("path", ".")).as_posix()
            return f"search for '{args.get('pattern', '')}' under  {where}"
        if tool == "edit_file":
            return f"edit a file:  {self._resolve(args.get('path', '')).as_posix()}"
        if tool == "move_file":
            src = self._resolve(args.get("src", "")).as_posix()
            dest = self._resolve(args.get("dest", "")).as_posix()
            return f"move  {src}  ->  {dest}"
        if tool == "run_shell":
            return f"run a shell command:  {args.get('command', '')}"
        return f"{tool}  {args}"

    def execute(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if tool == "create_dir":
                p = self._resolve(args["path"])
                if not p.exists() and p.parent.exists():
                    similar = _similar_entry(p.parent, p.name)
                    if similar:
                        return False, (f"Did not create '{p.name}': a folder named '{similar}' "
                                       f"already exists here ({p.parent.as_posix()}). If you meant "
                                       "that folder, use it directly instead of creating a new one.")
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
                    similar = _similar_entry(p.parent, p.name) if p.parent.exists() else None
                    hint = f" Did you mean '{similar}'?" if similar else ""
                    return False, f"File not found: {p.as_posix()}.{hint}"
                if args.get("offset"):
                    return self._read_file_range(p, args)
                from .filereader import read_any
                return True, read_any(p)
            if tool == "list_dir":
                p = self._resolve(args.get("path", "."))
                if not p.exists():
                    similar = _similar_entry(p.parent, p.name) if p.parent.exists() else None
                    hint = f" Did you mean '{similar}'?" if similar else ""
                    return False, f"Path not found: {p.as_posix()}.{hint}"
                items = sorted(("d " if c.is_dir() else "f ") + c.name for c in p.iterdir())
                return True, "\n".join(items) or "(empty)"
            if tool == "search_files":
                return self._search_files(args)
            if tool == "edit_file":
                return self._edit_file(args)
            if tool == "move_file":
                return self._move_file(args)
            if tool == "run_shell":
                cmd = args["command"]
                low = cmd.lower()
                for bad in _SHELL_DENY:
                    if bad in low:
                        return False, f"Blocked by safety policy: '{bad.strip()}'"
                if _DELETE_CMD_RE.search(cmd):
                    return False, ("Blocked by safety policy: deletion commands (rm/rmdir/del/"
                                    "unlink/shred) are not permitted through run_shell. If the "
                                    "user explicitly asked to delete something, tell them to do "
                                    "it themselves — do not work around this block.")
                proc = subprocess.run(cmd, shell=True, cwd=str(self.cwd),
                                      capture_output=True, text=True, timeout=120)
                out = ((proc.stdout or "") + (proc.stderr or "")).strip()
                return proc.returncode == 0, out or f"(exit {proc.returncode})"
            return False, f"Unknown tool: {tool}"
        except KeyError as exc:
            return False, f"Missing argument {exc} for tool {tool}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def _read_file_range(self, p: Path, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Read a line window of a text file — how a file bigger than the
        context budget gets read fully: one window at a time, driven by the
        line numbers search_files already returned, instead of one blind
        truncated dump of the whole file."""
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not read {p.as_posix()} as text: {exc}"
        offset = max(1, int(args.get("offset", 1) or 1))
        limit = max(1, min(int(args.get("limit", 200) or 200), 500))
        window = lines[offset - 1: offset - 1 + limit]
        if not window:
            return False, f"{p.as_posix()} has {len(lines)} lines — offset {offset} is past the end"
        body = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(window, start=offset))
        more = len(lines) - (offset - 1 + len(window))
        if more > 0:
            body += f"\n… {more} more line(s); call read_file again with offset={offset + limit} to continue"
        return True, body

    def _search_files(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Grep-like recursive text search across the codebase.

        Small models can't hold a whole repo in context, so this is how the
        agent finds *where* something lives before reading or editing it —
        the step that was missing (only single-file read/list existed).
        """
        pattern = args.get("pattern") or args.get("query")
        if not pattern:
            return False, "Missing argument 'pattern' for tool search_files"
        root = self._resolve(args.get("path", "."))
        if not root.exists():
            return False, f"Path not found: {root.as_posix()}"
        glob = args.get("glob")
        max_results = min(int(args.get("max_results", 50) or 50), 200)

        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))

        files = [root] if root.is_file() else _walk_files(root)
        matches: List[str] = []
        for f in files:
            if glob and not f.match(glob):
                continue
            try:
                if f.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    continue
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    try:
                        rel = f.relative_to(self.cwd).as_posix()
                    except ValueError:
                        rel = f.as_posix()
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        if not matches:
            return False, f"No matches for '{pattern}' under {root.as_posix()}"
        suffix = f"\n… ({max_results} result cap reached, narrow the pattern or glob)" \
            if len(matches) >= max_results else ""
        return True, "\n".join(matches) + suffix

    def _edit_file(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Targeted find/replace edit — the alternative to rewriting a whole
        file just to change one part of it (write_file overwrites everything)."""
        p = self._resolve(args["path"])
        if not p.exists():
            return False, f"File not found: {p.as_posix()}"
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        if not old:
            return False, "Missing argument 'old_string' for tool edit_file"
        text = p.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return False, "old_string not found in file — read the file first and copy the exact text"
        if count > 1 and not args.get("replace_all"):
            return False, (f"old_string is not unique ({count} occurrences) — include more "
                           "surrounding context, or pass replace_all=true")
        new_text = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        p.write_text(new_text, encoding="utf-8")
        n = count if args.get("replace_all") else 1
        return True, f"Edited {p.as_posix()} ({n} replacement{'s' if n != 1 else ''})"

    def _move_file(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Move or rename a file or directory — e.g. moving a file into a
        project folder, or renaming one, without a full read+write+delete
        round trip (and without needing a delete tool, which doesn't exist
        by design; this is not a delete, the source stops existing only
        because it now exists at dest)."""
        src = self._resolve(args.get("src") or args.get("path", ""))
        if not src.exists():
            similar = _similar_entry(src.parent, src.name) if src.parent.exists() else None
            hint = f" Did you mean '{similar}'?" if similar else ""
            return False, f"Source not found: {src.as_posix()}.{hint}"
        dest = self._resolve(args.get("dest", ""))
        if not dest.parts or str(args.get("dest", "")) == "":
            return False, "Missing argument 'dest' for tool move_file"
        if dest.exists() and dest.is_dir():
            dest = dest / src.name
        if dest.exists():
            return False, (f"Destination already exists: {dest.as_posix()} — move_file never "
                            "overwrites; choose a different name or move the existing file first.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        return True, f"Moved {src.as_posix()} -> {dest.as_posix()}"


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


def parse_final(text: str) -> Optional[str]:
    """Extract the answer text from a {"final": "..."} object, if present."""
    candidates: List[str] = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "final" in obj:
            return str(obj["final"])
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
        retriever: Optional[Any] = None,
        initial_history: Optional[List[dict]] = None,
    ):
        self.model = model
        self.permissions = permissions
        self.system_prompt = system_prompt or DEFAULT_AGENT_PROMPT
        self.executor = ToolExecutor(cwd)
        self.reporter = reporter or Reporter()
        self.max_steps = max_steps
        self.keep_history = keep_history
        # compact memory across turns. A caller that keeps this same
        # AgentLoop object alive across turns (the TUI, the terminal chat
        # loop) grows this via _remember() as it goes; a caller that
        # constructs a fresh AgentLoop per turn (sabi serve — one per HTTP
        # request) MUST pass the prior turns back in here, or every request
        # is amnesiac: "list them" right after "list my Desktop folders" has
        # no idea what "them" refers to, since there is no history to see.
        self.history: List[dict] = list(initial_history or [])[-8:]
        # Long-term recall beyond the rolling `history` window: each finished
        # turn is indexed here so a later, unrelated turn can still retrieve it
        # (see runtime.py — the same retriever backs a project-wide codebase
        # index too, so this is the one place "memory" actually lives).
        self.retriever = retriever

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
        note = ""
        if result.actions:
            note = "\n[done: " + "; ".join(result.actions[:6]) + "]"
        answer = (result.answer or "")[:600] + note
        self.history.append({"role": "user", "content": request[:500]})
        self.history.append({"role": "assistant", "content": answer})
        # keep the last 8 messages (4 turns) to stay within the context window
        self.history = self.history[-8:]
        # The rolling window above is what the model sees NEXT turn; the
        # retriever is what lets it recall THIS turn many turns later, once
        # it has scrolled out of that window.
        if self.retriever is not None:
            try:
                self.retriever.add_text(f"USER: {request}\nSABI: {answer}", source="conversation")
            except Exception:
                pass

    def run(self, request: str, context: str = "") -> AgentResult:
        messages = [{"role": "system", "content": self._system(context)}]
        messages += self.history
        messages.append({"role": "user", "content": request})
        result = AgentResult(ok=False)
        # Small models sometimes keep "double checking" a completed task
        # (re-reading/re-running the same file) instead of recognizing it's
        # done. Track exact-repeat calls and force termination rather than
        # burning the whole step budget on redundant verification.
        seen_calls: set = set()
        # The last successful read-only tool's raw output. Small models
        # sometimes fetch the data (list_dir/read_file/search_files all
        # succeed) and then fail to transcribe it into {"final": ...} —
        # instead repeating the same call or running out of steps. When that
        # happens we still have the real data right here, so the fallback
        # answers below show THAT instead of a content-free "I did a thing".
        last_info_output: Optional[str] = None
        _INFO_TOOLS = {"list_dir", "read_file", "search_files"}

        def _enrich_with_details(answer: str, raw_output: str) -> str:
            """If the model's {"final": ...} doesn't actually name the specific
            items from the last lookup (e.g. "There are 14 folders" with no
            names, confirmed live even when list_dir just returned all 14)),
            append the real result so the user gets the actual answer
            regardless of how well the model summarized it. A prompt rule
            asking for this exists too, but has proven unreliable on a small
            model — this makes it true by construction instead of by hoping."""
            lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
            if len(lines) < 2:
                return answer  # nothing meaningfully omittable from 0-1 lines
            def _name(l: str) -> str:
                return l[2:].strip() if l[:2] in ("d ", "f ") else l
            mentioned = sum(1 for l in lines if _name(l) and _name(l) in answer)
            if mentioned >= max(1, len(lines) // 2):
                return answer  # the model already named most of them
            return answer.rstrip() + "\n\n" + raw_output

        def _fallback_answer() -> str:
            if last_info_output is not None:
                return last_info_output
            return ("Here is what I completed:\n" + "\n".join(result.actions)
                    if result.actions else "Nothing further to do.")

        for step in range(self.max_steps):
            self.reporter.thinking()
            try:
                # json_mode forces the backend's JSON grammar so every reply is
                # syntactically valid JSON — either a tool call or {"final": ...}.
                # Low temperature keeps the choice of tool/args deterministic.
                # (A plain low-temperature nudge was NOT enough on its own: the
                # 3B model still narrated code as prose instead of emitting the
                # write_file call — confirmed by live testing 2026-08-13/14.
                # json_mode structurally rules that failure mode out.)
                gen = self.model.chat(messages, temperature=0.1, json_mode=True)
            except ModelUnavailable as exc:
                result.error = str(exc)
                return result

            result.tokens += gen.prompt_tokens + gen.completion_tokens
            result.elapsed_s += gen.elapsed_s
            result.steps_taken = step + 1

            call = parse_tool_call(gen.text)
            if not call:
                final = parse_final(gen.text)
                answer = final if final is not None else gen.text
                if last_info_output is not None:
                    answer = _enrich_with_details(answer, last_info_output)
                result.ok = True
                result.answer = answer
                self.reporter.final(answer)
                self._remember(request, result)
                return result

            tool, args = call["tool"], call["args"]
            args = self._maybe_locate(tool, args, request)
            desc = self.executor.describe(tool, args)

            signature = (tool, json.dumps(args, sort_keys=True))
            if signature in seen_calls:
                # Exact repeat of a call already made this run — the model is
                # looping instead of finishing. Stop here rather than spend
                # the rest of the step budget re-verifying the same thing.
                result.ok = True
                result.answer = _fallback_answer()
                self.reporter.final(result.answer)
                self._remember(request, result)
                return result
            seen_calls.add(signature)

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
            last_info_output = output if (ok and tool in _INFO_TOOLS) else None
            messages.append({"role": "user", "content":
                             f"TOOL RESULT ({'success' if ok else 'error'}):\n{output}\n"
                             "Call another tool if needed, or give your final answer in plain text."})

        result.ok = True
        result.answer = _fallback_answer()
        self._remember(request, result)
        return result


DEFAULT_AGENT_PROMPT = """You are SABI, an offline AI coding coworker. You CAN read, write and edit files in ANY programming language, create folders, run shell commands, and build whole projects on this machine. You are a capable agent, not just a chat bot.

EVERY reply is exactly one JSON object, nothing else: {"tool": "<name>", "args": {...}} to act, or {"final": "<answer, may include ```lang code```>"} to finish. Never bare prose or bare code.

Tools (use EXACTLY these names — there is no delete_file/delete_dir):
- create_dir(path) — new folder only. "go into/open/look inside" an EXISTING folder is list_dir, not create_dir.
- write_file(path, content) — create or fully overwrite a file with complete code.
- read_file(path, offset=None, limit=200) — read any file; use offset/limit for files longer than ~200 lines.
- list_dir(path) — list one folder level.
- search_files(pattern, path=".", glob=None) — regex-search file contents across a codebase. Use this first to find a file instead of guessing its path.
- edit_file(path, old_string, new_string, replace_all=false) — change part of an EXISTING file; old_string must be copied verbatim from a prior read/search and be unique. Prefer this over write_file for existing files (write_file erases everything else, and can truncate a large file mid-write).
- move_file(src, dest) — move/rename a file or folder into dest (or, if dest is a dir, into it by the same name). Never overwrites.
- run_shell(command) — run a shell command.

After a tool runs you get its result, then call another tool or reply {"final": ...}.

Hard rules, no exceptions:
1. No deletion, ever — not run_shell rm/rmdir/del, not overwriting something to empty, not even a file you just created — unless the user explicitly names that exact thing and asks for it. Finishing a task never implies cleaning up after it. (run_shell blocks rm-family commands outright regardless.)
2. Bare greeting/pleasantry ("hi", "thanks", "how are you") -> {"final": "..."} only, never a tool call.
3. One JSON object per reply. Stop with {"final": ...} the moment the task is done — no repeat lookups "to double-check", no extra steps nobody asked for.

Building a project: for "create/build a <lang> project", make a real structure, not one lone file (unless a single script was clearly wanted) — e.g. Python: project/main.py + requirements.txt + README.md; Java: package dirs matching the package declaration, one public class per file named after that class. Write complete, real code — no TODO stubs — unless a stub was asked for. After writing runnable code, run_shell it once to sanity-check it starts without a syntax/import error; if it needs stdin, pipe a value in rather than running it bare.

Paths: always ABSOLUTE. "~"/{home} is home. "on Desktop" -> {home}/Desktop/<name>. Reuse a path you created earlier in this conversation exactly. Only use {cwd} when the user gives no location.

Examples:
- "hi" -> {"final": "Hello! What would you like to work on?"}
- "create a folder app on my Desktop" -> {"tool": "create_dir", "args": {"path": "{home}/Desktop/app"}}
- "in the app folder you made, create main.py that prints hello" -> {"tool": "write_file", "args": {"path": "{home}/Desktop/app/main.py", "content": "print('hello')"}}
- "move config.py into the utils folder" -> {"tool": "move_file", "args": {"src": "{cwd}/config.py", "dest": "{cwd}/utils"}}
- "fix the bug where login fails" (no file named) -> {"tool": "search_files", "args": {"path": "{cwd}", "pattern": "login"}}
- after list_dir/read_file/search_files: your NEXT reply must be {"final": ...} stating the actual result (file names, matched lines, contents) by name — not a repeat lookup, and not "I checked it" without saying what was found.

Never say you can't access files — you can, via the tools above. Never state a file/folder's contents or existence unless a TOOL RESULT for that exact path already appeared in this conversation.

Current working directory: {cwd}
Home directory: {home}
"""
