"""SABI full-screen terminal interface.

A polished, distinct TUI (Textual) that:
  * answers greetings / questions instantly and STREAMS replies,
  * acts on the filesystem when asked (create/edit files, run commands, any language),
  * asks opencode-style permission for external folders / shell (Allow once / always),
  * shows a large, always-visible input box with a Send button,
  * shows live activity and tracks tokens, context %, timing, cwd + git branch.

Requires the optional `textual` dependency:  pip install "sabi-llm[tui]"
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widgets import Button, Input, Static
    from rich.markdown import Markdown
    from rich.text import Text
    _HAS_TEXTUAL = True
except Exception:  # pragma: no cover
    _HAS_TEXTUAL = False

from ..runtime import Runtime
from ..permissions import PermissionManager, Decision
from ..agent import Reporter, wants_action


def textual_available() -> bool:
    return _HAS_TEXTUAL


def _git_branch(cwd: Path) -> str:
    head = cwd / ".git" / "HEAD"
    try:
        txt = head.read_text(encoding="utf-8").strip()
        return txt.split("/")[-1] if txt.startswith("ref:") else txt[:7]
    except Exception:
        return ""


def _short(path: str, n: int = 24) -> str:
    base = Path(path).name or path
    return base if len(base) <= n else "…" + base[-(n - 1):]


_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


if _HAS_TEXTUAL:

    class PermissionScreen(ModalScreen):
        CSS = """
        PermissionScreen { align: center middle; }
        #pbox {
            width: 76; height: auto; padding: 1 2; background: #131c2b;
            border: round #e6b450;
        }
        #pbox .title { color: #e6b450; text-style: bold; }
        #pbox .desc { color: #e7edf5; margin: 1 0; }
        #prow { height: 3; margin-top: 1; }
        #prow Button { margin-right: 2; }
        """

        def __init__(self, desc: str):
            super().__init__()
            self.desc = desc

        def compose(self) -> ComposeResult:
            with Vertical(id="pbox"):
                yield Static("⚠ Permission required", classes="title")
                yield Static(self.desc, classes="desc")
                with Horizontal(id="prow"):
                    yield Button("Allow once", id="once", variant="warning")
                    yield Button("Allow always", id="always")
                    yield Button("Reject", id="reject", variant="error")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss({"once": Decision.ALLOW_ONCE,
                          "always": Decision.ALLOW_ALWAYS,
                          "reject": Decision.DENY}[event.button.id])

    class ConfirmScreen(ModalScreen):
        CSS = """
        ConfirmScreen { align: center middle; }
        #cbox { width: 70; height: auto; padding: 1 2; background: #131c2b; border: round #e6b450; }
        #cbox .title { color: #e6b450; text-style: bold; }
        #cbox .desc { color: #cdd7e3; margin: 1 0; }
        #crow { height: 3; margin-top: 1; }
        #crow Button { margin-right: 2; }
        """

        def __init__(self, desc: str):
            super().__init__()
            self.desc = desc

        def compose(self) -> ComposeResult:
            with Vertical(id="cbox"):
                yield Static("⚠ Always allow", classes="title")
                yield Static(f"This will allow {self.desc} for the rest of this session.",
                             classes="desc")
                with Horizontal(id="crow"):
                    yield Button("Confirm", id="confirm", variant="warning")
                    yield Button("Cancel", id="cancel")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "confirm")

    class TUIReporter(Reporter):
        def __init__(self, app: "SabiTUI"):
            self.app = app

        def thinking(self) -> None:
            self.app.call_from_thread(self.app.set_busy, "thinking")

        def proposing(self, tool: str, desc: str) -> None:
            verb = {"create_dir": "creating folder", "write_file": "writing file",
                    "read_file": "reading", "list_dir": "scanning",
                    "run_shell": "running"}.get(tool, "working on")
            target = desc.split(":", 1)[1].strip() if ":" in desc else desc
            label = f"{verb} {_short(target)}"
            self.app.call_from_thread(self.app.activity_add, label)
            self.app.call_from_thread(self.app.set_busy, label)

        def ran(self, ok: bool, output: str) -> None:
            self.app.call_from_thread(self.app.activity_done, ok)

        def denied(self, desc: str) -> None:
            self.app.call_from_thread(self.app.activity_done, False)

    class SabiTUI(App):
        CSS = """
        Screen { background: #0c1118; }
        #header { height: 1; background: #0c1118; color: #36d6c8; padding: 0 2; }
        #body { height: 1fr; }
        #chat { width: 3fr; padding: 1 2; background: #0c1118; }
        #side { width: 32; padding: 1 2; background: #0f1622; border-left: solid #1d2a3d; }
        .h { color: #e6b450; text-style: bold; margin-top: 1; }
        .t { color: #e7edf5; text-style: bold; }
        .dim { color: #7d8fa8; }
        .u { color: #e7edf5; margin: 1 0 0 0; }
        .s { border: round #1d2a3d; padding: 0 1; color: #e7edf5; }
        .m { color: #7d8fa8; margin: 0 0 1 0; }
        .act { border: round #e6b450; padding: 0 1; color: #cdd7e3; }
        #composer {
            height: 7; margin: 1 2 1 2; padding: 0 1;
            border: round #2a3a52; background: #0f1622;
        }
        #inputrow { height: 3; }
        #prompt { width: 1fr; height: 3; border: none; background: #0f1622; color: #e7edf5; }
        #prompt:focus { background: #122036; }
        #send {
            width: 14; min-width: 14; height: 3; margin-left: 1;
            background: #167f78; color: #ffffff;
        }
        #send:hover { background: #1aa39a; }
        #cstatus { height: 1; color: #7d8fa8; }
        """

        BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+q", "quit", "Quit")]

        def __init__(self, runtime: Runtime, cwd: Optional[str] = None):
            super().__init__()
            self.rt = runtime
            self.cwd = Path(cwd or os.getcwd())
            self.model_label = runtime.config.abs_model_path().stem or "offline model"
            self.total_tokens = 0
            self.ctx = runtime.config.context_length
            self.activity: List[Tuple[str, str]] = []
            self._busy = False
            self._spin_i = 0
            self._busy_label = ""
            # Permission manager that prompts (opencode-style) for external paths / shell.
            self.perms = PermissionManager(prompter=self._prompt_permission,
                                           confirmer=self._confirm_permission,
                                           auto_approve=False, prompt_all=False)
            self.agent = runtime.make_agent(permissions=self.perms,
                                            reporter=TUIReporter(self), cwd=str(self.cwd))

        # ---------------------------------------------------------- layout
        def compose(self) -> ComposeResult:
            ready = bool(self.rt.model and self.rt.model.is_available())
            right = "online" if ready else "model not loaded"
            yield Static(Text.from_markup(
                f"[b #36d6c8]◆ SABI[/]  [#7d8fa8]the offline AI coding coworker[/]"
                f"      [#7d8fa8]{self.model_label} · {right}[/]"), id="header")
            with Horizontal(id="body"):
                yield VerticalScroll(id="chat")
                with Vertical(id="side"):
                    yield Static("Session", classes="t")
                    yield Static("Context", classes="h")
                    yield Static("", id="ctx", classes="dim")
                    yield Static("Activity", classes="h")
                    yield Static("ready", id="act", classes="dim")
                    yield Static("", id="loc", classes="dim")
            with Vertical(id="composer"):
                with Horizontal(id="inputrow"):
                    yield Input(placeholder='Message SABI…  e.g. "create a folder app and a main.py"',
                                id="prompt")
                    yield Button("Send  ⏎", id="send", variant="primary")
                yield Static("", id="cstatus")

        def on_mount(self) -> None:
            self.query_one("#prompt", Input).focus()
            ready = bool(self.rt.model and self.rt.model.is_available())
            self._mount_sabi(
                "**Welcome to SABI** — your offline AI coding coworker.\n\n"
                "I can chat, plan, and **write code, create/edit files and folders, "
                "and run commands** in any language. I'll ask permission before "
                "touching anything outside this project.\n"
                + ("" if ready else "\n*No model loaded — run `sabi download` first.*"))
            self._refresh_side()
            self._set_status("ready")
            self.set_interval(0.1, self._tick)

        # ---------------------------------------------------------- permission I/O
        def _prompt_permission(self, key: str, desc: str) -> Decision:
            ev = threading.Event(); box = {}

            def show():
                def cb(result):
                    box["d"] = result; ev.set()
                self.push_screen(PermissionScreen(desc), cb)
            self.call_from_thread(show)
            ev.wait(timeout=300)
            return box.get("d", Decision.DENY)

        def _confirm_permission(self, desc: str) -> bool:
            ev = threading.Event(); box = {}

            def show():
                def cb(result):
                    box["d"] = result; ev.set()
                self.push_screen(ConfirmScreen(desc), cb)
            self.call_from_thread(show)
            ev.wait(timeout=300)
            return bool(box.get("d", False))

        # ---------------------------------------------------------- status
        def _tick(self) -> None:
            if not self._busy:
                return
            self._spin_i = (self._spin_i + 1) % len(_SPIN)
            self._set_status(f"{_SPIN[self._spin_i]} SABI is {self._busy_label}…")

        def set_busy(self, label: str) -> None:
            self._busy = True; self._busy_label = label

        def _set_status(self, msg: str) -> None:
            self.query_one("#cstatus", Static).update(
                Text.from_markup(f"[#36d6c8]Agent[/] · {self.model_label} · {msg}"))

        def _refresh_side(self) -> None:
            pct = (self.total_tokens / self.ctx * 100) if self.ctx else 0
            self.query_one("#ctx", Static).update(Text.from_markup(
                f"{self.total_tokens:,} tokens\n{pct:.0f}% of {self.ctx:,}\n$0.00 (offline)"))
            loc = str(self.cwd); home = str(Path.home())
            if loc.startswith(home):
                loc = "~" + loc[len(home):]
            branch = _git_branch(self.cwd)
            self.query_one("#loc", Static).update(Text.from_markup(
                f"\n[#7d8fa8]{loc}[/]" + (f"\n[#36d6c8]⎇ {branch}[/]" if branch else "")))

        # ---------------------------------------------------------- activity
        def activity_reset(self) -> None:
            self.activity = []; self._render_activity()

        def activity_add(self, label: str) -> None:
            self.activity.append(("doing", label)); self._render_activity()

        def activity_done(self, ok: bool) -> None:
            for i in range(len(self.activity) - 1, -1, -1):
                if self.activity[i][0] == "doing":
                    self.activity[i] = ("done" if ok else "fail", self.activity[i][1]); break
            self._render_activity()

        def _render_activity(self) -> None:
            if not self.activity:
                self.query_one("#act", Static).update("ready"); return
            out = []
            for state, label in self.activity[-8:]:
                if state == "done":
                    out.append(f"[#4ec98a]✓ {label}[/]")
                elif state == "fail":
                    out.append(f"[#e0584e]✗ {label}[/]")
                else:
                    out.append(f"[#e6b450]• {label}[/]")
            self.query_one("#act", Static).update(Text.from_markup("\n".join(out)))

        # ---------------------------------------------------------- chat io
        def _scroll(self) -> None:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

        def _mount_user(self, text: str) -> None:
            self.query_one("#chat", VerticalScroll).mount(
                Static(Text.from_markup(f"[b #36d6c8]›[/] {text}"), classes="u")); self._scroll()

        def _mount_sabi(self, text: str) -> Static:
            w = Static(Markdown(text, code_theme="monokai"), classes="s")
            self.query_one("#chat", VerticalScroll).mount(w); self._scroll()
            return w

        def _mount_actions(self, actions: List[str]) -> None:
            if not actions:
                return
            self.query_one("#chat", VerticalScroll).mount(
                Static(Text("\n".join(actions)), classes="act")); self._scroll()

        def _mount_meta(self, meta: str) -> None:
            self.query_one("#chat", VerticalScroll).mount(
                Static(Text.from_markup(f"[#7d8fa8]{meta}[/]"), classes="m")); self._scroll()

        def _update_stream(self, widget: Static, text: str, done: bool) -> None:
            widget.update(Markdown(text, code_theme="monokai") if done else Text(text)); self._scroll()

        # ---------------------------------------------------------- submit
        def _submit(self, text: str) -> None:
            text = text.strip()
            if not text:
                return
            self.query_one("#prompt", Input).value = ""
            if text.lower() in ("/exit", "/quit", ":q"):
                self.exit(); return
            self._mount_user(text)
            self.set_busy("thinking")
            self.process(text)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self._submit(event.value)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "send":
                self._submit(self.query_one("#prompt", Input).value)

        # ---------------------------------------------------------- worker
        @work(thread=True, exclusive=True)
        def process(self, message: str) -> None:
            if wants_action(message):
                self._run_action(message)
            else:
                self._run_chat(message)
            self._busy = False
            self.call_from_thread(self._refresh_side)
            self.call_from_thread(self._set_status, "ready")

        def _run_chat(self, message: str) -> None:
            bubble = self.call_from_thread(self._mount_sabi, "● thinking…")
            system = self.rt.prompts.get("system", "") or None
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})
            t0 = time.perf_counter(); buf = ""
            try:
                streamed = False
                for delta in self.rt.model.chat_stream(messages):
                    streamed = True; buf += delta
                    self.call_from_thread(self._update_stream, bubble, buf, False)
                if not streamed:
                    buf = self.rt.model.generate(message, system=system).text
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._update_stream, bubble, f"⚠ {exc}", False); return
            elapsed = time.perf_counter() - t0
            self.call_from_thread(self._update_stream, bubble, buf, True)
            try:
                toks = self.rt.model.count_tokens(buf)
            except Exception:
                toks = max(1, len(buf) // 4)
            self.total_tokens += toks
            self.call_from_thread(self._mount_meta, f"CHAT · {toks} tokens · {elapsed:.1f}s")
            try:
                self.rt.memory.add_turn("user", message, "CHAT")
                self.rt.memory.add_turn("assistant", buf, "CHAT")
            except Exception:
                pass

        def _run_action(self, message: str) -> None:
            self.call_from_thread(self.activity_reset)
            try:
                context = self.rt.retriever.context(message)
                res = self.agent.run(message, context=context)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._mount_sabi, f"⚠ {exc}"); return
            if res.error:
                self.call_from_thread(self._mount_sabi, f"⚠ {res.error}"); return
            self.total_tokens += res.tokens
            self.call_from_thread(self._mount_actions, res.actions)
            self.call_from_thread(self._mount_sabi, res.answer or "Done.")
            self.call_from_thread(self._mount_meta,
                                  f"{res.tokens:,} tokens · {res.elapsed_s:.1f}s · {res.steps_taken} step(s)")


def run_tui(runtime: Runtime, cwd: Optional[str] = None) -> None:
    if not _HAS_TEXTUAL:
        raise RuntimeError("The TUI needs the 'textual' package. Install it with:\n"
                           '    pip install "sabi-llm[tui]"')
    runtime.start()
    SabiTUI(runtime, cwd=cwd).run()
