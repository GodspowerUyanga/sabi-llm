"""SABI interactive interface.

An app-like terminal session (in the spirit of Claude Code / opencode). Every
message runs through the agentic tool loop, so SABI can actually create folders
and files and run commands -- always after showing what it will do and asking
permission.

Status the user sees:
  * "SABI is thinking..."        while the model works
  * "SABI wants to <action>"     when an action is proposed
  * Allow once / Allow always    permission choice
  * Confirm / Cancel             final gate before it runs (after Allow always)
"""

from __future__ import annotations

from . import console
from ..runtime import Runtime
from ..agent import Reporter
from ..permissions import PermissionManager, Decision

HELP = """
Commands:
  /think <text>   force the planning engine (analysis, no actions)
  /code <text>    force the code engine (writes code as text, no actions)
  /project        show detected project context
  /memory         show memory stats
  /trust          show which actions you've set to "Allow always"
  /help           show this help
  /exit           quit

Anything else is handled by the agent, which can create folders/files and run
commands -- it will always ask permission first.
"""

C = console  # shorthand


# --------------------------------------------------------- interactive I/O
def _ask_permission(tool_name: str, action_desc: str) -> Decision:
    C.rule("permission")
    C.warn(f"SABI wants to {action_desc}")
    C.info("  [1] Allow once     [2] Allow always     [3] Deny")
    while True:
        choice = input("  choose 1/2/3 > ").strip().lower()
        if choice in ("1", "o", "once"):
            return Decision.ALLOW_ONCE
        if choice in ("2", "a", "always"):
            return Decision.ALLOW_ALWAYS
        if choice in ("3", "d", "deny", "n", "no"):
            return Decision.DENY
        C.info("  please type 1, 2, or 3")


def _confirm(action_desc: str) -> bool:
    ans = input(f"  Confirm — {action_desc}?  [y] Yes  [n] Cancel > ").strip().lower()
    return ans in ("y", "yes")


class ConsoleReporter(Reporter):
    """Prints live status as the agent works."""

    def thinking(self) -> None:
        C.info("[dim]· SABI is thinking…[/dim]" if C.has_rich() else "· SABI is thinking…")

    def proposing(self, tool: str, desc: str) -> None:
        verb = "access your files" if tool in ("read_file", "list_dir") else "make changes"
        C.info(f"[dim]· SABI wants to {verb}…[/dim]" if C.has_rich()
               else f"· SABI wants to {verb}…")

    def ran(self, ok: bool, output: str) -> None:
        if ok:
            C.success("done")
            if output:
                snippet = output if len(output) < 600 else output[:600] + " …"
                C.info(f"[dim]{snippet}[/dim]" if C.has_rich() else snippet)
        else:
            C.error(f"action failed: {output}")

    def denied(self, desc: str) -> None:
        C.warn("skipped (you denied this action)")

    def final(self, text: str) -> None:
        pass  # printed by the caller


def chat_loop(runtime: Runtime, auto_approve: bool = False, cwd: str | None = None) -> None:
    runtime.start()
    model_label = runtime.config.abs_model_path().stem or "offline model"
    C.intro_screen(model_label=model_label,
                   ready=bool(runtime.model and runtime.model.is_available()))

    if runtime.model and not runtime.model.is_available():
        C.warn("Model not loaded: " + runtime.model.status() +
               "\n  Run `sabi doctor`, or download the model "
               "(`sabi download`).\n")

    permissions = PermissionManager(
        prompter=_ask_permission,
        confirmer=_confirm,
        auto_approve=auto_approve,
    )
    reporter = ConsoleReporter()
    agent = runtime.make_agent(permissions=permissions, reporter=reporter, cwd=cwd)

    while True:
        try:
            user = input("\nyou  > ").strip()
        except (EOFError, KeyboardInterrupt):
            C.info("\nbye 👋")
            return
        if not user:
            continue

        low = user.lower()
        if low in ("/exit", "/quit", ":q"):
            C.info("bye 👋")
            return
        if low == "/help":
            C.info(HELP)
            continue
        if low == "/project":
            C.kv_table("Project context", [("root", runtime.project.root),
                                           ("summary", runtime.project.summary())])
            continue
        if low == "/memory":
            C.kv_table("Memory", [(k, str(v)) for k, v in runtime.memory.stats().items()])
            continue
        if low == "/trust":
            trusted = ", ".join(sorted(permissions.trusted)) or "(none yet)"
            C.kv_table("Allow-always actions", [("trusted tools", trusted)])
            continue
        if low.startswith("/think "):
            _run_engine(runtime, "think", user[len("/think "):])
            continue
        if low.startswith("/code "):
            _run_engine(runtime, "code", user[len("/code "):])
            continue

        # Default: run through the agentic loop (can act, asks permission).
        try:
            res = agent.run(user, context=runtime.retriever.context(user))
        except Exception as exc:  # noqa: BLE001
            C.error(str(exc))
            continue

        if res.error:
            C.error(res.error)
            continue
        C.rule("sabi")
        C.markdown(res.answer or "(done)")
        runtime.memory.add_turn("user", user, "AGENT")
        runtime.memory.add_turn("assistant", res.answer, "AGENT")


def _run_engine(runtime: Runtime, which: str, text: str) -> None:
    try:
        engine = runtime.think if which == "think" else runtime.code
        gen = engine.run(text)
        C.markdown(gen.text)
    except Exception as exc:  # noqa: BLE001
        C.error(str(exc))
