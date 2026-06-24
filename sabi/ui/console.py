"""Terminal output helpers.

Uses `rich` when available for colour/panels, and degrades cleanly to plain
``print`` so SABI always starts, even on a bare Python install.
"""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - import guard
    from rich.console import Console as _RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text

    _HAS_RICH = True
    _console = _RichConsole()
except Exception:  # pragma: no cover
    _HAS_RICH = False
    _console = None


# SABI brand palette (matches the project spec sheet)
NAVY = "#16263d"
TEAL = "#1f8a8c"
GOLD = "#c8901f"


def _plain(*args, **kwargs):
    print(*args)


def info(msg: str) -> None:
    if _HAS_RICH:
        _console.print(msg)
    else:
        _plain(msg)


def success(msg: str) -> None:
    if _HAS_RICH:
        _console.print(f"[bold green]\u2714[/bold green] {msg}")
    else:
        _plain("[OK] " + msg)


def warn(msg: str) -> None:
    if _HAS_RICH:
        _console.print(f"[bold yellow]\u26a0[/bold yellow] {msg}")
    else:
        _plain("[WARN] " + msg)


def error(msg: str) -> None:
    if _HAS_RICH:
        _console.print(f"[bold red]\u2716[/bold red] {msg}")
    else:
        _plain("[ERROR] " + msg)


def rule(title: str = "") -> None:
    if _HAS_RICH:
        _console.rule(f"[bold {TEAL}]{title}")
    else:
        _plain("\n== " + title + " ==")


def banner() -> None:
    art = r"""
   ____    _    ____ ___
  / ___|  / \  | __ )_ _|
  \___ \ / _ \ |  _ \| |
   ___) / ___ \| |_) | |
  |____/_/   \_\____/___|
"""
    if _HAS_RICH:
        _console.print(Text(art, style=f"bold {TEAL}"))
        _console.print(
            "  The Offline AI Coworker  \u2014  on-device, private, "
            f"built for constrained hardware\n",
            style="dim",
        )
    else:
        _plain(art)
        _plain("  The Offline AI Coworker - on-device, private.\n")


def intro_screen(model_label: str = "offline model", ready: bool = True) -> None:
    """An opencode-style landing screen: centred logo, an input hint box,
    a status line, and key hints."""
    art = r""" ____    _    ____ ___
/ ___|  / \  | __ )_ _|
\___ \ / _ \ |  _ \| |
 ___) / ___ \| |_) | |
|____/_/   \_\____/___|"""
    if not _HAS_RICH:
        _plain(art)
        _plain("\n  Ask anything...  e.g. \"create a folder called app\"")
        _plain(f"  Agent · {model_label} · Offline")
        _plain("  /help commands    /exit quit")
        _plain("\n  Tip: SABI can create files & run commands \u2014 it always asks first.\n")
        return

    from rich.align import Align
    from rich.console import Group

    _console.print()
    _console.print(Align.center(Text(art, style=f"bold {TEAL}")))
    _console.print()

    status = Text()
    status.append("Agent", style=f"bold {TEAL}")
    status.append("  \u00b7  ", style="dim")
    status.append(model_label, style="white")
    status.append("  \u00b7  ", style="dim")
    status.append("Offline" if ready else "model not loaded",
                  style=f"bold {GOLD}" if ready else "bold red")

    box = Group(
        Text('Ask anything...  "create a folder called app"  ·  "plan a REST API"',
             style="grey70"),
        Text(""),
        status,
    )
    panel = Panel(box, border_style=TEAL, padding=(1, 2), width=72)
    _console.print(Align.center(panel))

    hints = Text()
    hints.append("/help ", style=f"bold {TEAL}")
    hints.append("commands", style="dim")
    hints.append("     ")
    hints.append("/trust ", style=f"bold {TEAL}")
    hints.append("approvals", style="dim")
    hints.append("     ")
    hints.append("/exit ", style=f"bold {TEAL}")
    hints.append("quit", style="dim")
    _console.print(Align.center(hints))

    tip = Text()
    tip.append(" Tip ", style=f"bold white on {GOLD}")
    tip.append("  SABI can create files & run commands \u2014 it always asks first.",
               style="dim")
    _console.print(Align.center(tip))
    _console.print()


def panel(body: str, title: str = "") -> None:
    if _HAS_RICH:
        _console.print(Panel.fit(body, title=title, border_style=TEAL))
    else:
        if title:
            _plain(f"--- {title} ---")
        _plain(body)


def markdown(text: str) -> None:
    if _HAS_RICH:
        _console.print(Markdown(text))
    else:
        _plain(text)


def kv_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    rows = list(rows)
    if _HAS_RICH:
        table = Table(title=title, title_style=f"bold {NAVY}", show_header=False, expand=False)
        table.add_column("k", style=f"bold {TEAL}")
        table.add_column("v")
        for k, v in rows:
            table.add_row(str(k), str(v))
        _console.print(table)
    else:
        if title:
            _plain("\n" + title)
        for k, v in rows:
            _plain(f"  {k:<24} {v}")


def has_rich() -> bool:
    return _HAS_RICH
