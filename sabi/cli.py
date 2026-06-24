"""SABI command-line interface.

Built on argparse (stdlib) so ``sabi`` always starts, even on a bare install.
Pretty output is provided by the console helper (rich with plain fallback).

Commands:
  sabi                 launch the runtime (interactive)
  sabi run             start the agent runtime in the current project
  sabi chat            launch the chat UI
  sabi ask <text>      one-off question (auto-routed)
  sabi think <text>    planning / analysis engine
  sabi code <text>     code generation / debugging engine
  sabi agent <text>    full plan->execute->verify loop
  sabi benchmark       run the local benchmark
  sabi profile         show RAM / CPU / thermal telemetry
  sabi doctor          diagnose the environment
  sabi workspace ...   inspect / reset the workspace
  sabi version         print version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .runtime import Runtime
from .ui import console
from .ui.chat import chat_loop
from . import doctor as doctor_mod
from . import benchmark as benchmark_mod
from . import profiler as profiler_mod
from . import project_scanner


def _runtime() -> Runtime:
    return Runtime(load_config()).start()


# --------------------------------------------------------------------- commands
def cmd_run(args) -> int:
    rt = _runtime()
    _ensure_model(rt)
    console.banner()
    console.kv_table("Runtime ready", [
        ("model", rt.model.status()),
        ("working dir", getattr(args, "cwd", None) or str(__import__("os").getcwd())),
        ("project", rt.project.summary()),
        ("tools", "create_dir, write_file, read_file, list_dir, run_shell"),
    ])
    if not rt.model.is_available():
        console.warn("Model not loaded. Run `sabi doctor` or download the model "
                     "(`sabi download`). Entering chat anyway.\n")
    chat_loop(rt, auto_approve=getattr(args, "yes", False), cwd=getattr(args, "cwd", None))
    return 0


def cmd_chat(args) -> int:
    rt = _runtime()
    _ensure_model(rt)
    chat_loop(rt, auto_approve=getattr(args, "yes", False),
              cwd=getattr(args, "cwd", None))
    return 0


def cmd_ask(args) -> int:
    rt = _runtime()
    res = rt.handle(" ".join(args.text))
    return _emit(res, args.json)


def cmd_think(args) -> int:
    rt = _runtime()
    try:
        gen = rt.think.run(" ".join(args.text))
        if args.json:
            print(json.dumps({"ok": True, "text": gen.text, "tps": gen.tokens_per_second}))
        else:
            console.markdown(gen.text)
        return 0
    except Exception as exc:  # noqa: BLE001
        console.error(str(exc))
        return 1


def cmd_code(args) -> int:
    rt = _runtime()
    try:
        gen = rt.code.run(" ".join(args.text))
        if args.json:
            print(json.dumps({"ok": True, "text": gen.text, "tps": gen.tokens_per_second}))
        else:
            console.markdown(gen.text)
        return 0
    except Exception as exc:  # noqa: BLE001
        console.error(str(exc))
        return 1


def cmd_agent(args) -> int:
    rt = _runtime()
    from .permissions import PermissionManager
    from .ui.chat import _ask_permission, _confirm, ConsoleReporter
    perms = PermissionManager(
        prompter=_ask_permission, confirmer=_confirm,
        auto_approve=getattr(args, "yes", False),
    )
    res = rt.agent(" ".join(args.text), permissions=perms,
                   reporter=ConsoleReporter(), cwd=getattr(args, "cwd", None))
    if args.json:
        print(json.dumps(res))
        return 0 if res.get("ok") else 1
    if res.get("ok"):
        if res.get("actions"):
            console.rule("actions")
            for a in res["actions"]:
                console.info("  " + a)
        console.rule("sabi")
        console.markdown(res.get("answer") or "(done)")
        return 0
    console.error(res.get("error", "agent failed"))
    return 1


def cmd_benchmark(args) -> int:
    cfg = load_config()
    console.rule("SABI Benchmark")
    report = benchmark_mod.run(cfg, limit=args.limit)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    console.kv_table("Summary", [
        ("model available", str(report.model_available)),
        ("avg accuracy", f"{report.avg_accuracy:.1%}"),
        ("avg speed", f"{report.avg_tps:.2f} tok/s"),
        ("peak RAM", f"{report.peak_rss_gb:.3f} GB / {cfg.ram_ceiling_gb} GB budget"),
        ("efficiency score", f"{report.efficiency_score:.1f} / 100"),
        ("thermal throttle", str(report.thermal_throttle)),
    ])
    for c in report.cases:
        status = "ok" if c.ok else "skip"
        console.info(f"  [{status}] {c.name:<22} {c.intent:<6} "
                     f"{c.tps:>6.1f} tok/s  acc={c.accuracy:.0%}  {c.note}")
    if not report.model_available:
        console.warn("\nModel not available - figures are placeholders. "
                     "Download the model to get real telemetry.")
    return 0


def cmd_profile(args) -> int:
    cfg = load_config()
    snap = profiler_mod.snapshot(cfg.thermal_ceiling_c)
    rows = [
        ("process RSS", f"{snap.rss_gb:.3f} GB"),
        ("available RAM", f"{snap.available_gb:.3f} GB" if snap.available_gb else "unknown"),
        ("CPU", f"{snap.cpu_percent:.1f}%"),
        ("temperature", f"{snap.temperature_c} \u00b0C" if snap.temperature_c is not None else "n/a"),
        ("thermal throttle", str(snap.thermal_throttle)),
        ("RAM ceiling", f"{cfg.ram_ceiling_gb} GB"),
    ]
    if args.json:
        print(json.dumps(dict(rows)))
    else:
        console.kv_table("Telemetry", rows)
    return 0


def cmd_doctor(args) -> int:
    cfg = load_config()
    console.rule("SABI Doctor")
    checks = doctor_mod.run_checks(cfg)
    for c in checks:
        (console.success if c.ok else console.warn)(f"{c.name}: {c.detail}")
    ok = doctor_mod.all_ok(checks)
    console.info("")
    (console.success if ok else console.warn)(
        "All checks passed." if ok else "Some checks need attention (see above)."
    )
    return 0 if ok else 1


def cmd_workspace(args) -> int:
    cfg = load_config()
    ws = cfg.abs_workspace()
    ws.mkdir(parents=True, exist_ok=True)
    if args.action == "reset":
        import shutil
        for child in ws.iterdir():
            if child.name == ".sabi":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        console.success(f"workspace reset: {ws}")
        return 0
    # default: info
    info = project_scanner.scan(ws)
    files = sorted(p.name for p in ws.iterdir())
    console.kv_table("Workspace", [
        ("path", str(ws)),
        ("entries", str(len(files))),
        ("contents", ", ".join(files) or "(empty)"),
    ])
    return 0


def cmd_serve(args) -> int:
    from . import server
    return server.serve(load_config(), host=args.host, port=args.port,
                        open_browser=not args.no_browser)


def cmd_download(args) -> int:
    from . import downloader
    cfg = load_config()
    out = cfg.abs_model_path()
    if out.exists() and not args.force:
        console.success(f"Model already present: {out}")
        console.info("(use `sabi download --force` to re-download)")
        return 0
    try:
        path = downloader.download_model(cfg, repo=args.repo, filename=args.file,
                                         force=args.force)
        console.success(f"Model ready at {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        console.error(str(exc))
        return 1


def _ensure_model(rt) -> None:
    """If the model is missing, offer to download it (interactive only)."""
    if rt.model and rt.model.is_available():
        return
    import sys as _sys
    from . import downloader
    size_hint = "~4.7 GB" if "7B" in rt.config.hf_filename or "Q4" in rt.config.hf_filename else "the model"
    if not _sys.stdin.isatty():
        console.warn("Model not found. Run `sabi download` to fetch it.")
        return
    console.warn("No model found locally.")
    ans = input(f"  Download it now from Hugging Face ({size_hint})?  [Y/n] > ").strip().lower()
    if ans in ("", "y", "yes"):
        try:
            downloader.download_model(rt.config)
            console.success("Model downloaded. Starting…\n")
        except Exception as exc:  # noqa: BLE001
            console.error(str(exc))
    else:
        console.info("Skipping download. You can run `sabi download` later.\n")


def cmd_version(args) -> int:
    console.info(f"SABI {__version__}")
    return 0


def _emit(res: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(res))
        return 0 if res.get("ok") else 1
    if res.get("ok"):
        console.info(f"[dim]intent={res['intent']} ({res['confidence']:.0%})[/dim]")
        console.markdown(res["text"])
        return 0
    console.error(res.get("error", "request failed"))
    return 1


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sabi",
        description="SABI - The Offline AI Coworker (on-device, private).",
    )
    p.add_argument("--version", action="version", version=f"SABI {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("run", help="start the agentic runtime in the current project")
    sp.add_argument("--yes", action="store_true", help="auto-approve all actions (no prompts)")
    sp.add_argument("--cwd", default=None, help="working directory the agent acts in")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("chat", help="launch the chat interface")
    sp.add_argument("--yes", action="store_true", help="auto-approve all actions (no prompts)")
    sp.add_argument("--cwd", default=None, help="working directory the agent acts in")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("ask", help="one-off question (auto-routed)")
    sp.add_argument("text", nargs="+")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("think", help="planning / analysis engine")
    sp.add_argument("text", nargs="+")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_think)

    sp = sub.add_parser("code", help="code generation / debugging engine")
    sp.add_argument("text", nargs="+")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_code)

    sp = sub.add_parser("agent", help="agentic loop that can create files/run commands")
    sp.add_argument("text", nargs="+")
    sp.add_argument("--yes", action="store_true", help="auto-approve all actions (no prompts)")
    sp.add_argument("--cwd", default=None, help="working directory the agent acts in")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_agent)

    sp = sub.add_parser("benchmark", help="run the local benchmark")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("profile", help="show RAM / CPU / thermal telemetry")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_profile)

    sub.add_parser("doctor", help="diagnose the environment").set_defaults(func=cmd_doctor)

    sp = sub.add_parser("workspace", help="inspect or reset the workspace")
    sp.add_argument("action", nargs="?", choices=["info", "reset"], default="info")
    sp.set_defaults(func=cmd_workspace)

    sp = sub.add_parser("download", help="download the model from Hugging Face into models/")
    sp.add_argument("--repo", default=None, help="override the Hugging Face repo id")
    sp.add_argument("--file", default=None, help="override the GGUF filename")
    sp.add_argument("--force", action="store_true", help="re-download even if present")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("serve", help="launch the web UI (browser chat with history)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    sp.set_defaults(func=cmd_serve)

    sub.add_parser("version", help="print version").set_defaults(func=cmd_version)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    # `sabi` with no subcommand -> launch the runtime (interactive).
    if not getattr(args, "command", None):
        return cmd_run(argparse.Namespace())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
