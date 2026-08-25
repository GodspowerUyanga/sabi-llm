"""Runtime core.

Wires together the model, router, engines, memory, RAG, tools and project
scanner. Initialisation follows the fixed order from the spec so startup stays
fast and deterministic:

    Load model -> Load prompts -> Initialize memory -> Initialize tools
    -> Start router -> Activate THINK + CODE -> Start runtime
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .config import Config, load_config
from .model import LLMModel
from .router import Router, THINK, CODE, CHAT
from .engines import ThinkEngine, CodeEngine
from .memory import MemoryStore
from .rag import HashingEmbedder, VectorStore, Retriever
from .tools import default_registry
from .agent import AgentLoop, Reporter
from .permissions import PermissionManager
from . import project_scanner
from . import translate
from .filereader import TEXT_EXTS

# Directories skipped when indexing a codebase — build artifacts, VCS
# internals and dependency trees are noise and can be enormous.
_INDEX_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", "target", ".egg-info",
}


class Runtime:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.prompts: dict[str, str] = {}
        self.model: Optional[LLMModel] = None
        self.router: Optional[Router] = None
        self.think: Optional[ThinkEngine] = None
        self.code: Optional[CodeEngine] = None
        self.memory: Optional[MemoryStore] = None
        self.retriever: Optional[Retriever] = None
        self.tools = None
        self.project = None
        self.cwd: Optional[Path] = None
        self._started = False

    # --------------------------------------------------------------- prompts
    def _load_prompts(self) -> None:
        prompt_dir = self.config.abs_prompts()
        wanted = {"system": "system.txt", "think": "think.txt",
                  "code": "code.txt", "router": "router.txt", "agent": "agent.txt"}
        for key, fname in wanted.items():
            fpath = prompt_dir / fname
            self.prompts[key] = fpath.read_text(encoding="utf-8") if fpath.exists() else ""

    # ----------------------------------------------------------------- start
    def start(self, cwd: Optional[str] = None) -> "Runtime":
        if self._started:
            return self

        # Per-project memory root: a ".sabi" folder INSIDE the project being
        # worked on (like ".git"), not inside wherever SABI itself is
        # installed. Getting this wrong means every project you ever point
        # SABI at shares one global memory/vector-store — later runs in repo B
        # retrieve "relevant context" chunks indexed from unrelated repo A and
        # feed them to the model, which quietly derails it. cwd defaults to
        # the directory the process was launched from, which is normally
        # already the project directory.
        self.cwd = Path(cwd or os.getcwd()).resolve()
        project_meta = self.cwd / ".sabi"

        # 1) Load model (lazy; not yet read into RAM)
        self.model = LLMModel(self.config)
        # 2) Load prompts
        self._load_prompts()
        # 3) Initialize memory
        project_meta.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryStore(project_meta / "memory.json")
        # 3b) Initialize RAG
        store = VectorStore(project_meta / "vector_store.json")
        self.retriever = Retriever(store, HashingEmbedder())
        # 4) Initialize tools
        self.tools = default_registry(self.config.abs_workspace())
        # 5) Start router
        self.router = Router(self.model)
        # 6) Activate THINK + CODE
        self.think = ThinkEngine(self.model, self.prompts.get("system", ""),
                                 self.prompts.get("think", ""))
        self.code = CodeEngine(self.model, self.prompts.get("system", ""),
                               self.prompts.get("code", ""))
        # 7) Scan current project context
        self.project = project_scanner.scan(self.cwd)
        self._started = True
        return self

    # --------------------------------------------------------- codebase memory
    def index_codebase(self, cwd: Optional[str] = None,
                       max_files: int = 400, max_file_bytes: int = 200_000) -> int:
        """Warm the retriever with this project's source so the agent can recall
        'what's in this codebase' from turn one, in any language, instead of only
        seeing what it happens to search/read mid-conversation.

        Only long-lived sessions (TUI, chat REPL) call this — one-shot CLI
        commands skip it and rely on search_files/read_file, since walking and
        embedding a whole tree on every invocation would be wasted work.
        Already-indexed files (by exact path, persisted in the vector store
        across runs) are skipped, so re-running this is cheap after the first
        time and never duplicates records.
        """
        if not self._started:
            self.start(cwd=cwd)
        root = Path(cwd or self.cwd or os.getcwd()).resolve()
        already = {rec.get("source") for rec in self.retriever.store.records}
        added = 0
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories outright (.git, .venv, .llama.cpp-style
            # vendored checkouts, .next, .cache, …) — a codebase's own
            # metadata/vendor trees are never what "index this project" means,
            # and enumerating them by name is a losing game across languages.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")
                           and d not in _INDEX_SKIP_DIRS and not d.endswith(".egg-info")]
            for name in filenames:
                if added >= max_files:
                    return added
                fp = Path(dirpath) / name
                if fp.suffix.lower() not in TEXT_EXTS:
                    continue
                source = str(fp)
                if source in already:
                    continue
                try:
                    if fp.stat().st_size > max_file_bytes:
                        continue
                    self.retriever.add_file(fp)
                except Exception:
                    continue
                already.add(source)
                added += 1
        return added

    # ------------------------------------------------------- sabi-yoruba-tts
    def _yoruba_status(self, text: str, force: bool = False) -> str:
        """'active' (translate this turn), 'unavailable' (wanted, not installed), or 'off'.

        ``force`` skips the looks_like_yoruba auto-detect — e.g. a UI-level
        "reply in Yoruba" toggle (sabi serve) that should apply regardless of
        what language the typed message happens to be in.
        """
        if not self.config.yoruba_enabled or not (force or translate.looks_like_yoruba(text)):
            return "off"
        if translate.available(str(self.config.abs_yoruba_model_path())):
            return "active"
        return "unavailable"

    def yoruba_available(self) -> bool:
        """True if sabi-yoruba-tts is downloaded and ready to use right now."""
        return translate.available(str(self.config.abs_yoruba_model_path()))

    def _to_english(self, text: str) -> str:
        return translate.to_english(text, str(self.config.abs_yoruba_model_path()))

    def _to_yoruba(self, text: str) -> str:
        try:
            return translate.to_yoruba(text, str(self.config.abs_yoruba_model_path()))
        except Exception as exc:  # never break the reply over a translation failure
            return text + f"\n\n_(Yoruba translation unavailable right now: {exc})_"

    _YORUBA_UNAVAILABLE_NOTE = (
        "\n\n_(sabi-yoruba-tts isn't installed yet, so this reply is in English — "
        "run `python scripts/download_yoruba_model.py` to enable Yoruba.)_"
    )

    # ------------------------------------------------------------- handling
    def handle(self, request: str, *, use_rag: bool = True, force_yoruba: bool = False) -> dict:
        """Route a request and run the appropriate engine. Returns a result dict."""
        if not self._started:
            self.start()

        yoruba = self._yoruba_status(request, force=force_yoruba)
        effective_request = self._to_english(request) if yoruba == "active" else request

        routing = self.router.route(effective_request, self.prompts.get("router", ""))
        context = self.retriever.context(effective_request) if use_rag else ""

        self.memory.add_turn("user", request, routing.intent)

        result = {"intent": routing.intent, "confidence": routing.confidence,
                  "reason": routing.reason, "context_used": bool(context),
                  "language": "yo" if yoruba != "off" else "en"}

        try:
            if routing.intent == CODE:
                gen = self.code.run(effective_request, context=context)
            elif routing.intent == THINK:
                gen = self.think.run(effective_request, context=context)
            else:  # CHAT - answer directly with the base model
                gen = self.model.generate(
                    effective_request, system=self.prompts.get("system", "") or None
                )
            text = gen.text
            if yoruba == "active":
                text = self._to_yoruba(text)
            elif yoruba == "unavailable":
                text += self._YORUBA_UNAVAILABLE_NOTE
            result.update({
                "ok": True,
                "text": text,
                "tps": round(gen.tokens_per_second, 2),
                "tokens": gen.prompt_tokens + gen.completion_tokens,
                "elapsed_s": round(gen.elapsed_s, 2),
            })
            self.memory.add_turn("assistant", text, routing.intent)
            self.memory.add_task(request[:80], "done", routing.intent)
            try:
                self.retriever.add_text(f"USER: {request}\nSABI: {text}", source="conversation")
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001 - surface as a clean message
            result.update({"ok": False, "text": "", "error": str(exc)})
        return result

    def make_agent(self, permissions: Optional[PermissionManager] = None,
                   reporter: Optional[Reporter] = None,
                   cwd: Optional[str] = None,
                   history: Optional[list] = None,
                   restricted: bool = False) -> AgentLoop:
        """Build a tool-calling agent loop bound to a permission manager.

        ``cwd`` defaults to the directory SABI was launched from, so the agent
        acts on the user's real project / files (with approval), not the
        internal workspace sandbox.

        ``history`` seeds prior conversation turns (role/content dicts) —
        needed by any caller that builds a fresh AgentLoop per request (sabi
        serve) rather than keeping one alive for the whole session (the TUI,
        the terminal chat loop already do this correctly on their own).

        ``restricted`` is sabi serve's Coding Assistant persona (code
        generation, debugging, programming tutoring) — no create_dir/
        write_file/edit_file/move_file, and it overrides
        ``self.prompts.get("agent", "")`` below with AgentLoop's own
        restricted prompt, since that file is the full file-creating prompt
        `sabi run`/`sabi agent`/the TUI use.
        """
        if not self._started:
            self.start(cwd=cwd)
        permissions = permissions or PermissionManager(auto_approve=False)
        return AgentLoop(
            model=self.model,
            permissions=permissions,
            system_prompt=self.prompts.get("agent", ""),
            cwd=Path(cwd) if cwd else (self.cwd or Path.cwd()),
            reporter=reporter,
            retriever=self.retriever,
            initial_history=history,
            restricted=restricted,
        )

    def agent(self, request: str, *, permissions: Optional[PermissionManager] = None,
              reporter: Optional[Reporter] = None, cwd: Optional[str] = None,
              use_rag: bool = True, force_yoruba: bool = False,
              history: Optional[list] = None, restricted: bool = False) -> dict:
        """Run the agentic loop for a request and return a result dict."""
        if not self._started:
            self.start(cwd=cwd)
        # Translate the natural-language request/reply only; tool-call JSON,
        # file paths and code (`actions`) are never routed through translation.
        yoruba = self._yoruba_status(request, force=force_yoruba)
        effective_request = self._to_english(request) if yoruba == "active" else request

        loop = self.make_agent(permissions=permissions, reporter=reporter, cwd=cwd,
                               history=history, restricted=restricted)
        context = self.retriever.context(effective_request) if use_rag else ""
        res = loop.run(effective_request, context=context)
        answer = res.answer
        if res.ok:
            if yoruba == "active":
                answer = self._to_yoruba(answer)
            elif yoruba == "unavailable":
                answer += self._YORUBA_UNAVAILABLE_NOTE
            self.memory.add_turn("user", request, "AGENT")
            self.memory.add_turn("assistant", answer, "AGENT")
            self.memory.add_task(request[:80], "done", "AGENT")
        return {"ok": res.ok, "answer": answer, "actions": res.actions, "error": res.error,
                "language": "yo" if yoruba != "off" else "en"}
