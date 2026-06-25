"""Runtime core.

Wires together the model, router, engines, memory, RAG, tools and project
scanner. Initialisation follows the fixed order from the spec so startup stays
fast and deterministic:

    Load model -> Load prompts -> Initialize memory -> Initialize tools
    -> Start router -> Activate THINK + CODE -> Start runtime
"""

from __future__ import annotations

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
    def start(self) -> "Runtime":
        if self._started:
            return self

        # 1) Load model (lazy; not yet read into RAM)
        self.model = LLMModel(self.config)
        # 2) Load prompts
        self._load_prompts()
        # 3) Initialize memory
        self.config.abs_workspace().mkdir(parents=True, exist_ok=True)
        self.memory = MemoryStore(self.config.abs_memory())
        # 3b) Initialize RAG
        store = VectorStore(self.config.abs_vector_store())
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
        self.project = project_scanner.scan(".")
        self._started = True
        return self

    # ------------------------------------------------------------- handling
    def handle(self, request: str, *, use_rag: bool = True) -> dict:
        """Route a request and run the appropriate engine. Returns a result dict."""
        if not self._started:
            self.start()

        routing = self.router.route(request, self.prompts.get("router", ""))
        context = self.retriever.context(request) if use_rag else ""

        self.memory.add_turn("user", request, routing.intent)

        result = {"intent": routing.intent, "confidence": routing.confidence,
                  "reason": routing.reason, "context_used": bool(context)}

        try:
            if routing.intent == CODE:
                gen = self.code.run(request, context=context)
            elif routing.intent == THINK:
                gen = self.think.run(request, context=context)
            else:  # CHAT - answer directly with the base model
                gen = self.model.generate(
                    request, system=self.prompts.get("system", "") or None
                )
            result.update({
                "ok": True,
                "text": gen.text,
                "tps": round(gen.tokens_per_second, 2),
                "tokens": gen.prompt_tokens + gen.completion_tokens,
                "elapsed_s": round(gen.elapsed_s, 2),
            })
            self.memory.add_turn("assistant", gen.text, routing.intent)
            self.memory.add_task(request[:80], "done", routing.intent)
        except Exception as exc:  # noqa: BLE001 - surface as a clean message
            result.update({"ok": False, "text": "", "error": str(exc)})
        return result

    def make_agent(self, permissions: Optional[PermissionManager] = None,
                   reporter: Optional[Reporter] = None,
                   cwd: Optional[str] = None) -> AgentLoop:
        """Build a tool-calling agent loop bound to a permission manager.

        ``cwd`` defaults to the directory SABI was launched from, so the agent
        acts on the user's real project / files (with approval), not the
        internal workspace sandbox.
        """
        if not self._started:
            self.start()
        permissions = permissions or PermissionManager(auto_approve=False)
        return AgentLoop(
            model=self.model,
            permissions=permissions,
            system_prompt=self.prompts.get("agent", ""),
            cwd=Path(cwd) if cwd else Path.cwd(),
            reporter=reporter,
        )

    def agent(self, request: str, *, permissions: Optional[PermissionManager] = None,
              reporter: Optional[Reporter] = None, cwd: Optional[str] = None,
              use_rag: bool = True) -> dict:
        """Run the agentic loop for a request and return a result dict."""
        if not self._started:
            self.start()
        loop = self.make_agent(permissions=permissions, reporter=reporter, cwd=cwd)
        context = self.retriever.context(request) if use_rag else ""
        res = loop.run(request, context=context)
        if res.ok:
            self.memory.add_turn("user", request, "AGENT")
            self.memory.add_turn("assistant", res.answer, "AGENT")
            self.memory.add_task(request[:80], "done", "AGENT")
        return {"ok": res.ok, "answer": res.answer, "actions": res.actions, "error": res.error}
