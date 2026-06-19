"""
The Sabi agent: a small, offline, tool-using orchestrator.

Per user turn it:
  1. detects the language (for the African-language support),
  2. retrieves relevant company context via RAG (and surfaces the sources),
  3. guarantees exact arithmetic by routing explicit calculations to the
     calculator tool *before* the model can guess (accuracy hardening),
  4. runs the model with a tool protocol, executing any tool calls locally,
  5. streams the final grounded answer token-by-token.

run() yields a stream of typed events so the UI can render sources, tool
traces, and the answer cleanly:
    {"type": "sources", "items": [{"name", "score"}...]}
    {"type": "tool",    "name": str, "summary": str}
    {"type": "delta",   "text": str}

Everything happens on-device. No network calls are ever made at inference.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from .config import Config
from .embeddings import make_embedder
from .languages import detect_language, language_directive
from .prompts import build_system_prompt
from .rag import RagIndex, format_context
from .tools import (build_registry, render_tool_specs, extract_expression,
                    auto_aggregate, auto_pivot, render_pivot_markdown, query_table,
                    show_table)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_OPEN = "<tool_call>"

# Questions about Sabi itself — answered from the fixed persona, never from
# the user's documents (prevents an uploaded file from polluting Sabi's identity).
_IDENTITY_RE = re.compile(
    r"\b(who\s+(are|r)\s+(you|u)|your\s+name|what'?s?\s+your\s+name|what\s+is\s+your\s+name|"
    r"who\s+(made|created|built|trained|develop(ed)?|owns?)\s+(you|u)|"
    r"how\s+(were|was|did)\s+(you|they|u)\b.*(train|made|make|built|build|create|offline)|"
    r"what\s+(model|llm|ai|version|system)\s+(are|is)\s+(you|u)|"
    r"are\s+you\s+(chatgpt|gpt|mistral|qwen|llama|gemini|claude|bard|ubuntulite)|"
    r"wetin\s+be\s+your\s+name|who\s+(train|build|make|own)s?\s+you|"
    r"what\s+(are|r)\s+you\s+(built|based|trained)\s+on|introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"what\s+language(s)?\s+(do|can)\s+you|which\s+language(s)?\s+(do|can)\s+you)", re.I)

# Explicit "answer in <language>" requests (English + Pidgin only).
_LANG_REQUEST = [
    (re.compile(r"\bpidgin\b", re.I), "pcm"),
    (re.compile(r"\b(in|reply in|respond in)\s+english\b", re.I), "en"),
]


# "Create / export an Excel sheet" intent.
_EXPORT_RE = re.compile(
    r"\b(create|make|export|generate|build|save|download|put|turn)\b[^.]*\b(excel|spreadsheet|xlsx|sheet|workbook)\b", re.I)
_AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|ok|okay|sure|please|go ahead|proceed|do it|create( it)?|generate( it)?)\b", re.I)


# Document availability vs. content. Availability = "do you have X / list files".
_DOC_AVAIL_RE = re.compile(
    r"(do|does|can)\s+you\s+(have|access|see|read|find)\b"
    r"|have\s+access\s+to\b"
    r"|you\s+(have|do\s*n'?t|don'?t|do\s+not)\b[^?]*\baccess\b"
    r"|(what|which)\s+(documents?|files?)\b"
    r"|\b(list|show)\b[^.?]*\b(documents?|files?)\b"
    r"|\bdocuments?\s+(do|can)\s+you\b"
    r"|(how|what)\s+about\b[^?]*\b(document|file|\.docx|\.pdf|\.xlsx|\.csv)\b",
    re.I)
# If the user is clearly asking about CONTENT, don't treat it as availability.
_DOC_CONTENT_RE = re.compile(
    r"\b(summar|explain|describe|draft|write|what does|what's in|whats in|"
    r"how many|how much|who (is|are|owes|made|paid)|total|average|pivot|"
    r"list (the|all)\s+\w+\s+(in|from|that|who|owing))\b", re.I)
_FILE_RE = re.compile(r"[\w\-]+\.(docx|pdf|xlsx|xls|csv|txt|md)\b", re.I)
_DOC_STOP = {"you", "have", "access", "the", "this", "that", "document", "documents",
             "file", "files", "doc", "docs", "about", "how", "really", "yes", "does",
             "your", "with", "and", "for", "can", "any", "got", "see", "read", "now",
             "what", "which", "list", "show", "all", "they", "their", "from", "into"}


def _explicit_language(text: str) -> str | None:
    for rx, code in _LANG_REQUEST:
        if rx.search(text or ""):
            return code
    return None


# "Save this to your knowledge base" intent.
_SAVE_RE = re.compile(
    r"\b(save|remember|memor(ize|ise)|note|store|keep)\b.*\b(this|that|it|information|note|fact|knowledge)\b"
    r"|\badd\s+(this|that|it)?\s*to\s+(your\s+)?(knowledge|memory|kb)\b"
    r"|^(save|remember|note)[:\- ]", re.I)
_SAVE_STRIP = re.compile(
    r"^\s*(please\s+)?(can you\s+)?(save|remember|memor(ize|ise)|note|store|keep)\s+"
    r"(this|that|it|the following|down)?\s*(to (your )?(knowledge base|memory|kb))?\s*[:\-]?\s*", re.I)

_OP_LABEL = {"sum": "Total", "mean": "Average", "count": "Count", "max": "Highest", "min": "Lowest"}
_MONEY_HINTS = ("ngn", "revenue", "naira", "price", "cost", "amount", "sales", "income", "profit")


def _fmt_num(column: str, value: float) -> str:
    money = any(h in column.lower() for h in _MONEY_HINTS)
    if float(value).is_integer():
        body = f"{value:,.0f}"
    else:
        body = f"{value:,.2f}"
    return f"₦{body}" if money else body


def _format_aggregate(res: dict) -> str:
    op, col = res["op"], res["value_column"]
    label = _OP_LABEL.get(op, op.title())
    pretty = col.replace("_", " ")
    # Single-group result, e.g. "total revenue for Port Harcourt".
    if res.get("filter_value"):
        return (f"**{label} {pretty} for {res['filter_value']}: "
                f"{_fmt_num(col, res['result'])}** "
                f"_(from {res['file']}, {res['n']} rows)_")
    lines = [f"**{label} {pretty}: {_fmt_num(col, res['result'])}** "
             f"_(from {res['file']}, {res['n']} rows)_", ""]
    bd = res.get("breakdown") or []
    if bd and res.get("group_column"):
        gcol = res["group_column"].replace("_", " ").title()
        lines.append(f"| {gcol} | {pretty.title()} |")
        lines.append("|---|---:|")
        for g, v in bd:
            lines.append(f"| {g} | {_fmt_num(col, v)} |")
        if op == "sum":
            lines.append(f"| **Total** | **{_fmt_num(col, res['result'])}** |")
    return "\n".join(lines)


class SabiAgent:
    def __init__(self, config: Config, chat, index: RagIndex | None = None):
        self.cfg = config
        self.chat = chat
        self.index = index
        self.history: list[dict[str, str]] = []

    # ----------------------------------------------------------------- RAG
    def _retrieve(self, query: str):
        """Return (context_text, [(source, score)...])."""
        if not (self.cfg.rag.enabled and self.index and self.index.size):
            return "", []
        embedder = make_embedder(
            str(self.cfg.embedding_path),
            n_ctx=self.cfg.embedding.n_ctx,
            n_threads=self.cfg.model.n_threads,
        )
        try:
            qv = embedder.embed([query])[0]
        finally:
            embedder.close()  # release embedder RAM immediately
        results = self.index.search(qv, self.cfg.rag.top_k, self.cfg.rag.min_score)
        return format_context(results), [(c.source, round(s, 2)) for c, s in results]

    # --------------------------------------------------------------- tools
    def _registry(self):
        return build_registry(
            corpus_dir=self.cfg.corpus_dir,
            search_docs=(lambda q: self._retrieve(q)[0] or "(nothing found)")
            if self.cfg.rag.enabled else None,
        )

    @staticmethod
    def _parse_tool_call(text: str):
        m = _TOOL_CALL_RE.search(text)
        if not m:
            # tolerate a missing closing tag (model stopped on the stop-token)
            m2 = re.search(r"<tool_call>\s*(\{.*)", text, re.DOTALL)
            if not m2:
                return None
            raw = m2.group(1)
            depth, end = 0, None
            for i, ch in enumerate(raw):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                return None
            raw = raw[:end]
        else:
            raw = m.group(1)
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) and "name" in data else None
        except json.JSONDecodeError:
            return None

    # ----------------------------------------------------------------- run
    def run(self, user_message: str, language: str | None = None,
            history: list[dict] | None = None) -> Iterator[dict]:
        hist = history if history is not None else self.history
        self._persist = (history is None)  # persist server-side only in CLI mode
        # Resolve reply language: explicit UI choice > "in <language>" in the
        # message > auto-detection > default.
        if language and language != "auto":
            lang = language
        else:
            lang = _explicit_language(user_message)
            if not lang:
                lang = (detect_language(user_message, self.cfg.language.supported)
                        if self.cfg.language.auto_detect else self.cfg.language.default)
        directive = language_directive(lang)

        # Identity / meta questions are answered ONLY from the persona — never
        # from the user's documents (so an uploaded file can't redefine Sabi).
        is_identity = bool(_IDENTITY_RE.search(user_message))

        # Document awareness: "what documents do you have", "do you have access
        # to X" — answered deterministically from the indexed files, so Sabi
        # never wrongly says it lacks a file it actually has.
        if not is_identity:
            doc_answer = self._doc_query(user_message)
            if doc_answer is not None:
                yield {"type": "tool", "name": "documents", "summary": "checked indexed files"}
                for line in doc_answer.split("\n"):
                    yield {"type": "delta", "text": line + "\n"}
                self._record(user_message, doc_answer)
                return

        # Knowledge-base save: "remember this / save this to your knowledge base".
        if not is_identity and self.cfg.rag.enabled and _SAVE_RE.search(user_message):
            saved = self._save_to_kb(user_message, hist)
            if saved:
                yield {"type": "tool", "name": "save_to_knowledge_base", "summary": saved}
                msg = f"Saved to my knowledge base. You can ask me about it anytime — it stays on this device."
                yield {"type": "delta", "text": msg}
                self._record(user_message, msg)
                return

        # Create / export an Excel sheet. If the request is data-driven (e.g. a
        # debtors sheet) we build it from the real cells; otherwise from the
        # latest summary text.
        if not is_identity and self.cfg.agent.enable_tools and self._wants_export(user_message, hist):
            from .tools import export_to_xlsx, export_table_to_xlsx, query_table as _qt, extract_sheet_name
            out_dir = self.cfg.abs_path("data/exports")
            sheet = extract_sheet_name(user_message) or "Summary"
            tbl = _qt(user_message, self.cfg.corpus_dir) if self._data_sheet_request(user_message) else None
            if tbl and tbl.get("rows"):
                name = export_table_to_xlsx(tbl["header"], tbl["rows"], sheet, out_dir)
                detail = f"with {len(tbl['rows'])} row(s) from {tbl['file']}"
            else:
                name = export_to_xlsx(self._export_content(hist) or "Summary", sheet, out_dir)
                detail = "from the summary"
            yield {"type": "tool", "name": "create_excel", "summary": f'"{sheet}" sheet {detail}'}
            msg = (f'I created the **{sheet}** sheet for you ({detail}) — saved on your device.\n\n'
                   f"[⬇ Download {name}](/download/{name})")
            yield {"type": "delta", "text": msg}
            self._record(user_message, msg)
            return

        if is_identity or not self.cfg.rag.enabled:
            context, sources = "", []
        else:
            context, sources = self._retrieve(user_message)

        tool_specs = render_tool_specs() if self.cfg.agent.enable_tools else None
        context_for_prompt = None if (is_identity or not self.cfg.rag.enabled) else context
        system = build_system_prompt(tool_specs, context_for_prompt, directive)

        messages = [{"role": "system", "content": system}]
        messages += hist[-8:]  # remember recent turns for follow-up context
        messages.append({"role": "user", "content": user_message})

        if sources:
            yield {"type": "sources", "items": [{"name": n, "score": s} for n, s in sources]}

        registry = self._registry()

        # --- Accuracy hardening: compute explicit arithmetic deterministically.
        if self.cfg.agent.enable_tools and not is_identity:
            expr = extract_expression(user_message)
            if expr:
                result = registry["calc"](expression=expr)
                messages.append({"role": "assistant",
                                 "content": f'<tool_call>{{"name":"calc","arguments":{{"expression":"{expr}"}}}}</tool_call>'})
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                yield {"type": "tool", "name": "calc", "summary": self._short(result)}
            else:
                # Structured question over a spreadsheet (who is owing, list debtors,
                # how many owing, highest/lowest payment)? Answer deterministically.
                q = query_table(user_message, self.cfg.corpus_dir)
                if q:
                    yield {"type": "tool", "name": "table_query",
                           "summary": f"{q['summary']} ({q['file']})"}
                    for line in q["markdown"].split("\n"):
                        yield {"type": "delta", "text": line + "\n"}
                    self._record(user_message, q["markdown"])
                    return
                # Pivot table request? (only when the word "pivot" is used)
                piv = auto_pivot(user_message, self.cfg.corpus_dir)
                if piv:
                    yield {"type": "tool", "name": "pivot_table",
                           "summary": f"{piv['agg']} {piv['value']} by {piv['index']}"}
                    answer = render_pivot_markdown(piv)
                    for line in answer.split("\n"):
                        yield {"type": "delta", "text": line + "\n"}
                    self._record(user_message, answer)
                    return
                # "Show / list / breakdown the data" — render the real rows
                # (optionally filtered to a region/product) deterministically.
                tv = show_table(user_message, self.cfg.corpus_dir)
                if tv:
                    yield {"type": "tool", "name": "show_table", "summary": tv["summary"]}
                    for line in tv["markdown"].split("\n"):
                        yield {"type": "delta", "text": line + "\n"}
                    self._record(user_message, tv["markdown"])
                    return
                # Otherwise: data question over a spreadsheet (total/average/...)?
                agg = auto_aggregate(user_message, self.cfg.corpus_dir)
                if agg:
                    yield {"type": "tool", "name": "analyze",
                           "summary": f"{agg['op']} {agg['value_column']} = {agg['result']}"}
                    answer = _format_aggregate(agg)
                    for line in answer.split("\n"):
                        yield {"type": "delta", "text": line + "\n"}
                    self._record(user_message, answer)
                    return

        final_text = ""
        for _ in range(self.cfg.agent.max_steps):
            decided = None          # None | "text" | "tool"
            buffer = ""
            for delta in self.chat.chat(
                messages,
                temperature=self.cfg.model.temperature,
                top_p=self.cfg.model.top_p,
                max_tokens=self.cfg.model.max_tokens,
                stop=["</tool_call>"],
            ):
                if decided is None:
                    buffer += delta
                    stripped = buffer.lstrip()
                    if _TOOL_OPEN in buffer:
                        decided = "tool"
                    elif stripped and not _TOOL_OPEN.startswith(stripped[:len(_TOOL_OPEN)]):
                        decided = "text"
                        final_text += buffer
                        yield {"type": "delta", "text": buffer}
                        buffer = ""
                elif decided == "text":
                    final_text += delta
                    yield {"type": "delta", "text": delta}
                else:  # decided == "tool": keep buffering silently
                    buffer += delta

            if decided == "tool" or _TOOL_OPEN in buffer:
                call = self._parse_tool_call(buffer)
                if call and self.cfg.agent.enable_tools:
                    name, args = call.get("name"), call.get("arguments", {}) or {}
                    fn = registry.get(name)
                    result = fn(**args) if fn else {"error": f"unknown tool '{name}'"}
                    messages.append({"role": "assistant", "content": buffer})
                    messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    yield {"type": "tool", "name": name, "summary": self._short(result)}
                    continue
                # malformed tool call -> surface the raw text instead of looping
                final_text += buffer
                yield {"type": "delta", "text": buffer}
                break
            else:
                if buffer:  # short final answer that never tripped the detector
                    final_text += buffer
                    yield {"type": "delta", "text": buffer}
                break
        else:
            if not final_text:
                msg = "(I could not complete the request within the step limit.)"
                yield {"type": "delta", "text": msg}

        self._record(user_message, final_text)

    def _record(self, user_message: str, answer: str) -> None:
        """Persist a turn to server-side history (CLI). In web mode the frontend
        owns the per-conversation history, so we don't duplicate it here."""
        if getattr(self, "_persist", True):
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": answer})

    def _export_content(self, hist=None) -> str | None:
        hist = self.history if hist is None else hist
        """The most recent substantial assistant message, to put in the sheet."""
        for m in reversed(hist):
            if m["role"] == "assistant" and len(m["content"].split()) >= 8:
                return m["content"]
        return None

    def _list_documents(self) -> list[str]:
        from .ingest import SUPPORTED_UPLOAD
        from pathlib import Path
        out = []
        d = self.cfg.corpus_dir
        for p in sorted(Path(d).glob("*")):
            if (p.is_file() and p.suffix.lower() in SUPPORTED_UPLOAD
                    and p.name != "knowledge_base.md"):
                out.append(p.name)
        return out

    def _doc_query(self, message: str) -> str | None:
        """Answer 'do you have X / list your documents' from the indexed files."""
        if _DOC_CONTENT_RE.search(message):
            return None  # it's a content/data question — let the model/tools handle it
        if not (_DOC_AVAIL_RE.search(message) or _FILE_RE.search(message)):
            return None
        files = self._list_documents()
        listing = bool(re.search(r"\b(list|what|which|all)\b", message, re.I)) and "access to" not in message.lower()

        # Specific document referenced (by filename or keywords)?
        fm = _FILE_RE.search(message)
        words = [w for w in re.findall(r"[a-zA-Z]{3,}", (fm.group(0) if fm else message).lower())
                 if w not in _DOC_STOP]
        best, score = None, 0
        for f in files:
            fl = f.lower()
            s = sum(1 for w in words if w in fl)
            if s > score:
                best, score = f, s

        if best and score > 0 and not listing:
            return (f"Yes — I have **{best}** indexed and ready. "
                    f"Ask me anything about it, or say \"summarise it\".")
        if listing or not words:
            if not files:
                return "I don't have any documents yet. Use the 📎 button to upload one (PDF, Word, Excel, CSV, text)."
            return ("I currently have access to these documents:\n\n"
                    + "\n".join(f"- {f}" for f in files))
        # Asked about a specific doc that isn't here.
        have = ("\n".join(f"- {f}" for f in files) if files else "- (none yet)")
        return (f"No — I don't have a document matching that. Here's what I do have:\n\n{have}\n\n"
                f"Upload it with the 📎 button and I'll index it instantly.")

    def _data_sheet_request(self, message: str) -> bool:
        return any(w in message.lower() for w in
                   ("debtor", "owing", "owe", "owed", "customer", "sales", "balance", "outstanding"))

    def _wants_export(self, message: str, hist=None) -> bool:
        hist = self.history if hist is None else hist
        explicit = bool(_EXPORT_RE.search(message))
        affirm = bool(_AFFIRM_RE.match(message.strip())) and len(message.split()) <= 4
        if affirm:
            last_ai = next((m["content"] for m in reversed(hist) if m["role"] == "assistant"), "")
            affirm = bool(re.search(r"\b(sheet|excel|spreadsheet|workbook|summary)\b", last_ai, re.I))
        if not (explicit or affirm):
            return False
        if explicit and self._data_sheet_request(message):
            return True  # data sheet built from cells — no prior summary needed
        return self._export_content(hist) is not None

    def _save_to_kb(self, user_message: str, hist=None) -> str | None:
        hist = self.history if hist is None else hist
        """Append a fact/note to the local knowledge base and index it."""
        content = _SAVE_STRIP.sub("", user_message).strip()
        if len(content.split()) < 2:
            # nothing substantive after the trigger -> save the last answer
            for m in reversed(hist):
                if m["role"] == "assistant" and m["content"].strip():
                    content = m["content"].strip()
                    break
        if len(content.split()) < 2:
            return None
        import datetime
        kb = self.cfg.corpus_dir / "knowledge_base.md"
        self.cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
        header = "" if kb.exists() else "# Saved knowledge base\n\n"
        stamp = datetime.date.today().isoformat()
        with open(kb, "a", encoding="utf-8") as fh:
            fh.write(f"{header}- ({stamp}) {content}\n")
        # (re)index the knowledge base file so it is immediately retrievable
        if self.index is None:
            self.index = RagIndex(self.cfg.index_dir)
            self.index.load()
        embedder = make_embedder(str(self.cfg.embedding_path), n_ctx=self.cfg.embedding.n_ctx,
                                 n_threads=self.cfg.model.n_threads)
        try:
            self.index.add_file(kb, embedder, self.cfg.rag.chunk_size, self.cfg.rag.chunk_overlap)
        finally:
            embedder.close()
        return content[:80] + ("…" if len(content) > 80 else "")

    @staticmethod
    def _short(result: dict) -> str:
        if "result" in result:
            return str(result["result"])
        if "error" in result:
            return f"error: {result['error']}"
        return json.dumps(result)[:80]

    def reset(self) -> None:
        self.history.clear()
