"""
Sabi application: FastAPI server (web UI + streaming API) and a CLI.

Endpoints
---------
GET  /                serves the single-page chat UI
GET  /api/health      model name, RAG size, RAM reading
GET  /api/stats       live RAM usage vs budget
POST /api/chat        streaming chat (Server-Sent Events)
POST /api/reset       clear conversation history

CLI (python -m sabi ...)
------------------------
  serve     run the web server
  chat      interactive terminal chat
  index     (re)build the RAG index from data/corpus
  bench     run the local profiler (see scripts/benchmark.py)
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import SabiAgent
from .config import load_config
from .embeddings import make_embedder
from .memory import available_gb, current_rss_gb
from .model import load_chat
from .rag import RagIndex

# FastAPI is a runtime dependency for `serve` only. Import it at module level so
# type hints resolve correctly, but keep it optional so the CLI commands that do
# not need a web server (index / chat / bench) still work if it is absent.
try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        message: str
        language: str | None = None
        history: list[dict] | None = None

    _FASTAPI_OK = True
except Exception:  # pragma: no cover
    _FASTAPI_OK = False


def _load_index(cfg) -> RagIndex | None:
    if not cfg.rag.enabled:
        return None
    idx = RagIndex(cfg.index_dir)
    return idx if idx.load() else idx  # empty index is fine; agent guards on size


def build_agent(cfg) -> SabiAgent:
    chat = load_chat(cfg)
    index = _load_index(cfg)
    return SabiAgent(cfg, chat, index)


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
def create_app(cfg=None):
    from pathlib import Path

    if not _FASTAPI_OK:
        raise RuntimeError("FastAPI is not installed. Run: pip install -r requirements.txt")

    cfg = cfg or load_config()
    agent = build_agent(cfg)
    web_dir = Path(__file__).resolve().parents[2] / "web"

    app = FastAPI(title="Sabi", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def home():
        html = (web_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/health")
    def health():
        return {
            "model": getattr(agent.chat, "name", "unknown"),
            "rag_chunks": agent.index.size if agent.index else 0,
            "languages": cfg.language.supported,
            "ram_gb": round(current_rss_gb(), 3),
            "ram_budget_gb": cfg.model.ram_budget_gb,
        }

    _peak = {"gb": round(current_rss_gb(), 3)}

    def _touch_peak():
        _peak["gb"] = round(max(_peak["gb"], current_rss_gb()), 3)
        return _peak["gb"]

    @app.get("/api/stats")
    def stats():
        return {
            "ram_gb": round(current_rss_gb(), 3),
            "available_gb": round(available_gb(), 3),
            "budget_gb": cfg.model.ram_budget_gb,
            "peak_gb": _touch_peak(),
        }

    @app.get("/api/compliance")
    def compliance_api():
        from .compliance import gather
        _touch_peak()
        return gather(cfg, agent, _peak["gb"])

    @app.get("/download/{name}")
    def download(name: str):
        from pathlib import Path as _P
        safe = _P(name).name
        path = cfg.abs_path("data/exports") / safe
        if not path.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        return FileResponse(str(path), filename=safe,
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.post("/api/reset")
    def reset():
        agent.reset()
        return {"ok": True}

    @app.post("/api/chat")
    def chat(body: ChatRequest):
        message = (body.message or "").strip()
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)

        def event_stream():
            for event in agent.run(message, language=body.language, history=body.history):
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'ram_gb': _touch_peak()})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        from pathlib import Path as _P
        from .ingest import SUPPORTED_UPLOAD
        from .rag import RagIndex as _RagIndex

        name = _P(file.filename or "document").name
        if _P(name).suffix.lower() not in SUPPORTED_UPLOAD:
            return JSONResponse(
                {"error": f"unsupported type; use {', '.join(sorted(SUPPORTED_UPLOAD))}"},
                status_code=400)

        cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
        dest = cfg.corpus_dir / name
        dest.write_bytes(await file.read())

        if agent.index is None:
            agent.index = _RagIndex(cfg.index_dir)
            agent.index.load()

        embedder = make_embedder(str(cfg.embedding_path), n_ctx=cfg.embedding.n_ctx,
                                 n_threads=cfg.model.n_threads)
        try:
            added = agent.index.add_file(dest, embedder, cfg.rag.chunk_size, cfg.rag.chunk_overlap)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        finally:
            embedder.close()

        if added == 0:
            return JSONResponse({"error": "no readable text found in that file"}, status_code=400)
        return {"ok": True, "name": name, "added": added, "total": agent.index.size}

    def _rebuild_index():
        from .rag import RagIndex as _RagIndex
        embedder = make_embedder(str(cfg.embedding_path), n_ctx=cfg.embedding.n_ctx,
                                 n_threads=cfg.model.n_threads)
        try:
            idx = _RagIndex(cfg.index_dir)
            n = idx.build(cfg.corpus_dir, embedder, cfg.rag.chunk_size, cfg.rag.chunk_overlap)
        finally:
            embedder.close()
        agent.index = idx
        return n

    @app.get("/api/documents")
    def list_documents():
        from .ingest import SUPPORTED_UPLOAD
        files = []
        for p in sorted(cfg.corpus_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_UPLOAD and p.name != "knowledge_base.md":
                files.append({"name": p.name, "kb": round(p.stat().st_size / 1024, 1)})
        return {"documents": files, "chunks": agent.index.size if agent.index else 0}

    @app.delete("/api/documents/{name}")
    def delete_document(name: str):
        from pathlib import Path as _P
        safe = _P(name).name
        target = cfg.corpus_dir / safe
        if not target.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        target.unlink()
        n = _rebuild_index()
        return {"ok": True, "deleted": safe, "chunks": n}

    @app.post("/api/documents/clear")
    def clear_documents():
        from .ingest import SUPPORTED_UPLOAD
        removed = 0
        for p in list(cfg.corpus_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_UPLOAD and p.name != "knowledge_base.md":
                p.unlink()
                removed += 1
        n = _rebuild_index()
        return {"ok": True, "removed": removed, "chunks": n}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_serve(args):
    import uvicorn
    cfg = load_config(args.config)
    app = create_app(cfg)
    print(f"\n  Sabi is running at http://{cfg.server.host}:{cfg.server.port}\n")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")


def cmd_chat(args):
    cfg = load_config(args.config)
    agent = build_agent(cfg)
    print(f"\n  {getattr(agent.chat, 'name', 'Sabi-1')} — offline. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if not msg:
            continue
        print("sabi> ", end="", flush=True)
        for event in agent.run(msg):
            kind = event.get("type")
            if kind == "delta":
                print(event["text"], end="", flush=True)
            elif kind == "tool":
                print(f"\n  [used {event['name']} → {event['summary']}]\n", end="", flush=True)
            elif kind == "sources":
                names = ", ".join(i["name"] for i in event["items"])
                print(f"\n  (sources: {names})\n", end="", flush=True)
        print("\n")


def cmd_index(args):
    cfg = load_config(args.config)
    embedder = make_embedder(str(cfg.embedding_path), n_ctx=cfg.embedding.n_ctx,
                             n_threads=cfg.model.n_threads)
    try:
        idx = RagIndex(cfg.index_dir)
        n = idx.build(cfg.corpus_dir, embedder, cfg.rag.chunk_size, cfg.rag.chunk_overlap)
    finally:
        embedder.close()
    print(f"Indexed {n} chunks from {cfg.corpus_dir} -> {cfg.index_dir}")


def cmd_bench(args):
    from scripts.benchmark import run_benchmark  # type: ignore
    run_benchmark(config_path=args.config, prompts_path=args.prompts, output=args.output)


def cmd_compliance(args):
    from .compliance import gather, render_markdown
    cfg = load_config(args.config)
    agent = build_agent(cfg)
    data = gather(cfg, agent, None)
    md = render_markdown(data)
    out = cfg.abs_path("COMPLIANCE.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved -> {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sabi", description="Sabi — offline enterprise assistant")
    parser.add_argument("--config", default=None, help="path to sabi.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="run the web server").set_defaults(func=cmd_serve)
    sub.add_parser("chat", help="interactive terminal chat").set_defaults(func=cmd_chat)
    sub.add_parser("index", help="build the RAG index").set_defaults(func=cmd_index)
    sub.add_parser("compliance", help="print + save the ADTC compliance report").set_defaults(func=cmd_compliance)

    b = sub.add_parser("bench", help="run the local profiler")
    b.add_argument("--prompts", default=None)
    b.add_argument("--output", default="benchmark_report.json")
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
