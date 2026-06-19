"""
Chat model backends for Sabi-1.

- ``LlamaChat`` wraps llama-cpp-python for offline GGUF inference on CPU.
- ``MockChat`` is a deterministic stand-in used by the test-suite and by
  ``--mock`` mode, so the full app (RAG, tools, UI, streaming) can be run and
  verified on any machine before the multi-gigabyte model is downloaded.

Both expose the same streaming ``chat`` interface that yields text deltas.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterator

Message = dict[str, str]  # {"role": "...", "content": "..."}


def budget_messages(messages, n_ctx, want_max, count_fn, margin=96):
    """Trim oldest history so (prompt + reply) fits the context window.

    Always keeps the system prompt (first) and the final message. Returns the
    (possibly trimmed) message list and a safe max_tokens for the reply.
    """
    msgs = list(messages)
    cap = min(want_max, 384)  # reserve at least this much for the reply
    while len(msgs) > 2 and count_fn(msgs) + cap + margin > n_ctx:
        del msgs[1]  # drop the oldest message after the system prompt
    used = count_fn(msgs)
    eff_max = max(64, min(want_max, n_ctx - used - margin))
    return msgs, eff_max


class LlamaChat:
    """Offline chat via llama-cpp-python (GGUF, CPU)."""

    def __init__(self, model_path: str, n_ctx: int, n_threads: int, n_batch: int,
                 use_mmap: bool, use_mlock: bool, n_gpu_layers: int = 0,
                 name: str = "Sabi-1", repeat_penalty: float = 1.18):
        from llama_cpp import Llama

        self.name = name
        self.n_ctx = n_ctx
        self.repeat_penalty = repeat_penalty
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _count_tokens(self, messages: list[Message]) -> int:
        text = "\n".join(m.get("content", "") for m in messages)
        try:
            return len(self._llama.tokenize(text.encode("utf-8"))) + 8 * len(messages)
        except Exception:
            return len(text) // 3 + 8 * len(messages)

    def chat(self, messages: list[Message], temperature: float, top_p: float,
             max_tokens: int, stop: list[str] | None = None) -> Iterator[str]:
        # Never exceed the context window — trim history and clamp the reply.
        msgs, eff_max = budget_messages(messages, self.n_ctx, max_tokens, self._count_tokens)
        try:
            stream = self._llama.create_chat_completion(
                messages=msgs,
                temperature=temperature,
                top_p=top_p,
                max_tokens=eff_max,
                repeat_penalty=self.repeat_penalty,
                stop=stop or [],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                piece = delta.get("content")
                if piece:
                    yield piece
        except Exception as exc:  # never crash the stream
            yield (f"\n\n_(I ran into a limit handling that: {exc}. "
                   f"Try a shorter question or start a new chat.)_")

    def close(self) -> None:
        try:
            del self._llama
        except Exception:
            pass


class MockChat:
    """Deterministic mock that imitates Sabi's behaviour without a real model.

    It understands the tool protocol enough to demonstrate the agent loop:
    if the latest user content contains arithmetic, it emits a calc tool call;
    otherwise it grounds a short answer in any provided context.
    """

    name = "Sabi-1 (mock)"
    _ARITH = re.compile(r"[-+/*]")

    def chat(self, messages: list[Message], temperature: float, top_p: float,
             max_tokens: int, stop: list[str] | None = None) -> Iterator[str]:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        last_tool = next((m["content"] for m in reversed(messages) if m["role"] == "tool"), "")

        # If a tool result is already present, summarise it as the final answer.
        if last_tool:
            text = f"Based on the computation, the result is {self._extract_result(last_tool)}."
            yield from self._stream(text)
            return

        # If the prompt asks for a number and tools are available, call calc.
        expr = self._find_expression(last_user)
        if expr and "calc" in system:
            yield '<tool_call>{"name": "calc", "arguments": {"expression": "' + expr + '"}}</tool_call>'
            return

        # Otherwise produce a grounded, concise answer.
        ctx = ""
        if "=== COMPANY CONTEXT ===" in system:
            ctx = system.split("=== COMPANY CONTEXT ===", 1)[1].split("=== END CONTEXT ===", 1)[0]
        if ctx.strip() and "No relevant" not in ctx:
            snippet = " ".join(ctx.split())[:160]
            text = f"From your documents: {snippet}…"
        else:
            text = (f"I am {self.name}. I run fully offline on your laptop. "
                    f"You asked: '{last_user[:80]}'. Share the relevant document and I will help.")
        yield from self._stream(text)

    # -- helpers -------------------------------------------------------------
    def _find_expression(self, text: str) -> str | None:
        m = re.search(r"[-+]?\d[\d,\.]*\s*[-+*/]\s*[-+]?\d[\d,\.]*(?:\s*[-+*/]\s*[-+]?\d[\d,\.]*)*", text)
        return m.group(0).replace(",", "").replace(" ", "") if m else None

    def _extract_result(self, tool_text: str) -> str:
        try:
            data = json.loads(tool_text)
            return str(data.get("result", tool_text))
        except Exception:
            return tool_text[:60]

    def _stream(self, text: str) -> Iterator[str]:
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.0)

    def close(self) -> None:
        pass


def load_chat(config) -> "LlamaChat | MockChat":
    """Return a LlamaChat if the model file exists, else a MockChat."""
    model_path = config.model_path
    if Path(model_path).exists():
        try:
            return LlamaChat(
                model_path=str(model_path),
                n_ctx=config.model.n_ctx,
                n_threads=config.model.n_threads,
                n_batch=config.model.n_batch,
                use_mmap=config.model.use_mmap,
                use_mlock=config.model.use_mlock,
                n_gpu_layers=config.model.n_gpu_layers,
                name=config.model.name,
                repeat_penalty=config.model.repeat_penalty,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[sabi] Could not load GGUF ({exc}); using MockChat.")
    else:
        print(f"[sabi] Model not found at {model_path}; using MockChat. "
              f"Run scripts/download_model.py to fetch the real Sabi-1.")
    return MockChat()
