"""Temporary public demo of sabi-v1's Coding Assistant persona (code
generation, debugging, programming tutoring) via a Gradio share link.

Deliberately routes through Runtime.handle() only — CODE/THINK/CHAT text
engines, never AgentLoop — so this public link can never read/write files or
run shell commands on this machine, regardless of what a visitor types.

Not part of the sabi-llm package: a throwaway helper for sharing the model
with people who can't run it locally, not a submission deliverable. Model
inference still runs on this machine; only the HTTPS front door is public.
"""

import gradio as gr

from sabi import downloader, translate
from sabi.config import load_config
from sabi.runtime import Runtime

# Fetch every model this demo can use up front (coder + Yoruba layer) so
# visitors never hit a mid-conversation "model not found" — same no-manual-
# step behavior as `sabi serve` and the CLI.
_cfg = load_config()
if not _cfg.abs_model_path().exists():
    print("No model found locally — downloading sabi-v1 (~2 GB)…")
    downloader.download_model(_cfg)
    print("sabi-v1 ready.\n")
if _cfg.yoruba_enabled and not translate.available(str(_cfg.abs_yoruba_model_path())):
    print("Downloading sabi-yoruba-llm (~635 MB)…")
    downloader.download_yoruba_model(_cfg)
    print("sabi-yoruba-llm ready.\n")

runtime = Runtime(_cfg).start()

DESCRIPTION = (
    "**sabi-v1** — offline Coding Assistant (code generation, debugging, "
    "programming tutoring). This public link is a text-only demo: no file "
    "access, no shell execution — inference runs locally on the host "
    "machine, only this chat front door is public."
)


def chat_fn(message: str, history):
    res = runtime.handle(message)
    if not res.get("ok"):
        return f"⚠ {res.get('error', 'request failed')}"
    return res.get("text", "")


demo = gr.ChatInterface(
    fn=chat_fn,
    title="SABI — Coding Assistant (sabi-v1)",
    description=DESCRIPTION,
    examples=[
        "Write a Python function that reverses a string",
        "Why does this raise IndexError: a = []; print(a[0])",
        "How do I get started learning Python?",
    ],
)

if __name__ == "__main__":
    demo.queue().launch(share=True)
