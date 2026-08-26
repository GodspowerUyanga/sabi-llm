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

from sabi.runtime import Runtime

runtime = Runtime().start()

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
