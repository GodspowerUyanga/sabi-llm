"""Full-featured public demo of sabi-v1 + sabi-yoruba-llm via a Gradio share
link — mirrors sabi serve's web UI (multi-conversation history, document
upload + summarization, Yoruba toggle with auto-detect, Auto/Think/Code mode)
but is deliberately restricted to Runtime.handle() / ThinkEngine / CodeEngine
only — never AgentLoop — so this public link can never read/write files or
run shell commands on this machine, regardless of what a visitor types.
(sabi serve's own "auto" mode routes non-small-talk through a restricted
*but still shell-executing* agent loop; that's fine on localhost for one
trusted user, not for an anonymous public link, so it's intentionally left
out here.)

Two more differences from sabi serve, both because this process is shared by
every visitor to the public link, not just one trusted local user:
  * Conversation history and uploaded-file text live in this session's
    browser state only (gr.State), never written to disk or to a store
    shared across visitors -- so nobody can browse another visitor's chats.
  * RAG is disabled (use_rag=False) to avoid one visitor's text surfacing in
    another's replies via semantic retrieval. The underlying Runtime's own
    short-term memory file is still shared across visitors (same as sabi
    serve across multiple local browser tabs) -- a low-severity, inherent
    trade-off of one shared model process, not a new one introduced here.

Not part of the sabi-llm package: a throwaway helper for sharing the model
with people who can't run it locally, not a submission deliverable. Model
inference still runs on this machine; only the HTTPS front door is public.
"""

import gradio as gr

from sabi import downloader, translate
from sabi.config import load_config
from sabi.filereader import read_any
from sabi.router import CODE, THINK
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
YORUBA_DIR = str(_cfg.abs_yoruba_model_path())

DESCRIPTION = (
    "**sabi-v1** — offline Coding Assistant (code generation, debugging, "
    "programming tutoring), with the **sabi-yoruba-llm** English↔Yoruba "
    "layer. Text-only demo: no file access, no shell execution — inference "
    "runs locally on the host machine, only this chat front door is public. "
    "Upload a document (PDF, Word, Excel, PowerPoint, CSV, HTML, image) and "
    "ask SABI to summarize or explain it."
)

EMPTY_CONV_STORE = {"order": [], "conversations": {}}


# --------------------------------------------------------------------- Yoruba
def _yoruba_active(message: str, toggle: bool) -> bool:
    if not runtime.config.yoruba_enabled:
        return False
    if not (toggle or translate.looks_like_yoruba(message)):
        return False
    return translate.available(YORUBA_DIR)


def _to_english(text: str, active: bool) -> str:
    return translate.to_english(text, YORUBA_DIR) if active else text


def _to_yoruba(text: str, active: bool) -> str:
    if not active:
        return text
    try:
        return translate.to_yoruba(text, YORUBA_DIR)
    except Exception as exc:  # noqa: BLE001
        return text + f"\n\n_(Yoruba translation unavailable right now: {exc})_"


# ------------------------------------------------------------- conversations
def _conv_title(text: str, limit: int = 40) -> str:
    text = text.strip().replace("\n", " ")
    return (text[:limit] + "…") if len(text) > limit else (text or "New chat")


def _history_context(conv_store: dict, cid: str, limit: int = 6) -> str:
    conv = conv_store["conversations"].get(cid) if cid else None
    if not conv or len(conv["messages"]) < 2:
        return ""
    prior = conv["messages"][:-1][-limit:]  # exclude the just-added current turn
    if not prior:
        return ""
    lines = [f"{m['role'].upper()}: {m['content']}" for m in prior]
    return "Recent conversation so far:\n" + "\n".join(lines)


def _dropdown_choices(conv_store: dict):
    return [(conv_store["conversations"][cid]["title"], cid) for cid in conv_store["order"]]


def new_chat(conv_store: dict, uploads_store: dict):
    import uuid
    cid = uuid.uuid4().hex[:10]
    conv_store = dict(conv_store)
    conv_store["conversations"] = {**conv_store["conversations"],
                                    cid: {"title": "New chat", "messages": []}}
    conv_store["order"] = [cid] + conv_store["order"]
    uploads_store = {**uploads_store, cid: []}
    return conv_store, uploads_store, cid, gr.update(choices=_dropdown_choices(conv_store), value=cid), []


def select_chat(cid: str, conv_store: dict):
    conv = conv_store["conversations"].get(cid) if cid else None
    return conv["messages"] if conv else []


def delete_chat(cid: str, conv_store: dict, uploads_store: dict):
    if not cid or cid not in conv_store["conversations"]:
        return conv_store, uploads_store, None, gr.update(choices=_dropdown_choices(conv_store)), []
    conv_store = dict(conv_store)
    conv_store["conversations"] = {k: v for k, v in conv_store["conversations"].items() if k != cid}
    conv_store["order"] = [x for x in conv_store["order"] if x != cid]
    uploads_store = {k: v for k, v in uploads_store.items() if k != cid}
    new_cid = conv_store["order"][0] if conv_store["order"] else None
    history = conv_store["conversations"][new_cid]["messages"] if new_cid else []
    return conv_store, uploads_store, new_cid, gr.update(choices=_dropdown_choices(conv_store), value=new_cid), history


# ------------------------------------------------------------------- uploads
def upload_file(file, cid: str, conv_store: dict, uploads_store: dict):
    if file is None:
        return uploads_store, "", conv_store, gr.update(), None
    if not cid or cid not in conv_store["conversations"]:
        conv_store, uploads_store, cid, dd, _hist = new_chat(conv_store, uploads_store)
    else:
        dd = gr.update()
    text = read_any(file, max_chars=8000)
    uploads_store = {**uploads_store, cid: [*uploads_store.get(cid, []), {"name": file.split("/")[-1], "text": text}]}
    preview = text[:500] + ("…" if len(text) > 500 else "")
    note = f"📎 **{file.split('/')[-1]}** attached ({len(text)} chars extracted)\n\n> {preview}"
    return uploads_store, note, conv_store, dd, cid


def _file_context(uploads_store: dict, cid: str, budget: int = 3000) -> str:
    files = uploads_store.get(cid or "", [])
    if not files:
        return ""
    parts = [f"--- FILE: {f['name']} ---\n{f['text'][:budget]}" for f in files[-3:]]
    return "Attached files the user uploaded:\n" + "\n\n".join(parts)


# --------------------------------------------------------------------- reply
def _stream_think_or_code(engine, request: str, context: str):
    buf, streamed = "", False
    for delta in engine.stream(request, context=context):
        streamed = True
        buf += delta
        yield buf
    if not streamed:
        yield engine.run(request, context=context).text


def _stream_auto(full: str):
    routing = runtime.router.route(full, runtime.prompts.get("router", ""))
    if routing.intent == CODE:
        yield from _stream_think_or_code(runtime.code, full, "")
        return
    if routing.intent == THINK:
        yield from _stream_think_or_code(runtime.think, full, "")
        return
    system = runtime.prompts.get("system", "") or None
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": full}]
    buf, streamed = "", False
    for delta in runtime.model.chat_stream(messages):
        streamed = True
        buf += delta
        yield buf
    if not streamed:
        yield runtime.model.generate(full, system=system).text


def respond(message: str, mode: str, yoruba_toggle: bool, cid: str,
            conv_store: dict, uploads_store: dict, chat_display: list):
    message = (message or "").strip()
    if not message:
        yield chat_display, conv_store, cid, gr.update(), ""
        return

    if not cid or cid not in conv_store["conversations"]:
        conv_store, uploads_store, cid, _dd, chat_display = new_chat(conv_store, uploads_store)

    conv = conv_store["conversations"][cid]
    conv["messages"].append({"role": "user", "content": message})
    if conv["title"] == "New chat":
        conv["title"] = _conv_title(message)

    chat_display = chat_display + [{"role": "user", "content": message},
                                    {"role": "assistant", "content": ""}]
    yield chat_display, conv_store, cid, gr.update(choices=_dropdown_choices(conv_store), value=cid), ""

    active = _yoruba_active(message, yoruba_toggle)
    eng_msg = _to_english(message, active)
    ctx = "\n\n".join(p for p in (_history_context(conv_store, cid),
                                   _file_context(uploads_store, cid)) if p)
    full = eng_msg + (("\n\n" + ctx) if ctx else "")

    final_text = ""
    try:
        if mode == "Think":
            gen_iter = _stream_think_or_code(runtime.think, eng_msg, ctx)
        elif mode == "Code":
            gen_iter = _stream_think_or_code(runtime.code, eng_msg, ctx)
        else:
            gen_iter = _stream_auto(full)
        for partial in gen_iter:
            final_text = partial
            if not active:  # Yoruba needs the complete text before translating
                chat_display[-1] = {"role": "assistant", "content": final_text}
                yield chat_display, conv_store, cid, gr.update(), ""
    except Exception as exc:  # noqa: BLE001 (includes ModelUnavailable)
        final_text = f"⚠ {exc}"

    final_text = _to_yoruba(final_text, active)
    chat_display[-1] = {"role": "assistant", "content": final_text}
    conv["messages"].append({"role": "assistant", "content": final_text})
    yield chat_display, conv_store, cid, gr.update(choices=_dropdown_choices(conv_store), value=cid), ""


# ----------------------------------------------------------------------- UI
with gr.Blocks(title="SABI — Coding Assistant (sabi-v1)") as demo:
    conv_store = gr.State(dict(EMPTY_CONV_STORE))
    uploads_store = gr.State({})
    current_cid = gr.State(None)

    gr.Markdown("## SABI — Coding Assistant (sabi-v1)\n" + DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Conversations")
            new_chat_btn = gr.Button("+ New chat")
            history_dd = gr.Dropdown(choices=[], label="History", interactive=True)
            delete_btn = gr.Button("🗑 Delete chat")
            gr.Markdown("### Upload a document")
            file_upload = gr.File(label="PDF, Word, Excel, PowerPoint, CSV, HTML, image", type="filepath")
            upload_note = gr.Markdown("")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, label=None)
            with gr.Row():
                mode = gr.Radio(["Auto", "Think", "Code"], value="Auto", label="Mode", scale=2)
                yoruba_toggle = gr.Checkbox(label="Reply in Yoruba", scale=1)
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Message SABI…", show_label=False, scale=5)
                send_btn = gr.Button("Send", variant="primary", scale=1)

    new_chat_btn.click(
        new_chat, [conv_store, uploads_store],
        [conv_store, uploads_store, current_cid, history_dd, chatbot],
    )
    history_dd.change(select_chat, [history_dd, conv_store], [chatbot]).then(
        lambda cid: cid, [history_dd], [current_cid],
    )
    delete_btn.click(
        delete_chat, [current_cid, conv_store, uploads_store],
        [conv_store, uploads_store, current_cid, history_dd, chatbot],
    )
    file_upload.upload(
        upload_file, [file_upload, current_cid, conv_store, uploads_store],
        [uploads_store, upload_note, conv_store, history_dd, current_cid],
    )

    send_args = dict(
        fn=respond,
        inputs=[msg_box, mode, yoruba_toggle, current_cid, conv_store, uploads_store, chatbot],
        outputs=[chatbot, conv_store, current_cid, history_dd, msg_box],
    )
    msg_box.submit(**send_args)
    send_btn.click(**send_args)

if __name__ == "__main__":
    demo.queue().launch(share=True)
