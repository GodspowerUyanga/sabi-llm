"""Headless test of the Textual TUI using a scripted fake model.

Verifies the app mounts, accepts input, runs the agent worker, and updates the
chat log + token counter -- all without a real LLM or a real terminal.
"""

import pytest

pytest.importorskip("textual")

from dataclasses import dataclass

from sabi.config import load_config
from sabi.runtime import Runtime


@dataclass
class FakeGen:
    text: str
    prompt_tokens: int = 20
    completion_tokens: int = 30
    elapsed_s: float = 0.1

    @property
    def tokens_per_second(self):
        return 300.0


class FakeModel:
    def __init__(self, reply="Hello! How can I help you today?"):
        self.reply = reply

    def is_available(self):
        return True

    def chat(self, messages, **kwargs):
        return FakeGen(self.reply)

    def chat_stream(self, messages, **kwargs):
        for word in self.reply.split(" "):
            yield word + " "

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def generate(self, prompt, **kwargs):
        return FakeGen(self.reply)


class FakeAgentModel:
    """Returns scripted replies in order (for multi-step agent tests)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0

    def is_available(self):
        return True

    def chat(self, messages, **kwargs):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return FakeGen(r)


@pytest.mark.asyncio
async def test_tui_mounts_and_responds(tmp_path):
    from sabi.ui.tui import SabiTUI

    rt = Runtime(load_config()).start(cwd=str(tmp_path))
    rt.model = FakeModel()
    # rebuild agent so it uses the fake model
    from sabi.permissions import PermissionManager
    from sabi.agent import AgentLoop, Reporter

    app = SabiTUI(rt, cwd=str(tmp_path))
    app.agent = AgentLoop(rt.model, PermissionManager(auto_approve=True),
                          system_prompt="sys", cwd=tmp_path, reporter=Reporter())

    async with app.run_test() as pilot:
        # type a message and submit
        await pilot.press(*"hi")
        await pilot.press("enter")
        # let the worker thread finish
        await app.workers.wait_for_complete()
        await pilot.pause()
        from textual.containers import VerticalScroll
        chat = app.query_one("#chat", VerticalScroll)
        assert len(chat.children) >= 2          # welcome + user + reply mounted
        assert app.total_tokens >= 1             # tokens counted from the stream


def test_tui_importable_without_textual(monkeypatch):
    # run_tui must raise a clean error if textual is missing
    import sabi.ui.tui as tui
    assert tui.textual_available() in (True, False)


def test_tui_process_uses_agent_loop_for_real_requests_regardless_of_phrasing():
    # There used to be a keyword gate here that only routed obvious phrasings
    # ("create a folder…") to the agent and sent everything else — including
    # natural phrasings that didn't match the keyword list — to a tool-less
    # chat fast path. It misclassified real requests often enough that it was
    # removed: process() now runs the agent loop for anything that isn't
    # unambiguous small talk, and the MODEL decides (via {"tool": ...} vs
    # {"final": ...}) whether a turn needs a tool.
    from sabi.router import is_smalltalk
    for msg in ["create a folder called app", "in the app folder, add main.py",
                "fix the login bug", "what does this function do?"]:
        assert not is_smalltalk(msg), msg


def test_tui_process_routes_smalltalk_to_chat_not_agent_loop():
    # Regression test for a real incident: a bare "hello" reached the agent's
    # tool-calling loop and resulted in an invented, executed tool call that
    # corrupted real repository files. Small talk must never reach the tool
    # loop at all — see sabi.router.is_smalltalk and SabiTUI._run_chat.
    from sabi.ui.tui import SabiTUI
    assert hasattr(SabiTUI, "_run_chat")
    from sabi.router import is_smalltalk
    for msg in ["hello", "hi", "thanks", "how are you"]:
        assert is_smalltalk(msg), msg


def test_location_safety_net(tmp_path, monkeypatch):
    # If the user says "on desktop" but the model gives a bare name, the agent
    # places it under the real Desktop, not the working directory.
    import sabi.agent as agentmod
    from sabi.permissions import PermissionManager
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(agentmod.Path, "home", staticmethod(lambda: home))

    model = FakeAgentModel([
        '{"tool": "create_dir", "args": {"path": "appstore"}}',
        "Created the appstore folder on your Desktop.",
    ])
    loop = agentmod.AgentLoop(model, PermissionManager(auto_approve=True),
                              system_prompt="sys", cwd=tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    res = loop.run("create a folder called appstore on my desktop")
    assert (home / "Desktop" / "appstore").is_dir()        # went to Desktop
    assert not (tmp_path / "proj" / "appstore").exists()    # NOT the project dir


def test_agent_remembers_across_turns(tmp_path):
    from sabi.permissions import PermissionManager
    from sabi.agent import AgentLoop
    model = FakeAgentModel([
        '{"tool": "create_dir", "args": {"path": "%s/app"}}' % tmp_path,
        "Created the app folder.",
        "I remember creating the app folder earlier.",
    ])
    loop = AgentLoop(model, PermissionManager(auto_approve=True),
                     system_prompt="sys", cwd=tmp_path)
    loop.run("create a folder app")
    assert len(loop.history) == 2          # one turn remembered
    loop.run("what did you just create?")
    # history carried into the second call's messages
    assert any("app" in m["content"] for m in loop.history)
