"""Tests for the agentic tool loop.

Uses a scripted fake model (no real LLM) to prove that, given a tool call, SABI
actually creates folders/files on disk and then returns a final answer.
"""

from dataclasses import dataclass

from sabi.agent import AgentLoop, ToolExecutor, parse_tool_call
from sabi.permissions import PermissionManager


@dataclass
class FakeGen:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0

    @property
    def tokens_per_second(self):
        return 0.0


class FakeModel:
    """Returns scripted replies in order, ignoring the messages."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def is_available(self):
        return True

    def chat(self, messages, **kwargs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return FakeGen(text=reply)


# ----------------------------------------------------------------- parser
def test_parse_plain_prose_is_none():
    assert parse_tool_call("All done! Your folder is ready.") is None


def test_parse_bare_json():
    call = parse_tool_call('{"tool": "create_dir", "args": {"path": "~/x"}}')
    assert call["tool"] == "create_dir"
    assert call["args"]["path"] == "~/x"


def test_parse_fenced_json():
    txt = 'Sure.\n```json\n{"tool":"create_dir","args":{"path":"a"}}\n```'
    call = parse_tool_call(txt)
    assert call and call["tool"] == "create_dir"


# --------------------------------------------------------------- executor
def test_executor_creates_dir(tmp_path):
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("create_dir", {"path": "project1"})
    assert ok
    assert (tmp_path / "project1").is_dir()


def test_executor_writes_file(tmp_path):
    ex = ToolExecutor(cwd=tmp_path)
    ok, _ = ex.execute("write_file", {"path": "notes/todo.txt", "content": "hi"})
    assert ok
    assert (tmp_path / "notes" / "todo.txt").read_text() == "hi"


def test_executor_blocks_dangerous_shell(tmp_path):
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("run_shell", {"command": "rm -rf /"})
    assert not ok
    assert "safety" in msg.lower()


# ------------------------------------------------------------------- loop
def test_agent_creates_folder_end_to_end(tmp_path):
    # First reply: a tool call. Second reply: final prose.
    model = FakeModel([
        '{"tool": "create_dir", "args": {"path": "appfolder"}}',
        "Done — I created the folder 'appfolder' for you.",
    ])
    pm = PermissionManager(auto_approve=True)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("create a folder called appfolder")
    assert res.ok
    assert (tmp_path / "appfolder").is_dir()
    assert any("create a directory" in a for a in res.actions)
    assert "appfolder" in res.answer


def test_agent_respects_denial(tmp_path):
    model = FakeModel([
        '{"tool": "create_dir", "args": {"path": "secret"}}',
        "Okay, I won't create it.",
    ])
    pm = PermissionManager(prompter=lambda t, d: __import__("sabi.permissions",
                           fromlist=["Decision"]).Decision.DENY)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("create a folder called secret")
    assert res.ok
    assert not (tmp_path / "secret").exists()
    assert any("DENIED" in a for a in res.actions)
