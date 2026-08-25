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


def test_executor_create_dir_refuses_near_duplicate_of_existing_folder(tmp_path):
    (tmp_path / "MSC ARTIFICIAL INTELLIGENCE").mkdir()
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("create_dir", {"path": "MSc Artificial Intelligence"})
    assert not ok
    assert "MSC ARTIFICIAL INTELLIGENCE" in msg
    assert not (tmp_path / "MSc Artificial Intelligence").exists()
    # the real, existing folder must be untouched and still there
    assert (tmp_path / "MSC ARTIFICIAL INTELLIGENCE").is_dir()


def test_executor_list_dir_missing_path_suggests_close_match(tmp_path):
    (tmp_path / "projects").mkdir()
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("list_dir", {"path": "projcets"})
    assert not ok
    assert "projects" in msg


def test_executor_moves_file_into_existing_dir(tmp_path):
    (tmp_path / "config.py").write_text("X = 1\n")
    (tmp_path / "utils").mkdir()
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("move_file", {"src": "config.py", "dest": "utils"})
    assert ok, msg
    assert not (tmp_path / "config.py").exists()
    assert (tmp_path / "utils" / "config.py").read_text() == "X = 1\n"


def test_executor_moves_and_renames_file(tmp_path):
    (tmp_path / "old.py").write_text("hi")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("move_file", {"src": "old.py", "dest": "new.py"})
    assert ok, msg
    assert not (tmp_path / "old.py").exists()
    assert (tmp_path / "new.py").read_text() == "hi"


def test_executor_move_file_never_overwrites(tmp_path):
    (tmp_path / "a.py").write_text("A")
    (tmp_path / "b.py").write_text("B")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("move_file", {"src": "a.py", "dest": "b.py"})
    assert not ok
    assert "already exists" in msg.lower()
    assert (tmp_path / "a.py").read_text() == "A"
    assert (tmp_path / "b.py").read_text() == "B"


def test_executor_move_file_missing_source(tmp_path):
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("move_file", {"src": "nope.py", "dest": "elsewhere.py"})
    assert not ok
    assert "not found" in msg.lower()


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


def test_executor_blocks_deletion_commands(tmp_path):
    # Regression test for a real incident: after finishing an unrelated task,
    # the model deleted the very file it had just created via run_shell +
    # rm, when nobody asked it to delete anything. Deletion has no supported
    # tool by design; this closes the run_shell workaround at the code level.
    (tmp_path / "keep.txt").write_text("data")
    ex = ToolExecutor(cwd=tmp_path)
    for cmd in ["rm keep.txt", "rm -f keep.txt", "rmdir sub", "del keep.txt",
                "cd /tmp && rm keep.txt", "echo hi; rm keep.txt"]:
        ok, msg = ex.execute("run_shell", {"command": cmd})
        assert not ok, cmd
        assert "safety" in msg.lower(), cmd
    assert (tmp_path / "keep.txt").exists()


def test_executor_does_not_block_unrelated_words(tmp_path):
    # The deletion-command regex must not misfire on words that merely
    # contain "rm" as a substring (confirm, warm, term, ...).
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("run_shell", {"command": "echo confirm the warm term"})
    assert ok, msg


def test_executor_search_finds_match_across_subdirs(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def calculate_total():\n    return 1\n")
    (tmp_path / "pkg" / "b.py").write_text("x = 2\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("search_files", {"pattern": "calculate_total"})
    assert ok
    assert "pkg/a.py:1:" in msg
    assert "b.py" not in msg


def test_executor_search_skips_junk_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("needle\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("search_files", {"pattern": "needle"})
    assert not ok


def test_executor_search_no_match(tmp_path):
    (tmp_path / "a.py").write_text("hello\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("search_files", {"pattern": "nope"})
    assert not ok
    assert "No matches" in msg


def test_executor_edit_replaces_unique_snippet(tmp_path):
    f = tmp_path / "billing.py"
    f.write_text("def total(x):\n    return x\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("edit_file", {
        "path": "billing.py", "old_string": "    return x", "new_string": "    return x * 1.1",
    })
    assert ok
    assert f.read_text() == "def total(x):\n    return x * 1.1\n"


def test_executor_edit_rejects_missing_snippet(tmp_path):
    f = tmp_path / "billing.py"
    f.write_text("def total(x):\n    return x\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("edit_file", {
        "path": "billing.py", "old_string": "not in file", "new_string": "x",
    })
    assert not ok
    assert "not found" in msg.lower()


def test_executor_edit_rejects_ambiguous_snippet(tmp_path):
    f = tmp_path / "billing.py"
    f.write_text("x = 1\nx = 1\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("edit_file", {
        "path": "billing.py", "old_string": "x = 1", "new_string": "x = 2",
    })
    assert not ok
    assert "not unique" in msg.lower()


def test_executor_read_file_with_offset_and_limit(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("read_file", {"path": "big.py", "offset": 3, "limit": 2})
    assert ok
    assert "line3" in msg and "line4" in msg
    assert "line5" not in msg
    assert "more line(s)" in msg


def test_executor_read_file_offset_past_end(tmp_path):
    f = tmp_path / "small.py"
    f.write_text("only line\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("read_file", {"path": "small.py", "offset": 50})
    assert not ok
    assert "past the end" in msg


def test_executor_edit_replace_all(tmp_path):
    f = tmp_path / "billing.py"
    f.write_text("x = 1\nx = 1\n")
    ex = ToolExecutor(cwd=tmp_path)
    ok, msg = ex.execute("edit_file", {
        "path": "billing.py", "old_string": "x = 1", "new_string": "x = 2",
        "replace_all": True,
    })
    assert ok
    assert f.read_text() == "x = 2\nx = 2\n"


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
    from sabi.permissions import Decision
    model = FakeModel([
        '{"tool": "create_dir", "args": {"path": "secret"}}',
        "Okay, I won't create it.",
    ])
    # prompt_all=True makes the agent ask for every action (the simple-REPL mode)
    pm = PermissionManager(prompter=lambda k, d: Decision.DENY, prompt_all=True)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("create a folder called secret")
    assert res.ok
    assert not (tmp_path / "secret").exists()
    assert any("DENIED" in a for a in res.actions)


class FakeRetriever:
    """Records what AgentLoop indexes, without a real embedder/vector store."""

    def __init__(self):
        self.indexed = []

    def add_text(self, text, source=""):
        self.indexed.append((source, text))


def test_agent_indexes_each_turn_into_retriever(tmp_path):
    model = FakeModel(["Hi there! How can I help?"])
    pm = PermissionManager(auto_approve=True)
    retriever = FakeRetriever()
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path, retriever=retriever)
    loop.run("hello")
    assert len(retriever.indexed) == 1
    source, text = retriever.indexed[0]
    assert source == "conversation"
    assert "hello" in text.lower()
    assert "Hi there" in text


def test_agent_shows_real_data_when_model_repeats_list_dir_instead_of_finishing(tmp_path):
    # Reproduces a real failure: user asks "what files are on my desktop", the
    # model successfully lists the directory, then — instead of replying with
    # {"final": "..."} describing what it found — emits the exact same
    # list_dir call again. The loop-guard used to end the turn with a
    # content-free "Here is what I completed: OK: list a directory: X", which
    # never told the user what was actually IN the directory. It must now
    # surface the real listing instead.
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "photos").mkdir()
    call = '{"tool": "list_dir", "args": {"path": "."}}'
    model = FakeModel([call, call, call])  # repeats the same call every time
    pm = PermissionManager(auto_approve=True)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("what files exist in this folder")
    assert res.ok
    assert "notes.txt" in res.answer
    assert "photos" in res.answer
    assert "Here is what I completed" not in res.answer


def test_agent_enriches_vague_final_answer_with_real_listing(tmp_path):
    # Reproduces a real, different failure from the one above: the model
    # DOES produce a well-formed {"final": ...} (so the repeat/dedup fallback
    # never triggers) but it's a bare count with no names — "There are 14
    # folders in your Desktop directory." — after a successful list_dir with
    # 14 real entries. The answer must still surface what was actually found.
    for i in range(1, 15):
        (tmp_path / f"Project{i}").mkdir()
    model = FakeModel([
        '{"tool": "list_dir", "args": {"path": "."}}',
        "There are 14 folders in your Desktop directory.",
    ])
    pm = PermissionManager(auto_approve=True)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("list how many folders are in there")
    assert res.ok
    assert "There are 14 folders" in res.answer  # original answer kept
    assert "Project1" in res.answer and "Project14" in res.answer  # real names appended


def test_agent_does_not_duplicate_when_answer_already_names_items(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "photos").mkdir()
    model = FakeModel([
        '{"tool": "list_dir", "args": {"path": "."}}',
        "This folder contains notes.txt and a photos folder.",
    ])
    pm = PermissionManager(auto_approve=True)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("what's in this folder")
    assert res.ok
    # already named both real entries -> nothing appended, answer untouched
    assert res.answer == "This folder contains notes.txt and a photos folder."


# ----------------------------------------------------- conversation history
def test_agent_loop_seeds_and_uses_initial_history(tmp_path):
    # Regression test for a real incident: sabi serve builds a fresh
    # AgentLoop per HTTP request, so without seeding prior turns a follow-up
    # like "list them" has no idea what "them" refers to (and, live-tested,
    # didn't even re-run list_dir — it just guessed). initial_history is how
    # a per-request caller restores continuity that the TUI/terminal chat
    # get for free by keeping one AgentLoop alive for the whole session.
    captured = {}

    class RecordingModel(FakeModel):
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            return super().chat(messages, **kwargs)

    model = RecordingModel(["Sure, following up on that."])
    pm = PermissionManager(auto_approve=True)
    prior = [
        {"role": "user", "content": "list my desktop folders"},
        {"role": "assistant", "content": "There are 2 folders: Work, Personal"},
    ]
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path, initial_history=prior)
    assert loop.history == prior
    loop.run("list them")
    roles_and_content = [(m["role"], m["content"]) for m in captured["messages"]]
    assert ("user", "list my desktop folders") in roles_and_content
    assert ("assistant", "There are 2 folders: Work, Personal") in roles_and_content


def test_agent_loop_truncates_initial_history_to_last_8():
    model = FakeModel(["ok"])
    pm = PermissionManager(auto_approve=True)
    long_history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    loop = AgentLoop(model, pm, system_prompt="sys", initial_history=long_history)
    assert len(loop.history) == 8
    assert loop.history[0]["content"] == "turn 12"  # last 8 of 20 (indices 12..19)


def test_agent_in_project_action_not_prompted(tmp_path):
    # Default TUI manager only prompts for external/shell, so an in-project
    # create runs without consulting the prompter.
    from sabi.permissions import Decision
    model = FakeModel([
        '{"tool": "create_dir", "args": {"path": "proj"}}',
        "Created the folder.",
    ])
    calls = []
    pm = PermissionManager(prompter=lambda k, d: calls.append(k) or Decision.DENY,
                           prompt_all=False)
    loop = AgentLoop(model, pm, system_prompt="sys", cwd=tmp_path)
    res = loop.run("create a folder called proj")
    assert (tmp_path / "proj").is_dir()
    assert calls == []   # never prompted for an in-project path
