"""Tests for the permission manager (allow once/always sticks, reject)."""

from sabi.permissions import PermissionManager, Decision


def test_allow_once_proceeds_and_is_not_remembered():
    pm = PermissionManager(prompter=lambda k, d: Decision.ALLOW_ONCE)
    assert pm.request("/home", "create a directory: /home/x") is True
    assert "/home" not in pm.trusted


def test_reject_blocks():
    pm = PermissionManager(prompter=lambda k, d: Decision.DENY)
    assert pm.request("shell", "run: ls") is False


def test_allow_always_confirms_then_sticks():
    confirms = []
    pm = PermissionManager(
        prompter=lambda k, d: Decision.ALLOW_ALWAYS,
        confirmer=lambda d: confirms.append(d) or True,
    )
    assert pm.request("/home", "access /home/a") is True
    assert "/home" in pm.trusted
    assert len(confirms) == 1
    # second time for the same key: no prompt, no confirm — it sticks
    assert pm.request("/home", "access /home/b") is True
    assert len(confirms) == 1


def test_always_then_cancel_does_not_stick():
    pm = PermissionManager(
        prompter=lambda k, d: Decision.ALLOW_ALWAYS,
        confirmer=lambda d: False,
    )
    assert pm.request("/home", "access /home/a") is False
    assert "/home" not in pm.trusted


def test_auto_approve():
    pm = PermissionManager(auto_approve=True)
    assert pm.request("shell", "run: anything") is True


def test_should_prompt_modes():
    tui = PermissionManager(prompt_all=False)
    assert tui.should_prompt(external=True, is_shell=False) is True
    assert tui.should_prompt(external=False, is_shell=True) is True
    assert tui.should_prompt(external=False, is_shell=False) is False  # in-project: no prompt
    repl = PermissionManager(prompt_all=True)
    assert repl.should_prompt(external=False, is_shell=False) is True  # REPL prompts everything
    auto = PermissionManager(auto_approve=True)
    assert auto.should_prompt(external=True, is_shell=True) is False
