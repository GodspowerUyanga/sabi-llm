"""Tests for the permission manager (allow once/always, confirm/cancel)."""

from sabi.permissions import PermissionManager, Decision


def test_allow_once_proceeds():
    pm = PermissionManager(prompter=lambda t, d: Decision.ALLOW_ONCE)
    assert pm.request("create_dir", "create a directory: /tmp/x") is True
    assert "create_dir" not in pm.trusted  # not remembered


def test_deny_blocks():
    pm = PermissionManager(prompter=lambda t, d: Decision.DENY)
    assert pm.request("run_shell", "run: ls") is False


def test_allow_always_trusts_and_confirms():
    confirms = []
    pm = PermissionManager(
        prompter=lambda t, d: Decision.ALLOW_ALWAYS,
        confirmer=lambda d: confirms.append(d) or True,
    )
    assert pm.request("write_file", "write a file: a.txt") is True
    assert "write_file" in pm.trusted
    assert len(confirms) == 1
    # second time: no prompt, just confirm
    assert pm.request("write_file", "write a file: b.txt") is True
    assert len(confirms) == 2


def test_always_then_cancel():
    pm = PermissionManager(
        prompter=lambda t, d: Decision.ALLOW_ALWAYS,
        confirmer=lambda d: False,   # user cancels at confirm
    )
    assert pm.request("write_file", "write a file: a.txt") is False
    assert "write_file" in pm.trusted


def test_auto_approve():
    pm = PermissionManager(auto_approve=True)
    assert pm.request("run_shell", "run: anything") is True
