"""Tests for the intent router heuristics (no model required)."""

from sabi.router import Router, THINK, CODE, CHAT, is_smalltalk


def test_code_intent_detected():
    r = Router(model=None)
    routing = r.route("Write a Python function to sort a list and fix the bug")
    assert routing.intent == CODE


def test_think_intent_detected():
    r = Router(model=None)
    routing = r.route("Create a PRD and roadmap for our new product strategy")
    assert routing.intent == THINK


def test_chat_intent_default():
    r = Router(model=None)
    routing = r.route("hello there")
    assert routing.intent == CHAT


def test_code_fence_boosts_code():
    r = Router(model=None)
    routing = r.route("```\nprint(1)\n``` what does this do")
    assert routing.intent == CODE


# ------------------------------------------------------------- is_smalltalk
# Regression coverage for a real incident: a bare "hello" reached the agent's
# tool-calling loop and resulted in an invented, executed tool call that
# corrupted real repository files. is_smalltalk is the hard gate that keeps
# unambiguous greetings out of the tool loop entirely (see sabi/ui/chat.py,
# sabi/ui/tui.py).
def test_smalltalk_matches_bare_greetings():
    for msg in ["hello", "Hello", "hi", "Hi!", "hey", "thanks", "thank you",
                "ok", "okay", "bye", "how are you", "what's up", " hello  "]:
        assert is_smalltalk(msg), msg


def test_smalltalk_does_not_match_real_requests():
    for msg in ["create a file main.py that prints hello",
                "hi, can you create a folder called app",
                "fix this bug", "hello world program in python",
                "thanks, now delete the temp folder",
                "", "   "]:
        assert not is_smalltalk(msg), msg
