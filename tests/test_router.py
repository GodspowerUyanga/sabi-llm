"""Tests for the intent router heuristics (no model required)."""

from sabi.router import Router, THINK, CODE, CHAT


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
