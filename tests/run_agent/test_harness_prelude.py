"""Tests for the harness history prelude builder.

Covers ``_build_harness_history_prelude`` in run_agent.py — the belt-and-
suspenders fallback that feeds prior conversation into ``user_message``
when the Claude Code SDK's ``resume`` chain is broken.
"""

from __future__ import annotations

import pytest

from run_agent import _build_harness_history_prelude, _flatten_message_text


class TestFlattenMessageText:
    def test_string_passthrough(self):
        assert _flatten_message_text("hello") == "hello"

    def test_none(self):
        assert _flatten_message_text(None) == ""

    def test_multimodal_text_parts_only(self):
        content = [
            {"type": "text", "text": "Look at this:"},
            {"type": "image", "source": {"data": "AAAA" * 100}},
            {"type": "text", "text": "What colour?"},
        ]
        out = _flatten_message_text(content)
        assert "Look at this" in out
        assert "What colour" in out
        assert "AAAA" not in out

    def test_multimodal_ignores_tool_use(self):
        content = [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "name": "Read", "input": {"path": "x"}},
            {"type": "tool_result", "content": "secret"},
        ]
        out = _flatten_message_text(content)
        assert "Let me check" in out
        assert "secret" not in out
        assert "Read" not in out

    def test_list_of_strings(self):
        assert _flatten_message_text(["a", "b"]) == "a\nb"


class TestPreludeBuilder:
    def test_empty_history(self):
        prelude, meta = _build_harness_history_prelude([])
        assert prelude == ""
        assert meta == {"turns": 0, "chars": 0}

    def test_all_tool_messages_filtered(self):
        msgs = [
            {"role": "tool", "content": "r1"},
            {"role": "system", "content": "s1"},
        ]
        prelude, meta = _build_harness_history_prelude(msgs)
        assert prelude == ""
        assert meta["turns"] == 0

    def test_empty_assistant_dropped(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]
        prelude, meta = _build_harness_history_prelude(msgs)
        assert meta["turns"] == 1
        assert "<prior_conversation>" in prelude

    def test_error_marker_dropped(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "oops", "_error": True},
            {"role": "user", "content": "again"},
        ]
        prelude, meta = _build_harness_history_prelude(msgs)
        assert meta["turns"] == 2
        assert "oops" not in prelude
        assert "again" in prelude

    def test_under_cap_full_transcript(self):
        msgs = [
            {"role": "user", "content": "Remember 73529."},
            {"role": "assistant", "content": "Remembered."},
            {"role": "user", "content": "What number?"},
        ]
        prelude, meta = _build_harness_history_prelude(msgs)
        assert meta["turns"] == 3
        assert "<prior_conversation>" in prelude
        assert "</prior_conversation>" in prelude
        assert "earlier turns omitted" not in prelude
        assert "Remember 73529" in prelude
        assert "User:" in prelude and "Assistant:" in prelude
        assert "User's latest message:" in prelude

    def test_over_cap_tail_only(self):
        msgs = [
            {"role": "user", "content": "A" * 500},
            {"role": "assistant", "content": "B" * 500},
            {"role": "user", "content": "MARKER"},
        ]
        prelude, meta = _build_harness_history_prelude(
            msgs, max_chars=200, tail_turns=1,
        )
        assert meta["turns"] == 1
        assert "earlier turns omitted" in prelude
        assert "MARKER" in prelude
        # The bulk of the earlier AAAA/BBBB content must be omitted.
        assert "A" * 400 not in prelude
        assert "B" * 400 not in prelude

    def test_over_cap_tail_turns_min_one(self):
        """tail_turns=0 should still emit at least 1 (we never return empty when kept)."""
        msgs = [
            {"role": "user", "content": "X" * 10_000},
            {"role": "assistant", "content": "Y" * 10_000},
        ]
        prelude, meta = _build_harness_history_prelude(
            msgs, max_chars=100, tail_turns=0,
        )
        assert meta["turns"] >= 1
        assert "earlier turns omitted" in prelude

    def test_multimodal_content_flattened(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Q?"},
                {"type": "image", "source": {"data": "ZZZZ" * 100}},
            ]},
            {"role": "assistant", "content": "A!"},
        ]
        prelude, meta = _build_harness_history_prelude(msgs)
        assert "Q?" in prelude
        assert "A!" in prelude
        assert "ZZZZ" not in prelude
        assert meta["turns"] == 2

    def test_envelope_format(self):
        msgs = [{"role": "user", "content": "hi"}]
        prelude, _ = _build_harness_history_prelude(msgs)
        # Must end with the user-message handoff so the model knows where
        # the prior context stops and the new question begins.
        assert prelude.rstrip().endswith("User's latest message:")

    def test_non_dict_messages_skipped(self):
        msgs = [
            "not a dict",
            {"role": "user", "content": "real"},
            42,
        ]
        prelude, meta = _build_harness_history_prelude(msgs)
        assert meta["turns"] == 1
        assert "real" in prelude
