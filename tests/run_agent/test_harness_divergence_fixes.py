"""Tests for the five Claude Code harness divergence fixes.

Covers:
  Fix 1 — `valid_tool_names` filters both the SDK's built-in surface
          (``filter_claude_builtins``) and the MCP bridge
          (``_filter_mcp_tools``); ``_create_harness`` wires the result
          through to ``ClaudeAgentSession``.
  Fix 2 — caller-provided ``system_message`` is prepended to the harness
          system prompt instead of being silently dropped.
  Fix 3 — ``_run_claude_code_conversation`` honours ``skip_context_files``
          and discovers context files using ``TERMINAL_CWD`` instead of
          ``os.getcwd()``.
  Fix 4 — end-of-turn lifecycle parity with ``agent/conversation_loop.py``:
          drain ``/steer``, clear ``_stream_callback``, sync external
          memory, optionally spawn background review, fire
          ``on_session_end`` plugin hook.
  Fix 5 — ``_safe_call`` swallows callback exceptions so a TUI render
          error can't abort the SDK exchange.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(
    *,
    valid_tool_names=None,
    skip_context_files=True,
    skip_memory=True,
    session_id="test-harness-fixes",
):
    """Construct a minimal AIAgent for harness tests.

    Defaults to ``skip_context_files=True`` and ``skip_memory=True`` so
    no real state.db / SOUL.md / providers fire up during the test.
    Pass ``skip_context_files=False`` when the test specifically needs
    to exercise context-file behaviour.
    """
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_id=session_id,
            skip_context_files=skip_context_files,
            skip_memory=skip_memory,
        )
    if valid_tool_names is not None:
        agent.valid_tool_names = set(valid_tool_names)
    return agent


# ---------------------------------------------------------------------------
# Fix 1 — toolset filtering
# ---------------------------------------------------------------------------

class TestFilterClaudeBuiltins:
    def test_none_returns_full_set(self):
        from tools.mcp_tools_server import filter_claude_builtins, _BUILTIN_TO_HERMES
        assert filter_claude_builtins(None) == list(_BUILTIN_TO_HERMES.keys())

    def test_empty_returns_full_set(self):
        """Empty/falsy treated as 'no filter' so existing callers (tests that
        construct an agent without a real toolset) keep the legacy surface."""
        from tools.mcp_tools_server import filter_claude_builtins, _BUILTIN_TO_HERMES
        assert filter_claude_builtins(set()) == list(_BUILTIN_TO_HERMES.keys())

    def test_unmapped_tool_excludes_all_builtins(self):
        from tools.mcp_tools_server import filter_claude_builtins
        # `todo` has no Claude built-in equivalent — all built-ins dropped.
        assert filter_claude_builtins({"todo"}) == []

    def test_file_tools_enable_read_write_edit_glob_grep(self):
        from tools.mcp_tools_server import filter_claude_builtins
        out = set(filter_claude_builtins({"read_file", "write_file", "patch", "search_files"}))
        assert out == {"Read", "Write", "Edit", "Glob", "Grep"}

    def test_terminal_enables_bash(self):
        from tools.mcp_tools_server import filter_claude_builtins
        assert "Bash" in filter_claude_builtins({"terminal"})
        # `process` alone is also enough — both map to Bash.
        assert "Bash" in filter_claude_builtins({"process"})

    def test_web_tools_enable_websearch_webfetch(self):
        from tools.mcp_tools_server import filter_claude_builtins
        out = set(filter_claude_builtins({"web_search", "web_extract"}))
        assert out == {"WebSearch", "WebFetch"}

    def test_partial_enable(self):
        from tools.mcp_tools_server import filter_claude_builtins
        # web_search alone → WebSearch but NOT WebFetch
        out = set(filter_claude_builtins({"web_search"}))
        assert out == {"WebSearch"}


class TestFilterMcpTools:
    def test_none_returns_universe(self):
        from tools.mcp_tools_server import _filter_mcp_tools, _get_hermes_mcp_tools
        assert _filter_mcp_tools(None) == _get_hermes_mcp_tools()

    def test_filter_keeps_only_enabled(self):
        from tools.mcp_tools_server import _filter_mcp_tools
        out = _filter_mcp_tools({"todo", "memory", "session_search"})
        assert out == frozenset({"todo", "memory", "session_search"})

    def test_filter_drops_claude_builtin_equivalents(self):
        """Even when read_file/terminal/etc are in valid_tool_names, they
        must NOT appear in the MCP surface — they're served by Claude
        Code's built-ins, and exposing both would duplicate."""
        from tools.mcp_tools_server import _filter_mcp_tools
        out = _filter_mcp_tools({"read_file", "terminal", "web_search", "todo"})
        assert out == frozenset({"todo"})


class TestCreateHarnessWiresFilters:
    def test_create_harness_passes_filtered_allowed_tools(self):
        agent = _make_agent(valid_tool_names={"terminal", "read_file", "web_search"})
        # Force the harness adapter to be a MagicMock so we don't need the
        # real claude-agent-sdk available, and so we can inspect kwargs.
        with patch("agent.claude_agent_adapter.ClaudeAgentSession") as MockSession:
            mock_instance = MagicMock()
            MockSession.return_value = mock_instance
            agent._create_harness()
            assert MockSession.called
            kwargs = MockSession.call_args.kwargs
            assert "allowed_tools" in kwargs
            assert set(kwargs["allowed_tools"]) == {"Read", "Bash", "WebSearch"}

    def test_create_harness_no_tools_passes_empty_list(self):
        agent = _make_agent(valid_tool_names={"todo"})
        with patch("agent.claude_agent_adapter.ClaudeAgentSession") as MockSession:
            MockSession.return_value = MagicMock()
            agent._create_harness()
            kwargs = MockSession.call_args.kwargs
            assert kwargs["allowed_tools"] == []


# ---------------------------------------------------------------------------
# Fix 2 — system_message threading
# ---------------------------------------------------------------------------

class TestSystemMessageThreading:
    def test_system_message_prepended_to_effective_system(self):
        agent = _make_agent()
        captured = {}

        def _fake_send(user_message, **kwargs):
            captured["system_prompt"] = kwargs.get("system_prompt")
            return {
                "final_response": "ok",
                "interrupted": False,
                "completed": True,
                "session_id": "sdk-test",
                "messages": [],
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0,
            }

        agent._claude_agent_session = MagicMock()
        agent._claude_agent_session.send_message.side_effect = _fake_send

        agent._run_claude_code_conversation(
            user_message="hi",
            original_user_message="hi",
            system_message="CUSTOM-SYSTEM-PROMPT-TOKEN",
        )

        assert captured["system_prompt"] is not None
        # CUSTOM token must appear, and it must be at the very top so caller
        # intent dominates over any cwd-discovered context.
        assert captured["system_prompt"].startswith("CUSTOM-SYSTEM-PROMPT-TOKEN")

    def test_no_system_message_no_caller_text(self):
        agent = _make_agent()
        captured = {}
        agent._claude_agent_session = MagicMock()
        agent._claude_agent_session.send_message.side_effect = lambda u, **kw: (
            captured.setdefault("sp", kw.get("system_prompt")),
            {
                "final_response": "ok", "interrupted": False, "completed": True,
                "session_id": "x", "messages": [],
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0,
            },
        )[1]
        agent._run_claude_code_conversation("hi", "hi", system_message=None)
        # No caller system_message + skip_context_files=True + no memory ⇒
        # effective_system is None (handled as "no system prompt").
        assert captured["sp"] in (None, "")


# ---------------------------------------------------------------------------
# Fix 3 — skip_context_files + TERMINAL_CWD
# ---------------------------------------------------------------------------

class TestContextFileBehaviour:
    def test_skip_context_files_suppresses_context_loading(self):
        agent = _make_agent(skip_context_files=True)
        with patch("agent.prompt_builder.build_context_files_prompt") as mock_build:
            mock_build.return_value = "AGENTS.md content"
            agent._claude_agent_session = MagicMock()
            agent._claude_agent_session.send_message.return_value = {
                "final_response": "ok", "interrupted": False, "completed": True,
                "session_id": "x", "messages": [],
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0,
            }
            agent._run_claude_code_conversation("hi", "hi")
            mock_build.assert_not_called()

    def test_uses_terminal_cwd_not_process_cwd(self):
        agent = _make_agent(skip_context_files=False)
        captured = {}
        with patch("agent.prompt_builder.build_context_files_prompt") as mock_build:
            mock_build.side_effect = lambda **kw: captured.update(kw) or ""
            agent._claude_agent_session = MagicMock()
            agent._claude_agent_session.send_message.return_value = {
                "final_response": "ok", "interrupted": False, "completed": True,
                "session_id": "x", "messages": [],
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0,
            }
            with patch.dict(os.environ, {"TERMINAL_CWD": "/tmp/some-where"}):
                agent._run_claude_code_conversation("hi", "hi")
            assert mock_build.called
            assert captured.get("cwd") == "/tmp/some-where"

    def test_no_terminal_cwd_passes_none(self):
        agent = _make_agent(skip_context_files=False)
        captured = {}
        with patch("agent.prompt_builder.build_context_files_prompt") as mock_build:
            mock_build.side_effect = lambda **kw: captured.update(kw) or ""
            agent._claude_agent_session = MagicMock()
            agent._claude_agent_session.send_message.return_value = {
                "final_response": "ok", "interrupted": False, "completed": True,
                "session_id": "x", "messages": [],
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0,
            }
            env = {k: v for k, v in os.environ.items() if k != "TERMINAL_CWD"}
            with patch.dict(os.environ, env, clear=True):
                agent._run_claude_code_conversation("hi", "hi")
            # Caller didn't set TERMINAL_CWD ⇒ cwd is None (matches
            # build_context_files_prompt's own default).
            assert captured.get("cwd") is None


# ---------------------------------------------------------------------------
# Fix 4 — end-of-turn lifecycle parity
# ---------------------------------------------------------------------------

def _stub_session(agent):
    """Wire a stub SDK session that returns a canned successful turn."""
    sess = MagicMock()
    sess.send_message.return_value = {
        "final_response": "ok", "interrupted": False, "completed": True,
        "session_id": "sdk-x", "messages": [],
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "total_tokens": 0, "estimated_cost_usd": 0.0,
    }
    agent._claude_agent_session = sess
    return sess


class TestEndOfTurnLifecycle:
    def test_pending_steer_surfaces_in_result(self):
        agent = _make_agent()
        _stub_session(agent)
        agent._pending_steer = "please pause"
        result = agent._run_claude_code_conversation("hi", "hi")
        assert result.get("pending_steer") == "please pause"

    def test_stream_callback_cleared_after_turn(self):
        agent = _make_agent()
        _stub_session(agent)
        agent._stream_callback = lambda chunk: None
        agent._run_claude_code_conversation("hi", "hi")
        assert agent._stream_callback is None

    def test_sync_external_memory_called(self):
        agent = _make_agent()
        _stub_session(agent)
        with patch.object(agent, "_sync_external_memory_for_turn") as m:
            agent._run_claude_code_conversation("hi", "hello-user")
            assert m.called
            call_kwargs = m.call_args.kwargs
            assert call_kwargs.get("original_user_message") == "hello-user"
            assert call_kwargs.get("interrupted") is False

    def test_on_session_end_hook_fired(self):
        agent = _make_agent()
        _stub_session(agent)
        with patch("hermes_cli.plugins.invoke_hook") as inv:
            agent._run_claude_code_conversation("hi", "hi")
            # invoke_hook may also be called for post_llm_call — check that
            # AT LEAST one call is on_session_end.
            hooks_fired = [c.args[0] for c in inv.call_args_list]
            assert "on_session_end" in hooks_fired

    def test_background_review_spawned_at_nudge_interval(self):
        agent = _make_agent()
        _stub_session(agent)
        agent._memory_nudge_interval = 1
        agent._turns_since_memory = 5
        # Mirror conversation_loop guards: memory tool enabled + store wired.
        agent.valid_tool_names = {"memory"}
        agent._memory_store = MagicMock()
        agent._memory_store.format_for_system_prompt.return_value = ""
        with patch.object(agent, "_spawn_background_review") as spawn:
            agent._run_claude_code_conversation("hi", "hi")
            assert spawn.called
            kwargs = spawn.call_args.kwargs
            assert kwargs.get("review_memory") is True
            assert kwargs.get("review_skills") is False
        assert agent._turns_since_memory == 0

    def test_background_review_skipped_when_memory_tool_disabled(self):
        agent = _make_agent()
        _stub_session(agent)
        agent._memory_nudge_interval = 1
        agent._turns_since_memory = 5
        agent.valid_tool_names = set()  # `memory` tool NOT enabled
        agent._memory_store = MagicMock()
        agent._memory_store.format_for_system_prompt.return_value = ""
        with patch.object(agent, "_spawn_background_review") as spawn:
            agent._run_claude_code_conversation("hi", "hi")
            assert not spawn.called

    def test_background_review_skipped_when_no_memory_store(self):
        agent = _make_agent()
        _stub_session(agent)
        agent._memory_nudge_interval = 1
        agent._turns_since_memory = 5
        agent.valid_tool_names = {"memory"}
        agent._memory_store = None
        with patch.object(agent, "_spawn_background_review") as spawn:
            agent._run_claude_code_conversation("hi", "hi")
            assert not spawn.called

    def test_background_review_skipped_when_under_nudge(self):
        agent = _make_agent()
        _stub_session(agent)
        agent._memory_nudge_interval = 10
        agent._turns_since_memory = 1
        agent.valid_tool_names = {"memory"}
        agent._memory_store = MagicMock()
        agent._memory_store.format_for_system_prompt.return_value = ""
        with patch.object(agent, "_spawn_background_review") as spawn:
            agent._run_claude_code_conversation("hi", "hi")
            assert not spawn.called

    def test_background_review_skipped_on_interrupt(self):
        agent = _make_agent()
        sess = _stub_session(agent)
        sess.send_message.return_value = {
            **sess.send_message.return_value,
            "interrupted": True,
        }
        agent._memory_nudge_interval = 1
        agent._turns_since_memory = 5
        agent.valid_tool_names = {"memory"}
        agent._memory_store = MagicMock()
        agent._memory_store.format_for_system_prompt.return_value = ""
        with patch.object(agent, "_spawn_background_review") as spawn:
            agent._run_claude_code_conversation("hi", "hi")
            assert not spawn.called


# ---------------------------------------------------------------------------
# Fix 5 — callback safety
# ---------------------------------------------------------------------------

class TestSafeCall:
    def test_none_callback_is_noop(self):
        from agent.claude_agent_adapter import _safe_call
        # Should not raise.
        _safe_call(None, "x", label="test")

    def test_raising_callback_swallowed(self):
        from agent.claude_agent_adapter import _safe_call

        def boom(*a, **k):
            raise RuntimeError("UI is on fire")

        # Should not raise — exception is logged at debug level.
        _safe_call(boom, "anything", label="stream_delta")

    def test_callback_args_and_kwargs_forwarded(self):
        from agent.claude_agent_adapter import _safe_call
        seen = {}

        def cb(*a, **k):
            seen["args"] = a
            seen["kwargs"] = k

        _safe_call(cb, "x", "y", label="test", extra=1)
        assert seen["args"] == ("x", "y")
        # `label` is consumed by _safe_call — must NOT leak into kwargs.
        assert seen["kwargs"] == {"extra": 1}
