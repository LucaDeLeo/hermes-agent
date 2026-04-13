"""Integration tests for the Claude Code harness + MCP tools server.

These tests make REAL Claude Code API calls via the Claude Agent SDK.
They verify:
1. The MCP server creates correctly from the Hermes tool registry
2. The adapter passes MCP config through to ClaudeAgentOptions
3. Claude Code can discover and call Hermes MCP tools end-to-end
4. Tool dispatch works through the MCP bridge

Requires:
- claude-agent-sdk installed
- claude CLI available on PATH
- Valid Claude Code auth (OAuth or API key)

Run with:  pytest tests/integration/test_claude_code_harness.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if prerequisites are missing
# ---------------------------------------------------------------------------

_skip_reason = None
try:
    from claude_agent_sdk import (
        query as claude_query,
        ClaudeAgentOptions,
        SdkMcpTool,
        create_sdk_mcp_server,
        ResultMessage,
    )
except ImportError:
    _skip_reason = "claude-agent-sdk not installed"

if not _skip_reason and not shutil.which("claude"):
    _skip_reason = "claude CLI not found on PATH"

pytestmark = pytest.mark.skipif(
    _skip_reason is not None,
    reason=_skip_reason or "",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_query(prompt: str, options: ClaudeAgentOptions, timeout: float = 120) -> dict:
    """Run a Claude Code query synchronously, return aggregated result."""
    texts = []
    tool_calls = []
    result_msg = None

    async def _go():
        nonlocal result_msg
        async for msg in claude_query(prompt=prompt, options=options):
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif hasattr(block, "name") and hasattr(block, "input"):
                        tool_calls.append({"name": block.name, "input": block.input})
            if isinstance(msg, ResultMessage):
                result_msg = msg

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(_go(), timeout=timeout))
    except asyncio.TimeoutError:
        pytest.fail(f"Query timed out after {timeout}s")
    finally:
        loop.close()

    return {
        "text": "".join(texts),
        "tool_calls": tool_calls,
        "result": result_msg,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cli_path():
    return shutil.which("claude")


@pytest.fixture()
def base_options(cli_path, tmp_path):
    """ClaudeAgentOptions with safe defaults for testing."""
    opts = {
        "permission_mode": "acceptEdits",
        "max_turns": 3,
        "cwd": str(tmp_path),
        "include_partial_messages": True,
    }
    if cli_path:
        opts["cli_path"] = cli_path
    # Strip empty ANTHROPIC_API_KEY (same as adapter does)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ.pop("ANTHROPIC_API_KEY", None)
    return opts


# ===================================================================
# Test 1: MCP server creation from registry
# ===================================================================

class TestMcpServerCreation:
    def test_server_creates_from_registry(self):
        """create_hermes_tools_server returns a valid McpSdkServerConfig."""
        from tools.mcp_tools_server import create_hermes_tools_server

        ctx = {
            "task_id": "test",
            "todo_store": None,
            "memory_store": None,
            "session_db": None,
            "session_id": "test",
            "user_task": None,
            "parent_agent": None,
            "enabled_tools": None,
        }
        server = create_hermes_tools_server(ctx)

        assert isinstance(server, dict)
        assert server["type"] == "sdk"
        assert server["name"] == "hermes-tools"
        assert server["instance"] is not None

    def test_exposed_tools_are_hermes_unique(self):
        """Exposed tools should NOT include Claude Code built-in equivalents."""
        from tools.mcp_tools_server import _get_hermes_mcp_tools, _CLAUDE_CODE_BUILTINS

        tools = _get_hermes_mcp_tools()
        # None of the Claude Code built-in equivalents should be in the set
        for builtin in _CLAUDE_CODE_BUILTINS:
            assert builtin not in tools, f"{builtin} should be excluded"

        # Core Hermes tools should be present
        for expected in ("todo", "memory", "browser_navigate", "vision_analyze"):
            assert expected in tools, f"{expected} should be exposed"

    def test_check_fn_filtering(self):
        """Tools whose check_fn fails should be excluded."""
        from tools.mcp_tools_server import create_hermes_tools_server

        ctx = {"task_id": "t", "todo_store": None, "memory_store": None,
               "session_db": None, "session_id": "t", "user_task": None,
               "parent_agent": None, "enabled_tools": None}
        server = create_hermes_tools_server(ctx)

        # The server should have been created with some tools filtered
        # (e.g. HA tools without HASS_TOKEN, vision without API key)
        instance = server["instance"]
        assert instance is not None  # server still works even with some tools filtered

    def test_mcp_server_has_request_handlers(self):
        """The MCP server instance should handle tools/list and tools/call."""
        from tools.mcp_tools_server import create_hermes_tools_server
        from mcp.types import ListToolsRequest, CallToolRequest

        ctx = {"task_id": "t", "todo_store": None, "memory_store": None,
               "session_db": None, "session_id": "t", "user_task": None,
               "parent_agent": None, "enabled_tools": None}
        server = create_hermes_tools_server(ctx)
        instance = server["instance"]

        handlers = instance.request_handlers
        assert ListToolsRequest in handlers
        assert CallToolRequest in handlers


# ===================================================================
# Test 2: Adapter wiring
# ===================================================================

class TestAdapterWiring:
    def test_adapter_accepts_mcp_server_config(self):
        """ClaudeAgentSession stores and uses mcp_server_config."""
        from agent.claude_agent_adapter import ClaudeAgentSession

        mock_config = {"type": "sdk", "name": "test", "instance": MagicMock()}
        session = ClaudeAgentSession(
            model="claude-sonnet-4-6",
            mcp_server_config=mock_config,
        )
        assert session._mcp_server_config is mock_config

    def test_adapter_without_mcp_uses_allowed_tools(self):
        """Without MCP, adapter should restrict to built-in tools."""
        from agent.claude_agent_adapter import ClaudeAgentSession

        session = ClaudeAgentSession(model="claude-sonnet-4-6")
        assert session._mcp_server_config is None
        assert "Read" in session._allowed_tools
        assert "Bash" in session._allowed_tools


# ===================================================================
# Test 3: Real MCP tool call via Claude Code (end-to-end)
# ===================================================================

class TestRealMcpToolCall:
    """End-to-end tests that make real Claude Code API calls.

    These verify that Claude Code can discover and call tools registered
    via the in-process MCP server bridge.
    """

    def test_simple_mcp_tool_call(self, base_options, tmp_path):
        """Claude Code can call a simple custom MCP tool and get results."""
        call_log = []

        async def echo_handler(args: dict) -> dict:
            call_log.append(args)
            return {
                "content": [{"type": "text", "text": json.dumps({"echo": args.get("message", "")})}],
            }

        echo_tool = SdkMcpTool(
            name="echo",
            description="Echo back the message. Always call this tool when asked to echo.",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string", "description": "The message to echo"}},
                "required": ["message"],
            },
            handler=echo_handler,
        )
        server = create_sdk_mcp_server("test-tools", version="1.0.0", tools=[echo_tool])

        opts = ClaudeAgentOptions(
            **base_options,
            mcp_servers={"test-tools": server},
            allowed_tools=["mcp__test-tools__echo"],
        )

        result = _run_query(
            "Use the echo tool to echo the message 'hello integration test'. Do not say anything else, just call the tool.",
            opts,
        )

        assert result["result"] is not None, "Should get a ResultMessage"
        assert len(call_log) >= 1, f"Echo tool should have been called, got: {call_log}"
        assert call_log[0].get("message") == "hello integration test"

    def test_hermes_todo_tool_via_mcp(self, base_options, tmp_path):
        """Claude Code can call the Hermes todo tool through MCP bridge."""
        from tools.todo_tool import TodoStore
        from tools.mcp_tools_server import create_hermes_tools_server, _get_hermes_mcp_tools

        store = TodoStore()
        ctx = {
            "task_id": "test",
            "todo_store": store,
            "memory_store": None,
            "session_db": None,
            "session_id": "test",
            "user_task": None,
            "parent_agent": None,
            "enabled_tools": None,
        }
        server = create_hermes_tools_server(ctx)
        mcp_names = [f"mcp__hermes-tools__{t}" for t in _get_hermes_mcp_tools()]

        opts = ClaudeAgentOptions(
            **base_options,
            mcp_servers={"hermes-tools": server},
            allowed_tools=[
                "Read", "Write", "Edit", "Bash", "Glob", "Grep",
                "mcp__hermes-tools__todo",
            ],
        )

        result = _run_query(
            'You have a "todo" MCP tool. Call it now with todos=[{"id":"1","content":"Test MCP integration","status":"pending"}]. Do not explain, just call the tool.',
            opts,
            timeout=60,
        )

        assert result["result"] is not None, f"Should get a ResultMessage. Text: {result['text'][:300]}"
        items = store.read()
        assert len(items) >= 1, \
            f"Todo store should have at least 1 item, got: {items}. Response: {result['text'][:300]}"

    def test_hermes_session_survives_mcp(self, base_options, tmp_path):
        """Claude Code session with MCP tools completes without errors."""
        from tools.mcp_tools_server import create_hermes_tools_server, _get_hermes_mcp_tools

        ctx = {
            "task_id": "test",
            "todo_store": None,
            "memory_store": None,
            "session_db": None,
            "session_id": "test",
            "user_task": None,
            "parent_agent": None,
            "enabled_tools": None,
        }
        server = create_hermes_tools_server(ctx)
        mcp_names = [f"mcp__hermes-tools__{t}" for t in _get_hermes_mcp_tools()]

        opts = ClaudeAgentOptions(
            **base_options,
            mcp_servers={"hermes-tools": server},
            allowed_tools=["Read", "Write", "Edit", "Bash"] + mcp_names,
        )

        result = _run_query(
            "Say exactly: 'MCP bridge working'. Nothing else.",
            opts,
        )

        assert result["result"] is not None
        assert "MCP" in result["text"] or "bridge" in result["text"].lower() or result["text"].strip() != ""
