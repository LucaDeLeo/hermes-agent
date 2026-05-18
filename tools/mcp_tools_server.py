"""In-process MCP server that exposes Hermes-unique tools to Claude Code.

When the Claude Code harness is active, this module wraps tools from the
Hermes tool registry as ``SdkMcpTool`` instances and returns an
``McpSdkServerConfig`` that the adapter passes to ``ClaudeAgentOptions``.

Claude Code discovers the tools via MCP ``tools/list`` and calls them via
``tools/call``.  Each handler delegates to ``registry.dispatch()`` so all
existing tool logic is reused without duplication.

The server is created once at harness init time.  A mutable *context* dict
is captured by closures so late-initialized stores (MemoryStore, etc.) are
available to tool handlers at call time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools to expose — Hermes-unique, not duplicated by Claude Code built-ins.
# Claude Code already has: Read, Write, Edit, Bash, Glob, Grep, WebSearch,
# WebFetch.  We skip Hermes equivalents and only expose the extras.
# Derived from toolsets._HERMES_CORE_TOOLS minus Claude Code equivalents.
# ---------------------------------------------------------------------------

# Maps each Claude Code built-in tool to the Hermes tool name(s) that must
# be enabled for it to remain in the SDK's allowed_tools list. Source of
# truth for both directions — _CLAUDE_CODE_BUILTINS below is derived.
_BUILTIN_TO_HERMES: Dict[str, frozenset] = {
    "Read":      frozenset({"read_file"}),
    "Write":     frozenset({"write_file"}),
    "Edit":      frozenset({"patch"}),
    "Glob":      frozenset({"search_files"}),
    "Grep":      frozenset({"search_files"}),
    "Bash":      frozenset({"terminal", "process"}),
    "WebSearch": frozenset({"web_search"}),
    "WebFetch":  frozenset({"web_extract"}),
}

# Hermes tools NOT exposed via MCP because Claude Code already serves them
# (via _BUILTIN_TO_HERMES) or requires a TUI callback we don't bridge (clarify).
_CLAUDE_CODE_BUILTINS = (
    frozenset().union(*_BUILTIN_TO_HERMES.values()) | {"clarify"}
)


def filter_claude_builtins(valid_tool_names) -> list:
    """Return the built-in tool names that should be exposed to Claude Code.

    A built-in is allowed iff at least one of its Hermes equivalents is in
    *valid_tool_names*. If *valid_tool_names* is None or empty, returns the
    full built-in set (the un-filtered default that pre-2026-05 code shipped).
    """
    if not valid_tool_names:
        return list(_BUILTIN_TO_HERMES.keys())
    enabled = set(valid_tool_names)
    return [
        builtin
        for builtin, mapped in _BUILTIN_TO_HERMES.items()
        if mapped & enabled
    ]


def _build_hermes_mcp_tools() -> frozenset:
    from toolsets import _HERMES_CORE_TOOLS
    return frozenset(_HERMES_CORE_TOOLS) - _CLAUDE_CODE_BUILTINS


# Lazy-initialized on first use to avoid import-time dependency on toolsets.
_hermes_mcp_tools: frozenset | None = None


def _get_hermes_mcp_tools() -> frozenset:
    global _hermes_mcp_tools
    if _hermes_mcp_tools is None:
        _hermes_mcp_tools = _build_hermes_mcp_tools()
    return _hermes_mcp_tools


def _filter_mcp_tools(enabled_tools) -> frozenset:
    """Return MCP-exposable tools intersected with *enabled_tools*.

    *enabled_tools* may be a set/list/tuple of tool names, or None. None
    keeps every Hermes-unique tool exposed (matches the legacy behaviour).
    """
    universe = _get_hermes_mcp_tools()
    if not enabled_tools:
        return universe
    return universe & frozenset(enabled_tools)


# ---------------------------------------------------------------------------
# Per-tool context mapping — tools that need specific kwargs from agent state.
# ---------------------------------------------------------------------------

_TOOL_KWARGS: Dict[str, Callable[[dict], dict]] = {
    "todo":           lambda ctx: {"store": ctx.get("todo_store")},
    "memory":         lambda ctx: {"store": ctx.get("memory_store")},
    "session_search": lambda ctx: {
        "db": ctx.get("session_db"),
        "current_session_id": ctx.get("session_id"),
    },
    "delegate_task":  lambda ctx: {"parent_agent": ctx.get("parent_agent")},
    "execute_code":   lambda ctx: {
        "task_id": ctx.get("task_id"),
        "enabled_tools": ctx.get("enabled_tools"),
    },
}

_DEFAULT_KWARGS: Callable[[dict], dict] = lambda ctx: {
    "task_id": ctx.get("task_id"),
    "user_task": ctx.get("user_task"),
}


def create_hermes_tools_server(context: dict):
    """Build an in-process MCP server config from the Hermes tool registry.

    *context* is a **mutable** dict that handler closures read at call time.
    This allows late-initialized stores (e.g. MemoryStore) to be injected
    after server creation.

    Returns a ``McpSdkServerConfig`` suitable for
    ``ClaudeAgentOptions(mcp_servers={"hermes-tools": ...})``.

    Raises ``ImportError`` if ``claude_agent_sdk`` is not installed.
    """
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

    # Ensure all tool modules are imported so the registry is populated.
    import model_tools  # noqa: F401
    from tools.registry import registry

    # When context["enabled_tools"] is populated (set by AIAgent._create_harness
    # from self.valid_tool_names), restrict the MCP surface to that intersection.
    # Falling back to the full Hermes-unique universe preserves test-harness
    # callers that build the server without an agent.
    hermes_tools = _filter_mcp_tools(context.get("enabled_tools"))
    mcp_tools: list = []
    exposed: list = []

    # Deduplicate check_fn calls — same callable may be shared across a
    # toolset (e.g. all browser_* tools share check_browser_requirements).
    check_cache: Dict[Callable, bool] = {}

    for tool_name in sorted(hermes_tools):
        entry = registry._tools.get(tool_name)
        if entry is None:
            logger.debug("MCP tools server: %s not in registry, skipping", tool_name)
            continue

        if entry.check_fn:
            if entry.check_fn not in check_cache:
                try:
                    check_cache[entry.check_fn] = bool(entry.check_fn())
                except Exception:
                    check_cache[entry.check_fn] = False
            if not check_cache[entry.check_fn]:
                continue

        fn_schema = entry.schema
        description = fn_schema.get("description", entry.description or tool_name)
        input_schema = fn_schema.get("parameters", {"type": "object", "properties": {}})
        kwargs_resolver = _TOOL_KWARGS.get(tool_name, _DEFAULT_KWARGS)

        # Handler runs dispatch in a thread executor because registry.dispatch()
        # may call blocking I/O (browser automation, HTTP requests, etc.).
        def _make_handler(name: str, resolver: Callable[[dict], dict]):
            async def handler(args: dict) -> dict:
                loop = asyncio.get_running_loop()
                kwargs = resolver(context)
                result_str = await loop.run_in_executor(
                    None, lambda: registry.dispatch(name, args, **kwargs)
                )
                is_error = result_str.startswith('{"error":') if result_str else False
                return {
                    "content": [{"type": "text", "text": result_str}],
                    "is_error": is_error,
                }
            return handler

        mcp_tools.append(SdkMcpTool(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            handler=_make_handler(tool_name, kwargs_resolver),
        ))
        exposed.append(tool_name)

    logger.info("MCP tools server: exposing %d tools: %s", len(exposed), ", ".join(exposed))

    return create_sdk_mcp_server("hermes-tools", version="1.0.0", tools=mcp_tools)
