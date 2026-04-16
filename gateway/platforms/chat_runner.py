"""Shared agent creation + SSE streaming for chat endpoints.

Extracted from ``api_server.py`` so the web dashboard can reuse the same
agent pipeline without duplicating non-obvious edge cases (delta-None
filtering, tool-progress tagged tuples, disconnect interruption, session
rotation, etc.).

Both ``api_server.py`` (aiohttp) and ``web_server.py`` (FastAPI) call
into these functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL_SECONDS = 30


# ------------------------------------------------------------------
# Agent creation
# ------------------------------------------------------------------

def create_chat_agent(
    *,
    platform: str = "dashboard",
    session_id: Optional[str] = None,
    stream_delta_callback: Optional[Callable] = None,
    tool_progress_callback: Optional[Callable] = None,
    tool_complete_callback: Optional[Callable] = None,
    ephemeral_system_prompt: Optional[str] = None,
    session_db: Any = None,
) -> Any:
    """Create an AIAgent from the current config.

    Mirrors ``api_server._create_agent()`` — resolves model, provider,
    toolsets, and fallback chain from config.yaml / env.
    """
    from run_agent import AIAgent
    from gateway.run import (
        _resolve_runtime_agent_kwargs,
        _resolve_gateway_model,
        _load_gateway_config,
        GatewayRunner,
    )
    from hermes_cli.tools_config import _get_platform_tools

    runtime_kwargs = _resolve_runtime_agent_kwargs()
    model = _resolve_gateway_model()

    user_config = _load_gateway_config()
    # _get_platform_tools looks up platform_toolsets.<platform> in config.
    # "dashboard" isn't a registered platform — fall back to "cli" toolset
    # which gives the full interactive tool surface.
    try:
        enabled_toolsets = sorted(_get_platform_tools(user_config, platform))
    except KeyError:
        enabled_toolsets = sorted(_get_platform_tools(user_config, "cli"))

    max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
    fallback_model = GatewayRunner._load_fallback_model()

    if session_db is None:
        try:
            from hermes_state import SessionDB
            session_db = SessionDB()
        except Exception:
            pass

    agent = AIAgent(
        model=model,
        **runtime_kwargs,
        max_iterations=max_iterations,
        quiet_mode=True,
        verbose_logging=False,
        ephemeral_system_prompt=ephemeral_system_prompt or None,
        enabled_toolsets=enabled_toolsets,
        session_id=session_id,
        platform=platform,
        stream_delta_callback=stream_delta_callback,
        tool_progress_callback=tool_progress_callback,
        tool_complete_callback=tool_complete_callback,
        session_db=session_db,
        fallback_model=fallback_model,
    )
    return agent


# ------------------------------------------------------------------
# Stream callback factory
# ------------------------------------------------------------------

_TOOL_RESULT_MAX_CHARS = 5000


def make_stream_callbacks() -> Tuple[Callable, Callable, Callable, queue.Queue]:
    """Create the (on_delta, on_tool_progress, on_tool_complete, stream_queue) quad.

    ``on_delta`` filters out ``None`` (the agent fires it to signal the
    CLI display to close its response box before tool execution, but SSE
    uses ``None`` as end-of-stream sentinel).

    ``on_tool_progress`` pushes tagged tuples for custom SSE events so
    tool markers are NOT stored in conversation history (see #6972).
    Also forwards ``tool.completed`` events with duration/error status.

    ``on_tool_complete`` sends full tool input/output for expandable cards.
    """
    stream_q: queue.Queue = queue.Queue()

    def on_delta(delta):
        if delta is not None:
            stream_q.put(delta)

    def on_tool_progress(event_type, name, preview, args, **kwargs):
        if name and name.startswith("_"):
            return
        if event_type == "tool.started":
            try:
                from agent.display import get_tool_emoji
                emoji = get_tool_emoji(name)
            except Exception:
                emoji = "⚡"
            payload = {
                "tool": name,
                "emoji": emoji,
                "label": preview or name,
            }
            tool_call_id = kwargs.get("tool_call_id")
            if tool_call_id:
                payload["tool_call_id"] = tool_call_id
            stream_q.put(("__tool_progress__", payload))
        elif event_type == "tool.completed":
            stream_q.put(("__tool_completed__", {
                "tool": name,
                "duration": kwargs.get("duration"),
                "is_error": kwargs.get("is_error", False),
            }))

    def on_tool_complete(tool_call_id, name, args, result):
        stream_q.put(("__tool_result__", {
            "tool_call_id": tool_call_id or "",
            "name": name or "",
            "input": str(args)[:_TOOL_RESULT_MAX_CHARS],
            "output": str(result)[:_TOOL_RESULT_MAX_CHARS],
        }))

    return on_delta, on_tool_progress, on_tool_complete, stream_q


# ------------------------------------------------------------------
# Agent execution
# ------------------------------------------------------------------

async def run_agent_async(
    agent: Any,
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    agent_ref: Optional[list] = None,
) -> Tuple[dict, dict]:
    """Run ``agent.run_conversation()`` in a thread executor.

    Returns ``(result_dict, usage_dict)``.  If *agent_ref* is a
    one-element list, the agent is stored at ``agent_ref[0]`` before
    execution so callers can interrupt it from another coroutine.
    """
    loop = asyncio.get_running_loop()

    def _run():
        if agent_ref is not None:
            agent_ref[0] = agent
        result = agent.run_conversation(
            user_message=user_message,
            conversation_history=conversation_history or [],
            task_id="default",
        )
        usage = {
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
        }
        return result, usage

    return await loop.run_in_executor(None, _run)


# ------------------------------------------------------------------
# SSE event generator
# ------------------------------------------------------------------

async def sse_event_stream(
    stream_queue: queue.Queue,
    agent_task: asyncio.Task,
    agent_ref: Optional[list] = None,
) -> AsyncGenerator[bytes, None]:
    """Yield SSE-formatted bytes from the stream queue.

    Handles delta chunks, tool-progress custom events, keepalive pings,
    queue draining on agent completion, and disconnect interruption.

    Consumers wrap this in their framework's streaming response:
    - FastAPI: ``StreamingResponse(sse_event_stream(...), media_type="text/event-stream")``
    - aiohttp: iterate and write to ``StreamResponse``
    """
    loop = asyncio.get_running_loop()
    last_activity = time.monotonic()

    try:
        while True:
            try:
                item = await loop.run_in_executor(
                    None, lambda: stream_queue.get(timeout=0.5)
                )
            except queue.Empty:
                if agent_task.done():
                    # Drain remaining items
                    while True:
                        try:
                            item = stream_queue.get_nowait()
                            if item is None:
                                break
                            yield _format_sse_item(item)
                            last_activity = time.monotonic()
                        except queue.Empty:
                            break
                    break
                # Keepalive
                if time.monotonic() - last_activity >= KEEPALIVE_INTERVAL_SECONDS:
                    yield b": keepalive\n\n"
                    last_activity = time.monotonic()
                continue

            if item is None:
                break

            yield _format_sse_item(item)
            last_activity = time.monotonic()

        # Done event with usage from the completed agent task
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        session_id = None
        try:
            result, agent_usage = await agent_task
            usage = agent_usage or usage
            # Prefer the Hermes session_id (used for DB lookups and history
            # continuity) over result["session_id"] which may be the Claude
            # SDK's internal session_id in harness mode.
            session_id = (
                (agent_ref[0].session_id if agent_ref and agent_ref[0] else None)
                or result.get("session_id")
            )
        except Exception:
            pass

        done_data = {"usage": usage}
        if session_id:
            done_data["session_id"] = session_id
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n".encode()

    except (GeneratorExit, asyncio.CancelledError):
        # Client disconnected — interrupt the agent
        agent = agent_ref[0] if agent_ref else None
        if agent is not None:
            try:
                agent.interrupt("Client disconnected")
            except Exception:
                pass
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("SSE client disconnected; interrupted agent")


def _format_sse_item(item) -> bytes:
    """Format a single queue item as an SSE event."""
    if isinstance(item, tuple) and len(item) == 2:
        tag, payload = item
        if tag == "__tool_progress__":
            return f"event: hermes.tool.progress\ndata: {json.dumps(payload)}\n\n".encode()
        elif tag == "__tool_completed__":
            return f"event: hermes.tool.completed\ndata: {json.dumps(payload)}\n\n".encode()
        elif tag == "__tool_result__":
            return f"event: hermes.tool.result\ndata: {json.dumps(payload)}\n\n".encode()
    return f"event: delta\ndata: {json.dumps({'text': item})}\n\n".encode()
