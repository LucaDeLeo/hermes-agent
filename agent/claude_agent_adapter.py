"""Claude Agent SDK adapter — runs Claude Code as the LLM backend.

When ``model.harness: claude_code`` is set in the Hermes config, this module
bridges the Claude Agent SDK's async ``ClaudeSDKClient`` into Hermes's
synchronous ``AIAgent.run_conversation`` call.  Claude Code owns the tool
loop (Read, Edit, Bash, …); Hermes maps streamed events to its callbacks.

Uses a dedicated daemon-thread event loop (not model_tools._run_async)
because ClaudeSDKClient starts persistent background tasks during connect()
that require the loop to stay running between calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_sdk_available: Optional[bool] = None


def _check_sdk():
    global _sdk_available
    if _sdk_available is not None:
        return _sdk_available
    try:
        import claude_agent_sdk  # noqa: F401
        _sdk_available = True
    except ImportError:
        _sdk_available = False
    return _sdk_available


# ---------------------------------------------------------------------------
# Persistent event loop for the SDK's long-lived subprocess connection.
# A single daemon thread runs loop.run_forever(); callers submit coroutines
# via run_coroutine_threadsafe and block on the future.
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            t = threading.Thread(target=_loop.run_forever, daemon=True, name="claude-sdk-loop")
            t.start()
        return _loop


def _run_sync(coro):
    """Submit *coro* to the persistent loop and block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class ClaudeAgentSession:
    """Wraps ``ClaudeSDKClient`` for use inside Hermes's synchronous agent."""

    def __init__(
        self,
        *,
        model: str = "",
        cwd: str = ".",
        permission_mode: str = "acceptEdits",
        max_turns: Optional[int] = None,
        allowed_tools: Optional[list] = None,
        resume_session_id: Optional[str] = None,
    ):
        if not _check_sdk():
            raise ImportError(
                "claude-agent-sdk is not installed.  "
                "Install it with:  pip install 'hermes-agent[claude-agent]'"
            )

        from claude_agent_sdk import ClaudeAgentOptions

        opts_kwargs: Dict[str, Any] = {
            "permission_mode": permission_mode,
            "include_partial_messages": True,
        }
        if model:
            opts_kwargs["model"] = model
        if cwd:
            opts_kwargs["cwd"] = cwd
        if max_turns:
            opts_kwargs["max_turns"] = max_turns
        if resume_session_id:
            opts_kwargs["resume"] = resume_session_id
        if allowed_tools:
            opts_kwargs["allowed_tools"] = allowed_tools
        else:
            opts_kwargs["allowed_tools"] = [
                "Read", "Write", "Edit", "Bash", "Glob", "Grep",
                "WebSearch", "WebFetch",
            ]

        self._options = ClaudeAgentOptions(**opts_kwargs)
        self._client = None
        self._session_id: Optional[str] = None
        self._connected = False

    def connect(self):
        if self._connected:
            return
        from claude_agent_sdk import ClaudeSDKClient

        async def _connect():
            self._client = ClaudeSDKClient(options=self._options)
            await self._client.connect()

        _run_sync(_connect())
        self._connected = True
        logger.info("Claude Agent SDK client connected")

    def close(self):
        if self._client and self._connected:
            try:
                _run_sync(self._client.disconnect())
            except Exception:
                pass
            self._connected = False
            self._client = None

    def send_message(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        thinking_callback: Optional[Callable] = None,
        stream_delta_callback: Optional[Callable] = None,
        tool_progress_callback: Optional[Callable] = None,
        tool_complete_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
        interrupt_check: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Send *user_message* to Claude Code and stream events back.

        Callback signatures match Hermes conventions:
        - tool_progress_callback(event_type, name, preview, args, **kwargs)
        - tool_complete_callback(tool_call_id, function_name, function_args, function_result)
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            ToolResultBlock,
        )

        if system_prompt and self._client:
            self._options.append_system_prompt = system_prompt

        final_text_parts: list[str] = []
        last_reasoning: Optional[str] = None
        result_msg: Optional[ResultMessage] = None
        in_flight_tools: Dict[str, tuple] = {}
        api_calls = 0
        interrupted = False
        start_time = time.monotonic()

        async def _exchange():
            nonlocal result_msg, api_calls, interrupted, last_reasoning

            await self._client.query(user_message)

            async for message in self._client.receive_response():
                if interrupt_check and interrupt_check():
                    try:
                        await self._client.interrupt()
                    except Exception:
                        pass
                    interrupted = True
                    break

                if isinstance(message, SystemMessage):
                    if message.subtype == "init" and hasattr(message, "data"):
                        sid = message.data.get("session_id")
                        if sid:
                            self._session_id = sid

                elif isinstance(message, AssistantMessage):
                    api_calls += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_text_parts.append(block.text)
                            if stream_delta_callback:
                                stream_delta_callback(block.text)

                        elif isinstance(block, ThinkingBlock):
                            last_reasoning = block.thinking
                            if thinking_callback:
                                thinking_callback(block.thinking)

                        elif isinstance(block, ToolUseBlock):
                            in_flight_tools[block.id] = (block.name, block.input or {})
                            if tool_progress_callback:
                                preview = str(block.input)[:120] if block.input else ""
                                tool_progress_callback("tool.started", block.name, preview, block.input)

                        elif isinstance(block, ToolResultBlock):
                            if tool_complete_callback:
                                name, args = in_flight_tools.pop(block.tool_use_id, ("", {}))
                                result_str = block.content if isinstance(block.content, str) else str(block.content)
                                tool_complete_callback(block.tool_use_id, name, args, result_str)

                elif isinstance(message, ResultMessage):
                    result_msg = message

        _run_sync(_exchange())

        duration_ms = int((time.monotonic() - start_time) * 1000)
        final_response = "".join(final_text_parts) or None

        cost = None
        usage = {}
        if result_msg:
            cost = result_msg.total_cost_usd
            usage = result_msg.usage or {}
            if not final_response and result_msg.result:
                final_response = result_msg.result

        return {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": [
                {"role": "user", "content": user_message},
                *([{"role": "assistant", "content": final_response}] if final_response else []),
            ],
            "api_calls": api_calls,
            "completed": not interrupted and (result_msg is not None),
            "partial": False,
            "interrupted": interrupted,
            "response_previewed": False,
            "model": self._options.model or "",
            "provider": "anthropic",
            "base_url": "",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
            "reasoning_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
            "last_prompt_tokens": 0,
            "estimated_cost_usd": cost or 0.0,
            "cost_status": "sdk" if cost else "unavailable",
            "cost_source": "claude_agent_sdk",
            "duration_ms": duration_ms,
            "session_id": self._session_id,
        }

    def interrupt(self):
        if self._client and self._connected:
            try:
                _run_sync(self._client.interrupt())
            except Exception as exc:
                logger.debug("Interrupt failed: %s", exc)
