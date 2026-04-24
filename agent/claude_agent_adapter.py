"""Claude Agent SDK adapter — runs Claude Code as the LLM backend.

When ``model.harness: claude_code`` is set in the Hermes config, this module
bridges the Claude Agent SDK's async ``query()`` into Hermes's synchronous
``AIAgent.run_conversation`` call.  Claude Code owns the tool loop (Read,
Edit, Bash, …); Hermes maps streamed events to its callbacks.

Uses ``query()`` per turn (not ``ClaudeSDKClient``) so that the system prompt
can be updated each turn.  Session continuity across turns is maintained via
the SDK's ``resume`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
# Persistent event loop — a single daemon thread runs loop.run_forever();
# callers submit coroutines via run_coroutine_threadsafe and block.
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
    """Wraps Claude Agent SDK ``query()`` for use inside Hermes's sync agent.

    Each ``send_message`` call invokes ``query()`` with the current system
    prompt and ``resume=<previous_session_id>`` so Claude Code sees the full
    conversation history.
    """

    def __init__(
        self,
        *,
        model: str = "",
        cwd: str = ".",
        permission_mode: str = "acceptEdits",
        max_turns: Optional[int] = None,
        allowed_tools: Optional[list] = None,
        resume_session_id: Optional[str] = None,
        mcp_server_config=None,
        on_session_captured: Optional[Callable[[str], None]] = None,
    ):
        if not _check_sdk():
            raise ImportError(
                "claude-agent-sdk is not installed.  "
                "Install it with:  pip install 'hermes-agent[claude-agent]'"
            )

        self._model = model
        self._cwd = cwd
        self._permission_mode = permission_mode
        self._max_turns = max_turns
        self._allowed_tools = allowed_tools or [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "WebSearch", "WebFetch",
        ]
        self._mcp_server_config = mcp_server_config
        self._session_id: Optional[str] = resume_session_id
        self._on_session_captured = on_session_captured
        self._connected = True  # no connect step needed for query()

    def connect(self):
        """No-op — query() doesn't need a persistent connection."""
        pass

    def close(self):
        """No-op — query() is stateless per call."""
        self._connected = False

    def _mark_session_captured(self, sid: Optional[str]) -> None:
        """Record ``sid`` as the current SDK session_id and notify the owner.

        Called from the init SystemMessage path and the final ResultMessage
        path; centralising both sites means the new-id guard and the
        callback error handling stay in sync.  A callback that raises is
        logged but never propagated — the SDK event loop must keep draining.
        """
        if not sid or sid == self._session_id:
            return
        self._session_id = sid
        if self._on_session_captured is None:
            return
        try:
            self._on_session_captured(sid)
        except Exception as cb_exc:
            logger.warning(
                "on_session_captured callback failed: %s",
                cb_exc,
                exc_info=True,
            )

    def send_message(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        thinking_callback: Optional[Callable] = None,
        thinking_delta_callback: Optional[Callable] = None,
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
            query as claude_query,
            ClaudeAgentOptions,
            AssistantMessage,
            UserMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            ToolResultBlock,
        )

        # Build fresh options each turn so system_prompt changes take effect
        opts_kwargs: Dict[str, Any] = {
            "permission_mode": self._permission_mode,
            "include_partial_messages": True,
        }
        if self._mcp_server_config is not None:
            opts_kwargs["mcp_servers"] = {"hermes-tools": self._mcp_server_config}
            # allowed_tools pre-approves tools so Claude Code doesn't prompt
            # for permission (prompts go to the subprocess stdin, not the TUI).
            # Include both Claude Code built-ins and MCP tool names.
            from tools.mcp_tools_server import _get_hermes_mcp_tools
            mcp_names = [f"mcp__hermes-tools__{t}" for t in _get_hermes_mcp_tools()]
            opts_kwargs["allowed_tools"] = self._allowed_tools + mcp_names
        else:
            opts_kwargs["allowed_tools"] = self._allowed_tools
        # Use the system-installed claude CLI, not the SDK's bundled one
        import shutil
        _cli = shutil.which("claude")
        if _cli:
            opts_kwargs["cli_path"] = _cli
        # Strip empty ANTHROPIC_API_KEY from the process env — Hermes's .env
        # often sets it to "" which overrides Claude Code's own OAuth creds
        # in the subprocess.  Done once at the process level because the SDK
        # inherits os.environ; the opts_kwargs["env"] dict is additive.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ.pop("ANTHROPIC_API_KEY", None)
        if self._model:
            opts_kwargs["model"] = self._model
        if self._cwd:
            opts_kwargs["cwd"] = self._cwd
        if self._max_turns:
            opts_kwargs["max_turns"] = self._max_turns
        if self._session_id:
            opts_kwargs["resume"] = self._session_id
        if system_prompt:
            # Use Hermes's system prompt directly instead of appending to
            # Claude Code's preset — the combined size would exceed the API
            # limit (Claude Code's own prompt is ~100K+ tokens).
            opts_kwargs["system_prompt"] = system_prompt

        options = ClaudeAgentOptions(**opts_kwargs)

        final_text_parts: list[str] = []
        last_reasoning: Optional[str] = None
        result_msg: Optional[ResultMessage] = None
        in_flight_tools: Dict[str, tuple] = {}
        api_calls = 0
        interrupted = False
        start_time = time.monotonic()
        # If StreamEvent text/thinking deltas fire, we've already delivered the
        # content to the callbacks and must not re-deliver from AssistantMessage.
        saw_text_delta = False
        saw_thinking_delta = False

        async def _exchange():
            nonlocal result_msg, api_calls, interrupted, last_reasoning
            nonlocal saw_text_delta, saw_thinking_delta

            logger.info("Claude SDK query: cli_path=%s cwd=%s model=%s resume=%s prompt_len=%d system_len=%d",
                        opts_kwargs.get("cli_path"), opts_kwargs.get("cwd"), opts_kwargs.get("model"),
                        opts_kwargs.get("resume"), len(user_message),
                        len(str(opts_kwargs.get("system_prompt", ""))))

            async for message in claude_query(prompt=user_message, options=options):
                if interrupt_check and interrupt_check():
                    interrupted = True
                    break

                # Debug: log all message types to understand the SDK protocol
                msg_type = type(message).__name__
                block_types = []
                if hasattr(message, "content") and isinstance(message.content, list):
                    block_types = [type(b).__name__ for b in message.content]
                logger.debug("SDK message: %s blocks=%s", msg_type, block_types)

                if isinstance(message, SystemMessage):
                    if message.subtype == "init" and hasattr(message, "data"):
                        self._mark_session_captured(message.data.get("session_id"))

                elif isinstance(message, StreamEvent):
                    ev = message.event or {}
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta") or {}
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            text = delta.get("text") or ""
                            if text and stream_delta_callback:
                                saw_text_delta = True
                                stream_delta_callback(text)
                        elif dtype == "thinking_delta":
                            thinking = delta.get("thinking") or ""
                            if thinking and thinking_delta_callback:
                                saw_thinking_delta = True
                                thinking_delta_callback(thinking)

                elif isinstance(message, AssistantMessage):
                    api_calls += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_text_parts.append(block.text)
                            if stream_delta_callback and not saw_text_delta:
                                stream_delta_callback(block.text)

                        elif isinstance(block, ThinkingBlock):
                            last_reasoning = block.thinking
                            if thinking_callback and not saw_thinking_delta:
                                thinking_callback(block.thinking)

                        elif isinstance(block, ToolUseBlock):
                            in_flight_tools[block.id] = (block.name, block.input or {})
                            if tool_progress_callback:
                                preview = str(block.input)[:120] if block.input else ""
                                tool_progress_callback("tool.started", block.name, preview, block.input, tool_call_id=block.id)

                        elif isinstance(block, ToolResultBlock):
                            if tool_complete_callback:
                                name, args = in_flight_tools.pop(block.tool_use_id, ("", {}))
                                result_str = block.content if isinstance(block.content, str) else str(block.content)
                                tool_complete_callback(block.tool_use_id, name, args, result_str)

                elif isinstance(message, UserMessage):
                    # Tool results come as UserMessage in the Claude Code
                    # protocol.  Extract ToolResultBlock for callbacks.
                    if hasattr(message, "content") and isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
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
            # Fallback capture: SDKs that skip the init SystemMessage still
            # reveal the session_id on the final ResultMessage.
            if not self._session_id and hasattr(result_msg, "session_id"):
                self._mark_session_captured(result_msg.session_id)

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
            "model": self._model or "",
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
        """No-op for query() mode — interrupts are handled via interrupt_check callback."""
        pass
