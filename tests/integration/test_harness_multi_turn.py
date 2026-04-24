"""Multi-turn context-preservation tests for the Claude Code harness.

These tests exercise ``AIAgent._run_claude_code_conversation`` end-to-end
with a mocked ``ClaudeAgentSession``.  No real Claude CLI or network is
required, so the suite runs in CI.

Scenarios covered (see ~/.claude/plans/plan-a-comprehensive-fix-temporal-rocket.md):
  A. Happy path — sdk_sid persists on turn 1, turn 2 resume works,
     no history prelude is injected.
  B. Induced persist failure — turn 1's sdk_sid doesn't land in state.db
     (DB patch raises); turn 2's fresh AIAgent sees no stored id, injects
     the prelude, and the fake SDK recalls the fact from the prelude.
  C. Resume mismatch — DB has a stale sdk_sid; fake SDK returns a
     different session_id; warning logged, state.db updated to the new
     id, known-dead flag set so the next turn uses the prelude.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fake ClaudeAgentSession — behaves enough like the real adapter for
# _run_claude_code_conversation to exercise all branches.
# ---------------------------------------------------------------------------

class FakeClaudeAgentSession:
    """Drop-in replacement for agent.claude_agent_adapter.ClaudeAgentSession.

    - Captures every ``send_message`` call's ``user_message`` and the
      resume_session_id in force at that point.
    - Returns a configurable session_id (controls the resume-mismatch
      scenario).
    - Its ``final_response`` is controlled by a per-instance ``responder``
      callable so tests can simulate "model recalls the fact" vs "model
      has no memory".
    """

    def __init__(
        self,
        *,
        model: str = "",
        cwd: str = ".",
        permission_mode: str = "acceptEdits",
        max_turns=None,
        allowed_tools=None,
        resume_session_id=None,
        mcp_server_config=None,
        on_session_captured=None,
        # Test-injected:
        session_id_to_emit: str = "sdk-fake-1",
        responder=None,
    ):
        self._resume_session_id = resume_session_id
        self._session_id = resume_session_id
        self._on_session_captured = on_session_captured
        self._session_id_to_emit = session_id_to_emit
        self._responder = responder or (lambda msg, resume: "ok")
        self.calls: list[dict] = []

    def connect(self):
        pass

    def close(self):
        pass

    def send_message(self, user_message: str, **kwargs):
        call = {
            "user_message": user_message,
            "resume_at_entry": self._resume_session_id,
            **{k: v for k, v in kwargs.items() if k == "system_prompt"},
        }
        self.calls.append(call)

        # Emit the fake session_id via the callback (mirrors real adapter
        # capturing from init SystemMessage).
        if self._session_id_to_emit and self._session_id_to_emit != self._session_id:
            self._session_id = self._session_id_to_emit
            if self._on_session_captured is not None:
                try:
                    self._on_session_captured(self._session_id_to_emit)
                except Exception:
                    pass  # mirror adapter — never propagate

        final = self._responder(user_message, self._resume_session_id)
        return {
            "final_response": final,
            "last_reasoning": None,
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final} if final else {},
            ],
            "api_calls": 1,
            "completed": True,
            "partial": False,
            "interrupted": False,
            "response_previewed": False,
            "model": "fake-model",
            "provider": "anthropic",
            "base_url": "",
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 15,
            "last_prompt_tokens": 0,
            "estimated_cost_usd": 0.0,
            "cost_status": "sdk",
            "cost_source": "claude_agent_sdk",
            "duration_ms": 1,
            "session_id": self._session_id_to_emit,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session_db():
    from hermes_state import SessionDB
    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")
        yield db


def _make_agent(session_db, session_id, *, fake_session=None):
    """Construct an AIAgent with its harness pre-wired to ``fake_session``.

    We bypass ``_create_harness`` (which would try to import the real SDK)
    and instead install ``fake_session`` directly as the live session.
    This is exactly the state the agent would be in after a successful
    ``_create_harness`` call.
    """
    from run_agent import AIAgent

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    if fake_session is not None:
        # We bypass _create_harness (which tries to import the real SDK
        # for its provider check), so replicate what that flow would have
        # done: load the persisted sdk_sid from DB and install the fake.
        agent._harness_sdk_session_id = agent._load_sdk_session_id()
        agent._claude_agent_session = fake_session
        fake_session._on_session_captured = agent._on_harness_sdk_session_captured
        fake_session._resume_session_id = agent._harness_sdk_session_id
        fake_session._session_id = agent._harness_sdk_session_id
    return agent


def _recall_responder(fact_marker: str):
    """Fake model: recalls ``fact_marker`` if it was in the prompt OR if a
    resume id was passed (simulating the SDK's own prior-session memory).
    Otherwise answers 'no memory'."""
    def _r(user_message: str, resume: str | None) -> str:
        if fact_marker in user_message:
            return f"Yes, I recall {fact_marker}."
        if resume:
            # Emulate "SDK resume carried prior context" — only if our
            # test path actually sets a resume id the fake pretends to
            # honor.  (In test C we want this to NOT fire.)
            return f"Yes, I recall {fact_marker}."
        return "I have no memory of that."
    return _r


def _sdk_model_config_sid(db, session_id: str) -> str | None:
    row = db.get_session(session_id)
    if not row or not row.get("model_config"):
        return None
    try:
        return json.loads(row["model_config"]).get("claude_sdk_session_id")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scenario A — Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_turn2_resume_works_no_prelude(self, session_db):
        """Turn 1 persists sdk_sid; turn 2 fresh AIAgent resumes successfully;
        no prelude is injected (the resume chain is healthy)."""
        sid = "sess-happy"

        # --- Turn 1 ---
        fake1 = FakeClaudeAgentSession(session_id_to_emit="sdk-1")
        agent1 = _make_agent(session_db, sid, fake_session=fake1)
        r1 = agent1._run_claude_code_conversation(
            "Remember the number 73529.", "Remember the number 73529.",
        )
        assert r1["final_response"]  # sanity
        # sdk_sid landed in state.db
        assert _sdk_model_config_sid(session_db, sid) == "sdk-1"

        # --- Turn 2 (fresh AIAgent, same session_id) ---
        fake2 = FakeClaudeAgentSession(
            session_id_to_emit="sdk-1",  # SDK honors our resume request
            responder=_recall_responder("73529"),
        )
        agent2 = _make_agent(session_db, sid, fake_session=fake2)

        # Assertion: the stored id was loaded.
        assert agent2._harness_sdk_session_id == "sdk-1"

        # Seed history like run_conversation does at line ~9020.
        agent2._harness_messages = [
            {"role": "user", "content": "Remember the number 73529."},
            {"role": "assistant", "content": r1["final_response"]},
        ]

        r2 = agent2._run_claude_code_conversation(
            "What number did I ask you to remember?",
            "What number did I ask you to remember?",
        )
        assert "73529" in r2["final_response"]

        # Critical: the adapter received the UNWRAPPED user message —
        # no <prior_conversation> envelope — because resume was healthy.
        sent = fake2.calls[0]["user_message"]
        assert "<prior_conversation>" not in sent, sent
        assert sent.startswith("What number")
        # And the resume id was the stored one.
        assert fake2.calls[0]["resume_at_entry"] == "sdk-1"


# ---------------------------------------------------------------------------
# Scenario B — Persist failure → prelude injection
# ---------------------------------------------------------------------------

class TestInducedPersistFailure:
    def test_turn2_injects_prelude_when_sdk_sid_not_persisted(
        self, session_db, caplog,
    ):
        """Turn 1's sdk_sid write fails; turn 2 sees no stored id and the
        fake SDK has no resume — so only the history prelude can carry
        context.  The fake "recalls the fact" only when it finds the
        marker inside the prelude."""
        sid = "sess-persist-fail"

        # --- Turn 1: patch_model_config raises, sdk_sid never lands ---
        fake1 = FakeClaudeAgentSession(session_id_to_emit="sdk-lost")
        agent1 = _make_agent(session_db, sid, fake_session=fake1)

        def boom(*a, **k):
            raise RuntimeError("disk full")

        caplog.set_level(logging.WARNING, logger="run_agent")
        with patch.object(session_db, "patch_model_config", side_effect=boom):
            r1 = agent1._run_claude_code_conversation(
                "Remember the number 73529.", "Remember the number 73529.",
            )
        assert r1["final_response"]
        # State.db has no sdk_sid (the raising patch).
        assert _sdk_model_config_sid(session_db, sid) is None
        # Loud failure logged (not swallowed).
        assert any(
            "Failed to persist claude_sdk_session_id" in rec.message
            for rec in caplog.records
        )

        # --- Turn 2: fresh AIAgent sees no stored id ---
        fake2 = FakeClaudeAgentSession(
            session_id_to_emit="sdk-new-session",
            # Crucially: responder ONLY recalls if the marker is in the
            # user_message (no resume honor) — so the prelude must carry
            # the fact.
            responder=lambda msg, resume: (
                "Yes, I recall 73529." if "73529" in msg
                else "I have no memory of that."
            ),
        )
        agent2 = _make_agent(session_db, sid, fake_session=fake2)
        assert agent2._harness_sdk_session_id is None  # nothing was persisted

        agent2._harness_messages = [
            {"role": "user", "content": "Remember the number 73529."},
            {"role": "assistant", "content": r1["final_response"]},
        ]

        caplog.clear()
        caplog.set_level(logging.INFO, logger="run_agent")
        r2 = agent2._run_claude_code_conversation(
            "What number did I ask you to remember?",
            "What number did I ask you to remember?",
        )

        # Context preserved despite broken resume chain.
        assert "73529" in r2["final_response"]

        # The adapter received the PRELUDE-WRAPPED message.
        sent = fake2.calls[0]["user_message"]
        assert "<prior_conversation>" in sent
        assert "73529" in sent
        assert sent.rstrip().endswith("What number did I ask you to remember?")

        # Prelude injection was logged.
        assert any(
            "Harness prelude injected" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Scenario C — Resume mismatch
# ---------------------------------------------------------------------------

class TestResumeMismatch:
    def test_mismatch_logs_warning_updates_db_marks_dead(self, session_db, caplog):
        """DB has sdk-stale; the fake SDK returns sdk-new instead (simulating
        the SDK silently dropping our resume).  We must: log the mismatch,
        update the DB to sdk-new, and set the known-dead flag so the NEXT
        turn (not this one) injects a prelude."""
        sid = "sess-mismatch"

        # Prime state.db with a stale sdk_sid.
        session_db.patch_model_config(sid, {"claude_sdk_session_id": "sdk-stale"})
        assert _sdk_model_config_sid(session_db, sid) == "sdk-stale"

        # Fake SDK returns a DIFFERENT id (resume silently ignored).
        fake = FakeClaudeAgentSession(
            session_id_to_emit="sdk-new",
            responder=_recall_responder("ANYTHING"),
        )
        agent = _make_agent(session_db, sid, fake_session=fake)
        assert agent._harness_sdk_session_id == "sdk-stale"

        # Simulate mid-session: history is non-empty but the current turn
        # proceeds with the stale resume still in force.
        agent._harness_messages = [
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "ack"},
        ]

        caplog.set_level(logging.WARNING, logger="run_agent")
        agent._run_claude_code_conversation("follow-up", "follow-up")

        # Mismatch was logged.
        assert any(
            "resume mismatch" in rec.message
            for rec in caplog.records
        ), [r.message for r in caplog.records]

        # DB was updated to the new id (via the adapter callback OR the
        # end-of-turn safety net).
        assert _sdk_model_config_sid(session_db, sid) == "sdk-new"

        # Chain marked dead → next turn will inject prelude.
        assert agent._harness_resume_known_dead is True

    def test_matching_sid_clears_known_dead_flag(self, session_db):
        """If a prior turn marked the chain dead (e.g. persist failed)
        but the current turn completes with a matching session_id, the
        flag should self-heal."""
        sid = "sess-heal"
        session_db.patch_model_config(sid, {"claude_sdk_session_id": "sdk-ok"})
        fake = FakeClaudeAgentSession(session_id_to_emit="sdk-ok")
        agent = _make_agent(session_db, sid, fake_session=fake)
        agent._harness_resume_known_dead = True  # set by some earlier failure

        agent._run_claude_code_conversation("hi", "hi")
        assert agent._harness_resume_known_dead is False
