"""Tests for Claude Code harness sdk_sid persistence + resume-chain health.

Covers:
- ``AIAgent._persist_sdk_sid`` — success, transient sqlite lock (info),
  structural failure (warning + flag flip).
- ``AIAgent._on_harness_sdk_session_captured`` — no-op on repeat id,
  writes on new id, survives a raising DB.
- Silent SDK emission (sdk_sid is None on a completed turn) — this is
  detected end-of-turn inside _run_claude_code_conversation and sets
  the known-dead flag so the next turn's prelude kicks in.  Exercised
  via a narrower unit that constructs the state directly (the full
  function requires a live SDK session and is covered by the mocked
  integration tests).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_agent(session_db, session_id="test-harness-persist"):
    """Construct a minimal AIAgent wired to a real SessionDB but no LLM."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

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
    return agent


@pytest.fixture
def agent_with_db():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = SessionDB(db_path=db_path)
        agent = _make_agent(db)
        yield agent, db


class TestPersistSdkSid:
    def test_happy_path_writes_and_returns_true(self, agent_with_db):
        agent, db = agent_with_db
        assert agent._persist_sdk_sid("sdk-abc") is True

        # Verify model_config now contains the id.
        import json
        row = db.get_session(agent.session_id)
        assert row is not None
        mc = json.loads(row["model_config"])
        assert mc.get("claude_sdk_session_id") == "sdk-abc"
        # Happy path must not flip the known-dead flag.
        assert agent._harness_resume_known_dead is False

    def test_empty_sdk_sid_is_noop(self, agent_with_db):
        agent, _ = agent_with_db
        assert agent._persist_sdk_sid("") is False
        assert agent._harness_resume_known_dead is False

    def test_no_session_db_returns_false(self):
        agent = _make_agent(session_db=None, session_id="x")
        assert agent._persist_sdk_sid("sdk-abc") is False

    def test_operational_error_logs_info_keeps_chain(self, agent_with_db, caplog):
        agent, db = agent_with_db

        def raise_locked(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        caplog.set_level(logging.INFO, logger="run_agent")
        with patch.object(db, "patch_model_config", side_effect=raise_locked):
            result = agent._persist_sdk_sid("sdk-abc")

        assert result is False
        # Transient contention must NOT mark the chain dead — the
        # end-of-turn retry may still succeed.
        assert agent._harness_resume_known_dead is False
        assert any(
            "Transient SQLite contention" in rec.message
            for rec in caplog.records
        ), caplog.records

    def test_structural_error_logs_warning_and_marks_dead(self, agent_with_db, caplog):
        agent, db = agent_with_db

        def raise_generic(*a, **k):
            raise RuntimeError("json encode failed")

        caplog.set_level(logging.WARNING, logger="run_agent")
        with patch.object(db, "patch_model_config", side_effect=raise_generic):
            result = agent._persist_sdk_sid("sdk-abc")

        assert result is False
        assert agent._harness_resume_known_dead is True
        assert any(
            "Failed to persist claude_sdk_session_id" in rec.message
            for rec in caplog.records
        ), caplog.records


class TestOnHarnessSdkSessionCaptured:
    """Callback fires on the SDK's async event loop — must be I/O-free.
    DB persistence is the reconcile step's job (see TestReconcile below)."""

    def test_new_id_updates_in_memory_state_only(self, agent_with_db):
        agent, db = agent_with_db
        with patch.object(db, "patch_model_config") as m:
            agent._on_harness_sdk_session_captured("sdk-new")
            m.assert_not_called()
        assert agent._harness_sdk_session_id == "sdk-new"

    def test_repeat_id_is_noop(self, agent_with_db):
        agent, _ = agent_with_db
        agent._harness_sdk_session_id = "sdk-same"
        agent._on_harness_sdk_session_captured("sdk-same")
        assert agent._harness_sdk_session_id == "sdk-same"

    def test_empty_id_is_noop(self, agent_with_db):
        agent, _ = agent_with_db
        agent._on_harness_sdk_session_captured("")
        assert agent._harness_sdk_session_id is None


class TestReconcileSdkSessionAfterTurn:
    def test_new_id_persists_and_updates_state(self, agent_with_db):
        agent, db = agent_with_db
        agent._reconcile_sdk_session_after_turn(
            sdk_sid_at_entry=None,
            sdk_sid="sdk-new",
            interrupted=False,
            final_response="ok",
        )
        assert agent._harness_sdk_session_id == "sdk-new"

        import json
        mc = json.loads(db.get_session(agent.session_id)["model_config"])
        assert mc.get("claude_sdk_session_id") == "sdk-new"

    def test_persist_failure_does_not_raise(self, agent_with_db):
        agent, db = agent_with_db

        def boom(*a, **k):
            raise RuntimeError("nope")

        with patch.object(db, "patch_model_config", side_effect=boom):
            agent._reconcile_sdk_session_after_turn(
                sdk_sid_at_entry=None,
                sdk_sid="sdk-x",
                interrupted=False,
                final_response="ok",
            )

        # In-memory state still advances; chain flagged dead so the next
        # turn falls back to the prelude path.
        assert agent._harness_sdk_session_id == "sdk-x"
        assert agent._harness_resume_known_dead is True


class TestInitialState:
    def test_flag_defaults_false(self, agent_with_db):
        agent, _ = agent_with_db
        assert agent._harness_resume_known_dead is False
        assert agent._harness_sdk_session_id is None
        assert agent._harness_messages == []
