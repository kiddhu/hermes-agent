"""Tests for the factory-build terminal-write receipt gate (AION-889 Phase B).

The gate is a minimal machine hard gate on ``complete_task``: a task flagged
``factory_build_gate = 1`` cannot terminalize until a 64-hex proof-kernel
``OUTCOME_ACCEPTED`` receipt sha256 is bound to
``factory_terminal_receipt_sha256``. Content-level (C1-C10 / 8-field)
verification deliberately stays in the aion-governance proof kernel; this
chokepoint only enforces the 64-hex shape so validation stays single-set.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


VALID_RECEIPT = "ab" * 32  # 64 hex chars


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (factory-gate variant)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task_columns(conn) -> set:
    return {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}


def _set_factory_gate(conn, task_id, gate=1, receipt=None) -> None:
    conn.execute(
        "UPDATE tasks SET factory_build_gate = ?, "
        "factory_terminal_receipt_sha256 = ? WHERE id = ?",
        (gate, receipt, task_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Schema / migration
# ---------------------------------------------------------------------------

def test_factory_gate_columns_present_on_fresh_db(kanban_home):
    with kb.connect() as conn:
        cols = _task_columns(conn)
    assert "factory_build_gate" in cols
    assert "factory_terminal_receipt_sha256" in cols


def test_factory_gate_defaults_are_legacy_safe(kanban_home):
    """A fresh task defaults to gate=0 / NULL receipt — legacy parity."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary")
        row = conn.execute(
            "SELECT factory_build_gate, factory_terminal_receipt_sha256 "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
    assert row["factory_build_gate"] == 0
    assert row["factory_terminal_receipt_sha256"] is None


def test_factory_gate_migration_idempotent(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="keep me")
    # Re-running the migration pass twice must not error or drop data.
    kb.init_db()
    kb.init_db()
    with kb.connect() as conn:
        cols = _task_columns(conn)
        t = kb.get_task(conn, tid)
    assert "factory_build_gate" in cols
    assert "factory_terminal_receipt_sha256" in cols
    assert t is not None and t.title == "keep me"


def test_factory_gate_legacy_db_migrates(tmp_path):
    """A pre-gate ``tasks`` table missing both columns migrates cleanly."""
    db_path = tmp_path / "legacy-factory.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    conn.commit()
    conn.close()

    with kb.connect(db_path) as migrated:
        cols = _task_columns(migrated)
        row = migrated.execute(
            "SELECT factory_build_gate, factory_terminal_receipt_sha256 "
            "FROM tasks WHERE id = 'legacy'"
        ).fetchone()
    assert "factory_build_gate" in cols
    assert "factory_terminal_receipt_sha256" in cols
    assert row["factory_build_gate"] == 0
    assert row["factory_terminal_receipt_sha256"] is None


# ---------------------------------------------------------------------------
# Terminal-write guard (RED/GREEN)
# ---------------------------------------------------------------------------

def test_complete_task_factory_gate_no_receipt_rejected_zero_delta(kanban_home):
    """gate=1 with no receipt -> complete rejected, zero DB delta."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="factory task")
        _set_factory_gate(conn, tid, gate=1, receipt=None)

        before = conn.execute(
            "SELECT status, completed_at, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        events_before = len(kb.list_events(conn, tid))

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError) as exc_info:
            kb.complete_task(conn, tid, result="done", summary="done")

        after = conn.execute(
            "SELECT status, completed_at, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        events_after = len(kb.list_events(conn, tid))

    assert exc_info.value.code == "FACTORY_TERMINAL_RECEIPT_REQUIRED"
    assert exc_info.value.task_id == tid
    # Zero mutation: status / completed_at / current_run_id unchanged, and no
    # event was appended (unlike the created_cards gate, this gate emits none).
    assert before["status"] == "ready"
    assert after["status"] == "ready"
    assert after["completed_at"] is None
    assert after["current_run_id"] is None
    assert events_after == events_before


@pytest.mark.parametrize(
    "bad_receipt",
    [
        "",
        "not-hex",
        "z" * 64,          # 64 chars but not hex
        "g" + "a" * 63,    # 64 chars with a non-hex leading char
        "a" * 63,          # too short
        "a" * 65,          # too long
        "A" * 63 + "z",    # 64 chars, trailing non-hex
    ],
)
def test_complete_task_factory_gate_invalid_receipt_rejected(
    kanban_home, bad_receipt
):
    """gate=1 with a non-64-hex receipt -> complete rejected."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="factory task")
        _set_factory_gate(conn, tid, gate=1, receipt=bad_receipt)
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, tid, result="done")
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["status"] == "ready"


def test_complete_task_factory_gate_valid_receipt_succeeds(kanban_home):
    """gate=1 with a valid 64-hex receipt -> complete succeeds."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="factory task")
        _set_factory_gate(conn, tid, gate=1, receipt=VALID_RECEIPT)
        ok = kb.complete_task(conn, tid, result="done", summary="done")
        row = conn.execute(
            "SELECT status, completed_at FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert ok is True
    assert row["status"] == "done"
    assert row["completed_at"] is not None


def test_complete_task_factory_gate_uppercase_hex_accepted(kanban_home):
    """Receipt shape check is case-insensitive (0-9a-fA-F)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="factory task")
        _set_factory_gate(conn, tid, gate=1, receipt="CDEF" * 16)
        ok = kb.complete_task(conn, tid, result="done")
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert ok is True
    assert row["status"] == "done"


# ---------------------------------------------------------------------------
# Legacy parity (gate=0)
# ---------------------------------------------------------------------------

def test_complete_task_legacy_gate_zero_parity(kanban_home):
    """gate=0 (default) must behave byte-identically to the pre-gate path."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary task")
        ok = kb.complete_task(conn, tid, result="ok")
        row = conn.execute(
            "SELECT status, completed_at FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert ok is True
    assert row["status"] == "done"
    assert row["completed_at"] is not None
