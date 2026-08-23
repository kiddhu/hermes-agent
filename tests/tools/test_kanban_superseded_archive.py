"""Regression coverage for orchestrator-owned superseded-task archival."""
from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.fixture
def orchestrator_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "factory-controller")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _topology(kb, conn, prefix: str):
    parent = kb.create_task(conn, title=f"{prefix} superseded parent")
    child = kb.create_task(
        conn,
        title=f"{prefix} failed legacy child",
        assignee="elder-senate",
        parents=[parent],
    )
    downstream = kb.create_task(
        conn,
        title=f"{prefix} intended downstream convergence",
        assignee="factory-controller",
        parents=[child],
    )
    return parent, child, downstream


def test_archive_superseded_todo_child_is_audited_idempotent_and_preserves_fanin(
    orchestrator_env,
):
    """Exact parent -> failed todo child -> downstream fan-in stays safe.

    The control topology proves the old defect: completing the parent makes the
    legacy todo child dispatchable.  The repaired topology archives that child
    first; parent completion can never re-promote it, while its downstream
    dependency remains satisfied through Native archived-parent semantics.
    """
    kb = orchestrator_env
    with kb.connect() as conn:
        control_parent, control_child, _ = _topology(kb, conn, "control")
        assert kb.get_task(conn, control_child).status == "todo"
        assert kb.complete_task(conn, control_parent)
        assert kb.get_task(conn, control_child).status == "ready"

        parent, child, downstream = _topology(kb, conn, "repair")
        assert kb.get_task(conn, child).status == "todo"
        assert kb.get_task(conn, downstream).status == "todo"

    from tools import kanban_tools as kt

    reason = "superseded by completed same-logical recovery t_recovery"
    first = json.loads(kt._handle_archive({"task_id": child, "reason": reason}))
    assert first == {
        "ok": True,
        "task_id": child,
        "status": "archived",
        "already_archived": False,
    }

    with kb.connect() as conn:
        assert kb.get_task(conn, parent).status == "ready"
        assert kb.get_task(conn, child).status == "archived"
        assert kb.claim_task(conn, child) is None
        assert kb.get_task(conn, downstream).status == "ready"
        archived_events = [e for e in kb.list_events(conn, child) if e.kind == "archived"]
        assert len(archived_events) == 1
        assert archived_events[0].payload == {
            "reason": reason,
            "actor": "factory-controller",
            "source": "kanban_archive",
        }

    replay = json.loads(kt._handle_archive({"task_id": child, "reason": reason}))
    assert replay == {
        "ok": True,
        "task_id": child,
        "status": "archived",
        "already_archived": True,
    }

    with kb.connect() as conn:
        assert len([e for e in kb.list_events(conn, child) if e.kind == "archived"]) == 1
        assert kb.complete_task(conn, parent)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "archived"
        assert kb.claim_task(conn, child) is None
        assert kb.get_task(conn, downstream).status == "ready"


def test_archive_factory_gated_todo_child_uses_transaction_scoped_grant(
    orchestrator_env,
):
    """Strict orchestrator archive may terminalize an idle factory-gated todo."""
    kb = orchestrator_env
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="factory archive hold")
        task_id = kb.create_task(
            conn,
            title="superseded factory child",
            parents=[parent],
            factory_build_gate=1,
        )
        downstream = kb.create_task(
            conn,
            title="post-archive convergence",
            parents=[task_id],
        )
        assert kb.get_task(conn, task_id).status == "todo"
        assert kb.get_task(conn, downstream).status == "todo"
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_terminal_write_grants"
        ).fetchone()[0] == 0

    from tools import kanban_tools as kt

    reason = "superseded factory candidate"
    archived = json.loads(
        kt._handle_archive({"task_id": task_id, "reason": reason})
    )
    assert archived == {
        "ok": True,
        "task_id": task_id,
        "status": "archived",
        "already_archived": False,
    }
    assert kb._STRICT_ORCHESTRATOR_ARCHIVE_AUTH.get() is False

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "archived"
        assert task.current_run_id is None
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert kb.list_runs(conn, task_id) == []
        assert kb.get_task(conn, downstream).status == "ready"
        assert conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()[0] is None
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_terminal_write_grants"
        ).fetchone()[0] == 0
        archived_events = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "archived"
        ]
        assert len(archived_events) == 1
        assert archived_events[0].payload == {
            "reason": reason,
            "actor": "factory-controller",
            "source": "kanban_archive",
        }


def test_archive_factory_grant_and_transition_roll_back_together(
    orchestrator_env, monkeypatch,
):
    kb = orchestrator_env
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="factory rollback hold")
        task_id = kb.create_task(
            conn,
            title="factory rollback candidate",
            parents=[parent],
            factory_build_gate=1,
        )
        assert kb.get_task(conn, task_id).status == "todo"

    original_append_event = kb._append_event

    def fail_archive_event(conn, event_task_id, kind, *args, **kwargs):
        if event_task_id == task_id and kind == "archived":
            raise RuntimeError("injected archive event failure")
        return original_append_event(conn, event_task_id, kind, *args, **kwargs)

    monkeypatch.setattr(kb, "_append_event", fail_archive_event)
    from tools import kanban_tools as kt

    refused = json.loads(
        kt._handle_archive({"task_id": task_id, "reason": "rollback probe"})
    )
    assert "injected archive event failure" in refused["error"]
    assert kb._STRICT_ORCHESTRATOR_ARCHIVE_AUTH.get() is False

    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "todo"
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_terminal_write_grants"
        ).fetchone()[0] == 0
        assert not [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "archived"
        ]


def test_archive_requires_reason_and_rejects_unknown_task(orchestrator_env):
    from tools import kanban_tools as kt

    missing_reason = json.loads(kt._handle_archive({"task_id": "t_missing"}))
    assert "reason is required" in missing_reason["error"]

    unknown = json.loads(
        kt._handle_archive({"task_id": "t_missing", "reason": "superseded"})
    )
    assert "not found" in unknown["error"]


def test_archive_running_task_fails_closed_without_reclaim_or_promotion(
    orchestrator_env,
):
    kb = orchestrator_env
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="actively running superseded candidate")
        downstream = kb.create_task(
            conn, title="must remain gated", parents=[task_id],
        )
        claim = kb.claim_task(conn, task_id)
        assert claim is not None
        before = kb.get_task(conn, task_id)
        run_id = before.current_run_id
        assert before.status == "running"
        assert run_id is not None

    from tools import kanban_tools as kt

    refused = json.loads(
        kt._handle_archive({"task_id": task_id, "reason": "superseded"})
    )
    assert "refusing to archive active task" in refused["error"]
    assert "status=running" in refused["error"]

    with kb.connect() as conn:
        after = kb.get_task(conn, task_id)
        assert after.status == "running"
        assert after.current_run_id == run_id
        assert after.claim_lock == before.claim_lock
        assert kb.get_task(conn, downstream).status == "todo"
        run = kb.get_run(conn, run_id)
        assert run.status == "running"
        assert run.ended_at is None
        assert not [e for e in kb.list_events(conn, task_id) if e.kind == "archived"]


def test_archive_fails_closed_on_detached_open_run(orchestrator_env):
    """An inconsistent task row must not bypass an open run obligation."""
    kb = orchestrator_env
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="detached active run")
        assert kb.claim_task(conn, task_id) is not None
        run_id = kb.get_task(conn, task_id).current_run_id
        conn.execute(
            "UPDATE tasks SET status = 'ready', current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ?",
            (task_id,),
        )

    from tools import kanban_tools as kt

    refused = json.loads(
        kt._handle_archive({"task_id": task_id, "reason": "superseded"})
    )
    assert "open_run=set" in refused["error"]

    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
        run = kb.get_run(conn, run_id)
        assert run.status == "running"
        assert run.ended_at is None
        assert not [e for e in kb.list_events(conn, task_id) if e.kind == "archived"]


@pytest.mark.parametrize("status", ["ready", "fenced", "blocked", "done"])
def test_archive_fails_closed_on_expected_status_drift(orchestrator_env, status):
    kb = orchestrator_env
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=f"unexpected {status} candidate")
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))

    from tools import kanban_tools as kt

    refused = json.loads(
        kt._handle_archive({"task_id": task_id, "reason": "superseded"})
    )
    assert f"expected status todo, found {status}" in refused["error"]

    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == status
        assert not [e for e in kb.list_events(conn, task_id) if e.kind == "archived"]


def test_archive_is_orchestrator_only_even_if_handler_is_called_stale(
    orchestrator_env, monkeypatch,
):
    kb = orchestrator_env
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="foreign parent")
        task_id = kb.create_task(
            conn,
            title="foreign child",
            parents=[parent],
            factory_build_gate=1,
        )
        assert kb.get_task(conn, task_id).status == "todo"

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    from tools import kanban_tools as kt

    refused = json.loads(
        kt._handle_archive({"task_id": task_id, "reason": "superseded"})
    )
    assert "orchestrator-only" in refused["error"]
    with kb.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="authenticated receipt"):
            kb.archive_task(
                conn,
                task_id,
                reason="stale direct caller",
                actor="factory-controller",
                source="kanban_archive",
                fail_if_active_run=True,
                expected_status="todo",
            )
        assert kb.get_task(conn, task_id).status == "todo"
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_terminal_write_grants"
        ).fetchone()[0] == 0
