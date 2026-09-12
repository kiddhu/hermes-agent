"""Regression coverage for GM/GM2 SAME-author reviewed-handoff recovery.

The deadlock under test: an author hands a candidate off to a direct
independent auditor child (``request_review_handoff`` -> author ``review``,
child ``ready``), the child runs and fails closed before emitting a Native
verdict (``block_task`` -> child ``blocked``/``triage``), and now no installed
transition can return the author to a runnable state: ``REQUEST_CHANGES``
requires a live running child, ``PASS`` requires a structured handoff
envelope, and neither ``claim_task`` nor ``recompute_ready`` can reach a
``review`` author. ``resume_reviewed_author`` is the narrow authenticated
recovery that returns the SAME author to ``ready`` (or ordinary parent-gated
``todo``) and resets only the SAME audit child to ``todo``.

The recovery is fingerprint-bound: it fires ONLY for a malformed (prose)
handoff envelope whose ``reason`` fails the strict candidate-envelope parser,
and whose direct auditor child's latest run ended ``blocked``. It is issued
only by the durable GM controller lane (``gm``/``gm2``), resolved from
``HERMES_HOME`` (never a bare mutable ``HERMES_PROFILE``).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated board for the reviewed-author recovery fixture."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    # Durable controller identity is resolved from the active Hermes profile
    # (HERMES_HOME), NOT from the mutable HERMES_PROFILE env var.
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "gm2"
    )
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task(conn, task_id):
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def _deadlocked_review(conn):
    """Author in ``review`` + direct auditor child blocked before a verdict."""
    author = kb.create_task(
        conn,
        title="implementation",
        assignee="author",
        workspace_kind="dir",
        workspace_path="/tmp/exact-review-workspace",
    )
    author_run = kb.claim_task(conn, author)
    assert author_run is not None and author_run.current_run_id is not None
    review_task = kb.create_task(
        conn,
        title="independent audit",
        assignee="auditor",
        parents=[author],
        workspace_kind="dir",
        workspace_path="/tmp/exact-review-workspace",
        provider_override="openai-codex",
        model_override="gpt-5.6-sol",
    )
    receipt = kb.request_review_handoff(
        conn,
        author,
        expected_run_id=author_run.current_run_id,
        review_task_id=review_task,
        reason="candidate frozen for independent audit (prose envelope)",
    )
    assert receipt is not None
    review_run = kb.claim_task(conn, review_task, claimer="host:first-review")
    assert review_run is not None and review_run.current_run_id is not None
    assert kb.block_task(
        conn,
        review_task,
        reason="audit failed closed before a Native verdict",
        kind="capability",
        expected_run_id=review_run.current_run_id,
    )
    return author, receipt, review_task, review_run.current_run_id


def _rewrite_handoff_reason(conn, author_task_id, receipt, reason):
    """Rewrite a review_handoff event's reason with a valid receipt hash."""
    core = {
        "version": 1,
        "expected_run_id": receipt.expected_run_id,
        "review_task_id": receipt.review_task_id,
        "reason": reason,
        "recovery": receipt.recovery,
    }
    sha = hashlib.sha256(
        json.dumps(
            {"task_id": author_task_id, **core},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    conn.execute(
        "UPDATE task_events SET payload = ? WHERE id = ?",
        (json.dumps({**core, "receipt_sha256": sha}), receipt.event_id),
    )


_STRUCTURED_REASON = json.dumps({
    "version": 1,
    "candidate": {
        "repository": "kiddhu/hermes-agent",
        "pr": 101,
        "head": "a" * 40,
        "tree": "b" * 40,
        "base": "c" * 40,
    },
    "summary": "structured candidate",
})


def _identity_snapshot(conn, author, review_task):
    author_row = tuple(
        conn.execute("SELECT * FROM tasks WHERE id = ?", (author,)).fetchone()
    )
    child_row = tuple(
        conn.execute("SELECT * FROM tasks WHERE id = ?", (review_task,)).fetchone()
    )
    runs = tuple(
        tuple(r)
        for r in conn.execute(
            "SELECT * FROM task_runs WHERE task_id IN (?, ?) ORDER BY id",
            (author, review_task),
        ).fetchall()
    )
    events = tuple(
        tuple(e)
        for e in conn.execute(
            "SELECT * FROM task_events WHERE task_id IN (?, ?) ORDER BY id",
            (author, review_task),
        ).fetchall()
    )
    return (author_row, child_row, runs, events)


def test_deadlock_has_no_existing_resume_path(kanban_home):
    """RED: a reviewed author whose auditor child blocked has no runnable path."""
    with kb.connect() as conn:
        author, receipt, review_task, review_run_id = _deadlocked_review(conn)
        assert _task(conn, author).status == "review"
        assert _task(conn, review_task).status in {"blocked", "triage"}
        # REQUEST_CHANGES is refused: the child is no longer running.
        assert kb.record_review_verdict(
            conn,
            author,
            review_task_id=review_task,
            expected_review_run_id=review_run_id,
            verdict="request_changes",
            reason="reissue structured handoff",
        ) is False
        # Neither task is claimable.
        assert kb.claim_task(conn, author) is None
        assert kb.claim_task(conn, review_task) is None
        # recompute_ready cannot promote a review author back to ready.
        assert kb.recompute_ready(conn) == 0
        assert _task(conn, author).status == "review"
        assert _task(conn, review_task).status in {"blocked", "triage"}


def test_resume_reviewed_author_recovers_deadlock(kanban_home):
    """GREEN: the recovery returns the SAME author to ready and child to todo."""
    with kb.connect() as conn:
        author, receipt, review_task, review_run_id = _deadlocked_review(conn)
        recovered = kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        )
        assert recovered is not None
        assert recovered["author_task_id"] == author
        assert recovered["audit_task_id"] == review_task
        assert recovered["review_handoff_event_id"] == receipt.event_id
        assert recovered["controller_profile"] == "gm2"
        assert recovered["author_run_id"] == receipt.expected_run_id
        assert recovered["audit_run_id"] == review_run_id
        assert recovered["blocker"] == kb.REVIEW_HANDOFF_MALFORMED_ENVELOPE_BLOCKER
        assert recovered["handoff_reason_sha256"] == hashlib.sha256(
            receipt.reason.encode("utf-8")
        ).hexdigest()
        assert recovered["author_target_status"] == "ready"
        assert len(recovered["receipt_sha256"]) == 64

        assert _task(conn, author).status == "ready"
        assert _task(conn, review_task).status == "todo"

        # One typed recovery event, and no review_verdict was fabricated.
        recovery_events = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ?",
            (author, kb.REVIEW_HANDOFF_RECOVERY_EVENT_KIND),
        ).fetchall()
        assert len(recovery_events) == 1
        assert recovery_events[0]["id"] == recovered["event_id"]
        verdict_count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'review_verdict'",
            (author,),
        ).fetchone()[0]
        assert verdict_count == 0

        # The author is now naturally claimable by the Dispatcher.
        claimed = kb.claim_task(conn, author, claimer="host:dispatcher")
        assert claimed is not None
        assert claimed.current_run_id != receipt.expected_run_id


def test_resume_reviewed_author_is_idempotent(kanban_home):
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        first = kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        )
        assert first is not None
        snapshot = _identity_snapshot(conn, author, review_task)

        second = kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        )
        assert second is not None
        assert second["event_id"] == first["event_id"]
        assert second["receipt_sha256"] == first["receipt_sha256"]
        assert _identity_snapshot(conn, author, review_task) == snapshot

        # A recovery for a DIFFERENT handoff on the same author fails closed.
        different = kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id + 1,
        )
        assert different is None
        assert _identity_snapshot(conn, author, review_task) == snapshot


@pytest.mark.parametrize("controller", ["gm", "gm2"])
def test_resume_reviewed_author_accepts_gm_and_gm2(kanban_home, monkeypatch, controller):
    monkeypatch.setenv("HERMES_PROFILE", controller)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: controller
    )
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        recovered = kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        )
        assert recovered is not None
        assert recovered["controller_profile"] == controller


@pytest.mark.parametrize("controller", ["", "agent007", "bafuxunan", "merger", "worker"])
def test_resume_reviewed_author_rejects_non_gm_controller(kanban_home, monkeypatch, controller):
    monkeypatch.setenv("HERMES_PROFILE", controller)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: controller
    )
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        before = _identity_snapshot(conn, author, review_task)
        with pytest.raises(PermissionError):
            kb.resume_reviewed_author(
                conn,
                author_task_id=author,
                audit_task_id=review_task,
                review_handoff_event_id=receipt.event_id,
            )
        assert _identity_snapshot(conn, author, review_task) == before


def test_resume_reviewed_author_rejects_env_escalation(kanban_home, monkeypatch):
    """A mutable HERMES_PROFILE=gm cannot override a non-GM durable profile."""
    monkeypatch.setenv("HERMES_PROFILE", "gm")
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "agent007"
    )
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        before = _identity_snapshot(conn, author, review_task)
        with pytest.raises(PermissionError):
            kb.resume_reviewed_author(
                conn,
                author_task_id=author,
                audit_task_id=review_task,
                review_handoff_event_id=receipt.event_id,
            )
        assert _identity_snapshot(conn, author, review_task) == before


def test_resume_reviewed_author_rejects_env_mismatch(kanban_home, monkeypatch):
    """A non-GM HERMES_PROFILE contradicting a GM durable profile fails closed."""
    monkeypatch.setenv("HERMES_PROFILE", "agent007")
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "gm2"
    )
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        with pytest.raises(PermissionError):
            kb.resume_reviewed_author(
                conn,
                author_task_id=author,
                audit_task_id=review_task,
                review_handoff_event_id=receipt.event_id,
            )


def test_resume_reviewed_author_rejects_delegated_child(kanban_home, monkeypatch):
    """A delegate_task child can never act as the GM controller."""
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    with kb.connect() as conn:
        before = _identity_snapshot(conn, author, review_task)
        with pytest.raises(PermissionError):
            kb.resume_reviewed_author(
                conn,
                author_task_id=author,
                audit_task_id=review_task,
                review_handoff_event_id=receipt.event_id,
            )
        assert _identity_snapshot(conn, author, review_task) == before


def test_resume_reviewed_author_cli_denylist_and_delegated_guard(kanban_home, monkeypatch):
    """The CLI fast-fail denylist covers ``resume-reviewed-author``."""
    from hermes_cli import kanban

    assert "resume-reviewed-author" in kanban._DELEGATED_CHILD_DENIED_ACTIONS

    class _Args:
        kanban_action = "resume-reviewed-author"
        boards_action = None

    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    assert kanban._is_delegated_child_cli_mutation(_Args()) is True
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT")
    assert kanban._is_delegated_child_cli_mutation(_Args()) is False


def test_resume_reviewed_author_rejects_valid_structured_handoff(kanban_home):
    """The recovery must NOT fire for an already-valid structured candidate."""
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        _rewrite_handoff_reason(conn, author, receipt, _STRUCTURED_REASON)
        conn.commit()
        before = _identity_snapshot(conn, author, review_task)
        assert kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        ) is None
        assert _identity_snapshot(conn, author, review_task) == before


def test_resume_reviewed_author_rejects_timed_out_audit_run(kanban_home):
    """The recovery requires the exact latest audit run to end ``blocked``."""
    with kb.connect() as conn:
        author, receipt, review_task, review_run_id = _deadlocked_review(conn)
        conn.execute(
            "UPDATE task_runs SET status = 'timed_out', outcome = 'timed_out' "
            "WHERE id = ?",
            (review_run_id,),
        )
        conn.commit()
        before = _identity_snapshot(conn, author, review_task)
        assert kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        ) is None
        assert _identity_snapshot(conn, author, review_task) == before


def test_resume_reviewed_author_parent_gated_todo(kanban_home):
    """An author with an unfinished parent returns to ``todo``, not ``ready``."""
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)
        upstream = kb.create_task(conn, title="unfinished parent", assignee="upstream")
        kb.link_tasks(conn, upstream, author)
        recovered = kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=receipt.event_id,
        )
        assert recovered is not None
        assert recovered["author_target_status"] == "todo"
        assert _task(conn, author).status == "todo"
        assert _task(conn, review_task).status == "todo"


@pytest.mark.parametrize(
    "invalidity",
    [
        "missing_handoff_event",
        "mismatched_audit",
        "self_audit",
        "author_not_review",
        "author_active_identity",
        "audit_active_identity",
        "multiple_parent",
        "existing_verdict",
        "terminal_author",
        "active_author_run",
        "stale_author_run",
    ],
)
def test_resume_reviewed_author_fails_closed_without_mutation(kanban_home, invalidity):
    with kb.connect() as conn:
        author, receipt, review_task, review_run_id = _deadlocked_review(conn)

        if invalidity == "missing_handoff_event":
            event_id = receipt.event_id + 9999
        elif invalidity == "mismatched_audit":
            event_id = receipt.event_id
            other = kb.create_task(conn, title="other audit", assignee="other-auditor")
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (json.dumps({
                    "version": 1, "expected_run_id": receipt.expected_run_id,
                    "review_task_id": other, "reason": "prose",
                    "recovery": False, "receipt_sha256": "0" * 64,
                }), receipt.event_id),
            )
        elif invalidity == "self_audit":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE tasks SET assignee = 'author' WHERE id = ?", (review_task,)
            )
        elif invalidity == "author_not_review":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ?", (author,)
            )
        elif invalidity == "author_active_identity":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (receipt.expected_run_id, author),
            )
        elif invalidity == "audit_active_identity":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (review_run_id, review_task),
            )
        elif invalidity == "multiple_parent":
            event_id = receipt.event_id
            other = kb.create_task(conn, title="extra parent", assignee="builder")
            kb.link_tasks(conn, other, review_task)
        elif invalidity == "existing_verdict":
            event_id = receipt.event_id
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'review_verdict', ?, ?)",
                (author, review_run_id, json.dumps({
                    "version": 1, "review_task_id": review_task,
                    "review_run_id": review_run_id, "verdict": "request_changes",
                    "reason": "prior verdict",
                }), 1),
            )
        elif invalidity == "terminal_author":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = 1 WHERE id = ?",
                (author,),
            )
        elif invalidity == "active_author_run":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE task_runs SET ended_at = NULL WHERE id = ?",
                (receipt.expected_run_id,),
            )
        elif invalidity == "stale_author_run":
            event_id = receipt.event_id
            conn.execute(
                "UPDATE task_runs SET status = 'blocked', outcome = 'blocked' WHERE id = ?",
                (receipt.expected_run_id,),
            )
        else:  # pragma: no cover
            raise AssertionError(invalidity)
        conn.commit()

        before = _identity_snapshot(conn, author, review_task)
        assert kb.resume_reviewed_author(
            conn,
            author_task_id=author,
            audit_task_id=review_task,
            review_handoff_event_id=event_id,
        ) is None
        assert _identity_snapshot(conn, author, review_task) == before
        assert _task(conn, author).status == ("review" if invalidity not in {
            "author_not_review", "terminal_author",
        } else ("todo" if invalidity == "author_not_review" else "done"))


def test_resume_reviewed_author_is_concurrent_safe(kanban_home):
    with kb.connect() as conn:
        author, receipt, review_task, _ = _deadlocked_review(conn)

    barrier = threading.Barrier(2)

    def resume():
        with kb.connect() as thread_conn:
            barrier.wait()
            return kb.resume_reviewed_author(
                thread_conn,
                author_task_id=author,
                audit_task_id=review_task,
                review_handoff_event_id=receipt.event_id,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: resume(), range(2)))

    # Exactly one logical recovery event, and both callers observe the same
    # receipt (one winner, one idempotent replay).
    receipts = [r for r in results if r is not None]
    assert len(receipts) == 2
    assert receipts[0]["event_id"] == receipts[1]["event_id"]
    assert receipts[0]["receipt_sha256"] == receipts[1]["receipt_sha256"]

    with kb.connect() as conn:
        assert _task(conn, author).status == "ready"
        assert _task(conn, review_task).status == "todo"
        recovery_events = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = ?",
            (author, kb.REVIEW_HANDOFF_RECOVERY_EVENT_KIND),
        ).fetchall()
        assert len(recovery_events) == 1
