"""Exact terminal-evidence SAME-author AMEND resume regressions."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        kb,
        "REVIEW_AUTHOR_AMEND_INCIDENT_V1",
        dict(kb.REVIEW_AUTHOR_AMEND_INCIDENT_V1),
    )
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _snapshot(conn):
    return "\n".join(conn.iterdump()).encode()


def _fixture(conn):
    controller = kb.create_task(conn, title="controller", assignee="gm2")
    controller_task = kb.claim_task(conn, controller, claimer="gm2:controller")
    assert controller_task and controller_task.current_run_id

    author = kb.create_task(
        conn, title="implementation", assignee="agent007",
        workspace_kind="dir", workspace_path="/tmp/same-workspace",
        provider_override="openai-codex", model_override="gpt-5.6-sol",
    )
    author_task = kb.claim_task(conn, author, claimer="dispatcher:author")
    assert author_task and author_task.current_run_id
    review = kb.create_task(
        conn, title="audit", assignee="bafuxunan", parents=[author],
        workspace_kind="dir", workspace_path="/tmp/same-workspace",
        provider_override="openai-codex", model_override="gpt-5.6-sol",
    )
    handoff = kb.request_review_handoff(
        conn, author, expected_run_id=author_task.current_run_id,
        review_task_id=review, reason="exact candidate submitted",
    )
    assert handoff
    review_task = kb.claim_task(conn, review, claimer="dispatcher:audit")
    assert review_task and review_task.current_run_id
    assert kb.complete_task(
        conn, review, expected_run_id=review_task.current_run_id,
        summary="APPROVED_EXACT_HEAD implementation evidence",
        metadata={"github_review_id": 5191125644},
    )
    fixture = {
        "controller": controller,
        "controller_run": controller_task.current_run_id,
        "author": author,
        "author_run": author_task.current_run_id,
        "review": review,
        "review_run": review_task.current_run_id,
        "handoff": handoff.event_id,
    }
    kb.REVIEW_AUTHOR_AMEND_INCIDENT_V1.update({
        "author_task_id": fixture["author"],
        "author_run_id": fixture["author_run"],
        "review_task_id": fixture["review"],
        "review_run_id": fixture["review_run"],
        "review_handoff_event_id": fixture["handoff"],
    })
    return fixture


def _call(conn, f, **changes):
    args = {
        "author_run_id": f["author_run"],
        "review_task_id": f["review"],
        "review_run_id": f["review_run"],
        "review_handoff_event_id": f["handoff"],
        "amend_reason": kb.REVIEW_AUTHOR_AMEND_REASON_V1,
        "amend_receipt_sha256": kb.REVIEW_AUTHOR_AMEND_INCIDENT_V1["decision_record_sha256"],
        "controller_task_id": f["controller"],
        "controller_run_id": f["controller_run"],
    }
    args.update(changes)
    return kb.resume_reviewed_author_for_amend(conn, f["author"], **args)


def test_pre_fix_paths_cannot_resume_terminal_review_author(kanban_home):
    with kb.connect() as conn:
        f = _fixture(conn)
        assert kb.get_task(conn, f["author"]).status == "review"
        assert kb.get_task(conn, f["review"]).status == "done"
        assert kb.claim_task(conn, f["author"]) is None
        assert kb.recompute_ready(conn) == 0
        assert kb.repromote_blocked_review_child(
            conn, f["author"], author_run_id=f["author_run"],
            review_task_id=f["review"], prior_review_run_id=f["review_run"],
            handoff_receipt_sha256="0" * 64, correction_reason="AMEND",
            controller_task_id=f["controller"], controller_run_id=f["controller_run"],
            exact_candidate={
                "repository": "kiddhu/aion-governance", "pr": 963,
                "head": "a" * 40, "tree": "b" * 40, "base": "c" * 40,
            },
        ) is None
        assert kb.get_task(conn, f["author"]).status == "review"


def test_resume_same_author_preserves_terminal_child_and_lineage(kanban_home):
    with kb.connect() as conn:
        f = _fixture(conn)
        child_before = tuple(conn.execute("SELECT * FROM tasks WHERE id=?", (f["review"],)).fetchone())
        links_before = [tuple(row) for row in conn.execute("SELECT * FROM task_links ORDER BY parent_id, child_id")]
        author_before = kb.get_task(conn, f["author"])
        receipt = _call(conn, f)
        assert receipt and receipt["version"] == 1
        assert receipt["author_target_status"] == "ready"
        assert len(receipt["terminal_evidence_sha256"]) == 64
        assert len(receipt["receipt_sha256"]) == 64
        author_after = kb.get_task(conn, f["author"])
        assert author_after.status == "ready"
        assert author_after.assignee == author_before.assignee == "agent007"
        assert author_after.workspace_path == author_before.workspace_path
        assert author_after.provider_override == author_before.provider_override
        assert author_after.model_override == author_before.model_override
        assert tuple(conn.execute("SELECT * FROM tasks WHERE id=?", (f["review"],)).fetchone()) == child_before
        assert [tuple(row) for row in conn.execute("SELECT * FROM task_links ORDER BY parent_id, child_id")] == links_before
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind='review_verdict'").fetchone()[0] == 0
        claimed = kb.claim_task(conn, f["author"], claimer="dispatcher:resume")
        assert claimed and claimed.current_run_id not in {f["author_run"], f["review_run"]}


def test_exact_replay_is_byte_equivalent_idempotent(kanban_home):
    with kb.connect() as conn:
        f = _fixture(conn)
        first = _call(conn, f)
        assert first
        before = _snapshot(conn)
        second = _call(conn, f)
        assert second == first
        assert _snapshot(conn) == before
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind=?",
            (kb.REVIEW_AUTHOR_AMEND_RESUMED_EVENT_KIND,),
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("author_run_id", 999999),
        ("review_task_id", "t_deadbeef"),
        ("review_run_id", 999999),
        ("review_handoff_event_id", 999999),
        ("controller_run_id", 999999),
        ("amend_receipt_sha256", "0" * 63),
        ("amend_receipt_sha256", "f" * 64),
        ("amend_reason", ""),
        ("amend_reason", "caller-minted AMEND prose"),
    ],
)
def test_stale_wrong_identity_inputs_fail_closed_zero_mutation(kanban_home, field, value):
    with kb.connect() as conn:
        f = _fixture(conn)
        before = _snapshot(conn)
        assert _call(conn, f, **{field: value}) is None
        assert _snapshot(conn) == before


def test_unrelated_future_audit_fails_closed_zero_mutation(kanban_home):
    with kb.connect() as conn:
        _fixture(conn)
        frozen_binding = dict(kb.REVIEW_AUTHOR_AMEND_INCIDENT_V1)
        unrelated = _fixture(conn)
        kb.REVIEW_AUTHOR_AMEND_INCIDENT_V1.clear()
        kb.REVIEW_AUTHOR_AMEND_INCIDENT_V1.update(frozen_binding)
        before = _snapshot(conn)
        assert _call(conn, unrelated) is None
        assert _snapshot(conn) == before


@pytest.mark.parametrize(
    "drift",
    [
        "wrong_role", "already_running", "terminal_author", "wrong_child_parent",
        "stale_latest_author_run", "stale_latest_review_run", "wrong_review_outcome",
        "duplicate_completion", "native_verdict_present",
    ],
)
def test_persisted_drift_fails_closed_zero_mutation(kanban_home, drift):
    with kb.connect() as conn:
        f = _fixture(conn)
        if drift == "wrong_role":
            conn.execute("UPDATE tasks SET assignee='worker' WHERE id=?", (f["author"],))
        elif drift == "already_running":
            conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (f["author_run"], f["author"]))
        elif drift == "terminal_author":
            conn.execute("UPDATE tasks SET status='done', completed_at=1 WHERE id=?", (f["author"],))
        elif drift == "wrong_child_parent":
            conn.execute("DELETE FROM task_links WHERE parent_id=? AND child_id=?", (f["author"], f["review"]))
        elif drift == "stale_latest_author_run":
            conn.execute("INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome) VALUES (?,'agent007','blocked',1,2,'blocked')", (f["author"],))
        elif drift == "stale_latest_review_run":
            conn.execute("INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome) VALUES (?,'bafuxunan','blocked',1,2,'blocked')", (f["review"],))
        elif drift == "wrong_review_outcome":
            conn.execute("UPDATE task_runs SET status='blocked', outcome='blocked' WHERE id=?", (f["review_run"],))
        elif drift == "duplicate_completion":
            row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='completed'", (f["review"], f["review_run"])).fetchone()
            conn.execute("INSERT INTO task_events(task_id,run_id,kind,payload,created_at) VALUES (?,?,'completed',?,3)", (f["review"], f["review_run"], row["payload"]))
        elif drift == "native_verdict_present":
            conn.execute("INSERT INTO task_events(task_id,run_id,kind,payload,created_at) VALUES (?,?,'review_verdict','{}',3)", (f["review"], f["review_run"]))
        conn.commit()
        before = _snapshot(conn)
        assert _call(conn, f) is None
        assert _snapshot(conn) == before


def test_wrong_controller_role_fails_closed(kanban_home, monkeypatch):
    with kb.connect() as conn:
        f = _fixture(conn)
        monkeypatch.setenv("HERMES_PROFILE", "agent007")
        before = _snapshot(conn)
        assert _call(conn, f) is None
        assert _snapshot(conn) == before


def test_gm_controller_role_is_authorized(kanban_home, monkeypatch):
    with kb.connect() as conn:
        f = _fixture(conn)
        conn.execute("UPDATE tasks SET assignee='gm' WHERE id=?", (f["controller"],))
        conn.execute("UPDATE task_runs SET profile='gm' WHERE id=?", (f["controller_run"],))
        conn.commit()
        monkeypatch.setenv("HERMES_PROFILE", "gm")
        receipt = _call(conn, f)
        assert receipt and receipt["controller_profile"] == "gm"
        author = kb.get_task(conn, f["author"])
        assert author is not None and author.status == "ready"


def test_resumed_author_can_amend_then_request_same_audit_child(kanban_home):
    with kb.connect() as conn:
        f = _fixture(conn)
        assert _call(conn, f)
        amended = kb.claim_task(conn, f["author"], claimer="author:amend")
        assert amended and amended.current_run_id is not None
        amended_run_id = amended.current_run_id
        assert amended_run_id > f["author_run"]
        handoff = kb.request_review_handoff(
            conn,
            f["author"],
            expected_run_id=amended_run_id,
            review_task_id=f["review"],
            reason="Amended same PR; request fresh exact-head audit",
        )
        assert handoff and handoff.review_task_id == f["review"]
        author = kb.get_task(conn, f["author"])
        assert author is not None and author.status == "review"
        review = kb.get_task(conn, f["review"])
        assert review is not None and review.status == "ready"


def test_unfinished_parent_returns_author_to_todo(kanban_home):
    with kb.connect() as conn:
        f = _fixture(conn)
        parent = kb.create_task(conn, title="upstream", assignee="worker")
        kb.link_tasks(conn, parent, f["author"])
        receipt = _call(conn, f)
        assert receipt and receipt["author_target_status"] == "todo"
        assert kb.get_task(conn, f["author"]).status == "todo"


def test_concurrent_replay_commits_one_receipt(kanban_home):
    with kb.connect() as conn:
        f = _fixture(conn)
    barrier = threading.Barrier(2)

    def resume():
        with kb.connect() as thread_conn:
            barrier.wait()
            return _call(thread_conn, f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: resume(), range(2)))
    assert results[0] == results[1]
    assert results[0] is not None
    with kb.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind=?",
            (kb.REVIEW_AUTHOR_AMEND_RESUMED_EVENT_KIND,),
        ).fetchone()[0] == 1
