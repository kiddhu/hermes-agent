from __future__ import annotations

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


EVIDENCE = {
    "repository": "kiddhu/hermes-agent",
    "pr": 105,
    "head": "d" * 40,
    "tree": "b" * 40,
    "base": "9bdb8f9401d61836297f877995f6646deac4098b",
    "github_review_id": 5186993241,
    "github_review_url": (
        "https://github.com/kiddhu/hermes-agent/pull/105#pullrequestreview-5186993241"
    ),
    "github_review_state": "APPROVED",
}
PASS_REASON = "Exact-head review is APPROVED and all focused gates pass."
LEGACY_HANDOFF = (
    "AION-790 repair R2: PR kiddhu/hermes-agent#105 new head "
    f"{EVIDENCE['head']} (tree {EVIDENCE['tree']}, base "
    f"{EVIDENCE['base'][:10]}). Candidate is frozen for independent audit."
)


@pytest.fixture
def isolated_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with kb.isolated_kanban_env(tmp_path):
        kb.init_db()
        yield


def _fixture(conn: sqlite3.Connection, monkeypatch) -> dict:
    author = kb.create_task(conn, title="author", assignee="agent007")
    auditor = kb.create_task(
        conn,
        title="same child",
        assignee="bafuxunan",
        parents=[author],
    )

    author_run_1 = kb.claim_task(conn, author)
    assert author_run_1 is not None
    assert author_run_1.current_run_id is not None
    assert (
        kb.request_review_handoff(
            conn,
            author,
            expected_run_id=author_run_1.current_run_id,
            review_task_id=auditor,
            reason="round one",
        )
        is not None
    )
    audit_run_1 = kb.claim_task(conn, auditor)
    assert audit_run_1 is not None
    assert audit_run_1.current_run_id is not None
    assert kb.record_review_verdict(
        conn,
        author,
        review_task_id=auditor,
        expected_review_run_id=audit_run_1.current_run_id,
        verdict="request_changes",
        reason="one exact-head repair required",
    )

    author_run_2 = kb.claim_task(conn, author)
    assert author_run_2 is not None
    assert author_run_2.current_run_id is not None
    assert (
        kb.request_review_handoff(
            conn,
            author,
            expected_run_id=author_run_2.current_run_id,
            review_task_id=auditor,
            reason=LEGACY_HANDOFF,
        )
        is not None
    )
    audit_run_2 = kb.claim_task(conn, auditor)
    assert audit_run_2 is not None
    assert audit_run_2.current_run_id is not None
    metadata = {
        "review_outcome": "approved_exact_head",
        "exact_candidate": {
            key: EVIDENCE[key] for key in ("repository", "pr", "head", "tree", "base")
        },
        "github_review": {
            "id": EVIDENCE["github_review_id"],
            "url": EVIDENCE["github_review_url"],
            "state": EVIDENCE["github_review_state"],
        },
        "tests": {"focused": "PASS"},
    }
    conn.execute(
        "UPDATE task_runs SET status='done',outcome='completed',summary=?,metadata=?,"
        "ended_at=strftime('%s','now') WHERE id=? AND task_id=?",
        (PASS_REASON, json.dumps(metadata), audit_run_2.current_run_id, auditor),
    )
    conn.execute(
        "UPDATE tasks SET status='done',current_run_id=NULL,claim_lock=NULL,"
        "claim_expires=NULL,worker_pid=NULL,worker_starttime=NULL,"
        "fence_lineage=NULL,fence_disposition=NULL WHERE id=?",
        (auditor,),
    )
    conn.execute(
        "UPDATE tasks SET factory_build_gate=1 WHERE id IN (?,?)",
        (author, auditor),
    )
    conn.commit()

    original_auth = kb._authenticated_factory_run_metadata

    def authenticated(run_conn, task_id):
        if task_id == auditor:
            return audit_run_2.current_run_id, "bafuxunan", metadata
        return original_auth(run_conn, task_id)

    monkeypatch.setattr(kb, "_authenticated_factory_run_metadata", authenticated)

    controller = kb.create_task(conn, title="controller", assignee="gm2")
    controller_run = kb.claim_task(conn, controller)
    assert controller_run is not None
    return {
        "author": author,
        "auditor": auditor,
        "author_run_1": author_run_1.current_run_id,
        "author_run_2": author_run_2.current_run_id,
        "audit_run_1": audit_run_1.current_run_id,
        "audit_run_2": audit_run_2.current_run_id,
        "controller": controller,
        "controller_run": controller_run.current_run_id,
    }


def _recover(conn, fixture, **changes):
    values = {
        "task_id": fixture["author"],
        "review_task_id": fixture["auditor"],
        "expected_review_run_id": fixture["audit_run_2"],
        "verdict": "pass",
        "reason": PASS_REASON,
        "evidence": copy.deepcopy(EVIDENCE),
        "recover_completed": True,
        "controller_task_id": fixture["controller"],
        "controller_run_id": fixture["controller_run"],
        "controller_profile": "gm2",
    }
    values.update(changes)
    return kb.record_review_verdict(conn, **values)


def _author_event_bytes(conn, task_id):
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT id,run_id,kind,payload,created_at FROM task_events "
            "WHERE task_id=? ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def test_latest_completed_same_child_pass_recovery_and_replay(
    isolated_board, monkeypatch
):
    with kb.connect() as conn:
        fixture = _fixture(conn, monkeypatch)
        before = _author_event_bytes(conn, fixture["author"])
        assert _recover(conn, fixture)
        after = _author_event_bytes(conn, fixture["author"])
        assert len(after) == len(before) + 1
        verdicts = conn.execute(
            "SELECT run_id,payload FROM task_events WHERE task_id=? "
            "AND kind='review_verdict' ORDER BY id",
            (fixture["author"],),
        ).fetchall()
        assert [row["run_id"] for row in verdicts] == [
            fixture["audit_run_1"],
            fixture["audit_run_2"],
        ]
        assert json.loads(verdicts[0]["payload"])["verdict"] == "request_changes"
        recovered = json.loads(verdicts[1]["payload"])
        assert recovered["verdict"] == "pass"
        assert recovered["recovery"] is True
        receipt = kb._canonical_audit_receipt(conn, fixture["author"])
        assert receipt is not None
        assert receipt["authenticated"] is True
        assert receipt["auditor_run_id"] == fixture["audit_run_2"]

        replay_before = _author_event_bytes(conn, fixture["author"])
        assert _recover(conn, fixture)
        assert _author_event_bytes(conn, fixture["author"]) == replay_before

        assert kb.complete_task(
            conn,
            fixture["controller"],
            summary="controller recovery committed",
            expected_run_id=fixture["controller_run"],
        )
        terminal_receipt = kb._canonical_audit_receipt(conn, fixture["author"])
        assert terminal_receipt is not None
        assert terminal_receipt["auditor_run_id"] == fixture["audit_run_2"]


def test_recovered_receipt_rejects_non_successful_controller(
    isolated_board,
    monkeypatch,
):
    with kb.connect() as conn:
        fixture = _fixture(conn, monkeypatch)
        assert _recover(conn, fixture)
        conn.execute(
            "UPDATE task_runs SET status='blocked',outcome='blocked',ended_at=started_at "
            "WHERE id=? AND task_id=?",
            (fixture["controller_run"], fixture["controller"]),
        )
        conn.execute(
            "UPDATE tasks SET status='blocked',current_run_id=NULL,claim_lock=NULL,"
            "claim_expires=NULL,worker_pid=NULL,worker_starttime=NULL,"
            "fence_lineage=NULL,fence_disposition=NULL WHERE id=?",
            (fixture["controller"],),
        )
        conn.commit()
        assert kb._canonical_audit_receipt(conn, fixture["author"]) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "stale_run",
        "head_drift",
        "wrong_controller",
        "wrong_role",
        "wrong_edge",
        "missing_generation",
        "malformed_evidence",
    ],
)
def test_latest_completed_same_child_pass_recovery_hostile_zero_mutation(
    isolated_board,
    monkeypatch,
    mutation,
):
    with kb.connect() as conn:
        fixture = _fixture(conn, monkeypatch)
        kwargs = {}
        if mutation == "stale_run":
            kwargs["expected_review_run_id"] = fixture["audit_run_1"]
        elif mutation == "head_drift":
            bad = copy.deepcopy(EVIDENCE)
            bad["head"] = "a" * 40
            kwargs["evidence"] = bad
        elif mutation == "wrong_controller":
            kwargs["controller_profile"] = "agent007"
        elif mutation == "wrong_role":
            conn.execute(
                "UPDATE tasks SET assignee='gm' WHERE id=?", (fixture["auditor"],)
            )
            conn.commit()
        elif mutation == "wrong_edge":
            sibling = kb.create_task(conn, title="extra parent", assignee="gm")
            kb.link_tasks(conn, sibling, fixture["auditor"])
        elif mutation == "missing_generation":
            conn.execute(
                "INSERT INTO task_runs(task_id,profile,status,outcome,started_at,ended_at) "
                "VALUES(?,?, 'review_required','review_required',1,2)",
                (fixture["author"], "agent007"),
            )
            conn.commit()
        elif mutation == "malformed_evidence":
            bad = copy.deepcopy(EVIDENCE)
            del bad["tree"]
            kwargs["evidence"] = bad
        before = _author_event_bytes(conn, fixture["author"])
        assert not _recover(conn, fixture, **kwargs)
        assert _author_event_bytes(conn, fixture["author"]) == before


def test_latest_completed_same_child_pass_recovery_concurrent_idempotency(
    isolated_board,
    monkeypatch,
):
    with kb.connect() as conn:
        fixture = _fixture(conn, monkeypatch)

    def worker():
        with kb.connect() as thread_conn:
            return _recover(thread_conn, fixture)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: worker(), range(2)))
    assert results == [True, True]
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='review_verdict' "
            "AND run_id=?",
            (fixture["author"], fixture["audit_run_2"]),
        ).fetchall()
        assert len(rows) == 1
