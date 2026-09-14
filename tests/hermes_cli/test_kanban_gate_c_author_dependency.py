"""Gate-C promotion from a canonical FINAL_ACCEPTED audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


_EVIDENCE = {
    "repository": "kiddhu/hermes-agent",
    "pr": 111,
    "head": "a" * 40,
    "tree": "b" * 40,
    "base": "c" * 40,
    "github_review_id": 5193486150,
    "github_review_url": (
        "https://github.com/kiddhu/hermes-agent/pull/111"
        "#pullrequestreview-5193486150"
    ),
    "github_review_state": "APPROVED",
}
_HANDOFF = json.dumps(
    {
        "version": 1,
        "candidate": {
            key: _EVIDENCE[key]
            for key in ("repository", "pr", "head", "tree", "base")
        },
        "summary": "exact Gate-B accepted candidate",
    },
    sort_keys=True,
    separators=(",", ":"),
)
_IDENTITY_BLOCK = (
    "merger_not_author: authoritative implementation identity required; "
    "Native audit-owned canonical PASS is missing, ambiguous, or drifted"
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _snapshot(conn):
    return "\n".join(conn.iterdump()).encode()


def _status(conn, task_id):
    row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert row is not None
    return row["status"]


def _final_accepted_gate_c_shape(conn, *, block_first=True):
    author = kb.create_task(
        conn, title="implementation", assignee=kb.FACTORY_REVIEW_AUTHOR_PROFILE,
        factory_build_gate=0,
    )
    author_claim = kb.claim_task(conn, author)
    assert author_claim is not None and author_claim.current_run_id is not None
    audit = kb.create_task(
        conn, title="independent audit", assignee=kb.FACTORY_REVIEW_AUDITOR_PROFILE,
        parents=[author], factory_build_gate=0,
    )
    assert kb.request_review_handoff(
        conn, author, expected_run_id=author_claim.current_run_id,
        review_task_id=audit, reason=_HANDOFF,
    )
    audit_claim = kb.claim_task(conn, audit)
    assert audit_claim is not None and audit_claim.current_run_id is not None
    assert kb.record_review_verdict(
        conn, author, review_task_id=audit,
        expected_review_run_id=audit_claim.current_run_id,
        verdict="pass", reason="exact head independently approved",
        evidence=_EVIDENCE,
    )
    outcome_row = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id=? "
        "AND kind='canonical_audit_outcome'",
        (audit,),
    ).fetchone()
    outcome = json.loads(outcome_row["payload"])
    assert outcome["disposition"] == "FINAL_ACCEPTED"
    assert outcome["continuation_ids"] == []

    merger = kb.create_task(
        conn, title="role-separated merge", assignee=kb.FACTORY_REVIEW_MERGER_PROFILE,
        parents=[audit], factory_build_gate=0,
    )
    if block_first:
        merger_claim = kb.claim_task(conn, merger)
        assert merger_claim is not None and merger_claim.current_run_id is not None
        assert kb.block_task(
            conn, merger, reason=_IDENTITY_BLOCK, kind="capability",
            expected_run_id=merger_claim.current_run_id,
        )
    kb.link_tasks(conn, author, merger)
    assert _status(conn, author) == "review"
    assert _status(conn, audit) == "done"
    assert _status(conn, merger) == ("blocked" if block_first else "todo")
    return {
        "author": author,
        "author_run": author_claim.current_run_id,
        "audit": audit,
        "audit_run": audit_claim.current_run_id,
        "outcome_event": int(outcome_row["id"]),
        "merger": merger,
    }


def test_final_accepted_author_parent_naturally_recovers_same_merger(kanban_home):
    with kb.connect() as conn:
        shape = _final_accepted_gate_c_shape(conn)
        task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        assert kb.recompute_ready(conn) == 1
        assert _status(conn, shape["author"]) == "review"
        assert _status(conn, shape["audit"]) == "done"
        assert _status(conn, shape["merger"]) == "ready"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == task_count
        promoted = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='promoted' "
            "ORDER BY id DESC LIMIT 1", (shape["merger"],),
        ).fetchone()
        assert json.loads(promoted["payload"]) == {
            "source": "canonical_final_accepted_gate_c",
            "author_task_id": shape["author"],
            "author_run_id": shape["author_run"],
            "audit_task_id": shape["audit"],
            "audit_run_id": shape["audit_run"],
            "canonical_audit_outcome_event_id": shape["outcome_event"],
        }

        committed = _snapshot(conn)
        assert kb.recompute_ready(conn) == 0
        assert _snapshot(conn) == committed


def test_final_accepted_author_parent_promotes_dependency_todo_merger(kanban_home):
    with kb.connect() as conn:
        shape = _final_accepted_gate_c_shape(conn, block_first=False)
        assert kb.recompute_ready(conn) == 1
        assert _status(conn, shape["author"]) == "review"
        assert _status(conn, shape["audit"]) == "done"
        assert _status(conn, shape["merger"]) == "ready"


@pytest.mark.parametrize(
    "drift",
    [
        "missing_outcome",
        "continuation_committed",
        "stale_outcome",
        "wrong_author",
        "wrong_auditor",
        "wrong_merger",
        "active_author_run",
        "active_audit_run",
        "ambiguous_parent",
        "wrong_block_reason",
    ],
)
def test_gate_c_author_binding_drift_fails_closed_zero_mutation(kanban_home, drift):
    with kb.connect() as conn:
        shape = _final_accepted_gate_c_shape(conn)
        if drift == "missing_outcome":
            conn.execute(
                "DELETE FROM task_events WHERE task_id=? "
                "AND kind IN ('canonical_audit_outcome','changed_fact')",
                (shape["audit"],),
            )
        elif drift == "continuation_committed":
            row = conn.execute(
                "SELECT id,payload FROM task_events WHERE id=?",
                (shape["outcome_event"],),
            ).fetchone()
            envelope = json.loads(row["payload"])
            core = {key: value for key, value in envelope.items() if key != "envelope_sha256"}
            core["disposition"] = "CONTINUATION_COMMITTED"
            core["continuation_ids"] = ["already-committed-continuation"]
            digest = hashlib.sha256(
                json.dumps(
                    core, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                ).encode()
            ).hexdigest()
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps({**core, "envelope_sha256": digest}), row["id"]),
            )
            fact = conn.execute(
                "SELECT id,payload FROM task_events WHERE task_id=? AND run_id=? "
                "AND kind='changed_fact'",
                (shape["audit"], shape["audit_run"]),
            ).fetchone()
            pointer = json.loads(fact["payload"])
            pointer["envelope_sha256"] = digest
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(pointer), fact["id"]),
            )
        elif drift == "stale_outcome":
            conn.execute(
                "INSERT INTO task_runs(task_id,profile,status,outcome,started_at,ended_at) "
                "VALUES (?,?, 'blocked','blocked',1,2)",
                (shape["author"], kb.FACTORY_REVIEW_AUTHOR_PROFILE),
            )
        elif drift == "wrong_author":
            conn.execute(
                "UPDATE tasks SET assignee='other-author' WHERE id=?",
                (shape["author"],),
            )
        elif drift == "wrong_auditor":
            conn.execute(
                "UPDATE tasks SET assignee='other-auditor' WHERE id=?",
                (shape["audit"],),
            )
        elif drift == "wrong_merger":
            conn.execute(
                "UPDATE tasks SET assignee='other-merger' WHERE id=?",
                (shape["merger"],),
            )
        elif drift == "active_author_run":
            run_id = conn.execute(
                "INSERT INTO task_runs(task_id,profile,status,started_at) "
                "VALUES (?,?,'running',3)",
                (shape["author"], kb.FACTORY_REVIEW_AUTHOR_PROFILE),
            ).lastrowid
            conn.execute(
                "UPDATE tasks SET current_run_id=? WHERE id=?",
                (run_id, shape["author"]),
            )
        elif drift == "active_audit_run":
            conn.execute(
                "INSERT INTO task_runs(task_id,profile,status,started_at) "
                "VALUES (?,?,'running',3)",
                (shape["audit"], kb.FACTORY_REVIEW_AUDITOR_PROFILE),
            )
        elif drift == "ambiguous_parent":
            extra = kb.create_task(
                conn, title="unrelated completed gate", assignee="other",
                factory_build_gate=0,
            )
            conn.execute(
                "UPDATE tasks SET status='done', completed_at=1 WHERE id=?",
                (extra,),
            )
            kb.link_tasks(conn, extra, shape["merger"])
        elif drift == "wrong_block_reason":
            row = conn.execute(
                "SELECT id, payload FROM task_events WHERE task_id=? "
                "AND kind='blocked' ORDER BY id DESC LIMIT 1",
                (shape["merger"],),
            ).fetchone()
            payload = json.loads(row["payload"])
            payload["reason"] = "human decision still required"
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload, sort_keys=True), row["id"]),
            )
        conn.commit()
        before = _snapshot(conn)

        assert kb.recompute_ready(conn) == 0
        assert _status(conn, shape["merger"]) == "blocked"
        assert _snapshot(conn) == before
