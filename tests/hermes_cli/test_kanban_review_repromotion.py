"""Exact-generation recovery for a blocked same-child audit."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

CANDIDATE = {
    "repository": "kiddhu/aion-governance",
    "pr": 961,
    "head": "650819825c0d92b819e33b26e8ebc4c32ab1fd56",
    "tree": "edebc7f8d22099f19d82437f8db77532107e1e77",
    "base": "adfccfef42a26df3e1c78fe311d1cae36a036ff2",
}

CANDIDATE_HANDOFF = json.dumps(
    {
        "version": 1,
        "candidate": CANDIDATE,
        "summary": "PR961 exact candidate is ready for independent audit",
    },
    sort_keys=True,
    separators=(",", ":"),
)

GENERIC_PROSE_HANDOFF = (
    "PR961 exact candidate 650819825c0d92b819e33b26e8ebc4c32ab1fd56 "
    "(tree edebc7f8d22099f19d82437f8db77532107e1e77, "
    "base adfccfef42a26df3e1c78fe311d1cae36a036ff2) "
    "is OPEN/CLEAN/MERGEABLE with 3/3 hosted checks PASS"
)

PR109_CANDIDATE = {
    "repository": "kiddhu/hermes-agent",
    "pr": 109,
    "head": "ec758e25523d2a618744f315f33836f41ac1a44c",
    "tree": "4a0d87a9900c89a1f5431a402db4fe7bec6fa2e7",
    "base": "edcc2d39258739cd0625366d91b20bac9a6a8096",
}

PR109_PROSE_HANDOFF = (
    "Round-3 repair candidate PR #109 is frozen at exact head "
    "ec758e25523d2a618744f315f33836f41ac1a44c "
    "(tree 4a0d87a9900c89a1f5431a402db4fe7bec6fa2e7, "
    "base edcc2d39258739cd0625366d91b20bac9a6a8096). "
    "The ambiguous historical live-identity gap now fails closed inside the "
    "terminal transaction with byte-equivalent state; deterministic RED, 753 "
    "relevant tests, and all hosted CI checks pass.Same role-separated audit "
    "child must issue a fresh commit-bound round-3 verdict."
)

PR962_CANDIDATE = {
    "repository": "kiddhu/aion-governance",
    "pr": 962,
    "head": "005d93b990f9b67962648ecf7c03d1e5d873dae6",
    "tree": "0313921167562df87490c31af9fa44f2661bcbd4",
    "base": "adfccfef42a26df3e1c78fe311d1cae36a036ff2",
}

PR962_PROSE_HANDOFF = (
    "ASTRA Gate B ACCEPT is bound to PR962 exact head "
    "005d93b990f9b67962648ecf7c03d1e5d873dae6 "
    "(tree 0313921167562df87490c31af9fa44f2661bcbd4, "
    "base adfccfef42a26df3e1c78fe311d1cae36a036ff2). "
    "Live OPEN/CLEAN/MERGEABLE readback and 3/3 hosted PASS checks verified "
    "immediately before handoff; exact audit packet is comment 7290. Fresh "
    "role-separated bafuxunan audit must independently verify incident binding, "
    "retirement, ambiguity/foreign-shape fail-closed behavior, and zero mutation "
    "before any Gate C action."
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _shape(conn, *, handoff_reason=None):
    author = kb.create_task(
        conn,
        title="author",
        assignee="agent007",
        body="formal_record: https://github.com/kiddhu/aion-governance/issues/790",
    )
    author_claim = kb.claim_task(conn, author, claimer="host:author")
    assert author_claim and author_claim.current_run_id
    author_run = author_claim.current_run_id
    child = kb.create_task(
        conn, title="audit", assignee="bafuxunan", parents=[author]
    )
    handoff = kb.request_review_handoff(
        conn,
        author,
        expected_run_id=author_run,
        review_task_id=child,
        reason=handoff_reason or CANDIDATE_HANDOFF,
    )
    assert handoff
    child_claim = kb.claim_task(conn, child, claimer="host:auditor")
    assert child_claim and child_claim.current_run_id
    first_child_run = child_claim.current_run_id
    assert kb.block_task(
        conn,
        child,
        reason="installed runtime was incorrectly required before audit PASS",
        kind="needs_input",
        expected_run_id=first_child_run,
    )
    blocked_child = kb.get_task(conn, child)
    assert blocked_child is not None and blocked_child.status == "blocked"
    assert kb.unblock_task(conn, child)
    kb.recompute_ready(conn)
    second_claim = kb.claim_task(conn, child, claimer="host:auditor-retry")
    assert second_claim and second_claim.current_run_id
    child_run = second_claim.current_run_id
    assert kb.block_task(
        conn,
        child,
        reason="same-child re-promotion requires a live author run",
        kind="dependency",
        expected_run_id=child_run,
    )
    assert kb.get_task(conn, child).status == "todo"
    controller = kb.create_task(conn, title="GM correction", assignee="gm2")
    controller_claim = kb.claim_task(
        conn, controller, claimer="gm2:controller"
    )
    assert controller_claim and controller_claim.current_run_id
    return (
        author, author_run, child, child_run, handoff,
        controller, controller_claim.current_run_id,
    )


def _call(conn, shape, *, reason="audit first; install follows", **changes):
    author, author_run, child, child_run, handoff, controller, controller_run = shape
    args = {
        "author_task_id": author,
        "author_run_id": author_run,
        "review_task_id": child,
        "prior_review_run_id": child_run,
        "handoff_receipt_sha256": handoff.receipt_sha256,
        "correction_reason": reason,
        "controller_task_id": controller,
        "controller_run_id": controller_run,
        "exact_candidate": CANDIDATE,
    }
    args.update(changes)
    return kb.repromote_blocked_review_child(conn, **args)


def _allow_historical_pr109_incident(monkeypatch, shape):
    author, author_run, child, child_run, handoff, *_ = shape
    monkeypatch.setattr(kb, "FACTORY_HISTORICAL_PROSE_REPROMOTION_INCIDENT", {
        "author_task_id": author,
        "author_run_id": author_run,
        "review_task_id": child,
        "prior_review_run_id": child_run,
        "handoff_receipt_sha256": handoff.receipt_sha256,
        "repository": PR109_CANDIDATE["repository"],
        "pr": PR109_CANDIDATE["pr"],
        "base": PR109_CANDIDATE["base"],
        "handoff_reason": PR109_PROSE_HANDOFF,
    })


def _history(conn, author, child):
    return (
        tuple(tuple(row) for row in conn.execute(
            "SELECT id, task_id, profile, status, outcome, summary, ended_at "
            "FROM task_runs WHERE task_id IN (?, ?) ORDER BY id",
            (author, child),
        )),
        tuple(tuple(row) for row in conn.execute(
            "SELECT id, task_id, run_id, kind, payload, created_at FROM task_events "
            "WHERE task_id IN (?, ?) AND kind != 'review_repromoted' "
            "AND NOT (task_id = ? AND kind = 'promoted') ORDER BY id",
            (author, child, child),
        )),
    )


def test_existing_tool_cannot_repromote_without_live_author(kanban_home, monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "agent007")
    with kb.connect() as conn:
        shape = _shape(conn)
        author, _, child, _, handoff, *_ = shape
    monkeypatch.setenv("HERMES_KANBAN_TASK", author)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    result = json.loads(kt._handle_request_review({
        "task_id": author,
        "review_task_id": child,
        "reason": handoff.reason,
    }))
    assert "current dispatcher run id is required" in result["error"]


def test_copied_live_two_blocked_run_shape_uses_latest_generation(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        child, latest_run = shape[2], shape[3]
        runs = conn.execute(
            "SELECT id, status, outcome FROM task_runs WHERE task_id=? ORDER BY id",
            (child,),
        ).fetchall()
        assert [(row["status"], row["outcome"]) for row in runs] == [
            ("blocked", "blocked"),
            ("blocked", "blocked"),
        ]
        assert int(runs[-1]["id"]) == latest_run
        assert _call(conn, shape) is not None


def test_gm_repromotes_one_exact_generation_and_replay_is_idempotent(
    kanban_home, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        author, _, child, child_run, *_ = shape
        history = _history(conn, author, child)
        blocked_state = conn.execute(
            "SELECT block_kind, block_recurrences FROM tasks WHERE id=?", (child,)
        ).fetchone()
        receipt = _call(conn, shape)
        assert receipt is not None
        assert receipt.correction_actor == "gm2"
        assert kb.get_task(conn, author).status == "review"
        assert kb.get_task(conn, child).status == "ready"
        assert _history(conn, author, child) == history
        assert tuple(conn.execute(
            "SELECT block_kind, block_recurrences FROM tasks WHERE id=?", (child,)
        ).fetchone()) == tuple(blocked_state)
        replay = _call(conn, shape)
        assert replay == receipt
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? "
            "AND kind='review_repromoted' AND run_id=?",
            (author, child_run),
        ).fetchone()[0] == 1


def test_repromotion_converts_copied_pr109_prose_handoff_to_strict_pass_target(
    kanban_home, monkeypatch,
):
    """RED for the live run4654 -> run4655 PASS-binding refusal."""
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn, handoff_reason=PR109_PROSE_HANDOFF)
        _allow_historical_pr109_incident(monkeypatch, shape)
        author, _, child, *_ = shape
        receipt = _call(
            conn,
            shape,
            exact_candidate=PR109_CANDIDATE,
            reason="PR109 strict PASS target recovery",
        )
        assert receipt is not None
        assert json.loads(receipt.strict_handoff_reason) == {
            "version": 1,
            "candidate": PR109_CANDIDATE,
            "summary": "PR109 strict PASS target recovery",
        }

        claimed = kb.claim_task(conn, child, claimer="host:fresh-auditor")
        assert claimed is not None and claimed.current_run_id is not None
        evidence = {
            **PR109_CANDIDATE,
            "github_review_id": 5189891566,
            "github_review_url": (
                "https://github.com/kiddhu/hermes-agent/pull/109"
                "#pullrequestreview-5189891566"
            ),
            "github_review_state": "APPROVED",
        }
        assert kb.record_review_verdict(
            conn,
            author,
            review_task_id=child,
            expected_review_run_id=claimed.current_run_id,
            verdict="pass",
            reason="fresh exact-head PASS",
            evidence=evidence,
        )


def test_frozen_pr109_incident_matches_unpatched_exact_live_tuple(
    kanban_home, monkeypatch,
):
    """The immutable run4654 handoff matches without patching migration data."""
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    task_ids = iter(["t_1d9142bd", "t_d2afcef5", "t_exact_controller"])
    monkeypatch.setattr(kb, "_new_task_id", lambda: next(task_ids))

    with kb.connect() as conn:
        conn.execute("DELETE FROM sqlite_sequence WHERE name='task_runs'")
        conn.execute(
            "INSERT INTO sqlite_sequence(name, seq) VALUES ('task_runs', 4653)"
        )
        author = kb.create_task(conn, title="author", assignee="agent007")
        author_claim = kb.claim_task(conn, author, claimer="host:author")
        assert author_claim is not None and author_claim.current_run_id == 4654
        child = kb.create_task(
            conn, title="audit", assignee="bafuxunan", parents=[author]
        )
        handoff = kb.request_review_handoff(
            conn,
            author,
            expected_run_id=4654,
            review_task_id=child,
            reason=PR109_PROSE_HANDOFF,
        )
        assert handoff is not None
        assert handoff.receipt_sha256 == (
            "1725939ccc2beb18d9555d1d7dc4e9039f01415cc5525b9d126bbc87c9429587"
        )
        child_claim = kb.claim_task(conn, child, claimer="host:auditor")
        assert child_claim is not None and child_claim.current_run_id == 4655
        assert kb.block_task(
            conn,
            child,
            reason="same-child re-promotion requires strict target recovery",
            kind="dependency",
            expected_run_id=4655,
        )
        controller = kb.create_task(conn, title="GM correction", assignee="gm2")
        controller_claim = kb.claim_task(
            conn, controller, claimer="gm2:controller"
        )
        assert controller_claim is not None
        shape = (
            author,
            4654,
            child,
            4655,
            handoff,
            controller,
            controller_claim.current_run_id,
        )

        receipt = _call(
            conn,
            shape,
            exact_candidate=PR109_CANDIDATE,
            reason="PR109 strict PASS target recovery",
        )
        assert receipt is not None
        assert receipt.strict_handoff_reason is not None
        assert json.loads(receipt.strict_handoff_reason) == {
            "version": 1,
            "candidate": PR109_CANDIDATE,
            "summary": "PR109 strict PASS target recovery",
        }


def _frozen_pr962_shape(conn, monkeypatch):
    task_ids = iter(["t_9fd7330a", "t_be1bf698", "t_exact_controller"])
    monkeypatch.setattr(kb, "_new_task_id", lambda: next(task_ids))
    conn.execute("DELETE FROM sqlite_sequence WHERE name='task_runs'")
    conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('task_runs', 4684)")
    author = kb.create_task(conn, title="author", assignee="agent007")
    author_claim = kb.claim_task(conn, author, claimer="host:author")
    assert author_claim is not None and author_claim.current_run_id == 4685
    child = kb.create_task(
        conn, title="audit", assignee="bafuxunan", parents=[author]
    )
    handoff = kb.request_review_handoff(
        conn,
        author,
        expected_run_id=4685,
        review_task_id=child,
        reason=PR962_PROSE_HANDOFF,
    )
    assert handoff is not None
    assert handoff.receipt_sha256 == (
        "148d46fae8368244a3a4d6f0122e5325788370f43c5a98ba1b81f4c65c3c3c23"
    )
    child_claim = kb.claim_task(conn, child, claimer="host:auditor")
    assert child_claim is not None and child_claim.current_run_id == 4686
    assert kb.block_task(
        conn,
        child,
        reason="same-child re-promotion requires strict target recovery",
        kind="dependency",
        expected_run_id=4686,
    )
    controller = kb.create_task(conn, title="GM correction", assignee="gm2")
    controller_claim = kb.claim_task(conn, controller, claimer="gm2:controller")
    assert controller_claim is not None
    return (
        author,
        4685,
        child,
        4686,
        handoff,
        controller,
        controller_claim.current_run_id,
    )


def test_frozen_pr962_incident_binds_strict_target_and_fresh_pass(
    kanban_home, monkeypatch,
):
    """The preserved run4685 -> run4686 prose handoff gets one strict target."""
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _frozen_pr962_shape(conn, monkeypatch)
        author, _, child, *_ = shape
        receipt = _call(
            conn,
            shape,
            exact_candidate=PR962_CANDIDATE,
            reason="PR962 strict PASS target recovery",
        )
        assert receipt is not None
        assert receipt.strict_handoff_reason is not None
        assert json.loads(receipt.strict_handoff_reason) == {
            "version": 1,
            "candidate": PR962_CANDIDATE,
            "summary": "PR962 strict PASS target recovery",
        }
        claimed = kb.claim_task(conn, child, claimer="host:fresh-auditor")
        assert claimed is not None and claimed.current_run_id is not None
        evidence = {
            **PR962_CANDIDATE,
            "github_review_id": 5190570082,
            "github_review_url": (
                "https://github.com/kiddhu/aion-governance/pull/962"
                "#pullrequestreview-5190570082"
            ),
            "github_review_state": "APPROVED",
        }
        assert kb.record_review_verdict(
            conn,
            author,
            review_task_id=child,
            expected_review_run_id=claimed.current_run_id,
            verdict="pass",
            reason="fresh exact-head PASS",
            evidence=evidence,
        )


@pytest.mark.parametrize("field", ["repository", "pr", "head", "tree", "base"])
def test_frozen_pr962_incident_rejects_candidate_drift_without_mutation(
    kanban_home, monkeypatch, field,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _frozen_pr962_shape(conn, monkeypatch)
        candidate = dict(PR962_CANDIDATE)
        candidate[field] = (
            963 if field == "pr"
            else "wrong/repo" if field == "repository"
            else "1" * 40
        )
        before = "\n".join(conn.iterdump())
        assert _call(
            conn,
            shape,
            exact_candidate=candidate,
            reason="must fail closed",
        ) is None
        assert "\n".join(conn.iterdump()) == before


@pytest.mark.parametrize("field", ["repository", "pr", "head", "tree", "base"])
def test_repromotion_strict_target_rejects_drifted_pass_evidence_without_mutation(
    kanban_home, monkeypatch, field,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn, handoff_reason=PR109_PROSE_HANDOFF)
        _allow_historical_pr109_incident(monkeypatch, shape)
        author, _, child, *_ = shape
        assert _call(
            conn,
            shape,
            exact_candidate=PR109_CANDIDATE,
            reason="PR109 strict PASS target recovery",
        ) is not None
        claimed = kb.claim_task(conn, child, claimer="host:fresh-auditor")
        assert claimed is not None and claimed.current_run_id is not None
        evidence = {
            **PR109_CANDIDATE,
            "github_review_id": 5189891566,
            "github_review_url": (
                "https://github.com/kiddhu/hermes-agent/pull/109"
                "#pullrequestreview-5189891566"
            ),
            "github_review_state": "APPROVED",
        }
        evidence[field] = (
            110 if field == "pr"
            else "wrong/repo" if field == "repository"
            else "1" * 40
        )
        if field in {"repository", "pr"}:
            evidence["github_review_url"] = (
                f"https://github.com/{evidence['repository']}/pull/{evidence['pr']}"
                "#pullrequestreview-5189891566"
            )
        before = "\n".join(conn.iterdump())
        assert not kb.record_review_verdict(
            conn,
            author,
            review_task_id=child,
            expected_review_run_id=claimed.current_run_id,
            verdict="pass",
            reason="must fail closed",
            evidence=evidence,
        )
        assert "\n".join(conn.iterdump()) == before


@pytest.mark.parametrize("drift", ["duplicate", "target", "run", "child"])
def test_repromotion_strict_correction_drift_is_zero_mutation_at_pass(
    kanban_home, monkeypatch, drift,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn, handoff_reason=PR109_PROSE_HANDOFF)
        _allow_historical_pr109_incident(monkeypatch, shape)
        author, _, child, *_ = shape
        assert _call(
            conn,
            shape,
            exact_candidate=PR109_CANDIDATE,
            reason="PR109 strict PASS target recovery",
        ) is not None
        claimed = kb.claim_task(conn, child, claimer="host:fresh-auditor")
        assert claimed is not None and claimed.current_run_id is not None
        row = conn.execute(
            "SELECT id,run_id,payload FROM task_events WHERE task_id=? "
            "AND kind='review_repromoted'", (author,),
        ).fetchone()
        if drift == "duplicate":
            conn.execute(
                "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) "
                "VALUES (?,?, 'review_repromoted',?,1)",
                (author, row["run_id"], row["payload"]),
            )
        else:
            payload = json.loads(row["payload"])
            if drift == "target":
                target = json.loads(payload["strict_handoff_reason"])
                target["candidate"]["head"] = "1" * 40
                payload["strict_handoff_reason"] = json.dumps(target)
            elif drift == "run":
                payload["prior_review_run_id"] += 1
            else:
                payload["review_task_id"] = "t_wrong"
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload), row["id"]),
            )
        conn.commit()
        evidence = {
            **PR109_CANDIDATE,
            "github_review_id": 5189891566,
            "github_review_url": (
                "https://github.com/kiddhu/hermes-agent/pull/109"
                "#pullrequestreview-5189891566"
            ),
            "github_review_state": "APPROVED",
        }
        before = "\n".join(conn.iterdump())
        assert not kb.record_review_verdict(
            conn,
            author,
            review_task_id=child,
            expected_review_run_id=claimed.current_run_id,
            verdict="pass",
            reason="must fail closed",
            evidence=evidence,
        )
        assert "\n".join(conn.iterdump()) == before


@pytest.mark.parametrize(
    "candidate",
    [
        {**PR109_CANDIDATE, "repository": "wrong/repo"},
        {**PR109_CANDIDATE, "pr": 110},
        {**PR109_CANDIDATE, "base": "1" * 40},
    ],
)
def test_prose_recovery_rejects_wrong_repo_pr_or_base_without_mutation(
    kanban_home, monkeypatch, candidate,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn, handoff_reason=PR109_PROSE_HANDOFF)
        _allow_historical_pr109_incident(monkeypatch, shape)
        before = "\n".join(conn.iterdump())
        assert _call(
            conn,
            shape,
            exact_candidate=candidate,
            reason="PR109 strict PASS target recovery",
        ) is None
        assert "\n".join(conn.iterdump()) == before


def test_prose_recovery_is_rejected_outside_frozen_historical_incident(
    kanban_home, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn, handoff_reason=PR109_PROSE_HANDOFF)
        before = "\n".join(conn.iterdump())
        assert _call(
            conn,
            shape,
            exact_candidate=PR109_CANDIDATE,
            reason="must not generalize prose authority",
        ) is None
        assert "\n".join(conn.iterdump()) == before


def test_generic_exact_candidate_prose_is_rejected_without_mutation(
    kanban_home, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn, handoff_reason=GENERIC_PROSE_HANDOFF)
        before = "\n".join(conn.iterdump())
        assert _call(conn, shape) is None
        assert "\n".join(conn.iterdump()) == before


def test_repromotion_scopes_receipts_to_latest_audit_generation(
    kanban_home, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        author, _, child, *_ = shape
        first = _call(conn, shape)
        assert first is not None

        claimed = kb.claim_task(conn, child, claimer="host:next-auditor")
        assert claimed is not None and claimed.current_run_id is not None
        assert kb.block_task(
            conn,
            child,
            reason="fresh exact-head audit needs one more supported recovery",
            kind="dependency",
            expected_run_id=claimed.current_run_id,
        )
        second_shape = (
            shape[0], shape[1], shape[2], claimed.current_run_id,
            shape[4], shape[5], shape[6],
        )
        second = _call(conn, second_shape)

        assert second is not None
        assert second.prior_review_run_id == claimed.current_run_id
        assert second.receipt_sha256 != first.receipt_sha256
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? "
            "AND kind='review_repromoted'", (author,),
        ).fetchone()[0] == 2

        final_claim = kb.claim_task(conn, child, claimer="host:final-auditor")
        assert final_claim is not None and final_claim.current_run_id is not None
        evidence = {
            **CANDIDATE,
            "github_review_id": 1,
            "github_review_url": (
                "https://github.com/kiddhu/aion-governance/pull/961"
                "#pullrequestreview-1"
            ),
            "github_review_state": "APPROVED",
        }
        assert kb.record_review_verdict(
            conn,
            author,
            review_task_id=child,
            expected_review_run_id=final_claim.current_run_id,
            verdict="pass",
            reason="fresh exact-head PASS",
            evidence=evidence,
        )


@pytest.mark.parametrize(
    ("profile", "change"),
    [
        ("agent007", {}),
        ("bafuxunan", {}),
        ("gm2", {"author_task_id": "t_wrong"}),
        ("gm2", {"author_run_id": 999999}),
        ("gm2", {"review_task_id": "t_wrong"}),
        ("gm2", {"prior_review_run_id": 999999}),
        ("gm2", {"handoff_receipt_sha256": "0" * 64}),
        ("gm2", {"controller_task_id": "t_wrong"}),
        ("gm2", {"controller_run_id": 999999}),
        ("gm2", {"exact_candidate": {**CANDIDATE, "repository": "wrong/repo"}}),
        ("gm2", {"exact_candidate": {**CANDIDATE, "head": "1" * 40}}),
    ],
)
def test_repromotion_hostile_identity_drift_is_zero_mutation(
    kanban_home, monkeypatch, profile, change,
):
    monkeypatch.setenv("HERMES_PROFILE", profile)
    with kb.connect() as conn:
        shape = _shape(conn)
        before = "\n".join(conn.iterdump())
        assert _call(conn, shape, **change) is None
        assert "\n".join(conn.iterdump()) == before


@pytest.mark.parametrize("drift", ["role", "edge", "candidate", "newer_run", "verdict"])
def test_repromotion_hostile_board_drift_is_zero_mutation(
    kanban_home, monkeypatch, drift,
):
    monkeypatch.setenv("HERMES_PROFILE", "gm")
    with kb.connect() as conn:
        shape = _shape(conn)
        author, _, child, child_run, handoff, *_ = shape
        if drift == "role":
            conn.execute("UPDATE tasks SET assignee='other' WHERE id=?", (child,))
        elif drift == "edge":
            conn.execute(
                "DELETE FROM task_links WHERE parent_id=? AND child_id=?", (author, child)
            )
        elif drift == "candidate":
            payload = json.loads(conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? "
                "AND kind='review_handoff'", (author,)
            ).fetchone()[0])
            payload["reason"] += " drift"
            conn.execute(
                "UPDATE task_events SET payload=? WHERE task_id=? AND kind='review_handoff'",
                (json.dumps(payload), author),
            )
        elif drift == "newer_run":
            conn.execute(
                "INSERT INTO task_runs(task_id, profile, status, started_at, ended_at, outcome) "
                "VALUES (?, 'bafuxunan', 'blocked', 1, 2, 'blocked')",
                (child,),
            )
        else:
            kb._append_event(
                conn, child, "review_verdict", {"verdict": "request_changes"},
                run_id=child_run,
            )
        conn.commit()
        before = "\n".join(conn.iterdump())
        assert _call(
            conn, shape, handoff_receipt_sha256=handoff.receipt_sha256
        ) is None
        assert "\n".join(conn.iterdump()) == before


def test_repromotion_waits_for_other_parent_then_succeeds(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        child = shape[2]
        gate = kb.create_task(conn, title="install gate", assignee="merger")
        kb.link_tasks(conn, gate, child)
        before = "\n".join(conn.iterdump())
        assert _call(conn, shape) is None
        assert "\n".join(conn.iterdump()) == before
        conn.execute(
            "UPDATE tasks SET status='done', completed_at=1 WHERE id=?", (gate,)
        )
        conn.commit()
        assert _call(conn, shape) is not None


def test_repromotion_partial_cas_rolls_back(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        conn.execute(
            "CREATE TRIGGER reject_repromotion BEFORE UPDATE ON tasks "
            "WHEN NEW.status='ready' BEGIN SELECT RAISE(IGNORE); END"
        )
        conn.commit()
        before = "\n".join(conn.iterdump())
        assert _call(conn, shape) is None
        assert "\n".join(conn.iterdump()) == before


def test_repromotion_conflicting_replay_is_zero_mutation(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        assert _call(conn, shape) is not None
        before = "\n".join(conn.iterdump())
        assert _call(conn, shape, reason="different correction") is None
        assert "\n".join(conn.iterdump()) == before


def test_repromotion_tool_requires_gm_controller_run(kanban_home, monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
    author, author_run, child, child_run, handoff, controller, controller_run = shape
    monkeypatch.setenv("HERMES_KANBAN_TASK", controller)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(controller_run))
    result = json.loads(kt._handle_repromote_review({
        "author_task_id": author,
        "author_run_id": author_run,
        "review_task_id": child,
        "prior_review_run_id": child_run,
        "handoff_receipt_sha256": handoff.receipt_sha256,
        "correction_reason": "audit first; install follows",
        "exact_candidate": CANDIDATE,
    }))
    assert result["ok"] is True
    assert result["review_task_id"] == child
    assert json.loads(result["strict_handoff_reason"]) == {
        "version": 1,
        "candidate": CANDIDATE,
        "summary": "audit first; install follows",
    }


def test_repromotion_tool_is_visible_to_controller_task_workers(monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_controller")
    assert kt._check_kanban_mode() is True


def test_repromotion_concurrent_race_emits_one_generation(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    with kb.connect() as conn:
        shape = _shape(conn)
        author = shape[0]
    barrier = threading.Barrier(2)

    def invoke():
        with kb.connect() as conn:
            barrier.wait()
            return _call(conn, shape)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert results[0] is not None and results[0] == results[1]
    with kb.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? "
            "AND kind='review_repromoted'", (author,)
        ).fetchone()[0] == 1
