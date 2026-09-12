"""Fail-closed completed-audit PASS recovery for the legacy base-omission reason.

The installed completed-audit PASS recovery authenticates a crashed
protocol-violation predecessor PASS by requiring its prose reason to re-state
the exact ``head``/``tree``/``base`` commit identity of the terminal receipt
(``_reason_bears_commit_identity``).  The real audited PR #87 path
(``t_d7a8d4dc`` run 4267) emitted a predecessor PASS whose reason re-states
``exact head`` and ``tree`` but omits ``base``::

    PASS exact head b563d45dcfec776599f5f3be8e366caca4e1783c / tree a9ffaff78f49b022adef753b7ef3e64ea2b27ce8: ...

The base is not lost: it is independently typed in the terminal run metadata
(``base`` = ef11089a57a1aec9adfe5d2b25e03ca0c50d1c69) and in the exact
commit-bound GitHub ``APPROVED`` recovery receipt supplied by the live
gm/gm2 controller, and both are compared byte-for-byte before the predecessor
reason is consulted.  Requiring ``base`` in the prose therefore fails closed
on that exact legacy shape even though the base is uniquely corroborated.

This suite pins the repair: ``_reason_bears_commit_identity`` must accept the
legacy head+tree-only reason (base exempt because it is already corroborated
from typed Native + live GitHub identity) while still failing closed on any
missing or drifted ``head``/``tree`` and on every hostile receipt/lineage/
verdict/run/replay shape already covered by the sibling suites.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# Exact commit identity copied from the live terminal runs (read-only).
EXACT_HEAD_4268 = "b563d45dcfec776599f5f3be8e366caca4e1783c"
TREE_4268 = "a9ffaff78f49b022adef753b7ef3e64ea2b27ce8"
BASE_4268 = "ef11089a57a1aec9adfe5d2b25e03ca0c50d1c69"
PR_4268 = 87
REVIEW_ID_4268 = 5151368236


def _receipt(head, tree, base, pr, review_id):
    return {
        "review_outcome": "approved",
        "repository": "kiddhu/hermes-agent",
        "pr": pr,
        "head": head,
        "tree": tree,
        "base": base,
        "github_review_id": review_id,
        "github_review_url": (
            f"https://github.com/kiddhu/hermes-agent/pull/{pr}"
            f"#pullrequestreview-{review_id}"
        ),
        "github_review_state": "APPROVED",
    }


RECEIPT_4268 = _receipt(
    EXACT_HEAD_4268, TREE_4268, BASE_4268, PR_4268, REVIEW_ID_4268,
)

# Faithful terminal-run metadata copy for run 4268 (``exact_head`` alias).
METADATA_4268 = {
    "base": BASE_4268,
    "exact_head": EXACT_HEAD_4268,
    "github_review_id": REVIEW_ID_4268,
    "github_review_state": "APPROVED",
    "github_review_url": (
        f"https://github.com/kiddhu/hermes-agent/pull/{PR_4268}"
        f"#pullrequestreview-{REVIEW_ID_4268}"
    ),
    "new_control_plane_count": 0,
    "pr": PR_4268,
    "repository": "kiddhu/hermes-agent",
    "review_outcome": "approved",
    "role_separation": {"author": "007AION", "auditor": "GemAION"},
    "tree": TREE_4268,
    "worker_session_id": "20260909_155707_afdc49",
}

# The real predecessor PASS reason for run 4267: head + tree, NO base.
PRECURSOR_REASON_OMITTING_BASE = (
    f"PASS exact head {EXACT_HEAD_4268} / tree {TREE_4268}: v6 fixes v5 "
    "null/empty digest collisions and unbounded/injectable profile framing; "
    "RED 3/3 on v5, GREEN 3/3 plus 723 focused suites on v6, independent "
    "75-finding hostile probe bounded/preserved evidence, canonical tasks "
    "unchanged, live CI passes, CLEAN/MERGEABLE, two-file scope, no "
    "secret/control-plane additions. Commit-bound GitHub APPROVED review "
    f"{REVIEW_ID_4268}: https://github.com/kiddhu/hermes-agent/pull/"
    f"{PR_4268}#pullrequestreview-{REVIEW_ID_4268}"
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


def _claim(conn, task_id) -> int:
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    return int(claimed.current_run_id)


def _fixture(conn, *, precursor_reason):
    """Repeated-audit PASS shape whose crashed predecessor reason omits base."""
    author = kb.create_task(
        conn, title="implementation", factory_build_gate=1, assignee="agent007",
    )
    audit = kb.create_task(
        conn, title="exact-head audit", assignee="bafuxunan", parents=[author],
    )
    for i in range(3):
        author_run = _claim(conn, author)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=audit,
            reason=f"PR #{PR_4268} round {i} head {'d' * 40}",
        ) is not None
        audit_run = _claim(conn, audit)
        assert kb.record_review_verdict(
            conn, author, review_task_id=audit,
            expected_review_run_id=audit_run, verdict="request_changes",
            reason=f"REQUEST_CHANGES round {i}",
        )
    author_run = _claim(conn, author)
    assert kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=audit,
        reason=f"PR #{PR_4268} frozen at exact head {EXACT_HEAD_4268}",
    ) is not None
    precursor_run = _claim(conn, audit)
    assert kb.record_review_verdict(
        conn, author, review_task_id=audit,
        expected_review_run_id=precursor_run, verdict="pass",
        reason=precursor_reason,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='crashed', outcome='crashed', "
            "summary=NULL, metadata=?, ended_at=11111, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (json.dumps({"pid": 1, "protocol_violation": True}), precursor_run),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (audit,),
        )
    terminal_run = _claim(conn, audit)
    terminal_summary = (
        f"Independent exact-head audit PASS for kiddhu/hermes-agent PR "
        f"#{PR_4268} at head {EXACT_HEAD_4268}/tree {TREE_4268}."
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='completed', summary=?, "
            "metadata=?, ended_at=22222, claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?",
            (terminal_summary, json.dumps(METADATA_4268), terminal_run),
        )
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (audit,),
        )
    controller = kb.create_task(conn, title="controller", assignee="gm2")
    controller_run = _claim(conn, controller)
    return {
        "author": author,
        "audit": audit,
        "terminal_run": terminal_run,
        "controller": controller,
        "controller_run": controller_run,
        "terminal_summary": terminal_summary,
    }


def _recover(conn, fixture, receipt, reason):
    return kb.record_review_verdict(
        conn,
        fixture["author"],
        review_task_id=fixture["audit"],
        expected_review_run_id=fixture["terminal_run"],
        verdict="pass",
        reason=reason,
        recovery_receipt=receipt,
        controller_task_id=fixture["controller"],
        controller_run_id=fixture["controller_run"],
        controller_profile="gm2",
    )


def _snapshot(conn):
    return "\n".join(conn.iterdump())


# ---------------------------------------------------------------------------
# GREEN: the legacy base-omission reason recovers (base corroborated from the
# typed terminal metadata + live commit-bound GitHub receipt).
# ---------------------------------------------------------------------------

def test_recovery_accepts_legacy_reason_omitting_base(kanban_home):
    with kb.connect() as conn:
        fixture = _fixture(conn, precursor_reason=PRECURSOR_REASON_OMITTING_BASE)
        before = _snapshot(conn)
        assert _recover(conn, fixture, RECEIPT_4268, fixture["terminal_summary"])
        after = _snapshot(conn)
        assert after != before
        # Idempotent replay writes nothing further.
        assert _recover(conn, fixture, RECEIPT_4268, fixture["terminal_summary"])
        assert _snapshot(conn) == after


# ---------------------------------------------------------------------------
# Hostile fail-closed: any head/tree drift or absence in the reason still
# fails closed even though base is omitted.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "precursor_reason",
    [
        # Wrong head.
        f"PASS exact head {'f' * 40} / tree {TREE_4268}: drifted head.",
        # Wrong tree.
        f"PASS exact head {EXACT_HEAD_4268} / tree {'e' * 40}: drifted tree.",
        # Head absent entirely.
        f"PASS tree {TREE_4268}: no head stated.",
        # Tree absent entirely.
        f"PASS exact head {EXACT_HEAD_4268}: no tree stated.",
        # Bare verdict with no commit identity at all.
        "PASS: no commit identity at all.",
    ],
)
def test_recovery_rejects_hostile_base_omission_reason(kanban_home, precursor_reason):
    with kb.connect() as conn:
        fixture = _fixture(conn, precursor_reason=precursor_reason)
        before = _snapshot(conn)
        assert not _recover(conn, fixture, RECEIPT_4268, fixture["terminal_summary"])
        assert _snapshot(conn) == before


def test_reason_bears_commit_identity_rejects_empty_or_non_string():
    assert not kb._reason_bears_commit_identity("", RECEIPT_4268)
    assert not kb._reason_bears_commit_identity(None, RECEIPT_4268)
    assert not kb._reason_bears_commit_identity(123, RECEIPT_4268)


# ---------------------------------------------------------------------------
# The base remains mandatory in the typed evidence: a missing or conflicting
# base in the terminal metadata still fails closed (independent of the reason).
# ---------------------------------------------------------------------------

def test_recovery_rejects_terminal_metadata_missing_base(kanban_home):
    metadata = copy.deepcopy(METADATA_4268)
    del metadata["base"]
    with kb.connect() as conn:
        # Build the shape by hand so we can inject terminal metadata without base.
        author = kb.create_task(
            conn, title="implementation", factory_build_gate=1,
            assignee="agent007",
        )
        audit = kb.create_task(
            conn, title="exact-head audit", assignee="bafuxunan", parents=[author],
        )
        author_run = _claim(conn, author)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=audit,
            reason=f"PR #{PR_4268} frozen at exact head {EXACT_HEAD_4268}",
        ) is not None
        precursor_run = _claim(conn, audit)
        assert kb.record_review_verdict(
            conn, author, review_task_id=audit,
            expected_review_run_id=precursor_run, verdict="pass",
            reason=PRECURSOR_REASON_OMITTING_BASE,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='crashed', outcome='crashed', "
                "summary=NULL, metadata=?, ended_at=11111, claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (json.dumps({"pid": 1, "protocol_violation": True}), precursor_run),
            )
            conn.execute(
                "UPDATE tasks SET status='ready', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (audit,),
            )
        terminal_run = _claim(conn, audit)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='done', outcome='completed', "
                "summary=?, metadata=?, ended_at=22222, claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (
                    "Independent exact-head audit PASS for kiddhu/hermes-agent "
                    f"PR #{PR_4268} at head {EXACT_HEAD_4268}/tree {TREE_4268}.",
                    json.dumps(metadata),
                    terminal_run,
                ),
            )
            conn.execute(
                "UPDATE tasks SET status='done', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (audit,),
            )
        controller = kb.create_task(conn, title="controller", assignee="gm2")
        controller_run = _claim(conn, controller)
        fixture = {
            "author": author,
            "audit": audit,
            "terminal_run": terminal_run,
            "controller": controller,
            "controller_run": controller_run,
        }
        before = _snapshot(conn)
        assert not _recover(
            conn, fixture, RECEIPT_4268,
            "Independent exact-head audit PASS for kiddhu/hermes-agent "
            f"PR #{PR_4268} at head {EXACT_HEAD_4268}/tree {TREE_4268}.",
        )
        assert _snapshot(conn) == before


def test_recovery_rejects_terminal_metadata_conflicting_base(kanban_home):
    """A terminal metadata base that differs from the caller receipt fails
    closed even with a head+tree-only reason."""
    metadata = copy.deepcopy(METADATA_4268)
    metadata["base"] = "d" * 40
    with kb.connect() as conn:
        author = kb.create_task(
            conn, title="implementation", factory_build_gate=1,
            assignee="agent007",
        )
        audit = kb.create_task(
            conn, title="exact-head audit", assignee="bafuxunan", parents=[author],
        )
        author_run = _claim(conn, author)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=audit,
            reason=f"PR #{PR_4268} frozen at exact head {EXACT_HEAD_4268}",
        ) is not None
        precursor_run = _claim(conn, audit)
        assert kb.record_review_verdict(
            conn, author, review_task_id=audit,
            expected_review_run_id=precursor_run, verdict="pass",
            reason=PRECURSOR_REASON_OMITTING_BASE,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='crashed', outcome='crashed', "
                "summary=NULL, metadata=?, ended_at=11111, claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (json.dumps({"pid": 1, "protocol_violation": True}), precursor_run),
            )
            conn.execute(
                "UPDATE tasks SET status='ready', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (audit,),
            )
        terminal_run = _claim(conn, audit)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='done', outcome='completed', "
                "summary=?, metadata=?, ended_at=22222, claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (
                    "Independent exact-head audit PASS for kiddhu/hermes-agent "
                    f"PR #{PR_4268} at head {EXACT_HEAD_4268}/tree {TREE_4268}.",
                    json.dumps(metadata),
                    terminal_run,
                ),
            )
            conn.execute(
                "UPDATE tasks SET status='done', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (audit,),
            )
        controller = kb.create_task(conn, title="controller", assignee="gm2")
        controller_run = _claim(conn, controller)
        fixture = {
            "author": author,
            "audit": audit,
            "terminal_run": terminal_run,
            "controller": controller,
            "controller_run": controller_run,
        }
        before = _snapshot(conn)
        assert not _recover(
            conn, fixture, RECEIPT_4268,
            "Independent exact-head audit PASS for kiddhu/hermes-agent "
            f"PR #{PR_4268} at head {EXACT_HEAD_4268}/tree {TREE_4268}.",
        )
        assert _snapshot(conn) == before
