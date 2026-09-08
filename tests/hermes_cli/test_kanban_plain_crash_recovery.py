"""Fail-closed recovery for an omitted PASS from a plain-crash audit run.

Reproduces t_aa59c305 run4217→4219: a terminal auditor run completed with a
typed plain-crash PASS (``audit_outcome=PASS_EXACT_HEAD``, an explicit
``prior_bound_review_run_id``, exact head/tree/base, and a clean
``no_side_effect_receipt``) after a same-child predecessor run emitted the
identical version-1 PASS and then crashed with ``exit_kind=unknown`` (no
``protocol_violation`` marker).  A live gm/gm2 controller may bind exactly one
omitted PASS to the latest terminal run via a closed typed plain-crash receipt.
Every repository/PR/head/tree/base/outcome/prior-run/no-side-effect/role/history
drift must fail closed, and the forgeable prose ``recovery_run_verification``
list must never grant authority.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


HEAD = "a" * 40
TREE = "b" * 40
BASE = "c" * 40
PR = 92
REPOSITORY = "kiddhu/hermes-agent"

NO_SIDE_EFFECT = {
    "broad_signal_count": 0,
    "manual_claim": 0,
    "manual_dispatch": 0,
    "raw_db_write": 0,
    "replacement_task_count": 0,
    "secret_exposure": "none",
}

PRECURSOR_REASON = (
    f"PASS_EXACT_HEAD for kiddhu/hermes-agent PR #{PR} at head {HEAD}/"
    f"tree {TREE}/base {BASE}. Role-separated author 007AION / auditor GemAION."
)
TERMINAL_SUMMARY = (
    f"Recovered and terminalized the commit-bound PASS for PR #{PR} at "
    f"head {HEAD}/tree {TREE}/base {BASE} after the predecessor worker exited "
    f"post-verdict."
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


def _plain_crash_fixture(
    conn, *, controller_profile="gm2", predecessor_metadata=None,
    terminal_metadata_override=None, no_side_effect=None,
):
    author = kb.create_task(
        conn, title="implementation", factory_build_gate=1, assignee="agent007",
    )
    author_run = _claim(conn, author)
    audit = kb.create_task(
        conn, title="exact-head audit", assignee="bafuxunan", parents=[author],
    )
    assert kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=audit,
        reason=f"PR kiddhu/hermes-agent#{PR} frozen at exact head {HEAD}",
    ) is not None
    precursor_run = _claim(conn, audit)
    assert kb.record_review_verdict(
        conn, author, review_task_id=audit, expected_review_run_id=precursor_run,
        verdict="pass", reason=PRECURSOR_REASON,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='crashed', outcome='crashed', summary=NULL, "
            "metadata=?, ended_at=11111, claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?",
            (
                json.dumps(
                    {"pid": 1, "claimer": "x", "exit_kind": "unknown"}
                    if predecessor_metadata is None else predecessor_metadata
                ),
                precursor_run,
            ),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (audit,),
        )
    terminal_run = _claim(conn, audit)
    metadata = {
        "audit_outcome": "PASS_EXACT_HEAD",
        "head": HEAD,
        "tree": TREE,
        "base": BASE,
        "prior_bound_review_run_id": precursor_run,
        "no_side_effect_receipt": NO_SIDE_EFFECT if no_side_effect is None else no_side_effect,
        "recovery_run_verification": ["prose that must never grant authority"],
        "auditor": "GemAION",
        "author": "007AION",
        "worker_session_id": "s",
    }
    if terminal_metadata_override:
        metadata.update(terminal_metadata_override)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='completed', summary=?, "
            "metadata=?, ended_at=22222, claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?",
            (TERMINAL_SUMMARY, json.dumps(metadata), terminal_run),
        )
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (audit,),
        )
    controller = kb.create_task(conn, title="controller", assignee=controller_profile)
    controller_run = _claim(conn, controller)
    return {
        "author": author,
        "author_run": author_run,
        "audit": audit,
        "precursor_run": precursor_run,
        "terminal_run": terminal_run,
        "controller": controller,
        "controller_run": controller_run,
    }


def _receipt(precursor_run, **overrides):
    receipt = {
        "audit_outcome": "PASS_EXACT_HEAD",
        "head": HEAD,
        "tree": TREE,
        "base": BASE,
        "prior_bound_review_run_id": precursor_run,
    }
    receipt.update(overrides)
    return receipt


def _recover(conn, fixture, *, receipt=None, profile="gm2", reason=TERMINAL_SUMMARY):
    return kb.record_review_verdict(
        conn,
        fixture["author"],
        review_task_id=fixture["audit"],
        expected_review_run_id=fixture["terminal_run"],
        verdict="pass",
        reason=reason,
        recovery_receipt=_receipt(fixture["precursor_run"]) if receipt is None else receipt,
        controller_task_id=fixture["controller"],
        controller_run_id=fixture["controller_run"],
        controller_profile=profile,
    )


def _recovered_verdicts(conn, fixture):
    return conn.execute(
        "SELECT id, run_id, payload FROM task_events WHERE task_id=? "
        "AND kind='review_verdict' ORDER BY id",
        (fixture["author"],),
    ).fetchall()


# --- GREEN ---------------------------------------------------------------


def test_plain_crash_recovery_emits_one_latest_run_pass(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn)
        assert _recover(conn, fixture) is True
        verdicts = _recovered_verdicts(conn, fixture)
        assert len(verdicts) == 2
        payload = json.loads(verdicts[-1]["payload"])
        assert payload["version"] == 2 and payload["recovery"] is True
        assert payload["verdict"] == "pass"
        assert payload["review_run_id"] == fixture["terminal_run"]
        assert payload["recovery_receipt"] == _receipt(fixture["precursor_run"])


def test_plain_crash_recovery_is_idempotent(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn)
        assert _recover(conn, fixture) is True
        assert _recover(conn, fixture) is True
        assert len(_recovered_verdicts(conn, fixture)) == 2


# --- HOSTILE: controller receipt drift -----------------------------------


@pytest.mark.parametrize(
    "mutate",
    ["head", "tree", "base", "audit_outcome", "prior_run"],
)
def test_plain_crash_recovery_rejects_receipt_drift(kanban_home, mutate):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn)
        receipt = _receipt(fixture["precursor_run"])
        if mutate in ("head", "tree", "base"):
            receipt[mutate] = "f" * 40
        elif mutate == "audit_outcome":
            receipt["audit_outcome"] = "REQUEST_CHANGES"
        elif mutate == "prior_run":
            receipt["prior_bound_review_run_id"] = fixture["precursor_run"] + 999
        assert _recover(conn, fixture, receipt=receipt) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


@pytest.mark.parametrize(
    "mutate",
    ["extra_key", "missing_key"],
)
def test_plain_crash_recovery_rejects_receipt_shape_drift(kanban_home, mutate):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn)
        receipt = _receipt(fixture["precursor_run"])
        if mutate == "extra_key":
            receipt["github_review_id"] = 123
        elif mutate == "missing_key":
            del receipt["prior_bound_review_run_id"]
        assert _recover(conn, fixture, receipt=receipt) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


# --- HOSTILE: terminal metadata drift ------------------------------------


def test_plain_crash_recovery_rejects_terminal_prior_run_drift(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn, terminal_metadata_override={"prior_bound_review_run_id": 999999},
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


@pytest.mark.parametrize(
    "bad_nse",
    [
        {**NO_SIDE_EFFECT, "manual_claim": 1},
        {**NO_SIDE_EFFECT, "raw_db_write": 1},
        {**NO_SIDE_EFFECT, "replacement_task_count": 1},
        {**NO_SIDE_EFFECT, "secret_exposure": "redacted"},
        {"broad_signal_count": 0},  # missing keys
    ],
)
def test_plain_crash_recovery_rejects_unclean_no_side_effect(kanban_home, bad_nse):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn, no_side_effect=bad_nse)
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def test_plain_crash_recovery_rejects_missing_no_side_effect(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn, terminal_metadata_override={"no_side_effect_receipt": None},
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def test_plain_crash_recovery_rejects_ambiguous_nested_receipt(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn, terminal_metadata_override={"recovery_receipt": {"x": 1}},
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


# --- HOSTILE: lane separation --------------------------------------------


def test_plain_crash_lane_rejects_protocol_violation_predecessor(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn, predecessor_metadata={"pid": 1, "protocol_violation": True},
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def test_plain_crash_lane_rejects_no_exit_kind_predecessor(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn, predecessor_metadata={"pid": 1, "claimer": "x"},
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


# --- HOSTILE: controller / role ------------------------------------------


def test_plain_crash_recovery_rejects_unauthorized_controller(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn, controller_profile="agent007")
        assert _recover(conn, fixture, profile="agent007") is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


# --- HOSTILE: signed handoff / event ordering ----------------------------


def test_plain_crash_recovery_requires_signed_handoff(kanban_home):
    # The recovery path must mint typed PASS authority only when a current
    # signed review-handoff binds the author to this exact auditor child.
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(conn)
        with kb.write_txn(conn):
            conn.execute(
                "DELETE FROM task_events WHERE task_id = ? AND kind = 'review_handoff'",
                (fixture["author"],),
            )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def _predating_handoff_fixture(conn):
    """Build a plain-crash recovery whose predecessor PASS predates the handoff.

    The predecessor v1 PASS is emitted under a first signed handoff, then the
    author re-arms and issues a second (newer) signed handoff before the
    terminal run completes.  The predecessor's author/mirror verdict events
    therefore predate the current (latest) signed handoff, which the recovery
    must reject even though the terminal 5-key plain-crash binding is exact.
    """
    author = kb.create_task(
        conn, title="implementation", factory_build_gate=1, assignee="agent007",
    )
    author_run = _claim(conn, author)
    audit = kb.create_task(
        conn, title="exact-head audit", assignee="bafuxunan", parents=[author],
    )
    first_handoff = kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=audit,
        reason=f"PR kiddhu/hermes-agent#{PR} first head {HEAD}",
    )
    assert first_handoff is not None
    precursor_run = _claim(conn, audit)
    assert kb.record_review_verdict(
        conn, author, review_task_id=audit, expected_review_run_id=precursor_run,
        verdict="pass", reason=PRECURSOR_REASON,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='crashed', outcome='crashed', summary=NULL, "
            "metadata=?, ended_at=11111, claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?",
            (json.dumps({"pid": 1, "claimer": "x", "exit_kind": "unknown"}), precursor_run),
        )
        conn.execute(
            "UPDATE tasks SET status='todo', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (audit,),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (author,),
        )
    second_author_run = _claim(conn, author)
    second_handoff = kb.request_review_handoff(
        conn, author, expected_run_id=second_author_run, review_task_id=audit,
        reason=f"PR kiddhu/hermes-agent#{PR} second head {HEAD}",
    )
    assert second_handoff is not None
    assert second_handoff.event_id > first_handoff.event_id
    terminal_run = _claim(conn, audit)
    metadata = {
        "audit_outcome": "PASS_EXACT_HEAD",
        "head": HEAD,
        "tree": TREE,
        "base": BASE,
        "prior_bound_review_run_id": precursor_run,
        "no_side_effect_receipt": NO_SIDE_EFFECT,
        "recovery_run_verification": ["prose that must never grant authority"],
        "auditor": "GemAION",
        "author": "007AION",
        "worker_session_id": "s",
    }
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='completed', summary=?, "
            "metadata=?, ended_at=22222, claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?",
            (TERMINAL_SUMMARY, json.dumps(metadata), terminal_run),
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
        "precursor_run": precursor_run,
        "terminal_run": terminal_run,
        "controller": controller,
        "controller_run": controller_run,
    }


def test_plain_crash_recovery_rejects_predecessor_verdict_predating_handoff(kanban_home):
    with kb.connect() as conn:
        fixture = _predating_handoff_fixture(conn)
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def _latest_handoff_row(conn, author):
    return conn.execute(
        "SELECT id, run_id, payload FROM task_events WHERE task_id=? "
        "AND kind='review_handoff' ORDER BY id DESC LIMIT 1",
        (author,),
    ).fetchone()


def test_review_handoff_event_for_child_does_not_fallback_on_deleted_latest(kanban_home):
    # A missing latest handoff must fail closed, not fall back to an older
    # valid handoff that bound a superseded review round.
    with kb.connect() as conn:
        fixture = _predating_handoff_fixture(conn)
        latest = _latest_handoff_row(conn, fixture["author"])
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_events WHERE id=?", (latest["id"],))
        assert kb._review_handoff_event_for_child(
            conn, fixture["author"], fixture["audit"]
        ) is None
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def test_review_handoff_event_for_child_does_not_fallback_on_hash_invalid_latest(kanban_home):
    # A hash-invalid latest handoff must fail closed, not fall back to an
    # older valid handoff that bound a superseded review round.
    with kb.connect() as conn:
        fixture = _predating_handoff_fixture(conn)
        latest = _latest_handoff_row(conn, fixture["author"])
        payload = json.loads(latest["payload"])
        payload["receipt_sha256"] = "0" * 64
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload), latest["id"]),
            )
        assert kb._review_handoff_event_for_child(
            conn, fixture["author"], fixture["audit"]
        ) is None
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def test_review_handoff_event_for_child_binds_only_latest_review_run(kanban_home):
    # Only the handoff binding the author's latest review-required run may
    # authenticate.  After deleting the latest handoff, the older handoff (for
    # an earlier run) must not bind, even though both runs are review_required.
    with kb.connect() as conn:
        fixture = _predating_handoff_fixture(conn)
        latest = _latest_handoff_row(conn, fixture["author"])
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_events WHERE id=?", (latest["id"],))
        # The older handoff is still present and structurally valid.
        older = conn.execute(
            "SELECT id, run_id, payload FROM task_events WHERE task_id=? "
            "AND kind='review_handoff' ORDER BY id",
            (fixture["author"],),
        ).fetchone()
        assert older is not None
        assert kb._review_handoff_receipt_from_row(fixture["author"], older) is not None
        # ...but it must not bind because it names a superseded run.
        assert kb._review_handoff_event_for_child(
            conn, fixture["author"], fixture["audit"]
        ) is None


# --- HOSTILE: closed crash-attribution grammar ---------------------------


@pytest.mark.parametrize(
    "exit_kind",
    ["   ", "gibberish", "unknown ", "UNKNOWN", "false", "not-an-exit-kind", "unknown\n"],
)
def test_plain_crash_lane_rejects_loose_exit_kind(kanban_home, exit_kind):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn, predecessor_metadata={"pid": 1, "claimer": "x", "exit_kind": exit_kind},
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1


def test_plain_crash_lane_rejects_explicit_false_protocol_violation(kanban_home):
    with kb.connect() as conn:
        fixture = _plain_crash_fixture(
            conn,
            predecessor_metadata={
                "pid": 1, "claimer": "x", "exit_kind": "unknown",
                "protocol_violation": False,
            },
        )
        assert _recover(conn, fixture) is False
        assert len(_recovered_verdicts(conn, fixture)) == 1
