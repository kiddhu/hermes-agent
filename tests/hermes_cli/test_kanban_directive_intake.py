"""Hostile/idempotency tests for the GM directive -> Native binding prefix.

Covers the AION-GM-DIRECTIVE-NATIVE-CONTINUITY-R1 repair: the durable
pre-claim events (``directive_observed`` -> ``directive_selected`` ->
``directive_bound_native``) recorded on the existing ``task_events`` surface,
plus their fail-closed binding invariants.

The observer/selector identity is kernel-authenticated (``HERMES_PROFILE``),
never caller-asserted, so a non-GM caller cannot forge gm/gm2 authority.
Selection is explicit: ``disposition="EXECUTE"`` records the full lineage;
omitting it (or passing anything else) records only awareness or fails closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Hermetic per-test board + authenticated gm2 observer/selector lane."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Kernel-authenticated observer/selector identity. The directive intake
    # path reads HERMES_PROFILE (never caller-supplied identity); run under
    # the gm2 patrol lane.
    monkeypatch.setenv("HERMES_PROFILE", "gm2")
    # Rebind every Native Kanban path/board pin so a probe that inherits the
    # dispatcher's live HERMES_KANBAN_DB/board cannot write synthetic cards
    # into the real board (AION-RL2-CORE-01-R10 class).
    with kb.isolated_kanban_env(home):
        yield home


def _mk_task(conn, **kw):
    kw.setdefault("assignee", "gm2")
    return kb.create_task(conn, title="gm directive carrier", **kw)


def _bind(conn, task_id, *, source_sha="sha-001",
          source_ref="#833 comment 5548300539",
          disposition: "str | None" = "EXECUTE", **kw):
    return kb.record_directive_intake(
        conn,
        task_id=task_id,
        source_ref=source_ref,
        source_sha_or_immutable_id=source_sha,
        disposition=disposition,
        **kw,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_records_observed_selected_bound_in_order(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        r = _bind(conn, t, source_ref="#833 comment 5548300539", source_sha="sha-001")
        assert r["already_bound"] is False
        assert r["selected"] is True

        lineage = kb.directive_binding_lineage(conn, t)
        assert [e.kind for e in lineage] == [
            kb.DIRECTIVE_OBSERVED_KIND,
            kb.DIRECTIVE_SELECTED_KIND,
            kb.DIRECTIVE_BOUND_NATIVE_KIND,
        ]

        observed, selected, bound = lineage
        assert observed.payload["source_ref"] == "#833 comment 5548300539"
        assert observed.payload["source_sha_or_immutable_id"] == "sha-001"
        assert observed.payload["observer_profile"] == "gm2"
        assert observed.payload["observed_at"] is not None

        assert selected.payload["selector_profile"] == "gm2"
        assert selected.payload["disposition"] == kb.DIRECTIVE_EXECUTE_DISPOSITION

        assert bound.payload["task_id"] == t
        # assignee is read from the task row (source of truth), not caller-
        # asserted nor derived from the observer profile.
        assert bound.payload["assignee"] == "gm2"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Idempotency: duplicate patrol read -> one logical lineage, no duplicate events
# ---------------------------------------------------------------------------

def test_duplicate_read_is_idempotent(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn, idempotency_key="directive-001")
        _bind(conn, t, source_sha="sha-dup")
        r2 = _bind(conn, t, source_sha="sha-dup")
        assert r2["already_bound"] is True
        assert r2["events"] == []

        lineage = kb.directive_binding_lineage(conn, t)
        # Still exactly one observed/selected/bound triple.
        assert [e.kind for e in lineage] == [
            kb.DIRECTIVE_OBSERVED_KIND,
            kb.DIRECTIVE_SELECTED_KIND,
            kb.DIRECTIVE_BOUND_NATIVE_KIND,
        ]
    finally:
        conn.close()


def test_observed_then_selected_reuses_observed(kanban_home):
    """Awareness (observed-only) followed by an explicit selection must not
    mint a duplicate ``directive_observed`` row."""
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        r1 = _bind(conn, t, source_sha="sha-seq", disposition=None)
        assert r1["selected"] is False
        assert [e.kind for e in kb.directive_binding_lineage(conn, t)] == [
            kb.DIRECTIVE_OBSERVED_KIND,
        ]

        r2 = _bind(conn, t, source_sha="sha-seq", disposition="EXECUTE")
        assert r2["selected"] is True
        assert r2["already_bound"] is False
        assert [e.kind for e in kb.directive_binding_lineage(conn, t)] == [
            kb.DIRECTIVE_OBSERVED_KIND,
            kb.DIRECTIVE_SELECTED_KIND,
            kb.DIRECTIVE_BOUND_NATIVE_KIND,
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fail-closed: one directive -> two tasks
# ---------------------------------------------------------------------------

def test_one_directive_cannot_bind_two_tasks(kanban_home):
    conn = kb.connect()
    try:
        t1 = _mk_task(conn)
        t2 = _mk_task(conn)
        _bind(conn, t1, source_sha="sha-two-tasks")
        with pytest.raises(RuntimeError):
            _bind(conn, t2, source_sha="sha-two-tasks")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Non-authoritative observer cannot become executable (finding 1)
# ---------------------------------------------------------------------------

def test_non_gm_authenticated_profile_fails_closed(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        # A non-GM authenticated profile cannot observe/select, even though
        # the caller previously would have been able to assert "gm2".
        monkeypatch.setenv("HERMES_PROFILE", "agent007")
        with pytest.raises(PermissionError):
            _bind(conn, t)
        # An unset profile is equally non-authoritative.
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        with pytest.raises(PermissionError):
            _bind(conn, t)
        # Nothing durable was written.
        assert kb.directive_binding_lineage(conn, t) == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Assignee evidence is read from the task row, not caller/observer (finding 2)
# ---------------------------------------------------------------------------

def test_bound_assignee_read_from_task_row(kanban_home):
    conn = kb.connect()
    try:
        # The carrier task's assignee is the source of truth. The authenticated
        # observer is gm2, but the task is assigned to gm — the bound payload
        # must reflect the task, not the observer/selector.
        t = _mk_task(conn, assignee="gm")
        _bind(conn, t)
        bound = kb.directive_binding_lineage(conn, t)[-1]
        assert bound.payload["assignee"] == "gm"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prose-only / missing immutable id fails closed
# ---------------------------------------------------------------------------

def test_missing_source_ref_or_sha_fails_closed(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        with pytest.raises(ValueError):
            kb.record_directive_intake(
                conn, task_id=t, source_ref="", source_sha_or_immutable_id="sha-x",
                disposition="EXECUTE",
            )
        with pytest.raises(ValueError):
            kb.record_directive_intake(
                conn, task_id=t, source_ref="#ref", source_sha_or_immutable_id="  ",
                disposition="EXECUTE",
            )
    finally:
        conn.close()


def test_awareness_is_not_selection(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        # A non-EXECUTE disposition fails closed (awareness is not selection).
        with pytest.raises(ValueError):
            _bind(conn, t, disposition="OBSERVED")
        assert kb.directive_binding_lineage(conn, t) == []

        # Omitting disposition records only directive_observed (awareness),
        # never a selection/binding.
        r = _bind(conn, t, disposition=None)
        assert r["selected"] is False
        assert [e.kind for e in kb.directive_binding_lineage(conn, t)] == [
            kb.DIRECTIVE_OBSERVED_KIND,
        ]
    finally:
        conn.close()


def test_bind_to_unknown_task_fails_closed(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError):
            _bind(conn, "t_does_not_exist")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Malformed existing bound event fails closed (finding 4)
# ---------------------------------------------------------------------------

def test_malformed_bound_event_fails_closed(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        # Plant a malformed directive_bound_native event (unparseable payload)
        # and confirm a new binding fails closed rather than skipping it.
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, ?)",
            (t, kb.DIRECTIVE_BOUND_NATIVE_KIND, "{not-valid-json", 1),
        )
        conn.commit()
        with pytest.raises(RuntimeError):
            _bind(conn, t, source_sha="sha-malformed")
    finally:
        conn.close()


def test_empty_bound_event_fails_closed(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, NULL, ?)",
            (t, kb.DIRECTIVE_BOUND_NATIVE_KIND, 1),
        )
        conn.commit()
        with pytest.raises(RuntimeError):
            _bind(conn, t, source_sha="sha-empty")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pre-claim status fence (finding 5): a directive cannot bind a claimed/terminal
# task
# ---------------------------------------------------------------------------

def test_bind_to_running_task_fails_closed(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        conn.execute(
            "UPDATE tasks SET status='running', current_run_id=1 WHERE id = ?",
            (t,),
        )
        conn.commit()
        with pytest.raises(ValueError):
            _bind(conn, t, source_sha="sha-running")
    finally:
        conn.close()


def test_bind_to_done_task_fails_closed(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        conn.execute("UPDATE tasks SET status='done' WHERE id = ?", (t,))
        conn.commit()
        with pytest.raises(ValueError):
            _bind(conn, t, source_sha="sha-done")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Observed-but-not-selected cannot claim a task by implication
# ---------------------------------------------------------------------------

def test_events_do_not_claim_task(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        _bind(conn, t)
        # Recording the prefix events must not claim/dispatch/run the task:
        # status stays ready, no current_run_id, no worker pid, no claimed run.
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.current_run_id is None
        runs = conn.execute(
            "SELECT 1 FROM task_runs WHERE task_id = ?", (t,),
        ).fetchall()
        assert runs == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stale/superseded directive does not silently overwrite current binding
# ---------------------------------------------------------------------------

def test_superseded_directive_does_not_overwrite(kanban_home):
    conn = kb.connect()
    try:
        t1 = _mk_task(conn)
        _bind(conn, t1, source_sha="sha-v1")
        # A new (superseding) directive has a different immutable id -> a new,
        # separate lineage on a different task; the v1 binding is preserved.
        t2 = _mk_task(conn)
        _bind(conn, t2, source_sha="sha-v2")

        l1 = kb.directive_binding_lineage(conn, t1)
        l2 = kb.directive_binding_lineage(conn, t2)
        assert l1[0].payload["source_sha_or_immutable_id"] == "sha-v1"
        assert l2[0].payload["source_sha_or_immutable_id"] == "sha-v2"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Restart/replay: lifecycle reconstructable from persisted events
# ---------------------------------------------------------------------------

def test_lineage_survives_reconnect(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        _bind(conn, t, source_sha="sha-replay")
    finally:
        conn.close()

    # Fresh connection (simulating restart): the lineage is fully reconstructable.
    conn2 = kb.connect()
    try:
        lineage = kb.directive_binding_lineage(conn2, t)
        assert [e.kind for e in lineage] == [
            kb.DIRECTIVE_OBSERVED_KIND,
            kb.DIRECTIVE_SELECTED_KIND,
            kb.DIRECTIVE_BOUND_NATIVE_KIND,
        ]
        assert lineage[0].payload["source_sha_or_immutable_id"] == "sha-replay"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Validating lineage (finding 6): malformed order/gaps/duplicates raise
# ---------------------------------------------------------------------------

def _plant_directive_event(conn, task_id, kind, payload, created_at):
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (task_id, kind, json.dumps(payload), created_at),
    )
    conn.commit()


def test_out_of_order_lineage_raises(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        # selected without observed (gap) -> malformed lineage.
        _plant_directive_event(conn, t, kb.DIRECTIVE_SELECTED_KIND, {"x": 1}, 1)
        with pytest.raises(ValueError):
            kb.directive_binding_lineage(conn, t)
    finally:
        conn.close()


def test_duplicate_observed_lineage_raises(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        _plant_directive_event(conn, t, kb.DIRECTIVE_OBSERVED_KIND, {"x": 1}, 1)
        _plant_directive_event(conn, t, kb.DIRECTIVE_OBSERVED_KIND, {"x": 2}, 2)
        with pytest.raises(ValueError):
            kb.directive_binding_lineage(conn, t)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Hostile isolation: inherited live board pins cannot redirect fixtures
# ---------------------------------------------------------------------------

def test_inherited_live_board_pins_do_not_redirect_directive_fixtures(
    monkeypatch, tmp_path,
):
    """A standalone probe that inherits the dispatcher's live
    ``HERMES_KANBAN_DB`` / board pins must not write synthetic directive
    bindings into the real board. Regression for the AION-RL2-CORE-01-R10
    pollution class."""
    live_home = tmp_path / "live-factory"
    live_home.mkdir()
    live_db = live_home / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "aion-factory")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_home))
    monkeypatch.setenv("HERMES_PROFILE", "gm2")

    # Seed the live board with a known task so any residue is detectable.
    with kb.connect() as live_conn:
        seed = kb.create_task(live_conn, title="real-factory-task", assignee="gm")
        live_ids_before = [
            r["id"] for r in live_conn.execute(
                "SELECT id FROM tasks ORDER BY id"
            ).fetchall()
        ]

    iso = tmp_path / "isolated-probe"
    iso.mkdir()
    with kb.isolated_kanban_env(iso):
        with kb.connect() as probe_conn:
            probe_task = kb.create_task(
                probe_conn, title="probe directive carrier", assignee="gm2",
            )
            kb.record_directive_intake(
                probe_conn,
                task_id=probe_task,
                source_ref="#833 comment 5548300539",
                source_sha_or_immutable_id="sha-probe",
                disposition="EXECUTE",
            )

    # The live board is untouched; the probe task/binding landed in the
    # isolated DB, not the live board.
    with kb.connect() as live_conn:
        live_ids_after = [
            r["id"] for r in live_conn.execute(
                "SELECT id FROM tasks ORDER BY id"
            ).fetchall()
        ]
    assert live_ids_after == live_ids_before
    assert seed in live_ids_after
    assert probe_task not in live_ids_after

    # After the context manager exits, the inherited live pins are restored.
    assert os.environ.get("HERMES_KANBAN_DB") == str(live_db)
    assert os.environ.get("HERMES_KANBAN_BOARD") == "aion-factory"
