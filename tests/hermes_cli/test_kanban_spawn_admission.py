"""Tests for R06 A — pre-spawn resource admission (host + cgroup headroom).

The dispatcher must refuse to spawn a new worker while aggregate host+cgroup
memory headroom is below a configured floor, leaving the task ``ready`` for a
later tick instead of failing it. Disabled by default (floor = 0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The ready-dispatch loop refuses to spawn for an unknown assignee profile.
    # Make the synthetic "gm2" assignee look real in the isolated env so the
    # task reaches the admission check instead of the nonspawnable bucket.
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: True
    )
    with kb.isolated_kanban_env(home):
        yield home


def _mk_task(conn, assignee="gm2"):
    return kb.create_task(conn, title="admission probe", assignee=assignee)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_spawn_admission_defers_only_below_floor():
    assert kb._spawn_admission_defers(500, 1000) is True
    assert kb._spawn_admission_defers(1000, 1000) is False
    assert kb._spawn_admission_defers(999, 0) is False  # disabled floor never defers


def test_spawn_resource_headroom_injects_probe():
    assert kb._spawn_resource_headroom_bytes(headroom_fn=lambda: 123) == 123


def test_spawn_resource_headroom_unknown_is_sentinel(monkeypatch):
    monkeypatch.setattr(kb, "_read_meminfo_memavailable_bytes", lambda: None)
    monkeypatch.setattr(kb, "_read_cgroup_memory_headroom_bytes", lambda: None)
    assert kb._spawn_resource_headroom_bytes() == kb._SPAWN_ADMISSION_UNKNOWN_HEADROOM


def test_spawn_resource_headroom_takes_min_of_sources(monkeypatch):
    monkeypatch.setattr(kb, "_read_meminfo_memavailable_bytes", lambda: 1024)
    monkeypatch.setattr(kb, "_read_cgroup_memory_headroom_bytes", lambda: 512)
    assert kb._spawn_resource_headroom_bytes() == 512


def test_resolve_spawn_admission_min_free_bytes(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_SPAWN_ADMISSION_MIN_FREE_BYTES", "12345")
    assert kb._resolve_spawn_admission_min_free_bytes() == 12345
    monkeypatch.setenv("HERMES_KANBAN_SPAWN_ADMISSION_MIN_FREE_BYTES", "not-an-int")
    assert kb._resolve_spawn_admission_min_free_bytes() == kb.DEFAULT_SPAWN_ADMISSION_MIN_FREE_BYTES
    monkeypatch.delenv("HERMES_KANBAN_SPAWN_ADMISSION_MIN_FREE_BYTES")
    assert kb._resolve_spawn_admission_min_free_bytes() == 0


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

def test_admission_defers_spawn_when_headroom_low(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        monkeypatch.setenv(
            "HERMES_KANBAN_SPAWN_ADMISSION_MIN_FREE_BYTES", str(1 << 30)
        )
        monkeypatch.setattr(
            kb, "_spawn_resource_headroom_bytes", lambda headroom_fn=None: 500
        )
        spawn_calls = []

        def spawn_fn(task, workspace, board=None):
            spawn_calls.append(task.id)
            return 424242

        result = kb.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        assert spawn_calls == []
        assert any(tid == t for (tid, _min, _head) in result.resource_deferred)
        # Task stayed ready/unclaimed (no failure counted).
        assert kb.get_task(conn, t).status == "ready"
        # A durable defer event is emitted.
        evs = [e for e in kb.list_events(conn, t) if e.kind == "spawn_deferred_resource"]
        assert len(evs) == 1
        assert evs[0].payload["min_free_bytes"] == (1 << 30)
        assert evs[0].payload["headroom_bytes"] == 500
    finally:
        conn.close()


def test_admission_allows_spawn_when_headroom_ok(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        monkeypatch.setenv(
            "HERMES_KANBAN_SPAWN_ADMISSION_MIN_FREE_BYTES", str(1 << 30)
        )
        monkeypatch.setattr(
            kb, "_spawn_resource_headroom_bytes", lambda headroom_fn=None: (2 << 30)
        )
        spawn_calls = []

        def spawn_fn(task, workspace, board=None):
            spawn_calls.append(task.id)
            return 424242

        result = kb.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        assert spawn_calls == [t]
        assert result.resource_deferred == []
    finally:
        conn.close()


def test_admission_disabled_by_default(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        monkeypatch.setattr(
            kb, "_spawn_resource_headroom_bytes", lambda headroom_fn=None: 0
        )
        spawn_calls = []

        def spawn_fn(task, workspace, board=None):
            spawn_calls.append(task.id)
            return 424242

        result = kb.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        assert spawn_calls == [t]  # disabled -> no defer
        assert result.resource_deferred == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# R06-B — automatic first heartbeat at spawn
# ---------------------------------------------------------------------------

def test_spawn_records_automatic_first_heartbeat(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        spawn_calls = []

        def spawn_fn(task, workspace, board=None):
            spawn_calls.append(task.id)
            return 424242

        kb.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        assert spawn_calls == [t]
        # Automatic first heartbeat event marks the worker as booted; the
        # ``last_heartbeat_at`` field stays a pure progress signal (set by
        # heartbeat_worker) so detect_stale_running can still reclaim a
        # booted-but-stalled worker. Verify the automatic boot event exists.
        evs = kb.list_events(conn, t)
        hb = [e for e in evs if e.kind == "heartbeat" and e.payload and e.payload.get("automatic")]
        assert len(hb) == 1
        assert hb[0].payload["phase"] == "spawn"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# R06-C — resource attribution on the spawned event
# ---------------------------------------------------------------------------

def test_spawned_event_carries_resource_attribution(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)

        def spawn_fn(task, workspace, board=None):
            return 424242

        kb.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        spawned = [e for e in kb.list_events(conn, t) if e.kind == "spawned"]
        assert len(spawned) == 1
        payload = spawned[0].payload
        assert payload["pid"] == 424242
        assert payload["assignee_profile"] == "gm2"
        # Deterministic fields regardless of cgroup readability.
        assert "resource_admission_verdict" in payload
        assert payload["resource_admission_verdict"] in {"admitted", "disabled"}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# R06-C — terminal cleanup / reaping (orphaned open run closure)
# ---------------------------------------------------------------------------

def test_reconcile_terminal_runs_closes_orphaned_open_run(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        claimed = kb.claim_task(conn, t)
        assert claimed is not None
        run_id = kb._current_run_id(conn, t)
        assert run_id is not None
        # Simulate the historical orphan: terminalize the task WITHOUT closing
        # its active run (current_run_id cleared, run left open).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (t,),
            )
        row = conn.execute(
            "SELECT status, ended_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["status"] == "running"
        assert row["ended_at"] is None

        closed = kb.reconcile_terminal_runs(conn)
        assert closed == 1
        row = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["status"] == "reclaimed"
        assert row["outcome"] == "reclaimed"
        assert row["ended_at"] is not None
        # Idempotent: second pass closes nothing.
        assert kb.reconcile_terminal_runs(conn) == 0
    finally:
        conn.close()


def test_reconcile_terminal_runs_ignores_live_pointer(kanban_home):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        claimed = kb.claim_task(conn, t)
        assert claimed is not None
        run_id = kb._current_run_id(conn, t)
        assert run_id is not None
        # A running task with a live current_run_id must NOT be reconciled —
        # _end_run owns that closure.
        assert kb.reconcile_terminal_runs(conn) == 0
        row = conn.execute(
            "SELECT status, ended_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["status"] == "running"
        assert row["ended_at"] is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Errno11 / EAGAIN attribution (platform_resource classification)
# ---------------------------------------------------------------------------

def test_classify_failure_errno11_is_platform_resource():
    assert kb._classify_failure("OSError: [Errno 11] Resource temporarily unavailable") == "platform_resource"
    assert kb._classify_failure("fork failed: EAGAIN") == "platform_resource"
    assert kb._classify_failure("something else entirely") == "task"


def test_read_cgroup_attribution_never_raises():
    # Must return a dict with the four expected keys regardless of host cgroup
    # state, and never raise (best-effort, fail-open attribution).
    out = kb._read_cgroup_attribution(99999999)  # nonexistent PID
    assert set(out) == {"cgroup_path", "memory_current", "memory_high", "memory_max"}

