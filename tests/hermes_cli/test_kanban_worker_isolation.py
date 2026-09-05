"""Tests for R06 B/C — per-worker cgroup isolation + process-level reaping.

R06-C closes the canonical t_e690dcc1 defect: a worker whose detached
descendants (background procs, LSP servers, temp subprocesses) survive the
worker's own termination and hold memory/swap in the gateway cgroup. The fix
has two halves, both best-effort / fail-open:

  * R06-B — a per-worker cgroup v2 (``hermes-kanban-<task>``) with optional
    ``memory.high`` / ``memory.max`` / ``pids.max`` limits.
  * R06-C — process-level descendant reaping: kill the cgroup members (or, when
    no cgroup exists, killpg + a /proc descendant walk) at every termination
    path (max-runtime timeout, crash, TTL/no-heartbeat reclaim, operator
    reclaim, self-fence) and for terminal tasks.

All helpers never raise and degrade to a no-op, so these tests exercise the
pure logic via monkeypatched cgroup/process primitives rather than real host
signals.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with kb.isolated_kanban_env(home):
        yield home


def _mk_task(conn, assignee="gm2", **kw):
    return kb.create_task(conn, title="isolation probe", assignee=assignee, **kw)


# ---------------------------------------------------------------------------
# R06-B — isolation settings resolution + cgroup create / assign
# ---------------------------------------------------------------------------

def test_resolve_worker_isolation_settings_env_bridge(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_WORKER_ISOLATION", "1")
    monkeypatch.setenv("HERMES_KANBAN_WORKER_MEMORY_HIGH_BYTES", "1024")
    monkeypatch.setenv("HERMES_KANBAN_WORKER_MEMORY_MAX_BYTES", "2048")
    monkeypatch.setenv("HERMES_KANBAN_WORKER_PIDS_MAX", "64")
    s = kb._resolve_worker_isolation_settings()
    assert s["enabled"] is True
    assert s["memory_high_bytes"] == 1024
    assert s["memory_max_bytes"] == 2048
    assert s["pids_max"] == 64

    monkeypatch.setenv("HERMES_KANBAN_WORKER_ISOLATION", "0")
    monkeypatch.delenv("HERMES_KANBAN_WORKER_MEMORY_HIGH_BYTES")
    s = kb._resolve_worker_isolation_settings()
    assert s["enabled"] is False
    assert s["memory_high_bytes"] == 0


def test_worker_cgroup_name_sanitizes():
    assert kb._worker_cgroup_name("t_abc123") == "hermes-kanban-t_abc123"
    assert kb._worker_cgroup_name("t/a/../b") == "hermes-kanban-t_a_.._b"


def test_create_worker_cgroup_writes_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "_CGROUP_FS_ROOT", str(tmp_path))
    monkeypatch.setattr(kb, "_dispatcher_cgroup_path", lambda: "/parent")
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: f"/parent/hermes-kanban-{task_id}")
    monkeypatch.setattr(kb, "_ensure_subtree_controllers", lambda parent: True)

    path = kb._create_worker_cgroup("t1", memory_high=100, memory_max=200, pids_max=50)
    assert path == "/parent/hermes-kanban-t1"
    base = tmp_path / "parent" / "hermes-kanban-t1"
    assert (base / "memory.high").read_text() == "100"
    assert (base / "memory.max").read_text() == "200"
    assert (base / "pids.max").read_text() == "50"


def test_create_worker_cgroup_zero_limits_write_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "_CGROUP_FS_ROOT", str(tmp_path))
    monkeypatch.setattr(kb, "_dispatcher_cgroup_path", lambda: "/parent")
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: f"/parent/hermes-kanban-{task_id}")
    monkeypatch.setattr(kb, "_ensure_subtree_controllers", lambda parent: True)

    path = kb._create_worker_cgroup("t2")
    assert path == "/parent/hermes-kanban-t2"
    base = tmp_path / "parent" / "hermes-kanban-t2"
    assert not (base / "memory.high").exists()
    assert not (base / "pids.max").exists()


def test_create_worker_cgroup_fails_open_when_no_cgroup(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "_dispatcher_cgroup_path", lambda: None)
    assert kb._create_worker_cgroup("t3", memory_high=100) is None


def test_assign_pid_to_cgroup(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "_CGROUP_FS_ROOT", str(tmp_path))
    (tmp_path / "cg").mkdir()
    assert kb._assign_pid_to_cgroup(12345, "/cg") is True
    assert (tmp_path / "cg" / "cgroup.procs").read_text() == "12345"


def test_assign_pid_to_cgroup_rejects_bad_pid():
    assert kb._assign_pid_to_cgroup(0, "/cg") is False
    assert kb._assign_pid_to_cgroup(None, "/cg") is False


# ---------------------------------------------------------------------------
# R06-C — cgroup reaping
# ---------------------------------------------------------------------------

def test_reap_worker_cgroup_signals_members_except_self(monkeypatch):
    import signal as _signal
    import time as _time

    own = os.getpid()
    monkeypatch.setattr(kb, "_read_cgroup_pids", lambda path: [111, own, 222])
    outcomes = []

    def fake_ladder(pid, kill, *, signal, time):
        outcomes.append(pid)
        return "terminated"

    monkeypatch.setattr(kb, "_signal_pid_ladder", fake_ladder)
    n = kb._reap_worker_cgroup(
        "/cg", kill=lambda p, s: None, signal=_signal, time=_time,
    )
    assert n == 2
    assert own not in outcomes
    assert 111 in outcomes and 222 in outcomes


def test_read_cgroup_pids_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "_CGROUP_FS_ROOT", str(tmp_path))
    assert kb._read_cgroup_pids("/nonexistent") == []


# ---------------------------------------------------------------------------
# R06-C — _reap_worker_descendants (cgroup-first, then fallback)
# ---------------------------------------------------------------------------

def test_reap_worker_descendants_cgroup_first(monkeypatch):
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: "/cg")
    monkeypatch.setattr(kb, "_cgroup_exists", lambda path: True)
    monkeypatch.setattr(
        kb, "_reap_worker_cgroup", lambda path, *, kill, signal, time: 5,
    )
    monkeypatch.setattr(kb, "_remove_worker_cgroup", lambda path: True)

    info = kb._reap_worker_descendants(4242, task_id="t1", signal_fn=lambda p, s: None)
    assert info["cgroup_path"] == "/cg"
    assert info["cgroup_reaped"] == 5
    assert info["killpg_attempted"] is False
    assert info["descendants"] == 0


def test_reap_worker_descendants_pid_none_still_reaps_cgroup(monkeypatch):
    # reconcile_terminal_runs calls with pid=None: cgroup membership survives
    # the worker root's exit, so the cgroup must still be reaped.
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: "/cg")
    monkeypatch.setattr(kb, "_cgroup_exists", lambda path: True)
    monkeypatch.setattr(
        kb, "_reap_worker_cgroup", lambda path, *, kill, signal, time: 3,
    )
    monkeypatch.setattr(kb, "_remove_worker_cgroup", lambda path: True)

    info = kb._reap_worker_descendants(None, task_id="t1", signal_fn=lambda p, s: None)
    assert info["cgroup_reaped"] == 3
    assert info["cgroup_path"] == "/cg"


def test_reap_worker_descendants_fallback_killpg_and_walk(monkeypatch):
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: None)
    monkeypatch.setattr(
        kb, "_read_process_identity",
        lambda pid: {"starttime": 123, "cwd": "/", "pgid": 1},
    )
    monkeypatch.setattr(
        kb, "_discover_descendant_pids",
        lambda pid, expected_starttime=None: {111, 222},
    )
    signalled = []

    def fake_ladder(pid, kill, *, signal, time):
        signalled.append(pid)
        return "terminated"

    monkeypatch.setattr(kb, "_signal_pid_ladder", fake_ladder)
    monkeypatch.setattr(os, "killpg", lambda *a, **k: None)
    # 4242 is a synthetic worker pid in a different (group-leader) session, so
    # the killpg guard passes: it is not our own pid and os.getpgid(4242)==4242.
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    info = kb._reap_worker_descendants(4242, task_id="t1", signal_fn=lambda p, s: None)
    assert info["killpg_attempted"] is True
    assert info["descendants"] == 2
    assert info["terminated"] == 2
    # The worker root is NOT re-signalled here — the caller already terminated
    # it and killpg(pid) covers it. Only the detached descendants are signalled.
    assert 4242 not in signalled
    assert 111 in signalled and 222 in signalled


def test_reap_worker_descendants_noop_without_pid_or_cgroup(monkeypatch):
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: None)
    info = kb._reap_worker_descendants(None, task_id="t1", signal_fn=lambda p, s: None)
    assert info["pid"] is None
    assert info["cgroup_reaped"] == 0


def test_reap_worker_descendants_never_killpg_own_group(monkeypatch):
    # A synthetic worker pid equal to the caller's own pid must never trigger
    # killpg: in a session-leader process (e.g. the test runner) that would
    # SIGTERM our own process group mid-run. The guard short-circuits.
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: None)
    monkeypatch.setattr(
        kb, "_read_process_identity",
        lambda pid: {"starttime": 123, "cwd": "/", "pgid": 1},
    )
    monkeypatch.setattr(kb, "_discover_descendant_pids", lambda pid, expected_starttime=None: set())
    monkeypatch.setattr(kb, "_signal_pid_ladder", lambda *a, **k: "gone")
    killpg_calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(pgid))

    info = kb._reap_worker_descendants(os.getpid(), task_id="t1", signal_fn=lambda p, s: None)
    assert info["killpg_attempted"] is False
    assert killpg_calls == []


def test_reap_worker_descendants_skips_killpg_for_non_group_leader(monkeypatch):
    # A synthetic worker pid that is NOT a process-group leader (pgid != pid)
    # must not be killpg'd — that would signal an unrelated group (possibly our
    # own). The /proc descendant walk still runs.
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: None)
    monkeypatch.setattr(
        kb, "_read_process_identity",
        lambda pid: {"starttime": 123, "cwd": "/", "pgid": 1},
    )
    monkeypatch.setattr(
        kb, "_discover_descendant_pids",
        lambda pid, expected_starttime=None: {111},
    )
    monkeypatch.setattr(kb, "_signal_pid_ladder", lambda *a, **k: "terminated")
    killpg_calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(pgid))
    monkeypatch.setattr(os, "getpgid", lambda pid: 1)  # not pid -> not a leader

    info = kb._reap_worker_descendants(4242, task_id="t1", signal_fn=lambda p, s: None)
    assert info["killpg_attempted"] is False
    assert killpg_calls == []
    assert info["descendants"] == 1


def test_reap_worker_descendants_windows_no_killpg_no_raise(monkeypatch):
    # RED/GREEN Windows-footgun guard: os.killpg / os.getpgid are POSIX-only
    # and absent on Windows, where bare access raises AttributeError. Simulate
    # their absence (monkeypatch.delattr) and prove the fallback path degrades
    # to a clean no-op for the killpg primitive — killpg_attempted stays False,
    # no AttributeError escapes — while the /proc descendant walk still runs.
    monkeypatch.setattr(kb, "_worker_cgroup_path", lambda task_id: None)
    monkeypatch.setattr(
        kb, "_read_process_identity",
        lambda pid: {"starttime": 123, "cwd": "/", "pgid": 1},
    )
    monkeypatch.setattr(
        kb, "_discover_descendant_pids",
        lambda pid, expected_starttime=None: {111},
    )
    monkeypatch.setattr(kb, "_signal_pid_ladder", lambda *a, **k: "terminated")
    monkeypatch.delattr(os, "killpg", raising=False)
    monkeypatch.delattr(os, "getpgid", raising=False)

    info = kb._reap_worker_descendants(4242, task_id="t1", signal_fn=lambda p, s: None)
    assert info["killpg_attempted"] is False
    assert info["descendants"] == 1
    assert info["terminated"] == 1


# ---------------------------------------------------------------------------
# R06-C — wiring into termination paths
# ---------------------------------------------------------------------------

def test_enforce_max_runtime_reaps_descendants(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        t = kb.create_task(conn, title="timeout", assignee="gm2", max_runtime_seconds=10)
        kb.claim_task(conn, t, claimer=f"{host}:t")
        run = kb.latest_run(conn, t)
        old_started = int(time.time()) - 20
        conn.execute(
            "UPDATE tasks SET started_at = ?, worker_pid = ? WHERE id = ?",
            (old_started, 999999, t),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ?, worker_pid = ? WHERE id = ?",
            (old_started, 999999, run.id),
        )

        reaped = []

        def _fake_reap(pid, *, task_id=None, signal_fn=None):
            reaped.append((pid, task_id))
            return {"descendants": 2}

        monkeypatch.setattr(kb, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(kb, "_reap_worker_descendants", _fake_reap)

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda _p, _s: None)
        assert timed_out == [t]
        assert reaped == [(999999, t)]
    finally:
        conn.close()


def test_detect_crashed_workers_reaps_descendants(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        t = kb.create_task(conn, title="crash", assignee="gm2")
        kb.claim_task(conn, t, claimer=f"{host}:c")
        # Back-date start so the crash grace window has elapsed.
        old_started = int(time.time()) - 120
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
            (999999, old_started, t),
        )

        reaped = []

        def _fake_reap(pid, *, task_id=None, signal_fn=None):
            reaped.append((pid, task_id))
            return {}

        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(kb, "_reap_worker_descendants", _fake_reap)
        # Avoid breaker tripping on the synthetic crash.
        monkeypatch.setattr(kb, "_record_task_failure", lambda *a, **k: False)

        crashed = kb.detect_crashed_workers(conn)
        assert crashed == [t]
        assert (999999, t) in reaped
    finally:
        conn.close()


def test_reconcile_terminal_runs_reaps_cgroup(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="terminal", assignee="gm2")
        kb.claim_task(conn, t)
        run_id = kb._current_run_id(conn, t)
        assert run_id is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (t,),
            )

        reaped = []

        def _fake_reap(pid, *, task_id=None, signal_fn=None):
            reaped.append(task_id)
            return {"cgroup_reaped": 4}

        monkeypatch.setattr(kb, "_reap_worker_descendants", _fake_reap)

        closed = kb.reconcile_terminal_runs(conn)
        assert closed == 1
        assert reaped == [t]
    finally:
        conn.close()


def test_spawned_event_carries_isolation_verdict(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        t = _mk_task(conn)
        monkeypatch.setattr(kb, "_resolve_worker_isolation_settings", lambda: {
            "enabled": False,
            "memory_high_bytes": 0,
            "memory_max_bytes": 0,
            "pids_max": 0,
        })
        spawn_calls = []

        def spawn_fn(task, workspace, board=None):
            spawn_calls.append(task.id)
            return 424242

        kb.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        spawned = [e for e in kb.list_events(conn, t) if e.kind == "spawned"]
        assert len(spawned) == 1
        payload = spawned[0].payload
        assert payload["pid"] == 424242
        assert "worker_isolation_enabled" in payload
        assert payload["worker_isolation"] in {"cgroup", "degraded"}
    finally:
        conn.close()
