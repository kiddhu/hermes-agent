"""RED→GREEN tests for AION-Core dependency-block process closure.

The defect (canonical incident t_e690dcc1 → bounded repair child; source task
t_f6b1a62f run4212, PID 2203158).  ``block_task(kind="dependency")`` — the
controller-originated cross-task ``kanban_link`` + ``kanban_block`` dependency
path — transitions ``running → todo`` (``dependency_wait``), clears
``worker_pid``, and calls ``_end_run`` to release Native ownership, but it
**returns early inside the write txn**, before the workspace-process-closure
epilogue that the ``blocked``/``triage`` branches share.  Net effect: Native
ends the run and reports ``todo``/``current_run_id=null`` (success) while the
exact bound worker/process remains alive in its exclusive worktree.

The repair.  The ``dependency`` branch must fall through to the same shared
epilogue as ``blocked``/``triage`` so the exact bound worker is closed via the
already-audited identity-revalidated ``close_workspace_processes`` path before
the block call returns success.  Ownership semantics must stay identical:
scratch/worktree close only in-workspace processes (cwd containment, exclusive
workspace); shared-dir closes only the exact run-owned worker lineage and
preserves unrelated same-cwd processes; PID-reuse/cwd-drift/identity mismatch
fail closed; and a self-blocking worker is never signalled (its own PID).

Contract under test:

* a ``dependency`` block on a running task with a live worker closes that
  worker before returning True (exclusive scratch AND worktree).
* a ``dependency`` block on a shared-dir task closes only the exact run-owned
  lineage and preserves unrelated same-dir processes.
* outside-workspace processes (unrelated gateway/other worker) are never
  signalled.
* an already-exited worker does not raise and the block still returns True.
* a TERM-ignoring owned child is escalated to SIGKILL (bounded TERM→KILL).
* a stale/incorrect ``expected_run_id`` fails closed (returns False, no signal).
* a duplicate/replayed block on the now-``todo`` task returns False and does
  not double-signal.
* a self-blocking worker (caller cwd inside the workspace) is never signalled.

All tests are hermetic under a dispatcher-pinned environment and spawn real OS
processes against an isolated temporary DB — never the live board.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + pinned Native Kanban DB (no live-board leak)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with kb.isolated_kanban_env(home):
        kb.init_db()
        yield home


def _spawn_worker(ws: Path, *, new_session: bool = True) -> subprocess.Popen:
    """Spawn a ``sleep 300`` worker whose cwd is inside *ws*."""
    return subprocess.Popen(
        ["sleep", "300"],
        cwd=str(ws),
        start_new_session=new_session,
    )


def _spawn_term_ignoring_worker(ws: Path) -> subprocess.Popen:
    """Spawn a worker that ignores SIGTERM (forces SIGKILL escalation)."""
    code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(300)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(ws),
        start_new_session=True,
    )


def _worker_spawn_payload(proc: subprocess.Popen) -> dict:
    """Build a canonical ``spawned`` payload from a live worker process."""
    identity = kb._read_process_identity(proc.pid)
    assert identity is not None
    return {"pid": proc.pid, "starttime": identity["starttime"]}


def _make_running_task(
    conn, ws: Path, *, kind: str = "scratch",
    spawn_payload: dict | None = None,
) -> str:
    """Create a *claimed* (running) task owning *ws*, with optional spawn identity.

    ``claim_task`` drives ``ready -> running`` and sets ``current_run_id``, so
    ``block_task(kind="dependency")`` acts exactly on a live running task —
    the run4212 shape (running → todo dependency_wait).
    """
    tid = kb.create_task(conn, title="block-dep-cleanup", assignee="a")
    conn.execute(
        "UPDATE tasks SET workspace_kind=?, workspace_path=? WHERE id=?",
        (kind, str(ws), tid),
    )
    claimed = kb.claim_task(conn, tid, claimer="host:test")
    assert claimed is not None
    if spawn_payload is not None:
        kb._append_event(
            conn, tid, "spawned", spawn_payload, run_id=claimed.current_run_id,
        )
    conn.commit()
    return tid


def _kill(*procs):
    for p in procs:
        try:
            p.kill()
            p.wait(timeout=2)
        except Exception:
            pass


# ── Core RED→GREEN: dependency block must close the exact bound worker ─────


def test_dependency_block_closes_scratch_worker(kanban_home, tmp_path):
    """running→todo dependency block closes the exclusive scratch worker."""
    ws = tmp_path / "ws"
    ws.mkdir()

    worker = _spawn_worker(ws)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="scratch")
            assert kb.block_task(conn, tid, reason="waiting on parent",
                                 kind="dependency")
            t = kb.get_task(conn, tid)
            assert t.status == "todo"
            assert t.block_kind == "dependency"

        worker.wait(timeout=6)
        assert worker.returncode != 0, (
            "dependency block returned success but left the bound worker alive"
        )
    finally:
        _kill(worker)


def test_dependency_block_closes_worktree_worker(kanban_home, tmp_path):
    """The exact run4212 shape: exclusive worktree worker must be closed."""
    ws = tmp_path / "wt"
    ws.mkdir()

    worker = _spawn_worker(ws)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="worktree")
            assert kb.block_task(conn, tid, reason="dependency_wait",
                                 kind="dependency")

        worker.wait(timeout=6)
        assert worker.returncode != 0, (
            "worktree dependency block left the bound worker alive"
        )
    finally:
        _kill(worker)


def test_dependency_block_dir_closes_owned_preserves_unrelated(
    kanban_home, tmp_path,
):
    """Shared-dir dependency block closes owned lineage, spares unrelated."""
    ws = tmp_path / "shared"
    ws.mkdir()

    worker = _spawn_worker(ws)
    unrelated = _spawn_worker(ws)
    try:
        time.sleep(0.1)
        payload = _worker_spawn_payload(worker)

        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="dir", spawn_payload=payload)
            assert kb.block_task(conn, tid, reason="wait", kind="dependency")

        worker.wait(timeout=6)
        assert worker.returncode != 0, "owned worker was not signalled"
        assert unrelated.poll() is None, (
            "unrelated same-dir worker was signalled by dependency block"
        )
    finally:
        _kill(worker, unrelated)


# ── Hostile / containment ───────────────────────────────────────────────────


def test_dependency_block_preserves_outside_workspace_process(
    kanban_home, tmp_path,
):
    """A process with cwd outside the workspace is never signalled."""
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    worker = _spawn_worker(ws)
    stranger = _spawn_worker(outside)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="scratch")
            assert kb.block_task(conn, tid, reason="wait", kind="dependency")

        worker.wait(timeout=6)
        assert worker.returncode != 0, "in-workspace worker was not signalled"
        assert stranger.poll() is None, (
            "outside-workspace process was signalled by dependency block"
        )
    finally:
        _kill(worker, stranger)


def test_dependency_block_already_exited_worker_returns_true(
    kanban_home, tmp_path,
):
    """An already-exited worker is a no-op: block still returns True."""
    ws = tmp_path / "ws"
    ws.mkdir()

    worker = _spawn_worker(ws)
    worker.kill()
    worker.wait(timeout=2)
    try:
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="scratch")
            assert kb.block_task(conn, tid, reason="wait", kind="dependency")
            assert kb.get_task(conn, tid).status == "todo"
    finally:
        _kill(worker)


@pytest.mark.live_system_guard_bypass
def test_dependency_block_term_ignoring_child_is_killed(kanban_home, tmp_path):
    """A TERM-ignoring owned worker is escalated to SIGKILL and does not survive."""
    ws = tmp_path / "ws"
    ws.mkdir()

    worker = _spawn_term_ignoring_worker(ws)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="scratch")
            assert kb.block_task(conn, tid, reason="wait", kind="dependency")

        worker.wait(timeout=8)
        assert worker.returncode != 0, (
            "TERM-ignoring worker survived dependency block (no KILL escalation)"
        )
    finally:
        _kill(worker)


def test_dependency_block_stale_expected_run_id_fails_closed(
    kanban_home, tmp_path,
):
    """A stale expected_run_id fails closed: block returns False, no signal."""
    ws = tmp_path / "ws"
    ws.mkdir()

    worker = _spawn_worker(ws)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="scratch")
            actual = kb.get_task(conn, tid).current_run_id
            assert actual is not None
            # A stale run id that does not match the live run.
            assert not kb.block_task(
                conn, tid, reason="wait", kind="dependency",
                expected_run_id=actual + 999_999,
            )
            # Ownership NOT released, worker untouched.
            assert kb.get_task(conn, tid).status == "running"

        assert worker.poll() is None, (
            "stale expected_run_id dependency block signalled the worker"
        )
    finally:
        _kill(worker)


def test_dependency_block_replay_returns_false_no_double_signal(
    kanban_home, tmp_path,
):
    """A replayed dependency block on a todo task is a no-op (False)."""
    ws = tmp_path / "ws"
    ws.mkdir()

    worker = _spawn_worker(ws)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="scratch")
            assert kb.block_task(conn, tid, reason="wait", kind="dependency")
            # Replay (idempotent orchestrator re-issue): task no longer running.
            assert not kb.block_task(conn, tid, reason="wait", kind="dependency")

        worker.wait(timeout=6)
        assert worker.returncode != 0, "worker not closed on first block"
    finally:
        _kill(worker)


def test_dependency_block_self_caller_never_signalled(kanban_home, tmp_path):
    """A self-blocking worker (caller cwd inside the workspace) is never signalled.

    The caller performs ``block_task(kind="dependency")`` on its own task while
    its own cwd is inside the workspace. ``close_workspace_processes`` must
    skip the caller PID (``skipped_self``) so the caller survives to receive its
    tool receipt, while an owned child in the same workspace is closed.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    marker = tmp_path / "survived.txt"

    script = (
        "import pathlib\n"
        "from hermes_cli import kanban_db as kb\n"
        "home = pathlib.Path.home()\n"
        "with kb.isolated_kanban_env(home):\n"
        "    kb.init_db()\n"
        "    with kb.connect() as conn:\n"
        "        tid = kb.create_task(conn, title='self-block', assignee='a')\n"
        "        conn.execute(\"UPDATE tasks SET workspace_kind='scratch',\"\n"
        "                     \" workspace_path=? WHERE id=?\","
        f" ({str(ws)!r}, tid))\n"
        "        claimed = kb.claim_task(conn, tid, claimer='host:self')\n"
        "        assert claimed is not None\n"
        "        conn.commit()\n"
        "        # Caller cwd IS the workspace; an owned child also lives there.\n"
        "        ok = kb.block_task(conn, tid, reason='dep', kind='dependency')\n"
        f"        pathlib.Path({str(marker)!r}).write_text(str(ok))\n"
    )

    child = subprocess.Popen(
        ["sleep", "300"], cwd=str(ws), start_new_session=True,
    )
    caller = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(ws),
        start_new_session=True,
    )
    try:
        caller.wait(timeout=15)
        assert caller.returncode == 0, (
            f"self-blocking caller died (exit {caller.returncode})"
        )
        assert marker.exists(), "caller did not reach post-block receipt write"
        assert marker.read_text() == "True"
        child.wait(timeout=6)
        assert child.returncode != 0, "owned child was not closed"
    finally:
        _kill(caller, child)


def test_dependency_block_dir_no_spawn_identity_fails_closed(
    kanban_home, tmp_path,
):
    """A dir task with no provable spawn identity signals nothing (fail closed)."""
    ws = tmp_path / "shared"
    ws.mkdir()

    unrelated = _spawn_worker(ws)
    try:
        time.sleep(0.1)
        with kb.connect() as conn:
            tid = _make_running_task(conn, ws, kind="dir", spawn_payload=None)
            assert kb.block_task(conn, tid, reason="wait", kind="dependency")

        assert unrelated.poll() is None, (
            "shared-dir process signalled despite no spawn identity"
        )
    finally:
        _kill(unrelated)
