"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _dispatch_native_lifecycle_restart_signal,
    _evaluate_native_lifecycle_restart_request,
)
from hermes_cli.kanban_db import NativeLifecycleRequest

SIGUSR1 = getattr(signal, "SIGUSR1", None)

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


def _request(**overrides):
    values = {
        "version": 1,
        "action": "planned_gateway_restart",
        "nonce": "restart-request-0001",
        "task_id": "t_1234abcd",
        "target_service": "hermes-gateway-gm2.service",
        "expected_pid": 4242,
        "expected_starttime": 98765,
        "expected_invocation_id": "invocation-123",
        "expected_cgroup": "/system.slice/hermes-gateway-gm2.service",
        "require_board_idle": True,
    }
    values.update(overrides)
    return NativeLifecycleRequest(**values)


def test_native_lifecycle_request_accepts_exact_gateway_generation():
    task = SimpleNamespace(id="t_1234abcd", current_run_id=73, claim_lock="claim-73")
    decision = _evaluate_native_lifecycle_restart_request(
        task,
        _request(),
        current_pid=4242,
        current_starttime=98765,
        current_invocation_id="invocation-123",
        current_service="hermes-gateway-gm2.service",
        current_cgroup="/system.slice/hermes-gateway-gm2.service",
    )

    assert decision.accepted is True
    assert decision.receipt == {
        "actual_pid": 4242,
        "actual_starttime": 98765,
        "actual_invocation_id": "invocation-123",
        "actual_service": "hermes-gateway-gm2.service",
        "actual_cgroup": "/system.slice/hermes-gateway-gm2.service",
    }


@pytest.mark.parametrize(
    "actuals, reason",
    [
        ({"current_pid": 4243}, "expected_pid mismatch"),
        ({"current_starttime": 98766}, "expected_starttime mismatch"),
        ({"current_invocation_id": "invocation-456"}, "expected_invocation_id mismatch"),
        ({"current_service": "hermes-gateway-agent007.service"}, "target_service mismatch"),
        ({"current_cgroup": "/system.slice/hermes-gateway-agent007.service"}, "expected_cgroup mismatch"),
    ],
)
def test_native_lifecycle_request_rejects_generation_or_target_drift(actuals, reason):
    task = SimpleNamespace(id="t_1234abcd", current_run_id=73, claim_lock="claim-73")
    runtime = {
        "current_pid": 4242,
        "current_starttime": 98765,
        "current_invocation_id": "invocation-123",
        "current_service": "hermes-gateway-gm2.service",
        "current_cgroup": "/system.slice/hermes-gateway-gm2.service",
    }
    runtime.update(actuals)

    decision = _evaluate_native_lifecycle_restart_request(task, _request(), **runtime)

    assert decision.accepted is False
    assert reason in decision.reason


def test_native_lifecycle_request_rejects_unbound_claim_provenance():
    task = SimpleNamespace(id="t_1234abcd", current_run_id=None, claim_lock=None)
    decision = _evaluate_native_lifecycle_restart_request(
        task,
        _request(),
        current_pid=4242,
        current_starttime=98765,
        current_invocation_id="invocation-123",
        current_service="hermes-gateway-gm2.service",
        current_cgroup="/system.slice/hermes-gateway-gm2.service",
    )

    assert decision.accepted is False
    assert "claim provenance" in decision.reason


@pytest.mark.skipif(SIGUSR1 is None, reason="SIGUSR1 is POSIX-only")
def test_dispatch_native_lifecycle_restart_signal_uses_existing_sigusr1_path():
    calls = []
    accepted = [{"task_id": "t_1234abcd", "nonce": "restart-request-0001"}]

    assert _dispatch_native_lifecycle_restart_signal(
        accepted,
        kill_fn=lambda pid, sig: calls.append((pid, sig)),
        current_pid=4242,
    )
    assert calls == [(4242, SIGUSR1)]


def test_dispatch_native_lifecycle_restart_signal_fails_closed_without_exact_single_request():
    calls = []
    assert not _dispatch_native_lifecycle_restart_signal([], kill_fn=lambda *args: calls.append(args))
    assert not _dispatch_native_lifecycle_restart_signal(
        [{"task_id": "a"}, {"task_id": "b"}],
        kill_fn=lambda *args: calls.append(args),
    )
    assert calls == []


@pytest.mark.skipif(SIGUSR1 is None, reason="SIGUSR1 is POSIX-only")
def test_native_lifecycle_request_temp_db_reaches_existing_sigusr1_path(
    tmp_path, monkeypatch
):
    """Exercise the real request parse→claim→receipt→signal handoff."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.gateway import get_service_name

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("INVOCATION_ID", "integration-invocation-123")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    service = f"{get_service_name()}.service"
    # The production evaluator deliberately requires physical membership in
    # the target unit cgroup. Patch only the /proc read boundary so this
    # isolated test can exercise the full DB/dispatcher path outside systemd.
    target_cgroup = f"/system.slice/{service}"
    monkeypatch.setattr(
        "gateway.kanban_watchers._read_self_cgroup",
        lambda: target_cgroup,
    )
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="restart exact gateway", assignee="agent007")
        body = {
            "native_lifecycle_request": {
                "version": 1,
                "action": "planned_gateway_restart",
                "nonce": "integration-request-0001",
                "task_id": task_id,
                "target_service": service,
                "expected_pid": os.getpid(),
                "expected_starttime": kb._process_starttime(),
                "expected_invocation_id": "integration-invocation-123",
                "expected_cgroup": target_cgroup,
                "require_board_idle": True,
            }
        }
        conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (json.dumps(body), task_id))
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: pytest.fail("control request must not spawn a worker"),
            lifecycle_request_fn=_evaluate_native_lifecycle_restart_request,
        )

    calls = []
    assert _dispatch_native_lifecycle_restart_signal(
        result.lifecycle_restart_requests,
        kill_fn=lambda pid, sig: calls.append((pid, sig)),
    )
    assert calls == [(os.getpid(), SIGUSR1)]


@pytest.mark.parametrize(
    "invalid_body",
    [
        lambda task_id, request: json.dumps(
            {"native_lifecycle_request": request, "unrelated": True}
        ),
        lambda task_id, request: (
            '{"native_lifecycle_request":'
            f'{json.dumps(request)},"native_lifecycle_request":{json.dumps(request)}'
            "}"
        ),
        lambda task_id, request: json.dumps(
            {"native_lifecycle_request": request}
        ).replace('"version": 1', '"version": 1, "version": 1', 1),
        lambda task_id, request: json.dumps(
            {"native_lifecycle_request": request}
        ).replace("native_lifecycle_request", r"native\u005flifecycle_request", 1),
    ],
    ids=[
        "extra-envelope-key",
        "duplicate-envelope-key",
        "duplicate-request-key",
        "escaped-envelope-key",
    ],
)
def test_non_closed_native_lifecycle_json_cannot_reach_callback_spawn_or_signal(
    tmp_path, monkeypatch, invalid_body
):
    from hermes_cli import kanban_db as kb
    from hermes_cli import profiles

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    kb.init_db()

    callback_calls = []
    spawn_calls = []
    signal_calls = []

    def evaluate(*_):
        callback_calls.append(True)
        return kb.NativeLifecycleDecision(True, "unexpected callback", {})

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="reject non-closed JSON", assignee="agent007")
        request = {
            "version": 1,
            "action": "planned_gateway_restart",
            "nonce": "integration-request-0001",
            "task_id": task_id,
            "target_service": "hermes-gateway-gm2.service",
            "expected_pid": os.getpid(),
            "expected_starttime": kb._process_starttime(),
            "expected_invocation_id": "integration-invocation-123",
            "expected_cgroup": "/system.slice/hermes-gateway-gm2.service",
            "require_board_idle": True,
        }
        conn.execute(
            "UPDATE tasks SET body = ? WHERE id = ?",
            (invalid_body(task_id, request), task_id),
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: spawn_calls.append(True),
            lifecycle_request_fn=evaluate,
        )

    assert not _dispatch_native_lifecycle_restart_signal(
        result.lifecycle_restart_requests,
        kill_fn=lambda *args: signal_calls.append(args),
    )
    assert callback_calls == []
    assert spawn_calls == []
    assert signal_calls == []
