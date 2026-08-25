"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.kanban_watchers as kanban_watchers
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
        "expires_at": int(time.time()) + 120,
    }
    values.update(overrides)
    return NativeLifecycleRequest(**values)


def test_native_lifecycle_request_accepts_exact_gateway_generation():
    task = SimpleNamespace(
        id="t_1234abcd",
        current_run_id=73,
        claim_lock="claim-73",
        claim_expires=int(time.time()) + 300,
    )
    request = _request()
    decision = _evaluate_native_lifecycle_restart_request(
        task,
        request,
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
        "expires_at": request.expires_at,
        "claim_expires": task.claim_expires,
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
    task = SimpleNamespace(
        id="t_1234abcd",
        current_run_id=73,
        claim_lock="claim-73",
        claim_expires=int(time.time()) + 300,
    )
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
    task = SimpleNamespace(
        id="t_1234abcd",
        current_run_id=None,
        claim_lock=None,
        claim_expires=None,
    )
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
    accepted = [
        {
            "action": "planned_gateway_restart",
            "task_id": "t_1234abcd",
            "nonce": "restart-request-0001",
            "target_service": "hermes-gateway-gm2.service",
        }
    ]

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


def _dashboard_request_body(task_id: str, **overrides) -> str:
    request = {
        "version": 1,
        "action": "planned_dashboard_generation_transition",
        "nonce": "dashboard-request-0001",
        "task_id": task_id,
        "target_service": "hermes-dashboard-mct.service",
        "expected_pid": 4242,
        "expected_starttime": 98765,
        "expected_invocation_id": "dashboard-invocation-123",
        "expected_cgroup": "/system.slice/hermes-dashboard-mct.service",
        "require_board_idle": True,
        "expires_at": int(time.time()) + 120,
    }
    request.update(overrides)
    return json.dumps({"native_lifecycle_request": request})


def test_native_lifecycle_parser_accepts_only_exact_dashboard_action_target_with_expiry():
    from hermes_cli import kanban_db as kb

    task = SimpleNamespace(id="t_1234abcd", body=_dashboard_request_body("t_1234abcd"))
    request, error = kb.parse_native_lifecycle_request(task)

    assert error is None
    assert request is not None
    assert request.action == "planned_dashboard_generation_transition"
    assert request.target_service == "hermes-dashboard-mct.service"
    assert request.expires_at is not None
    assert request.expires_at > int(time.time())


def test_native_lifecycle_parser_rejects_dashboard_request_without_expiry():
    from hermes_cli import kanban_db as kb

    raw = json.loads(_dashboard_request_body("t_1234abcd"))
    raw["native_lifecycle_request"].pop("expires_at")
    request, error = kb.parse_native_lifecycle_request(
        SimpleNamespace(id="t_1234abcd", body=json.dumps(raw))
    )

    assert request is None
    assert error is not None and "expires_at" in error


def test_native_lifecycle_parser_preserves_legacy_gateway_request_without_expiry():
    from hermes_cli import kanban_db as kb

    request_dict = _request().__dict__.copy()
    request_dict.pop("expires_at")
    body = json.dumps({"native_lifecycle_request": request_dict})
    request, error = kb.parse_native_lifecycle_request(
        SimpleNamespace(id="t_1234abcd", body=body)
    )

    assert error is None
    assert request is not None
    assert request.action == "planned_gateway_restart"
    assert request.expires_at is None


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"target_service": "hermes-gateway-gm2.service"}, "target_service"),
        ({"action": "planned_gateway_restart"}, "gateway unit"),
        ({"action": "restart_any_service"}, "action"),
    ],
)
def test_dashboard_parser_rejects_wrong_action_target_pairs(overrides, reason):
    from hermes_cli import kanban_db as kb

    task = SimpleNamespace(
        id="t_1234abcd",
        body=_dashboard_request_body("t_1234abcd", **overrides),
    )
    request, error = kb.parse_native_lifecycle_request(task)

    assert request is None
    assert error is not None and reason in error


def _dashboard_identity(**overrides):
    identity = {
        "actual_pid": 4242,
        "actual_starttime": 98765,
        "actual_invocation_id": "dashboard-invocation-123",
        "actual_service": "hermes-dashboard-mct.service",
        "actual_cgroup": "/system.slice/hermes-dashboard-mct.service",
        "active_state": "active",
        "sub_state": "running",
        "need_daemon_reload": "no",
        "fragment_path": "/etc/systemd/system/hermes-dashboard-mct.service",
        "dropin_paths": ["dropin-a", "dropin-b", "dropin-c"],
        "loaded_hashes": {"fragment": "sha256-a"},
        "loaded_exec_argv": "fixed-dashboard-exec",
    }
    identity.update(overrides)
    return identity


def test_dashboard_identity_binds_loaded_unit_dropins_and_exec(tmp_path, monkeypatch):
    fragment = tmp_path / "hermes-dashboard-mct.service"
    dropin = tmp_path / "exact.conf"
    fragment.write_text("[Service]\nExecStart=/fixed\n", encoding="utf-8")
    dropin.write_text("[Service]\nEnvironment=SAFE=1\n", encoding="utf-8")
    fragment_hash = hashlib.sha256(fragment.read_bytes()).hexdigest()
    dropin_hash = hashlib.sha256(dropin.read_bytes()).hexdigest()
    monkeypatch.setattr(kanban_watchers, "_DASHBOARD_FRAGMENT", fragment)
    monkeypatch.setattr(kanban_watchers, "_DASHBOARD_FRAGMENT_SHA256", fragment_hash)
    monkeypatch.setattr(kanban_watchers, "_DASHBOARD_DROPINS", {dropin: dropin_hash})
    monkeypatch.setattr(kanban_watchers, "_DASHBOARD_EXEC_ARGV", "/fixed --literal")
    monkeypatch.setattr(kanban_watchers, "_read_proc_starttime", lambda _pid: 98765)
    stdout = "\n".join(
        [
            "MainPID=4242",
            "InvocationID=dashboard-invocation-123",
            "ControlGroup=/system.slice/hermes-dashboard-mct.service",
            "ActiveState=active",
            "SubState=running",
            "NeedDaemonReload=no",
            f"FragmentPath={fragment}",
            f"DropInPaths={dropin}",
            "ExecStart={ path=/fixed ; argv[]=/fixed --literal ; ignore_errors=no ; }",
        ]
    )
    run_fn = lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, stdout=stdout
    )

    identity = kanban_watchers._read_dashboard_service_identity(run_fn=run_fn)
    assert identity["loaded_hashes"] == {
        str(fragment): fragment_hash,
        str(dropin): dropin_hash,
    }
    dropin.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        kanban_watchers._read_dashboard_service_identity(run_fn=run_fn)


def test_dashboard_identity_rejects_generation_change_during_snapshot(monkeypatch):
    monkeypatch.setattr(kanban_watchers, "_read_proc_starttime", lambda _pid: 98765)
    before = "\n".join(
        [
            "MainPID=4242",
            "InvocationID=dashboard-invocation-123",
            "ControlGroup=/system.slice/hermes-dashboard-mct.service",
        ]
    )
    after = before.replace("MainPID=4242", "MainPID=4343")
    outputs = iter((before, after))

    def run_fn(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=next(outputs))

    with pytest.raises(RuntimeError, match="changed during fresh read"):
        kanban_watchers._read_dashboard_service_identity(run_fn=run_fn)


def test_dashboard_request_requires_fresh_claim_identity_and_rollback_readiness():
    from hermes_cli import kanban_db as kb

    now = 1000
    request, error = kb.parse_native_lifecycle_request(
        SimpleNamespace(
            id="t_1234abcd",
            body=_dashboard_request_body("t_1234abcd", expires_at=1100),
        )
    )
    assert error is None and request is not None
    task = SimpleNamespace(
        id="t_1234abcd",
        current_run_id=73,
        claim_lock="claim-73",
        claim_expires=1200,
    )
    decision = _evaluate_native_lifecycle_restart_request(
        task,
        request,
        now=now,
        dashboard_identity_fn=_dashboard_identity,
        rollback_ready_fn=lambda: {
            "rollback_ref": "refs/aion/rollback/t_2066e6dc-pr44-run3246",
            "source_clean": True,
        },
    )

    assert decision.accepted is True
    assert decision.receipt["rollback_ref"].endswith("pr44-run3246")
    assert decision.receipt["actual_pid"] == 4242


def test_dashboard_request_rejects_stale_expiry_before_runtime_reads():
    from hermes_cli import kanban_db as kb

    request, error = kb.parse_native_lifecycle_request(
        SimpleNamespace(
            id="t_1234abcd",
            body=_dashboard_request_body("t_1234abcd", expires_at=1000),
        )
    )
    assert error is None and request is not None
    calls = []
    decision = _evaluate_native_lifecycle_restart_request(
        SimpleNamespace(
            id="t_1234abcd",
            current_run_id=73,
            claim_lock="claim-73",
            claim_expires=1200,
        ),
        request,
        now=1000,
        dashboard_identity_fn=lambda: calls.append("identity"),
        rollback_ready_fn=lambda: calls.append("rollback"),
    )

    assert decision.accepted is False
    assert "expired" in decision.reason
    assert calls == []


@pytest.mark.parametrize(
    "identity, rollback_fn, reason",
    [
        (_dashboard_identity(actual_pid=4243), lambda: {"source_clean": True}, "expected_pid mismatch"),
        (_dashboard_identity(), lambda: (_ for _ in ()).throw(RuntimeError("missing ref")), "rollback readiness"),
    ],
)
def test_dashboard_request_rejects_identity_or_rollback_drift(identity, rollback_fn, reason):
    from hermes_cli import kanban_db as kb

    request, error = kb.parse_native_lifecycle_request(
        SimpleNamespace(
            id="t_1234abcd",
            body=_dashboard_request_body("t_1234abcd", expires_at=1100),
        )
    )
    assert error is None and request is not None
    decision = _evaluate_native_lifecycle_restart_request(
        SimpleNamespace(
            id="t_1234abcd",
            current_run_id=73,
            claim_lock="claim-73",
            claim_expires=1200,
        ),
        request,
        now=1000,
        dashboard_identity_fn=lambda: identity,
        rollback_ready_fn=rollback_fn,
    )

    assert decision.accepted is False
    assert reason in decision.reason


def test_dashboard_action_cannot_reach_sigusr1_delivery():
    calls = []
    accepted = [
        {
            "action": "planned_dashboard_generation_transition",
            "task_id": "t_1234abcd",
            "nonce": "dashboard-request-0001",
            "target_service": "hermes-dashboard-mct.service",
        }
    ]

    assert not _dispatch_native_lifecycle_restart_signal(
        accepted,
        kill_fn=lambda *args: calls.append(args),
        current_pid=4242,
    )
    assert calls == []


@pytest.mark.parametrize(
    "returncode, exception, expected_status",
    [
        (0, None, "DELIVERED"),
        (9, None, "REJECTED"),
        (None, subprocess.TimeoutExpired("systemctl", 30), "AMBIGUOUS"),
    ],
)
def test_dashboard_delivery_has_literal_argv_and_terminal_result_classification(
    returncode, exception, expected_status
):
    dispatch = getattr(kanban_watchers, "_dispatch_native_lifecycle_action", None)
    assert callable(dispatch), "dashboard lifecycle delivery branch is missing"
    calls = []
    old_identity = _dashboard_identity()
    new_identity = _dashboard_identity(
        actual_pid=4343,
        actual_starttime=99876,
        actual_invocation_id="dashboard-invocation-456",
    )
    identities = iter((old_identity, new_identity))

    def run_fn(argv, **kwargs):
        calls.append((argv, kwargs))
        if exception is not None:
            raise exception
        return subprocess.CompletedProcess(argv, returncode)

    result = dispatch(
        [
            {
                "action": "planned_dashboard_generation_transition",
                "task_id": "t_1234abcd",
                "nonce": "dashboard-request-0001",
                "target_service": "hermes-dashboard-mct.service",
                "receipt": {
                    name: old_identity[name]
                    for name in (
                        "actual_pid",
                        "actual_starttime",
                        "actual_invocation_id",
                        "actual_service",
                        "actual_cgroup",
                        "fragment_path",
                        "dropin_paths",
                        "loaded_hashes",
                        "loaded_exec_argv",
                    )
                },
            }
        ],
        run_fn=run_fn,
        dashboard_identity_fn=lambda: next(identities),
        rollback_ready_fn=lambda: {"rollback_ref": "fixed"},
    )

    assert result.status == expected_status
    assert calls == [
        (
            ["/usr/bin/systemctl", "restart", "hermes-dashboard-mct.service"],
            {"check": False, "timeout": 30},
        )
    ]


def test_dashboard_delivery_rechecks_identity_before_literal_invocation():
    calls = []
    result = kanban_watchers._dispatch_native_lifecycle_action(
        [
            {
                "action": "planned_dashboard_generation_transition",
                "target_service": "hermes-dashboard-mct.service",
                "receipt": _dashboard_identity(),
            }
        ],
        run_fn=lambda *args, **kwargs: calls.append((args, kwargs)),
        dashboard_identity_fn=lambda: _dashboard_identity(actual_pid=4243),
        rollback_ready_fn=lambda: {"rollback_ref": "fixed"},
    )

    assert result.status == "REJECTED"
    assert "changed before delivery" in result.reason
    assert calls == []


def test_dashboard_delivery_rechecks_claim_immediately_before_invocation():
    calls = []
    result = kanban_watchers._dispatch_native_lifecycle_action(
        [
            {
                "action": "planned_dashboard_generation_transition",
                "target_service": "hermes-dashboard-mct.service",
                "receipt": _dashboard_identity(),
            }
        ],
        run_fn=lambda *args, **kwargs: calls.append((args, kwargs)),
        dashboard_identity_fn=_dashboard_identity,
        rollback_ready_fn=lambda: {"rollback_ref": "fixed"},
        pre_delivery_guard=lambda: False,
    )

    assert result.status == "REJECTED"
    assert "claim or expiry changed" in result.reason
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
                "expires_at": int(time.time()) + 120,
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
            "expires_at": int(time.time()) + 120,
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
