from __future__ import annotations

import errno
import json
import logging
from types import SimpleNamespace

import pytest

from agent.resource_denial_receipt import emit_resource_denial_receipt


def _receipt_from_logs(caplog) -> dict:
    record = next(
        record for record in caplog.records
        if record.getMessage().startswith("RESOURCE_ALLOCATION_DENIAL ")
    )
    return json.loads(record.getMessage().split(" ", 1)[1])


def test_eagain_receipt_binds_safe_event_time_identity_and_cgroup(monkeypatch, caplog):
    from agent import resource_denial_receipt as receipt

    files = {
        "/proc/self/stat": "321 (python worker) S " + " ".join(["0"] * 18) + " 98765 0",
        "/proc/self/cgroup": "0::/system.slice/hermes-gateway-gm2.service\n",
        "/sys/fs/cgroup/system.slice/hermes-gateway-gm2.service/pids.current": "51\n",
        "/sys/fs/cgroup/system.slice/hermes-gateway-gm2.service/pids.peak": "120\n",
        "/sys/fs/cgroup/system.slice/hermes-gateway-gm2.service/pids.max": "120\n",
        "/sys/fs/cgroup/system.slice/hermes-gateway-gm2.service/pids.events": "max 41\n",
    }
    monkeypatch.setattr(receipt, "_read_text", files.get)
    monkeypatch.setattr(receipt.os, "getpid", lambda: 321)
    monkeypatch.setattr(receipt.threading, "get_native_id", lambda: 654)

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        emitted = emit_resource_denial_receipt(
            BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable"),
            component="kanban_dispatcher",
            caller="worker_spawn",
            task_id="t_safe1234",
            run_id=3142,
            session_id="must-not-be-overridden",
            prompt="secret prompt must never be retained",
        )

    assert emitted is not None
    logged = _receipt_from_logs(caplog)
    assert logged == emitted
    assert logged["event"] == "resource_allocation_denial"
    assert logged["failure_kind"] == "process_spawn_eagain"
    assert logged["errno"] == errno.EAGAIN
    assert logged["component"] == "kanban_dispatcher"
    assert logged["caller"] == "worker_spawn"
    assert logged["identity"] == {
        "task_id": "t_safe1234",
        "run_id": 3142,
        "session_id": "must-not-be-overridden",
    }
    assert logged["process"] == {"pid": 321, "tid": 654, "starttime": 98765}
    assert logged["cgroup"] == {
        "path": "/system.slice/hermes-gateway-gm2.service",
        "pids_current": 51,
        "pids_peak": 120,
        "pids_max": 120,
        "pids_events_max": 41,
    }
    assert logged["event_utc"].endswith("Z")
    serialized = json.dumps(logged, sort_keys=True)
    assert "secret prompt" not in serialized
    assert "prompt" not in serialized


def test_thread_start_failure_is_classified_without_cross_domain_env_identity(
    monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_unrelated_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999")
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated-worker-session")

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        emitted = emit_resource_denial_receipt(
            RuntimeError("can't start new thread"),
            component="cron",
            caller="agent_thread_submit",
            job_id="job-1",
            execution_id="exec-1",
        )

    assert emitted is not None
    assert emitted["failure_kind"] == "thread_start_failed"
    assert emitted["errno"] is None
    assert emitted["identity"] == {"job_id": "job-1", "execution_id": "exec-1"}


def test_unrelated_exception_emits_no_receipt(caplog):
    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        emitted = emit_resource_denial_receipt(
            ValueError("ordinary failure"),
            component="cron",
            caller="dispatch",
            task_id="t_1",
        )

    assert emitted is None
    assert not any("RESOURCE_ALLOCATION_DENIAL" in record.getMessage() for record in caplog.records)


def test_identity_values_are_scalar_bounded_and_allowlisted(caplog):
    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        emitted = emit_resource_denial_receipt(
            OSError(errno.EAGAIN, "blocked"),
            component="x" * 300,
            caller="y" * 300,
            task_id="t" * 300,
            run_id=7,
            payload={"secret": "never"},
            unknown_key="never",
        )

    assert emitted is not None
    assert len(emitted["component"]) == 128
    assert len(emitted["caller"]) == 128
    assert len(emitted["identity"]["task_id"]) == 128
    assert emitted["identity"]["run_id"] == 7
    assert "payload" not in emitted["identity"]
    assert "unknown_key" not in emitted["identity"]


def test_worker_env_supplies_identity_when_low_level_caller_has_none(
    monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_env")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "99")
    monkeypatch.setenv("HERMES_SESSION_ID", "session-env")

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        emitted = emit_resource_denial_receipt(
            OSError(errno.EAGAIN, "blocked"),
            component="tool_environment",
            caller="foreground_process_spawn",
            inherit_environment_identity=True,
        )

    assert emitted is not None
    assert emitted["identity"] == {
        "task_id": "t_env",
        "run_id": 99,
        "session_id": "session-env",
    }


def test_dispatcher_spawn_eagain_emits_task_and_run_identity(
    monkeypatch, tmp_path, caplog
):
    from hermes_cli import kanban_db as kb

    task = SimpleNamespace(
        id="t_dispatch",
        assignee="worker",
        tenant=None,
        current_run_id=88,
        claim_lock="lock",
        goal_mode=False,
        goal_max_turns=None,
        branch_name=None,
        skills=None,
        model_override=None,
        provider_override=None,
        max_runtime_seconds=300,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: log_dir)
    monkeypatch.setattr(
        kb.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        ),
    )

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        with pytest.raises(BlockingIOError):
            kb._default_spawn(task, str(workspace))  # type: ignore[arg-type]

    receipt = _receipt_from_logs(caplog)
    assert receipt["component"] == "kanban_dispatcher"
    assert receipt["caller"] == "worker_process_spawn"
    assert receipt["identity"]["task_id"] == "t_dispatch"
    assert receipt["identity"]["run_id"] == 88


def test_cron_dispatch_thread_denial_emits_job_and_execution_identity(
    monkeypatch, caplog
):
    import cron.scheduler as scheduler

    class BrokenPool:
        def submit(self, _callable):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_args, **_kwargs: {"id": "exec-resource-denial"},
    )
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [{"id": "job-resource-denial"}])
    monkeypatch.setattr(scheduler, "advance_next_run", lambda _job_id: None)
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: BrokenPool())
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_unrelated_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999")
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated-worker-session")

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        assert scheduler.tick(verbose=False, sync=False) == 0

    receipt = _receipt_from_logs(caplog)
    assert receipt["component"] == "cron_scheduler"
    assert receipt["caller"] == "job_executor_submit"
    assert receipt["identity"] == {
        "job_id": "job-resource-denial",
        "execution_id": "exec-resource-denial",
    }


def test_tool_background_spawn_eagain_emits_task_and_session_identity(
    monkeypatch, tmp_path, caplog
):
    from tools import process_registry

    monkeypatch.setattr(
        process_registry.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        ),
    )
    registry = process_registry.ProcessRegistry()

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        with pytest.raises(BlockingIOError):
            registry.spawn_local(
                "true",
                cwd=str(tmp_path),
                task_id="t_tool",
                session_key="session-safe",
            )

    receipt = _receipt_from_logs(caplog)
    assert receipt["component"] == "tool_process_registry"
    assert receipt["caller"] == "background_process_spawn"
    assert receipt["identity"]["task_id"] == "t_tool"
    assert receipt["identity"]["session_id"] == "session-safe"
    assert receipt["identity"]["process_session_id"].startswith("proc_")


@pytest.mark.asyncio
async def test_gateway_executor_denial_does_not_inherit_process_global_identity(
    monkeypatch, caplog
):
    import gateway.run as gateway_run

    class BrokenLoop:
        def run_in_executor(self, *_args, **_kwargs):
            raise RuntimeError("can't start new thread")

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run.asyncio, "get_running_loop", lambda: BrokenLoop())
    monkeypatch.setattr(runner, "_get_executor", lambda: None)
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated-worker-session")

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            await runner._run_in_executor_with_context(lambda: None)

    receipt = _receipt_from_logs(caplog)
    assert receipt["component"] == "gateway"
    assert receipt["caller"] == "session_executor_submit"
    assert receipt["identity"] == {}


@pytest.mark.asyncio
async def test_gateway_executor_denial_uses_bound_session_context(monkeypatch, caplog):
    import gateway.run as gateway_run
    from gateway.session_context import clear_session_vars, set_session_vars

    class BrokenLoop:
        def run_in_executor(self, *_args, **_kwargs):
            raise RuntimeError("can't start new thread")

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run.asyncio, "get_running_loop", lambda: BrokenLoop())
    monkeypatch.setattr(runner, "_get_executor", lambda: None)
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated-worker-session")
    tokens = set_session_vars(session_id="gateway-session")

    try:
        with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
            with pytest.raises(RuntimeError, match="can't start new thread"):
                await runner._run_in_executor_with_context(lambda: None)
    finally:
        clear_session_vars(tokens)

    receipt = _receipt_from_logs(caplog)
    assert receipt["identity"] == {"session_id": "gateway-session"}


@pytest.mark.asyncio
async def test_gateway_does_not_mislabel_worker_function_failure_as_submit_denial(
    monkeypatch, caplog
):
    import gateway.run as gateway_run

    class SuccessfulSubmitLoop:
        def run_in_executor(self, *_args, **_kwargs):
            async def worker_result():
                raise RuntimeError("can't start new thread")

            return worker_result()

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(
        gateway_run.asyncio, "get_running_loop", lambda: SuccessfulSubmitLoop()
    )
    monkeypatch.setattr(runner, "_get_executor", lambda: None)

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            await runner._run_in_executor_with_context(lambda: None)

    assert not any(
        record.getMessage().startswith("RESOURCE_ALLOCATION_DENIAL ")
        for record in caplog.records
    )
