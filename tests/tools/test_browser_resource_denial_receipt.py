from __future__ import annotations

import errno
import json
import logging
import os
from collections.abc import Callable

import pytest

from tools import browser_tool


def _receipt_from_logs(caplog: pytest.LogCaptureFixture) -> dict:
    record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("RESOURCE_ALLOCATION_DENIAL ")
    )
    return json.loads(record.getMessage().split(" ", 1)[1])


def _run_browser_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    popen: Callable[..., object],
    *,
    snapshots: list[dict],
) -> dict:
    snapshot_iter = iter(snapshots)
    monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "/bin/true")
    monkeypatch.setattr(
        browser_tool, "_requires_real_termux_browser_install", lambda _command: False
    )
    monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_chromium_installed", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task_id: {
        "session_name": "h_denial_fixture",
        "cdp_url": None,
    })
    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(browser_tool, "_build_browser_env", lambda: {"PATH": "/bin"})
    monkeypatch.setattr(browser_tool, "_merge_browser_path", lambda value: value)
    monkeypatch.setattr(browser_tool, "_needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", popen)
    monkeypatch.setattr(
        browser_tool,
        "capture_cgroup_pids_snapshot",
        lambda: next(snapshot_iter),
        raising=False,
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_browser_denial")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "3172")

    return browser_tool._run_browser_command(
        "session-browser-denial",
        "open",
        ["https://example.com"],
    )


def test_browser_child_pids_delta_emits_attributable_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
):
    class FailedBrowserCommand:
        returncode = 1

        def __init__(self, *_args, stdout: int, **_kwargs):
            os.write(
                stdout,
                b'{"success":false,"error":"CDP response channel closed"}',
            )

        def wait(self, timeout: int):
            return self.returncode

    before = {
        "path": "/system.slice/hermes-gateway-gm2.service",
        "pids_current": 25,
        "pids_peak": 89,
        "pids_max": 120,
        "pids_events_max": 0,
    }
    after = {
        "path": "/system.slice/hermes-gateway-gm2.service",
        "pids_current": 20,
        "pids_peak": 120,
        "pids_max": 120,
        "pids_events_max": 26,
    }

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        result = _run_browser_failure(
            monkeypatch,
            tmp_path,
            FailedBrowserCommand,
            snapshots=[before, after],
        )

    assert result == {"success": False, "error": "CDP response channel closed"}
    receipt = _receipt_from_logs(caplog)
    assert receipt["failure_kind"] == "cgroup_pids_max_delta"
    assert receipt["component"] == "browser_tool"
    assert receipt["caller"] == "agent_browser_command"
    assert receipt["identity"] == {
        "task_id": "t_browser_denial",
        "run_id": 3172,
        "session_id": "session-browser-denial",
        "tool_name": "open",
        "process_session_id": "h_denial_fixture",
    }
    assert receipt["cgroup_before"] == before
    assert receipt["cgroup"] == after
    assert receipt["pids_events_max_delta"] == 26


def test_browser_direct_spawn_eagain_emits_existing_exception_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
):
    def denied_spawn(*_args, **_kwargs):
        raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

    snapshot = {
        "path": "/system.slice/hermes-gateway-gm2.service",
        "pids_current": 120,
        "pids_peak": 120,
        "pids_max": 120,
        "pids_events_max": 27,
    }

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        result = _run_browser_failure(
            monkeypatch,
            tmp_path,
            denied_spawn,
            snapshots=[snapshot],
        )

    assert result["success"] is False
    receipt = _receipt_from_logs(caplog)
    assert receipt["failure_kind"] == "process_spawn_eagain"
    assert receipt["component"] == "browser_tool"
    assert receipt["caller"] == "agent_browser_command_spawn"
    assert receipt["identity"]["task_id"] == "t_browser_denial"
    assert receipt["identity"]["run_id"] == 3172
    assert receipt["identity"]["session_id"] == "session-browser-denial"


def test_browser_protocol_failure_without_pids_delta_emits_no_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
):
    class FailedBrowserCommand:
        returncode = 1

        def __init__(self, *_args, stdout: int, **_kwargs):
            os.write(stdout, b'{"success":false,"error":"ordinary CDP failure"}')

        def wait(self, timeout: int):
            return self.returncode

    snapshot = {
        "path": "/system.slice/hermes-gateway-gm2.service",
        "pids_current": 20,
        "pids_peak": 120,
        "pids_max": 120,
        "pids_events_max": 26,
    }

    with caplog.at_level(logging.ERROR, logger="agent.resource_denial_receipt"):
        result = _run_browser_failure(
            monkeypatch,
            tmp_path,
            FailedBrowserCommand,
            snapshots=[snapshot, snapshot],
        )

    assert result == {"success": False, "error": "ordinary CDP failure"}
    assert not any(
        record.getMessage().startswith("RESOURCE_ALLOCATION_DENIAL ")
        for record in caplog.records
    )
