"""Safe event-time receipts for process/thread allocation denials.

This module is intentionally passive: existing failure catch points call it only
after an allocation has already failed.  It does not poll, reserve capacity, or
change admission behavior.  Receipts contain bounded kernel/process identity and
caller-supplied scalar IDs only; prompts, commands, provider payloads, and
credentials are never accepted into the emitted schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
import errno as errno_module
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RECEIPT_PREFIX = "RESOURCE_ALLOCATION_DENIAL "
_MAX_TEXT = 128
_IDENTITY_KEYS = frozenset({
    "task_id",
    "run_id",
    "session_id",
    "job_id",
    "execution_id",
    "tool_name",
    "process_session_id",
})
_ENV_IDENTITY = {
    "task_id": "HERMES_KANBAN_TASK",
    "run_id": "HERMES_KANBAN_RUN_ID",
    "session_id": "HERMES_SESSION_ID",
}


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _bounded_text(value: Any) -> str:
    return str(value)[:_MAX_TEXT]


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _process_starttime() -> Optional[int]:
    text = _read_text("/proc/self/stat")
    if not text:
        return None
    rparen = text.rfind(")")
    if rparen < 0:
        return None
    tail = text[rparen + 1 :].split()
    # /proc/<pid>/stat field 22; the tail starts at field 3.
    if len(tail) <= 19:
        return None
    return _parse_int(tail[19])


def _parse_events_max(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    for line in value.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "max":
            return _parse_int(parts[1])
    return None


def _cgroup_snapshot() -> dict[str, Any]:
    text = _read_text("/proc/self/cgroup")
    if not text:
        return {}
    cgroup_path = None
    for line in text.splitlines():
        if line.startswith("0::"):
            candidate = line[3:].strip()
            if candidate.startswith("/") and ".." not in candidate.split("/"):
                cgroup_path = candidate
            break
    if cgroup_path is None:
        return {}

    base = "/sys/fs/cgroup" + ("" if cgroup_path == "/" else cgroup_path)
    snapshot: dict[str, Any] = {"path": _bounded_text(cgroup_path)}
    values = {
        "pids_current": _parse_int(_read_text(f"{base}/pids.current")),
        "pids_peak": _parse_int(_read_text(f"{base}/pids.peak")),
        "pids_max": _parse_int(_read_text(f"{base}/pids.max")),
        "pids_events_max": _parse_events_max(_read_text(f"{base}/pids.events")),
    }
    snapshot.update({key: value for key, value in values.items() if value is not None})
    return snapshot


def _failure_kind(exc: BaseException) -> Optional[str]:
    message = str(exc).lower()
    if isinstance(exc, RuntimeError) and (
        "can't start new thread" in message or "cannot start new thread" in message
    ):
        return "thread_start_failed"
    if isinstance(exc, OSError) and exc.errno == errno_module.EAGAIN:
        return "process_spawn_eagain"
    # Some wrappers discard errno while retaining the canonical OS message.
    if "resource temporarily unavailable" in message:
        return "allocation_eagain"
    return None


def emit_resource_denial_receipt(
    exc: BaseException,
    *,
    component: str,
    caller: str,
    inherit_environment_identity: bool = False,
    **identity: Any,
) -> Optional[dict[str, Any]]:
    """Log and return a bounded receipt when *exc* is an allocation denial.

    Returns ``None`` for unrelated failures.  Receipt construction is
    best-effort and must never replace the original exception at the caller.
    Only allowlisted scalar identity fields are retained. Process-global worker
    identity is inherited only when the caller explicitly opts in; unrelated
    cron, gateway, and dispatcher domains must pass their own safe context.
    """
    try:
        kind = _failure_kind(exc)
        if kind is None:
            return None

        safe_identity: dict[str, Any] = {}
        for key in _IDENTITY_KEYS:
            value = identity.get(key)
            if value is None and inherit_environment_identity and key in _ENV_IDENTITY:
                value = os.environ.get(_ENV_IDENTITY[key])
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, int):
                safe_identity[key] = value
            elif isinstance(value, str):
                bounded = _bounded_text(value)
                if key == "run_id" and bounded.isdecimal():
                    safe_identity[key] = int(bounded)
                elif bounded:
                    safe_identity[key] = bounded

        receipt = {
            "event": "resource_allocation_denial",
            "event_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "failure_kind": kind,
            "errno": exc.errno if isinstance(exc, OSError) else None,
            "error_type": type(exc).__name__[:_MAX_TEXT],
            "component": _bounded_text(component),
            "caller": _bounded_text(caller),
            "identity": safe_identity,
            "process": {
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "starttime": _process_starttime(),
            },
            "cgroup": _cgroup_snapshot(),
        }
        logger.error(
            "%s%s",
            _RECEIPT_PREFIX,
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        )
        return receipt
    except Exception:
        logger.debug("resource denial receipt construction failed", exc_info=True)
        return None
