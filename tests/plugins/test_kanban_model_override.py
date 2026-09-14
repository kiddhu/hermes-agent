"""Per-task model/provider override — DB layer, worker spawn, dashboard API.

Covers the model-dropdown feature: kanban_db.set_model_override(),
create_task(model_override=..., provider_override=...), the dispatcher
passing ``-m <model> --provider <name>`` to the worker, and the dashboard
PATCH/bulk/model-options surfaces.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban as kanban_cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    yield c
    c.close()


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_model_override_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# DB layer — set_model_override
# ---------------------------------------------------------------------------


def test_set_and_clear_model_override(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    assert kb.set_model_override(conn, tid, "gpt-5.6-sol", provider="openai")
    t = kb.get_task(conn, tid)
    assert t.model_override == "gpt-5.6-sol"
    assert t.provider_override == "openai"

    # Clearing the model clears the provider too.
    assert kb.set_model_override(conn, tid, None)
    t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.provider_override is None


def test_set_model_override_events(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    kb.set_model_override(conn, tid, "sonnet-x", provider="anthropic")
    events = kb.list_events(conn, tid)
    kinds = [e.kind for e in events]
    assert "model_override_set" in kinds
    ev = next(e for e in events if e.kind == "model_override_set")
    assert ev.payload["model"] == "sonnet-x"
    assert ev.payload["provider"] == "anthropic"


def test_provider_without_model_rejected(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    with pytest.raises(ValueError):
        kb.set_model_override(conn, tid, None, provider="openrouter")
    with pytest.raises(ValueError):
        kb.create_task(
            conn, title="t2", assignee="worker", provider_override="openrouter",
        )


def test_set_model_override_unknown_task(conn):
    assert kb.set_model_override(conn, "t_nope", "some-model") is False


def test_set_model_override_archived_task_rejected(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    assert kb.archive_task(conn, tid)
    with pytest.raises(RuntimeError):
        kb.set_model_override(conn, tid, "some-model")


def test_set_model_override_allowed_on_running(conn):
    """The rate-limit recovery flow: override a running task so the NEXT
    dispatch (after reclaim/retry) picks up the new model."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    assert kb.set_model_override(conn, tid, "fallback-model", provider="nous")
    t = kb.get_task(conn, tid)
    assert t.status == "running"
    assert t.model_override == "fallback-model"
    assert t.provider_override == "nous"


def test_compare_and_set_model_override_exact_transition(conn, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )

    outcome = kb.compare_and_set_model_override(
        conn, tid,
        expected_task_id=tid,
        expected_model="deepseek-v4-pro",
        expected_provider="openrouter",
        model="gpt-5.6-sol",
        provider="openai-codex",
    )

    assert outcome == "updated"
    task = kb.get_task(conn, tid)
    assert task.model_override == "gpt-5.6-sol"
    assert task.provider_override == "openai-codex"
    event = kb.list_events(conn, tid)[-1]
    assert event.kind == "model_override_cas"
    assert event.payload == {
        "actor": "worker",
        "old_model": "deepseek-v4-pro",
        "old_provider": "openrouter",
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
    }


def test_compare_and_set_model_override_idempotent_replay_is_noop(conn, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    args = dict(
        expected_task_id=tid,
        expected_model="deepseek-v4-pro",
        expected_provider="openrouter",
        model="gpt-5.6-sol",
        provider="openai-codex",
    )
    assert kb.compare_and_set_model_override(conn, tid, **args) == "updated"
    event_count = len(kb.list_events(conn, tid))

    assert kb.compare_and_set_model_override(conn, tid, **args) == "already_applied"
    assert len(kb.list_events(conn, tid)) == event_count


def test_compare_and_set_model_override_already_desired_without_receipt_rejects(
    conn, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="gpt-5.6-sol", provider_override="openai-codex",
    )
    before = kb.get_task(conn, tid)
    before_events = len(kb.list_events(conn, tid))

    with pytest.raises(RuntimeError, match="CAS mismatch"):
        kb.compare_and_set_model_override(
            conn, tid,
            expected_task_id=tid,
            expected_model="definitely-stale-model",
            expected_provider="definitely-stale-provider",
            model="gpt-5.6-sol",
            provider="openai-codex",
        )

    assert kb.get_task(conn, tid) == before
    assert len(kb.list_events(conn, tid)) == before_events


def test_compare_and_set_model_override_replay_requires_exact_receipt(conn, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    assert kb.compare_and_set_model_override(
        conn, tid,
        expected_task_id=tid,
        expected_model="deepseek-v4-pro",
        expected_provider="openrouter",
        model="gpt-5.6-sol",
        provider="openai-codex",
    ) == "updated"
    before = kb.get_task(conn, tid)
    before_events = len(kb.list_events(conn, tid))

    with pytest.raises(RuntimeError, match="matching prior CAS receipt"):
        kb.compare_and_set_model_override(
            conn, tid,
            expected_task_id=tid,
            expected_model="some-other-old-model",
            expected_provider="openrouter",
            model="gpt-5.6-sol",
            provider="openai-codex",
        )

    assert kb.get_task(conn, tid) == before
    assert len(kb.list_events(conn, tid)) == before_events


@pytest.mark.parametrize(
    ("use_missing_task", "expected_model", "expected_provider", "profile"),
    [
        (False, "stale-model", "openrouter", "worker"),
        (False, "deepseek-v4-pro", "wrong-provider", "worker"),
        (False, "deepseek-v4-pro", "openrouter", "intruder"),
        (True, "deepseek-v4-pro", "openrouter", "worker"),
    ],
)
def test_compare_and_set_model_override_hostile_inputs_do_not_mutate(
    conn, monkeypatch, use_missing_task, expected_model, expected_provider, profile,
):
    monkeypatch.setenv("HERMES_PROFILE", profile)
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    target_id = "t_missing" if use_missing_task else tid
    before = kb.get_task(conn, tid)
    before_events = len(kb.list_events(conn, tid))

    with pytest.raises((LookupError, PermissionError, RuntimeError)):
        kb.compare_and_set_model_override(
            conn, target_id,
            expected_task_id=target_id,
            expected_model=expected_model,
            expected_provider=expected_provider,
            model="gpt-5.6-sol",
            provider="openai-codex",
        )

    assert kb.get_task(conn, tid) == before
    assert len(kb.list_events(conn, tid)) == before_events


def test_compare_and_set_model_override_rejects_active_run(conn, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    assert kb.claim_task(conn, tid, claimer="dispatcher") is not None
    before_events = len(kb.list_events(conn, tid))

    with pytest.raises(RuntimeError, match="active run"):
        kb.compare_and_set_model_override(
            conn, tid,
            expected_task_id=tid,
            expected_model="deepseek-v4-pro",
            expected_provider="openrouter",
            model="gpt-5.6-sol",
            provider="openai-codex",
        )

    task = kb.get_task(conn, tid)
    assert task.model_override == "deepseek-v4-pro"
    assert task.provider_override == "openrouter"
    assert len(kb.list_events(conn, tid)) == before_events


def test_compare_and_set_model_override_requires_authenticated_profile(conn, monkeypatch):
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    before = kb.get_task(conn, tid)
    before_events = len(kb.list_events(conn, tid))

    with pytest.raises(PermissionError, match="authenticated HERMES_PROFILE"):
        kb.compare_and_set_model_override(
            conn, tid,
            expected_task_id=tid,
            expected_model="deepseek-v4-pro",
            expected_provider="openrouter",
            model="gpt-5.6-sol",
            provider="openai-codex",
        )

    assert kb.get_task(conn, tid) == before
    assert len(kb.list_events(conn, tid)) == before_events


def test_compare_and_set_model_override_same_assignee_wrong_task_fails_identity_gate(
    conn, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    target = kb.create_task(
        conn, title="target", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    wrong = kb.create_task(
        conn, title="wrong", assignee="worker",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    target_before = kb.get_task(conn, target)
    wrong_before = kb.get_task(conn, wrong)

    with pytest.raises(ValueError, match="expected task"):
        kb.compare_and_set_model_override(
            conn, wrong,
            expected_task_id=target,
            expected_model="deepseek-v4-pro",
            expected_provider="openrouter",
            model="gpt-5.6-sol",
            provider="openai-codex",
        )

    assert kb.get_task(conn, target) == target_before
    assert kb.get_task(conn, wrong) == wrong_before


def test_compare_and_set_model_override_concurrent_replay_writes_once(
    kanban_home, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    with kb.connect_closing() as setup:
        tid = kb.create_task(
            setup, title="t", assignee="worker",
            model_override="deepseek-v4-pro", provider_override="openrouter",
        )
    barrier = threading.Barrier(2)
    outcomes = []

    def migrate():
        with kb.connect_closing() as worker_conn:
            barrier.wait()
            outcomes.append(kb.compare_and_set_model_override(
                worker_conn, tid,
                expected_task_id=tid,
                expected_model="deepseek-v4-pro",
                expected_provider="openrouter",
                model="gpt-5.6-sol",
                provider="openai-codex",
            ))

    threads = [threading.Thread(target=migrate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["already_applied", "updated"]
    with kb.connect_closing() as check:
        events = [e for e in kb.list_events(check, tid) if e.kind == "model_override_cas"]
        assert len(events) == 1


def test_compare_and_set_model_override_preserves_task_and_edge_fields(conn, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    parent = kb.create_task(conn, title="parent", assignee="worker")
    tid = kb.create_task(
        conn, title="target", body="immutable acceptance", assignee="worker",
        workspace_kind="dir", workspace_path="/tmp/exact-workspace",
        model_override="deepseek-v4-pro", provider_override="openrouter",
    )
    child = kb.create_task(conn, title="child", assignee="worker")
    kb.link_tasks(conn, parent, tid)
    kb.link_tasks(conn, tid, child)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    before_links = [tuple(row) for row in conn.execute(
        "SELECT parent_id, child_id FROM task_links "
        "WHERE parent_id=? OR child_id=? ORDER BY parent_id, child_id",
        (tid, tid),
    )]

    assert kb.compare_and_set_model_override(
        conn, tid,
        expected_task_id=tid,
        expected_model="deepseek-v4-pro", expected_provider="openrouter",
        model="gpt-5.6-sol", provider="openai-codex",
    ) == "updated"

    after = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    for key in ("model_override", "provider_override"):
        before.pop(key)
        after.pop(key)
    assert after == before
    assert [tuple(row) for row in conn.execute(
        "SELECT parent_id, child_id FROM task_links "
        "WHERE parent_id=? OR child_id=? ORDER BY parent_id, child_id",
        (tid, tid),
    )] == before_links


def test_set_model_cli_parses_exact_binding_cas_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kanban_cli.build_parser(subparsers)

    args = parser.parse_args([
        "kanban", "set-model", "t_exact", "gpt-5.6-sol",
        "--provider", "openai-codex",
        "--expect-task-id", "t_exact",
        "--expect-model", "deepseek-v4-pro",
        "--expect-provider", "openrouter",
    ])

    assert args.expect_task_id == "t_exact"
    assert args.expect_model == "deepseek-v4-pro"
    assert args.expect_provider == "openrouter"


def test_create_task_with_model_and_provider(conn):
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="qwen-max", provider_override="openrouter",
    )
    t = kb.get_task(conn, tid)
    assert t.model_override == "qwen-max"
    assert t.provider_override == "openrouter"
    # Creation event carries the override for auditability.
    ev = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
    assert ev.payload["model_override"] == "qwen-max"
    assert ev.payload["provider_override"] == "openrouter"


def test_migration_adds_provider_override_column(conn):
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "model_override" in cols
    assert "provider_override" in cols


# ---------------------------------------------------------------------------
# Worker spawn — argv carries -m and --provider
# ---------------------------------------------------------------------------


def _spawn_and_capture(monkeypatch, tmp_path, task):
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(task, str(workspace))
    return captured["cmd"]


def test_spawn_passes_model_and_provider(monkeypatch, tmp_path, conn):
    tid = kb.create_task(
        conn, title="t", assignee="elias",
        model_override="glm-5", provider_override="openrouter",
    )
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    i = cmd.index("-m")
    assert cmd[i + 1] == "glm-5"
    j = cmd.index("--provider")
    assert j == i + 2
    assert cmd[j + 1] == "openrouter"


def test_spawn_model_only_omits_provider_flag(monkeypatch, tmp_path, conn):
    tid = kb.create_task(
        conn, title="t", assignee="elias", model_override="glm-5",
    )
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    assert "-m" in cmd
    assert "--provider" not in cmd


def test_spawn_no_override_omits_both_flags(monkeypatch, tmp_path, conn):
    tid = kb.create_task(conn, title="t", assignee="elias")
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    assert "-m" not in cmd
    assert "--provider" not in cmd


# ---------------------------------------------------------------------------
# Dashboard API — PATCH / bulk / create / model-options
# ---------------------------------------------------------------------------


def _create(client, **kwargs):
    body = {"title": "task", "assignee": "worker"}
    body.update(kwargs)
    r = client.post("/api/plugins/kanban/tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()["task"]


def test_patch_sets_model_override(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"model_override": "gpt-5.6-sol", "provider_override": "openai"},
    )
    assert r.status_code == 200, r.text
    updated = r.json()["task"]
    assert updated["model_override"] == "gpt-5.6-sol"
    assert updated["provider_override"] == "openai"


def test_patch_clears_model_override(client):
    task = _create(
        client, model_override="gpt-5.6-sol", provider_override="openai",
    )
    assert task["model_override"] == "gpt-5.6-sol"
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"clear_model_override": True},
    )
    assert r.status_code == 200, r.text
    updated = r.json()["task"]
    assert updated["model_override"] is None
    assert updated["provider_override"] is None


def test_patch_provider_without_model_is_400(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"model_override": "", "provider_override": "openai"},
    )
    assert r.status_code == 400


def test_create_task_with_override_via_api(client):
    task = _create(
        client, model_override="qwen-max", provider_override="openrouter",
    )
    assert task["model_override"] == "qwen-max"
    assert task["provider_override"] == "openrouter"


def test_bulk_model_override(client):
    t1 = _create(client)
    t2 = _create(client)
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={
            "ids": [t1["id"], t2["id"]],
            "model_override": "fallback-model",
            "provider_override": "nous",
        },
    )
    assert r.status_code == 200, r.text
    assert all(entry["ok"] for entry in r.json()["results"])
    for tid in (t1["id"], t2["id"]):
        got = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert got["model_override"] == "fallback-model"
        assert got["provider_override"] == "nous"


def test_model_options_endpoint_shape(client, monkeypatch):
    """The endpoint returns {providers: [{slug,label,models}]} and degrades
    to an empty catalog when the inventory substrate raises."""
    r = client.get("/api/plugins/kanban/model-options")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)
    for row in data["providers"]:
        assert "slug" in row and "label" in row and "models" in row
        assert isinstance(row["models"], list)
        assert len(row["models"]) >= 1  # empty-model rows are filtered out
