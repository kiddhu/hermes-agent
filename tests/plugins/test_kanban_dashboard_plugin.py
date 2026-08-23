"""Tests for the Kanban dashboard plugin backend (plugins/kanban/dashboard/plugin_api.py).

The plugin mounts as /api/plugins/kanban/ inside the dashboard's FastAPI app,
but here we attach its router to a bare FastAPI instance so we can test the
REST surface without spinning up the whole dashboard.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /board on an empty DB
# ---------------------------------------------------------------------------


def test_board_empty(client):
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    # All canonical columns present (triage + the rest), each empty.
    names = [c["name"] for c in data["columns"]]
    assert set(names) == kb.VALID_STATUSES - {"archived"}
    for expected in ("triage", "todo", "scheduled", "ready", "running", "blocked", "done"):
        assert expected in names, f"missing column {expected}: {names}"
    assert all(len(c["tasks"]) == 0 for c in data["columns"])
    assert data["tenants"] == []
    assert data["assignees"] == []
    assert data["latest_event_id"] == 0


# ---------------------------------------------------------------------------
# POST /tasks then GET /board sees it
# ---------------------------------------------------------------------------


def test_create_task_appears_on_board(client):
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "Research LLM caching",
            "assignee": "researcher",
            "priority": 3,
            "tenant": "acme",
        },
    )
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    assert task["title"] == "Research LLM caching"
    assert task["assignee"] == "researcher"
    assert task["status"] == "ready"  # no parents -> immediately ready
    assert task["priority"] == 3
    assert task["tenant"] == "acme"
    task_id = task["id"]

    # Board now lists it under 'ready'.
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    ready = next(c for c in data["columns"] if c["name"] == "ready")
    assert len(ready["tasks"]) == 1
    assert ready["tasks"][0]["id"] == task_id
    assert "acme" in data["tenants"]
    assert "researcher" in data["assignees"]


def test_board_list_recommends_persistent_workspace_for_configured_workdir(
    client, tmp_path
):
    """Board metadata should tell the UI which safe task default to use."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    kb.write_board_metadata("default", default_workdir=str(repo))

    plain_dir = tmp_path / "notes"
    plain_dir.mkdir()
    kb.create_board("notes", default_workdir=str(plain_dir))
    kb.create_board("disposable")

    response = client.get("/api/plugins/kanban/boards")

    assert response.status_code == 200
    boards = {board["slug"]: board for board in response.json()["boards"]}
    assert boards["default"]["default_workspace_kind"] == "worktree"
    assert boards["notes"]["default_workspace_kind"] == "dir"
    assert boards["disposable"]["default_workspace_kind"] == "scratch"


def test_create_board_persists_project_directory(client, tmp_path):
    """The dashboard board form should anchor future tasks to its project."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    response = client.post(
        "/api/plugins/kanban/boards",
        json={
            "slug": "project-board",
            "name": "Project Board",
            "default_workdir": str(project_dir),
        },
    )

    assert response.status_code == 200, response.text
    board = response.json()["board"]
    assert board["default_workdir"] == str(project_dir.resolve())
    assert board["default_workspace_kind"] == "dir"
    assert kb.read_board_metadata("project-board")["default_workdir"] == str(
        project_dir.resolve()
    )


@pytest.mark.parametrize("path", ["relative/project", "~/missing-project"])
def test_create_board_rejects_invalid_project_directory(client, path):
    """A board must not persist a path that cannot anchor worker output."""
    response = client.post(
        "/api/plugins/kanban/boards",
        json={"slug": "invalid-project", "default_workdir": path},
    )

    assert response.status_code == 400
    assert "project directory" in response.json()["detail"].lower()


def test_patch_board_sets_project_directory(client, tmp_path):
    """Board-level default_workdir must be editable after creation."""
    kb.create_board("late-config")
    project_dir = tmp_path / "late-project"
    project_dir.mkdir()

    response = client.patch(
        "/api/plugins/kanban/boards/late-config",
        json={"default_workdir": str(project_dir)},
    )

    assert response.status_code == 200, response.text
    board = response.json()["board"]
    assert board["default_workdir"] == str(project_dir.resolve())
    # The recommendation flips from scratch to a persistent kind so the
    # create-task dialog's workspace default follows the board setting.
    assert board["default_workspace_kind"] == "dir"
    assert kb.read_board_metadata("late-config")["default_workdir"] == str(
        project_dir.resolve()
    )


def test_patch_board_clears_project_directory(client, tmp_path):
    """Empty string clears default_workdir; omitting it leaves it unchanged."""
    project_dir = tmp_path / "was-configured"
    project_dir.mkdir()
    kb.create_board("clearable", default_workdir=str(project_dir))

    # Omitted key → unchanged.
    r = client.patch(
        "/api/plugins/kanban/boards/clearable",
        json={"name": "Renamed Only"},
    )
    assert r.status_code == 200
    assert r.json()["board"]["default_workdir"] == str(project_dir.resolve())

    # Empty string → cleared, recommendation falls back to scratch.
    r = client.patch(
        "/api/plugins/kanban/boards/clearable",
        json={"default_workdir": ""},
    )
    assert r.status_code == 200
    board = r.json()["board"]
    assert not board.get("default_workdir")
    assert board["default_workspace_kind"] == "scratch"


@pytest.mark.parametrize("path", ["relative/project", "~/missing-project"])
def test_patch_board_rejects_invalid_project_directory(client, path):
    """PATCH must validate default_workdir like board creation does."""
    kb.create_board("strict")

    response = client.patch(
        "/api/plugins/kanban/boards/strict",
        json={"default_workdir": path},
    )

    assert response.status_code == 400
    assert "project directory" in response.json()["detail"].lower()


def test_new_board_dialog_collects_project_directory():
    """Board creation should expose the setting that controls safe task defaults."""
    bundle = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "dist"
        / "index.js"
    ).read_text(encoding="utf-8")

    assert 'const [projectDirectory, setProjectDirectory] = useState("");' in bundle
    assert "Project directory" in bundle
    assert "Absolute path to the project folder" in bundle
    assert "default_workdir: projectDirectory.trim() || undefined" in bundle


def test_dashboard_workspace_picker_explains_persistence_contract():
    """Task creation must make scratch deletion visible without a hover."""
    bundle = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "dist"
        / "index.js"
    ).read_text(encoding="utf-8")

    assert "Temporary — deleted on completion" in bundle
    assert "Git worktree — preserved" in bundle
    assert "Directory — preserved" in bundle
    assert "defaultWorkspacePath: (props.boardMeta && props.boardMeta.default_workdir) || \"\"" in bundle
    assert (
        "This workspace and any files left in it are deleted when the task completes."
        in bundle
    )


def test_scheduled_tasks_have_their_own_column_not_todo(client):
    """Scheduled/time-delay tasks must not be silently bucketed into todo."""

    task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "wait for indexed data", "assignee": "ops"},
    ).json()["task"]

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'scheduled' WHERE id = ?",
                (task["id"],),
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    columns = {c["name"]: c["tasks"] for c in r.json()["columns"]}
    assert any(t["id"] == task["id"] for t in columns["scheduled"])
    assert not any(t["id"] == task["id"] for t in columns["todo"])


def test_tenant_filter(client):
    client.post("/api/plugins/kanban/tasks", json={"title": "A", "tenant": "t1"})
    client.post("/api/plugins/kanban/tasks", json={"title": "B", "tenant": "t2"})

    r = client.get("/api/plugins/kanban/board?tenant=t1")
    counts = {c["name"]: len(c["tasks"]) for c in r.json()["columns"]}
    total = sum(counts.values())
    assert total == 1

    r = client.get("/api/plugins/kanban/board?tenant=t2")
    total = sum(len(c["tasks"]) for c in r.json()["columns"])
    assert total == 1


def test_board_query_param_default_overrides_current_board_pointer(client):
    """Dashboard ``?board=default`` must win even if the CLI's current-board
    pointer targets a non-default board.

    Regression: selecting the Default board in the dashboard must not fall
    through to whichever board ``hermes kanban boards switch`` last pinned.
    """
    default_task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "default-only"},
    ).json()["task"]

    kb.create_board("other")
    other_conn = kb.connect(board="other")
    try:
        kb.create_task(other_conn, title="other-only")
    finally:
        other_conn.close()

    kb.set_current_board("other")

    current_board = client.get("/api/plugins/kanban/board").json()
    current_ids = {
        task["id"]
        for column in current_board["columns"]
        for task in column["tasks"]
    }
    assert default_task["id"] not in current_ids

    pinned_default = client.get("/api/plugins/kanban/board?board=default").json()
    pinned_ids = {
        task["id"]
        for column in pinned_default["columns"]
        for task in column["tasks"]
    }
    assert pinned_ids == {default_task["id"]}


def test_dashboard_select_filters_use_sdk_value_change_handler():
    """Tenant/assignee filters must work with the dashboard SDK Select API.

    The dashboard Select component is shadcn-like and calls
    ``onValueChange(value)`` instead of native ``onChange(event)``. A native-only
    handler leaves the tenant dropdown visually selectable but never updates the
    filtered board query.
    """

    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert "function selectChangeHandler(setter)" in js
    assert "onValueChange: function (v)" in js
    assert "onChange: function (e)" in js
    assert "selectChangeHandler(props.setTenantFilter)" in js
    assert "selectChangeHandler(props.setAssigneeFilter)" in js


def test_dashboard_client_side_filtering_includes_tenant_filter():
    """The rendered board must also filter by tenant.

    The API request includes ``?tenant=...``, but the dashboard also filters the
    locally cached board for search/assignee changes. Without checking
    ``tenantFilter`` here, switching tenants can leave stale cards visible until a
    full reload finishes.
    """

    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert "if (tenantFilter && t.tenant !== tenantFilter) return false;" in js
    assert "[boardData, tenantFilter, assigneeFilter, search]" in js


def test_dashboard_initial_board_uses_backend_current_when_unpinned():
    """Fresh browsers should open the backend current board, not default.

    Explicit dashboard selections are stored in localStorage and should still
    win, but an empty localStorage state must adopt the API's ``current`` board
    so multi-board installs do not look empty on first load.
    """

    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert 'useState(() => readSelectedBoard() || null)' in js
    assert "const storedBoard = readSelectedBoard();" in js
    assert "if (!storedBoard && !board && data && data.current)" in js
    assert "setBoard(data.current);" in js
    assert 'readSelectedBoard() || "default"' not in js


def test_dashboard_markdown_html_is_sanitized_before_render():
    """Markdown rendering must sanitize HTML before dangerouslySetInnerHTML."""

    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert "function sanitizeMarkdownHtml(html)" in js
    assert "MARKDOWN_ALLOWED_TAGS" in js
    assert "sanitizeMarkdownHtml(renderMarkdown(props.source || \"\"))" in js
    assert "dangerouslySetInnerHTML: { __html: renderMarkdown(props.source || \"\") }" not in js


# ---------------------------------------------------------------------------
# GET /tasks/:id returns body + comments + events + links
# ---------------------------------------------------------------------------


def test_task_detail_includes_links_and_events(client):
    parent = client.post(
        "/api/plugins/kanban/tasks", json={"title": "parent"},
    ).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "child", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"  # parent not done yet

    # Detail for the child shows the parent link.
    r = client.get(f"/api/plugins/kanban/tasks/{child['id']}")
    assert r.status_code == 200
    data = r.json()
    assert data["task"]["id"] == child["id"]
    assert parent["id"] in data["links"]["parents"]

    # Detail for the parent shows the child.
    r = client.get(f"/api/plugins/kanban/tasks/{parent['id']}")
    assert child["id"] in r.json()["links"]["children"]

    # Events exist from creation.
    assert len(data["events"]) >= 1


def test_task_detail_404_on_unknown(client):
    r = client.get("/api/plugins/kanban/tasks/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /tasks/:id — status transitions
# ---------------------------------------------------------------------------


def test_patch_status_complete(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "done", "result": "shipped"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "done"

    # Board reflects the move.
    done = next(
        c for c in client.get("/api/plugins/kanban/board").json()["columns"]
        if c["name"] == "done"
    )
    assert any(x["id"] == t["id"] for x in done["tasks"])


def test_patch_block_then_unblock(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "blocked", "block_reason": "need input"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "blocked"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "ready"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "ready"


def test_patch_schedule_then_unblock(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "scheduled", "block_reason": "run tomorrow"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "scheduled"

    columns = client.get("/api/plugins/kanban/board").json()["columns"]
    assert "scheduled" in [c["name"] for c in columns]
    scheduled = next(c for c in columns if c["name"] == "scheduled")
    assert any(x["id"] == t["id"] for x in scheduled["tasks"])

    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "ready"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "ready"


def test_patch_drag_drop_move_todo_to_ready(client):
    """Direct status write: the drag-drop path for statuses without a
    dedicated verb (e.g. manually promoting todo -> ready).

    Promoting a child whose parent is not done is rejected (409).
    Promoting a child whose parent IS done is accepted (200)."""
    parent = client.post("/api/plugins/kanban/tasks", json={"title": "p"}).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "c", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"

    # Rejected: parent not done yet.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{child['id']}",
        json={"status": "ready"},
    )
    assert r.status_code == 409

    # The 409 detail must name the blocking parent so the dashboard can
    # render an actionable toast instead of a silent no-op (#26744).
    detail = r.json()["detail"]
    assert "Cannot move to 'ready'" in detail
    assert parent["id"] in detail
    assert "'p'" in detail
    assert "status=" in detail
    # Whatever non-``done`` status the parent currently has must show up
    # so the operator knows what to fix.
    assert f"status={parent['status']}" in detail
    assert parent["status"] != "done"

    # Complete the parent.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200

    # Now child auto-promoted by recompute_ready — already ready.
    child_after = client.get(f"/api/plugins/kanban/tasks/{child['id']}").json()["task"]
    assert child_after["status"] == "ready"


def test_reopening_parent_demotes_ready_child(client):
    """Reopening a completed parent must invalidate ready children immediately.

    The dispatcher re-checks parent completion on claim, but the dashboard
    should not keep showing a stale child as ready after an operator drags
    its parent back out of done for more work.
    """
    parent = client.post("/api/plugins/kanban/tasks", json={"title": "p"}).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "c", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200

    child_after_done = client.get(
        f"/api/plugins/kanban/tasks/{child['id']}"
    ).json()["task"]
    assert child_after_done["status"] == "ready"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "todo"},
    )
    assert r.status_code == 200

    child_after_reopen = client.get(
        f"/api/plugins/kanban/tasks/{child['id']}"
    ).json()["task"]
    assert child_after_reopen["status"] == "todo"


def test_patch_reassign(client):
    t = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "x", "assignee": "a"},
    ).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"assignee": "b"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["assignee"] == "b"


def test_patch_priority_and_edit(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"priority": 5, "title": "renamed"},
    )
    assert r.status_code == 200
    data = r.json()["task"]
    assert data["priority"] == 5
    assert data["title"] == "renamed"


def test_patch_invalid_status(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "banana"},
    )
    assert r.status_code == 400


def test_patch_status_running_rejected(client):
    """Dashboard PATCH cannot transition a task directly to 'running'.

    The only legitimate path into 'running' is through the dispatcher's
    ``claim_task`` — which atomically creates a ``task_runs`` row,
    claim_lock, expiry, and worker-PID metadata. Allowing a direct set
    creates orphaned 'running' tasks with no run row or claim, which
    violate the board's run-history invariants. See issue #19535.
    """
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "running"},
    )
    assert r.status_code == 400
    assert "running" in r.json()["detail"]
    # Task's status should still be its pre-request value — the direct-set
    # was rejected before any mutation.
    board = client.get("/api/plugins/kanban/board").json()
    statuses = {
        tt["id"]: col["name"]
        for col in board["columns"]
        for tt in col["tasks"]
    }
    assert statuses.get(t["id"]) != "running"


# ---------------------------------------------------------------------------
# DELETE /tasks/:id
# ---------------------------------------------------------------------------

def test_delete_task(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "to-delete"}).json()["task"]
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert r.json()["task_id"] == t["id"]

    # Gone from board
    board = client.get("/api/plugins/kanban/board").json()
    all_ids = [tt["id"] for col in board["columns"] for tt in col["tasks"]]
    assert t["id"] not in all_ids

    # Gone from detail
    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 404


def test_delete_task_not_found(client):
    r = client.delete("/api/plugins/kanban/tasks/t_nonexistent")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Comments + Links
# ---------------------------------------------------------------------------


def test_add_comment(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/comments",
        json={"body": "how's progress?", "author": "teknium"},
    )
    assert r.status_code == 200

    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    comments = r.json()["comments"]
    assert len(comments) == 1
    assert comments[0]["body"] == "how's progress?"
    assert comments[0]["author"] == "teknium"


def test_add_comment_empty_rejected(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/comments",
        json={"body": "   "},
    )
    assert r.status_code == 400


def test_add_link_and_delete_link(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]

    r = client.post(
        "/api/plugins/kanban/links",
        json={"parent_id": a["id"], "child_id": b["id"]},
    )
    assert r.status_code == 200

    r = client.get(f"/api/plugins/kanban/tasks/{b['id']}")
    assert a["id"] in r.json()["links"]["parents"]

    r = client.delete(
        "/api/plugins/kanban/links",
        params={"parent_id": a["id"], "child_id": b["id"]},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_add_link_cycle_rejected(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    client.post(
        "/api/plugins/kanban/links",
        json={"parent_id": a["id"], "child_id": b["id"]},
    )
    r = client.post(
        "/api/plugins/kanban/links",
        json={"parent_id": b["id"], "child_id": a["id"]},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Dispatch nudge
# ---------------------------------------------------------------------------


def test_dispatch_dry_run(client):
    client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "work", "assignee": "researcher"},
    )
    r = client.post("/api/plugins/kanban/dispatch?dry_run=true&max=4")
    assert r.status_code == 200
    body = r.json()
    # DispatchResult is serialized as a dataclass dict.
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# Triage column (new v1 status)
# ---------------------------------------------------------------------------


def test_create_triage_lands_in_triage_column(client):
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "rough idea, spec me", "triage": True},
    )
    assert r.status_code == 200
    task = r.json()["task"]
    assert task["status"] == "triage"

    r = client.get("/api/plugins/kanban/board")
    triage = next(c for c in r.json()["columns"] if c["name"] == "triage")
    assert len(triage["tasks"]) == 1
    assert triage["tasks"][0]["title"] == "rough idea, spec me"


def test_triage_task_not_promoted_to_ready(client):
    """Triage tasks must stay in triage even when they have no parents."""
    client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "must stay put", "triage": True},
    )
    # Run the dispatcher — it should NOT promote the triage task.
    client.post("/api/plugins/kanban/dispatch?dry_run=false&max=4")
    r = client.get("/api/plugins/kanban/board")
    triage = next(c for c in r.json()["columns"] if c["name"] == "triage")
    ready = next(c for c in r.json()["columns"] if c["name"] == "ready")
    assert len(triage["tasks"]) == 1
    assert len(ready["tasks"]) == 0


def test_patch_status_triage_works(client):
    """A user (or specifier) can push a task back into triage, and out of it."""
    t = client.post(
        "/api/plugins/kanban/tasks", json={"title": "x"},
    ).json()["task"]
    # Normal creation is 'ready'; push to triage.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "triage"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "triage"

    # Now promote to todo.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "todo"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "todo"


# ---------------------------------------------------------------------------
# Progress rollup (done children / total children)
# ---------------------------------------------------------------------------


def test_board_progress_rollup(client):
    parent = client.post(
        "/api/plugins/kanban/tasks", json={"title": "parent"},
    ).json()["task"]
    child_a = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "a", "parents": [parent["id"]]},
    ).json()["task"]
    child_b = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "b", "parents": [parent["id"]]},
    ).json()["task"]
    # Children start as "todo" because the parent isn't done yet.  Set the
    # parent to done so children auto-promote to ready via recompute_ready.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200
    # Verify children are now ready.
    for cid in (child_a["id"], child_b["id"]):
        t = client.get(f"/api/plugins/kanban/tasks/{cid}").json()["task"]
        assert t["status"] == "ready", f"{cid} should be ready after parent done"

    # 0/2 done.
    r = client.get("/api/plugins/kanban/board")
    parent_row = next(
        t for col in r.json()["columns"] for t in col["tasks"]
        if t["id"] == parent["id"]
    )
    assert parent_row["progress"] == {"done": 0, "total": 2}

    # Complete one child. 1/2.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{child_a['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200
    r = client.get("/api/plugins/kanban/board")
    parent_row = next(
        t for col in r.json()["columns"] for t in col["tasks"]
        if t["id"] == parent["id"]
    )
    assert parent_row["progress"] == {"done": 1, "total": 2}

    # Childless tasks report progress=None, not {0/0}.
    assert next(
        t for col in r.json()["columns"] for t in col["tasks"]
        if t["id"] == child_b["id"]
    )["progress"] is None


# ---------------------------------------------------------------------------
# Auto-init on first board read
# ---------------------------------------------------------------------------


def test_board_auto_initializes_missing_db(tmp_path, monkeypatch):
    """If kanban.db doesn't exist yet, GET /board must create it, not 500."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Deliberately DO NOT call kb.init_db().

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    c = TestClient(app)
    r = c.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    assert (home / "kanban.db").exists(), "init_db wasn't invoked by /board"


# ---------------------------------------------------------------------------
# WebSocket auth (query-param token)
# ---------------------------------------------------------------------------


def test_ws_events_rejects_when_token_required(tmp_path, monkeypatch):
    """Loopback mode: a missing or wrong ?token= must be rejected with
    policy-violation; the correct token is accepted. The kanban WS now
    delegates to web_server._ws_auth_ok, so we stub that with the real
    loopback-token semantics (auth_required False → constant-time token
    compare)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    # Stub web_server with a loopback-mode _ws_auth_ok (auth_required False →
    # accept only the correct ?token=). Mirrors the real gate's loopback path.
    import hermes_cli
    import types

    def _fake_ws_auth_ok(ws):
        return ws.query_params.get("token", "") == "secret-xyz"

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=_fake_ws_auth_ok,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    c = TestClient(app)

    # No token → policy violation close.
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events"):
            pass
    assert exc.value.code == 1008

    # Wrong token → policy violation close.
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events?token=nope"):
            pass
    assert exc.value.code == 1008

    # Correct token → accepted (connect then close cleanly from our side).
    with c.websocket_connect(
        "/api/plugins/kanban/events?token=secret-xyz"
    ) as ws:
        assert ws is not None  # handshake succeeded


def test_ws_events_accepts_gated_ticket(tmp_path, monkeypatch):
    """Gated OAuth mode: the WS must accept a single-use ?ticket= (and reject
    a bare ?token=, even one matching _SESSION_TOKEN). This is the regression
    for the hosted-dashboard bug where the kanban live-events WS 1008'd on
    every gated deployment because its bespoke check only knew _SESSION_TOKEN.
    We stub _ws_auth_ok with the real gated semantics (ticket-only)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    import hermes_cli
    import types

    def _fake_ws_auth_ok(ws):
        # Gated mode: only a known ticket is accepted; token path rejected.
        return ws.query_params.get("ticket", "") == "good-ticket"

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=_fake_ws_auth_ok,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    c = TestClient(app)

    from starlette.websockets import WebSocketDisconnect

    # Legacy token is rejected in gated mode, even if it's the real one.
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events?token=secret-xyz"):
            pass
    assert exc.value.code == 1008

    # A valid ticket is accepted.
    with c.websocket_connect(
        "/api/plugins/kanban/events?ticket=good-ticket"
    ) as ws:
        assert ws is not None


def test_ws_events_board_query_param_default_overrides_current_board_pointer(tmp_path, monkeypatch):
    """The event stream must honor ``board=default`` even when the global
    current-board pointer targets a different board.

    This is the live-update half of the dashboard regression: after the UI
    selects Default, the websocket must not subscribe to the CLI's current
    non-default board.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    default_conn = kb.connect()
    try:
        default_task = kb.create_task(default_conn, title="default-live")
    finally:
        default_conn.close()

    kb.create_board("other")
    other_conn = kb.connect(board="other")
    try:
        other_task = kb.create_task(other_conn, title="other-live")
    finally:
        other_conn.close()

    kb.set_current_board("other")

    import hermes_cli
    import types

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=lambda ws: ws.query_params.get("token", "") == "secret-xyz",
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    c = TestClient(app)

    with c.websocket_connect(
        "/api/plugins/kanban/events?token=secret-xyz&board=default&since=0"
    ) as ws:
        payload = ws.receive_json()

    task_ids = {event["task_id"] for event in payload["events"]}
    assert default_task in task_ids
    assert other_task not in task_ids


def test_ws_events_swallows_cancellation_on_shutdown(tmp_path, monkeypatch):
    """``asyncio.CancelledError`` while sleeping in the poll loop is the
    normal uvicorn-shutdown path (``BaseException``, so the bare
    ``except Exception:`` does NOT catch it). Without the explicit
    clause the cancellation surfaces as an application traceback.

    Regression test for #20790 (fix in #20938). Drives the coroutine
    directly (rather than through FastAPI TestClient) so we can observe
    the cancellation outcome deterministically.
    """
    import asyncio

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    # Short-circuit the auth check — this test is about the cancellation
    # path, not auth.
    import plugins.kanban.dashboard.plugin_api as pa
    monkeypatch.setattr(pa, "_ws_upgrade_authorized", lambda ws: True)

    class _FakeWS:
        def __init__(self):
            self.query_params = {"token": "x", "since": "0"}
            self.accepted = False
            self.closed = False

        async def accept(self):
            self.accepted = True

        async def send_json(self, data):
            pass

        async def close(self, code=None):
            self.closed = True

    async def _run():
        ws = _FakeWS()
        task = asyncio.create_task(pa.stream_events(ws))
        # Give the handler a tick to accept + start polling.
        await asyncio.sleep(0.05)
        assert ws.accepted is True
        task.cancel()
        # stream_events should swallow CancelledError and return cleanly.
        # If it doesn't, this await re-raises the CancelledError.
        result = await task
        return result, ws

    result, ws = asyncio.run(_run())
    assert result is None, (
        f"stream_events should return cleanly after cancellation, got {result!r}"
    )
    # The bug symptom was a traceback; we don't assert on stderr because
    # capturing asyncio's internal "exception was never retrieved" logging
    # is flaky. The assertion that matters is: no CancelledError escaped.


# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------


def test_bulk_status_ready(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    c2 = client.post("/api/plugins/kanban/tasks", json={"title": "c"}).json()["task"]
    # Parent-less tasks land in "ready" already; push them to blocked first.
    for tid in (a["id"], b["id"], c2["id"]):
        client.patch(f"/api/plugins/kanban/tasks/{tid}",
                     json={"status": "blocked", "block_reason": "wait"})

    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"], c2["id"]], "status": "ready"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert all(r["ok"] for r in results)
    # All three are now ready.
    board = client.get("/api/plugins/kanban/board").json()
    ready = next(col for col in board["columns"] if col["name"] == "ready")
    ids = {t["id"] for t in ready["tasks"]}
    assert {a["id"], b["id"], c2["id"]}.issubset(ids)


def test_bulk_status_done_forwards_completion_summary(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]

    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={
            "ids": [a["id"], b["id"]],
            "status": "done",
            "result": "DECIDED: ship it",
            "summary": "DECIDED: ship it",
            "metadata": {"source": "dashboard"},
        },
    )

    assert r.status_code == 200
    assert all(r["ok"] for r in r.json()["results"])
    conn = kb.connect()
    try:
        for tid in (a["id"], b["id"]):
            task = kb.get_task(conn, tid)
            run = kb.latest_run(conn, tid)
            assert task.status == "done"
            assert task.result == "DECIDED: ship it"
            assert run.summary == "DECIDED: ship it"
            assert run.metadata == {"source": "dashboard"}
    finally:
        conn.close()


def test_bulk_status_running_rejected(client):
    """Bulk updates must match single-task PATCH: direct 'running' is invalid."""
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]

    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [t["id"]], "status": "running"},
    )

    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == t["id"]
    assert results[0]["ok"] is False
    assert "running" in results[0]["error"]

    board = client.get("/api/plugins/kanban/board").json()
    statuses = {
        tt["id"]: col["name"]
        for col in board["columns"]
        for tt in col["tasks"]
    }
    assert statuses.get(t["id"]) != "running"


def test_dashboard_done_actions_prompt_for_completion_summary():
    repo_root = Path(__file__).resolve().parents[2]
    bundle = (
        repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    ).read_text()

    assert "withCompletionSummary" in bundle
    assert "Completion summary" in bundle
    assert "result: summary" in bundle
    assert "body: JSON.stringify(patch)" in bundle
    assert "body: JSON.stringify(finalPatch)" in bundle


def test_dashboard_surfaces_ready_blocked_error_inline():
    """Regression for #26744: failed status transitions must be surfaced
    inline, not swallowed.  The drag/drop banner and the drawer's action
    row each render the parsed API ``detail`` so operators see *why*
    their click did nothing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    bundle = (
        repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    ).read_text()

    # Helper that strips ``"409: {\"detail\":\"…\"}"`` down to the
    # human-readable message before it lands in any banner.
    assert "function parseApiErrorMessage(err)" in bundle
    assert "parsed.detail" in bundle

    # Drag/drop banner now uses the parsed message instead of raw
    # ``err.message`` so it no longer leaks HTTP plumbing.
    assert "setError(tx(t, \"moveFailed\", \"Move failed: \") + parseApiErrorMessage(err))" in bundle

    # Drawer action row has its own visible error surface and clears it
    # on success/refresh so stale failures don't follow the operator
    # around.
    assert "const [patchErr, setPatchErr] = useState(null);" in bundle
    assert "setPatchErr(parseApiErrorMessage(e))" in bundle
    assert "setPatchErr(null)" in bundle


def test_dashboard_dependency_selects_use_value_change_handler():
    """Regression for the dependency selects in the task drawer: the
    add-parent / add-child dropdowns must wire through the shared
    selectChangeHandler helper so their value actually lands on the
    underlying React state. Salvaged from #20019 @LeonSGP43.
    """
    repo_root = Path(__file__).resolve().parents[2]
    bundle = (
        repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    ).read_text()

    parent_select = (
        'value: newParent,\n'
        '          className: "h-7 text-xs flex-1",\n'
        '        }, selectChangeHandler(setNewParent))'
    )
    child_select = (
        'value: newChild,\n'
        '          className: "h-7 text-xs flex-1",\n'
        '        }, selectChangeHandler(setNewChild))'
    )

    assert parent_select in bundle
    assert child_select in bundle


def test_bulk_archive(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"]], "archive": True})
    assert r.status_code == 200
    assert all(r["ok"] for r in r.json()["results"])
    # Default board (archived hidden) — both gone.
    board = client.get("/api/plugins/kanban/board").json()
    ids = {t["id"] for col in board["columns"] for t in col["tasks"]}
    assert a["id"] not in ids
    assert b["id"] not in ids


def test_bulk_reassign(client):
    a = client.post("/api/plugins/kanban/tasks",
                    json={"title": "a", "assignee": "old"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks",
                    json={"title": "b", "assignee": "old"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"]], "assignee": "new"})
    assert r.status_code == 200
    for tid in (a["id"], b["id"]):
        t = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert t["assignee"] == "new"


def test_bulk_unassign_via_empty_string(client):
    a = client.post("/api/plugins/kanban/tasks",
                    json={"title": "a", "assignee": "x"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"]], "assignee": ""})
    assert r.status_code == 200
    t = client.get(f"/api/plugins/kanban/tasks/{a['id']}").json()["task"]
    assert t["assignee"] is None


def test_bulk_partial_failure_doesnt_abort_siblings(client):
    """One bad id in the middle of a batch must not prevent others from
    applying."""
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    c2 = client.post("/api/plugins/kanban/tasks", json={"title": "c"}).json()["task"]
    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], "bogus-id", c2["id"]], "priority": 7})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    ok_ids = {r["id"] for r in results if r["ok"]}
    assert a["id"] in ok_ids
    assert c2["id"] in ok_ids
    assert any(not r["ok"] and r["id"] == "bogus-id" for r in results)
    # Good siblings actually got the priority bump.
    for tid in (a["id"], c2["id"]):
        t = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert t["priority"] == 7


def test_bulk_empty_ids_400(client):
    r = client.post("/api/plugins/kanban/tasks/bulk", json={"ids": []})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /config endpoint
# ---------------------------------------------------------------------------


def test_config_returns_defaults_when_section_missing(client):
    r = client.get("/api/plugins/kanban/config")
    assert r.status_code == 200
    data = r.json()
    # Defaults when dashboard.kanban is missing.
    assert data["default_tenant"] == ""
    assert data["lane_by_profile"] is True
    assert data["include_archived_by_default"] is False
    assert data["render_markdown"] is True


def test_config_reads_dashboard_kanban_section(tmp_path, monkeypatch, client):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "dashboard:\n"
        "  kanban:\n"
        "    default_tenant: acme\n"
        "    lane_by_profile: false\n"
        "    include_archived_by_default: true\n"
        "    render_markdown: false\n"
    )
    r = client.get("/api/plugins/kanban/config")
    assert r.status_code == 200
    data = r.json()
    assert data["default_tenant"] == "acme"
    assert data["lane_by_profile"] is False
    assert data["include_archived_by_default"] is True
    assert data["render_markdown"] is False


# ---------------------------------------------------------------------------
# Runs surfacing (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------

def test_task_detail_includes_runs(client):
    """GET /tasks/:id carries a runs[] array with the attempt history."""
    r = client.post("/api/plugins/kanban/tasks",
                    json={"title": "port x", "assignee": "worker"}).json()
    tid = r["task"]["id"]

    # Drive status running to force a run creation: PATCH to running
    # doesn't call claim_task (the PATCH path uses _set_status_direct),
    # so use the bulk/claim indirection via the kernel.
    import hermes_cli.kanban_db as _kb
    conn = _kb.connect()
    try:
        _kb.claim_task(conn, tid)
        _kb.complete_task(
            conn, tid,
            result="done",
            summary="tested on rate limiter",
            metadata={"changed_files": ["limiter.py"]},
        )
    finally:
        conn.close()

    d = client.get(f"/api/plugins/kanban/tasks/{tid}").json()
    assert "runs" in d
    assert len(d["runs"]) == 1
    run = d["runs"][0]
    assert run["outcome"] == "completed"
    assert run["profile"] == "worker"
    assert run["summary"] == "tested on rate limiter"
    assert run["metadata"] == {"changed_files": ["limiter.py"]}
    assert run["ended_at"] is not None


def test_task_detail_runs_empty_before_claim(client):
    """A task that's never been claimed has an empty runs[] list, not
    a missing key."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "fresh"}).json()
    d = client.get(f"/api/plugins/kanban/tasks/{r['task']['id']}").json()
    assert d["runs"] == []


def test_patch_status_done_with_summary_and_metadata(client):
    """PATCH /tasks/:id with status=done + summary + metadata must
    reach complete_task, so the dashboard has CLI parity."""
    # Create + claim.
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={
            "status": "done",
            "summary": "shipped the thing",
            "metadata": {"changed_files": ["a.py", "b.py"], "tests_run": 7},
        },
    )
    assert r.status_code == 200, r.text

    # The run must have the summary + metadata attached.
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, tid)
        assert run.outcome == "completed"
        assert run.summary == "shipped the thing"
        assert run.metadata == {"changed_files": ["a.py", "b.py"], "tests_run": 7}
    finally:
        conn.close()


def test_patch_status_done_without_summary_still_works(client):
    """Back-compat: PATCH without the new fields still completes."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "y", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "done", "result": "legacy shape"},
    )
    assert r.status_code == 200, r.text
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, tid)
        assert run.outcome == "completed"
        assert run.summary == "legacy shape"  # falls back to result
    finally:
        conn.close()


def test_patch_status_archive_closes_running_run(client):
    """PATCH to archived while running must close the in-flight run."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "z", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
        open_run = kb.latest_run(conn, tid)
        assert open_run.ended_at is None
    finally:
        conn.close()
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "archived"},
    )
    assert r.status_code == 200, r.text
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        assert task.current_run_id is None
        assert kb.latest_run(conn, tid).outcome == "reclaimed"
    finally:
        conn.close()


def test_event_dict_includes_run_id(client):
    """GET /tasks/:id returns events with run_id populated."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "e", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
        run_id = kb.latest_run(conn, tid).id
        kb.complete_task(conn, tid, summary="wss")
    finally:
        conn.close()

    r = client.get(f"/api/plugins/kanban/tasks/{tid}")
    assert r.status_code == 200
    events = r.json()["events"]
    # Every event in the response must have a run_id key (None or int).
    for e in events:
        assert "run_id" in e, f"missing run_id in event: {e}"
    # completed event must have the actual run_id.
    comp = [e for e in events if e["kind"] == "completed"]
    assert comp[0]["run_id"] == run_id



# ---------------------------------------------------------------------------
# Per-task force-loaded skills via REST
# ---------------------------------------------------------------------------

def test_create_task_with_skills_roundtrips(client):
    """POST /tasks accepts `skills: [...]`, GET /tasks/:id returns it."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "translate docs",
            "assignee": "linguist",
            "skills": ["translation", "github-code-review"],
        },
    )
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    assert task["skills"] == ["translation", "github-code-review"]

    # Fetch via GET /tasks/:id as the drawer does.
    got = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()
    assert got["task"]["skills"] == ["translation", "github-code-review"]


def test_create_task_without_skills_defaults_to_empty_list(client):
    """_task_dict serializes Task.skills=None as [] so the drawer can
    always .length check without guarding against null."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "no skills", "assignee": "x"},
    )
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    # Task.skills is None in-memory; _task_dict serializes via
    # dataclasses.asdict which keeps it None. The drawer's
    # `t.skills && t.skills.length > 0` guard handles both null and [].
    assert task.get("skills") in (None, [])


def test_create_task_with_toolset_name_in_skills_is_rejected(client):
    """POST /tasks fails fast when callers confuse toolsets with skills."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "bad skills payload",
            "assignee": "linguist",
            "skills": ["web"],
        },
    )
    assert r.status_code == 400, r.text
    assert "toolset name" in r.json()["detail"]


def test_patch_blocked_task_skills_roundtrips_with_event(client, kanban_home):
    skill_dir = (
        kanban_home / "profiles" / "auditor" / "skills" / "test" / "audit-skill"
    )
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: audit-skill\ndescription: test\n---\n",
        encoding="utf-8",
    )
    task = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "recover audit",
            "assignee": "auditor",
            "skills": ["audit-skill"],
            "model_override": "test-model",
            "provider_override": "test-provider",
        },
    ).json()["task"]
    assert client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"status": "blocked", "block_reason": "bad pin"},
    ).status_code == 200

    response = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"skills": []},
    )

    assert response.status_code == 200, response.text
    updated = response.json()["task"]
    assert updated["skills"] == []
    assert updated["status"] == "blocked"
    assert updated["assignee"] == "auditor"
    assert updated["model_override"] == "test-model"
    assert updated["provider_override"] == "test-provider"
    detail = client.get(
        f"/api/plugins/kanban/tasks/{task['id']}"
    ).json()
    event = detail["events"][-1]
    assert event["kind"] == "skills_amended"
    assert event["payload"] == {
        "old_skills": ["audit-skill"],
        "new_skills": [],
    }


def test_patch_skills_rejects_running_task(client, kanban_home):
    task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "live", "assignee": "auditor"},
    ).json()["task"]
    conn = kb.connect()
    try:
        assert kb.claim_task(conn, task["id"], claimer="test") is not None
    finally:
        conn.close()

    response = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"skills": []},
    )
    assert response.status_code == 409
    assert "running" in response.json()["detail"]



# ---------------------------------------------------------------------------
# Dispatcher-presence warning in POST /tasks response
# ---------------------------------------------------------------------------

def test_create_task_includes_warning_when_no_dispatcher(client, monkeypatch):
    """ready+assigned task + no gateway -> response has `warning` field
    so the dashboard UI can surface a banner."""
    # Force the dispatcher probe to report "not running".
    monkeypatch.setattr(
        "hermes_cli.kanban._check_dispatcher_presence",
        lambda **kw: (False, "No gateway is running — start `hermes gateway start`."),
    )
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "warn-me", "assignee": "worker"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("warning")
    assert "gateway" in data["warning"].lower()


def test_create_task_no_warning_when_dispatcher_up(client, monkeypatch):
    """Dispatcher running -> no `warning` field in the response."""
    monkeypatch.setattr(
        "hermes_cli.kanban._check_dispatcher_presence",
        lambda **kw: (True, ""),
    )
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "silent", "assignee": "worker"},
    )
    assert r.status_code == 200
    assert "warning" not in r.json() or not r.json()["warning"]


def test_create_task_no_warning_on_triage(client, monkeypatch):
    """Triage tasks never get the warning (they can't be dispatched
    anyway until promoted)."""
    monkeypatch.setattr(
        "hermes_cli.kanban._check_dispatcher_presence",
        lambda **kw: (False, "oh no"),
    )
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "triage-task", "assignee": "worker", "triage": True},
    )
    assert r.status_code == 200
    assert "warning" not in r.json() or not r.json()["warning"]


# ---------------------------------------------------------------------------
# _task_dict — outer try/except fallback when task_age raises
#
# Background: kanban_db.task_age was hardened in 061a1830 to return None for
# corrupt timestamp values via _safe_int. The companion fix added a belt-and-
# suspenders try/except in plugin_api._task_dict so that *any future* exception
# from task_age (not just ValueError on '%s') still yields a usable dict
# instead of 500'ing GET /board for the entire org.
#
# kanban_db._safe_int / task_age corruption paths are covered in
# tests/hermes_cli/test_kanban_db.py. The OUTER fallback here is not, which
# means a refactor that drops the try/except would not be caught by CI. The
# tests below pin that contract.
# ---------------------------------------------------------------------------


_FALLBACK_AGE = {
    "created_age_seconds": None,
    "started_age_seconds": None,
    "time_to_complete_seconds": None,
}


def test_board_endpoint_survives_task_age_exception(client, monkeypatch):
    """If task_age raises for any reason, GET /board must NOT 500.

    Pre-fix behavior (without the try/except in _task_dict): a single corrupt
    row turned the entire board response into a 500. The fallback dict lets
    the dashboard render every other card normally.
    """
    create = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "doomed", "assignee": "alice"},
    )
    assert create.status_code == 200, create.text

    # Force task_age to raise an exception type _safe_int does NOT handle —
    # simulates a future regression where someone re-introduces an unguarded
    # operation in task_age. ValueError on '%s' would be absorbed by _safe_int
    # and never reach the outer try/except, so it would not exercise the
    # contract this test pins.
    def _boom(_task):
        raise RuntimeError("simulated future task_age bug")
    monkeypatch.setattr("hermes_cli.kanban_db.task_age", _boom)

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200, r.text

    payload = r.json()
    # /board returns columns as a list of {name, tasks} — not a dict — so
    # flatten across all columns to find our seeded task.
    tasks = [t for col in payload["columns"] for t in col["tasks"]]
    assert len(tasks) == 1, f"expected exactly the seeded task, got {tasks!r}"
    # Strict equality: the literal fallback dict from plugin_api._task_dict
    # is the published contract the dashboard UI relies on. Key renames or
    # silent additions should fail this test on purpose.
    assert tasks[0]["age"] == _FALLBACK_AGE


def test_single_task_endpoint_survives_task_age_exception(client, monkeypatch):
    """GET /tasks/:id also calls _task_dict — same fallback should kick in.

    This is the "drawer view" path: the user clicks one card and we serialize
    just that task. A corrupt timestamp on a single task should not block the
    user from opening its drawer.
    """
    create = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "drawer-target", "assignee": "bob"},
    )
    task_id = create.json()["task"]["id"]

    def _boom(_task):
        raise RuntimeError("simulated future task_age bug")
    monkeypatch.setattr("hermes_cli.kanban_db.task_age", _boom)

    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200, r.text
    assert r.json()["task"]["age"] == _FALLBACK_AGE


def test_create_task_probe_error_does_not_break_create(client, monkeypatch):
    """Probe failure must never break task creation."""
    def _raise(**kw):
        raise RuntimeError("probe crashed")
    monkeypatch.setattr(
        "hermes_cli.kanban._check_dispatcher_presence", _raise,
    )
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "resilient", "assignee": "worker"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["title"] == "resilient"



# ---------------------------------------------------------------------------
# Home-channel subscription endpoints (#19534 follow-up: GUI opt-in)
# ---------------------------------------------------------------------------
#
# Dashboard surface for per-task, per-platform notification toggles. The
# backend endpoints read the live GatewayConfig, so tests set env vars
# (BOT_TOKEN + HOME_CHANNEL) to simulate a user who has run /sethome on
# telegram and discord.


@pytest.fixture
def with_home_channels(monkeypatch):
    """Simulate a user with home channels set on telegram and discord."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:fake")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "1234567")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "42")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_NAME", "Main TG")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "disc_fake")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "9999999")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL_NAME", "Main Discord")
    # Slack has a token but NO home — should be excluded from the list.
    monkeypatch.setenv("SLACK_BOT_TOKEN", "slack_fake")


def test_home_channels_lists_only_platforms_with_home(client, with_home_channels):
    """GET /home-channels returns entries only for platforms where the
    user has set a home; untoggled-subscribed bool is false by default."""
    r = client.get("/api/plugins/kanban/home-channels")
    assert r.status_code == 200
    platforms = {h["platform"] for h in r.json()["home_channels"]}
    assert platforms == {"telegram", "discord"}, (
        f"slack has a token but no home — must not appear. got {platforms}"
    )
    for h in r.json()["home_channels"]:
        assert h["subscribed"] is False


def test_home_channels_no_task_id_all_unsubscribed(client, with_home_channels):
    """Without task_id, every entry's subscribed=false (UI "no task" state)."""
    r = client.get("/api/plugins/kanban/home-channels")
    assert r.status_code == 200
    assert all(not h["subscribed"] for h in r.json()["home_channels"])


def test_home_subscribe_creates_notify_sub_row(client, with_home_channels):
    """POST .../home-subscribe/telegram writes a kanban_notify_subs row
    keyed to the telegram home's (chat_id, thread_id)."""
    from hermes_cli import kanban_db as kb
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]

    r = client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, t["id"])
    finally:
        conn.close()
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "1234567"
    assert subs[0]["thread_id"] == "42"
    assert subs[0]["notifier_profile"] == "default"


def test_home_subscribe_flips_subscribed_flag_in_subsequent_get(client, with_home_channels):
    """After subscribe, the GET endpoint reports subscribed=true for that
    platform and false for the others."""
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")

    r = client.get(f"/api/plugins/kanban/home-channels?task_id={t['id']}")
    flags = {h["platform"]: h["subscribed"] for h in r.json()["home_channels"]}
    assert flags == {"telegram": True, "discord": False}


def test_home_subscribe_is_idempotent(client, with_home_channels):
    """Re-subscribing keeps a single row at the DB layer."""
    from hermes_cli import kanban_db as kb
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    conn = kb.connect()
    try:
        assert len(kb.list_notify_subs(conn, t["id"])) == 1
    finally:
        conn.close()


def test_home_subscribe_backfills_owner_on_legacy_row(client, with_home_channels):
    """Re-subscribing should backfill notifier ownership on ownerless rows."""
    from hermes_cli import kanban_db as kb
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]

    conn = kb.connect()
    try:
        kb.add_notify_sub(
            conn,
            task_id=t["id"],
            platform="telegram",
            chat_id="1234567",
            thread_id="42",
        )
    finally:
        conn.close()

    r = client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    assert r.status_code == 200

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, t["id"])
    finally:
        conn.close()

    assert len(subs) == 1
    assert subs[0]["notifier_profile"] == "default"


def test_home_subscribe_unknown_platform_returns_404(client, with_home_channels):
    """Platforms without a home configured (slack in the fixture) return 404."""
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/slack")
    assert r.status_code == 404
    assert "slack" in r.json()["detail"]


def test_home_subscribe_unknown_task_returns_404(client, with_home_channels):
    r = client.post("/api/plugins/kanban/tasks/t_nonexistent/home-subscribe/telegram")
    assert r.status_code == 404


def test_home_unsubscribe_removes_notify_sub_row(client, with_home_channels):
    """DELETE .../home-subscribe/telegram removes the matching row."""
    from hermes_cli import kanban_db as kb
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    assert r.status_code == 200

    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, t["id"]) == []
    finally:
        conn.close()


def test_home_subscribe_multiple_platforms_independent(client, with_home_channels):
    """Subscribing on telegram does not affect discord and vice versa."""
    from hermes_cli import kanban_db as kb
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]

    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    client.post(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/discord")

    conn = kb.connect()
    try:
        subs = {s["platform"]: s for s in kb.list_notify_subs(conn, t["id"])}
    finally:
        conn.close()
    assert set(subs) == {"telegram", "discord"}

    # Unsubscribe telegram only.
    client.delete(f"/api/plugins/kanban/tasks/{t['id']}/home-subscribe/telegram")
    conn = kb.connect()
    try:
        subs = {s["platform"]: s for s in kb.list_notify_subs(conn, t["id"])}
    finally:
        conn.close()
    assert set(subs) == {"discord"}


def test_home_channels_empty_when_no_homes_configured(client, monkeypatch):
    """Zero platforms with a home -> empty list (UI hides the section)."""
    # No BOT_TOKEN env vars set → load_gateway_config().platforms is empty.
    for var in [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL",
        "DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL",
        "SLACK_BOT_TOKEN",
    ]:
        monkeypatch.delenv(var, raising=False)
    r = client.get("/api/plugins/kanban/home-channels")
    assert r.status_code == 200
    assert r.json()["home_channels"] == []


# ---------------------------------------------------------------------------
# Recovery endpoints (reclaim + reassign) and warnings field
# ---------------------------------------------------------------------------

def test_board_surfaces_warnings_field_for_hallucinated_completions(client):
    """Tasks with a pending completion_blocked_hallucination event surface
    a ``warnings`` object on the /board payload so the UI can badge
    them without fetching per-task events. The warnings summary is
    keyed by diagnostic kind (``hallucinated_cards``) rather than the
    raw event kind — see hermes_cli.kanban_diagnostics for the rule
    that produces it.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        real = kb.create_task(conn, title="real", assignee="x", created_by="alice")

        import pytest as _pytest
        with _pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="claimed phantom",
                created_cards=[real, "t_deadbeefcafe"],
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    tasks = [t for col in data["columns"] for t in col["tasks"]]
    parent_dict = next(t for t in tasks if t["title"] == "parent")
    assert parent_dict.get("warnings") is not None
    w = parent_dict["warnings"]
    assert w["count"] >= 1
    assert "hallucinated_cards" in w["kinds"]
    assert w["highest_severity"] == "error"
    # Full diagnostic list also on the payload for drawer rendering.
    assert parent_dict.get("diagnostics") is not None
    assert parent_dict["diagnostics"][0]["kind"] == "hallucinated_cards"
    assert "t_deadbeefcafe" in parent_dict["diagnostics"][0]["data"]["phantom_ids"]


def test_board_warnings_cleared_after_clean_completion(client):
    """A completed or edited event after a hallucination event clears
    the warning badge — we don't mark tasks permanently."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        real = kb.create_task(conn, title="real", assignee="x", created_by="alice")

        import pytest as _pytest
        with _pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="first attempt phantom",
                created_cards=[real, "t_phantom11"],
            )

        # Second attempt drops the bad id — succeeds.
        ok = kb.complete_task(
            conn, parent,
            summary="retry without phantom",
            created_cards=[real],
        )
        assert ok is True
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/board", params={"include_archived": True})
    assert r.status_code == 200
    data = r.json()
    tasks = [t for col in data["columns"] for t in col["tasks"]]
    parent_dict = next(t for t in tasks if t["title"] == "parent")
    # The clean completion wiped the warning.
    assert parent_dict.get("warnings") is None


def test_reclaim_endpoint_releases_running_claim(client):
    """POST /tasks/<id>/reclaim drops the claim, returns ok, and emits
    a manual reclaimed event."""
    import secrets
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="running", assignee="x")
        lock = secrets.token_hex(8)
        future = int(time.time()) + 3600
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, future, 99999, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, future, 99999, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, t))
        conn.commit()
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reclaim",
        json={"reason": "browser recovery"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["task_id"] == t

    # Confirm the task is back to ready.
    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT status, claim_lock FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
    finally:
        conn2.close()


def test_reclaim_endpoint_409_for_non_running_task(client):
    """Reclaiming a task that's already ready returns 409."""
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="ready", assignee="x")
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reclaim",
        json={},
    )
    assert r.status_code == 409


def test_reassign_endpoint_switches_profile(client):
    """POST /tasks/<id>/reassign changes the assignee field."""
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="task", assignee="orig")
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reassign",
        json={"profile": "newbie", "reclaim_first": False},
    )
    assert r.status_code == 200, r.text
    assert r.json()["assignee"] == "newbie"

    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT assignee FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["assignee"] == "newbie"
    finally:
        conn2.close()


def test_reassign_endpoint_409_on_running_without_reclaim(client):
    """Reassigning a running task without reclaim_first returns 409."""
    import secrets
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="running", assignee="orig")
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=? WHERE id=?",
            (secrets.token_hex(4), t),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reassign",
        json={"profile": "new", "reclaim_first": False},
    )
    assert r.status_code == 409


def test_reassign_endpoint_with_reclaim_first_succeeds_on_running(client):
    """With reclaim_first=true, a running task is reclaimed+reassigned in
    one call."""
    import secrets
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="running", assignee="orig")
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 1234, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, int(time.time()) + 3600, 1234, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, t))
        conn.commit()
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reassign",
        json={"profile": "new", "reclaim_first": True, "reason": "switch"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["assignee"] == "new"

    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT status, assignee FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["assignee"] == "new"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Diagnostics endpoint (/api/plugins/kanban/diagnostics)
# ---------------------------------------------------------------------------

def test_diagnostics_endpoint_empty_for_clean_board(client):
    r = client.get("/api/plugins/kanban/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["diagnostics"] == []


def test_diagnostics_endpoint_surfaces_blocked_hallucination(client):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        real = kb.create_task(conn, title="real", assignee="x", created_by="alice")
        import pytest as _pytest
        with _pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent, summary="phantom",
                created_cards=[real, "t_ffff00001234"],
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    row = data["diagnostics"][0]
    assert row["task_id"] == parent
    assert row["diagnostics"][0]["kind"] == "hallucinated_cards"
    assert row["diagnostics"][0]["severity"] == "error"
    assert "t_ffff00001234" in row["diagnostics"][0]["data"]["phantom_ids"]


def test_diagnostics_endpoint_severity_filter(client):
    """Severity filter is at-or-above: warning includes warning+error+critical,
    error includes error+critical, critical is exact (no higher level)."""
    conn = kb.connect()
    try:
        # A warning-severity diagnostic (prose phantom) on one task.
        # Phantom id must be valid hex — the prose scanner regex
        # requires ``t_[a-f0-9]{8,}``.
        p1 = kb.create_task(conn, title="prose", assignee="a")
        kb.complete_task(conn, p1, summary="mentioned t_deadbeef1234")
        # An error-severity diagnostic (spawn failures) on another.
        # Keep this below critical severity (failure_threshold * 2).
        p2 = kb.create_task(conn, title="spawn", assignee="b")
        conn.execute(
            "UPDATE tasks SET consecutive_failures=2, last_failure_error='x' WHERE id=?",
            (p2,),
        )
        conn.commit()
    finally:
        conn.close()

    # warning filter is at-or-above → both the warning AND the error pass.
    r = client.get("/api/plugins/kanban/diagnostics?severity=warning")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    task_ids = {row["task_id"] for row in data["diagnostics"]}
    assert task_ids == {p1, p2}

    # error filter is at-or-above → only the error passes (warning is below).
    r = client.get("/api/plugins/kanban/diagnostics?severity=error")
    data = r.json()
    assert data["count"] == 1
    assert data["diagnostics"][0]["task_id"] == p2


def test_board_exposes_diagnostics_list_and_summary(client):
    """/board should attach both the full diagnostics list AND the
    compact warnings summary (with highest_severity) on each task
    that has any diagnostic.
    """
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="crashy", assignee="worker")
        # Simulate 2 consecutive crashes -> repeated_crashes error diag
        for i in range(2):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, outcome, started_at, "
                "ended_at, error) VALUES (?, 'crashed', 'crashed', ?, ?, ?)",
                (t, int(time.time()) - 100, int(time.time()) - 50, "OOM"),
            )
        conn.commit()
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/board")
    data = r.json()
    tasks = [x for col in data["columns"] for x in col["tasks"]]
    task_dict = next(x for x in tasks if x["title"] == "crashy")
    assert task_dict["warnings"] is not None
    assert task_dict["warnings"]["highest_severity"] == "error"
    assert task_dict["diagnostics"][0]["kind"] == "repeated_crashes"


# ---------------------------------------------------------------------------
# POST /tasks/:id/specify — triage specifier endpoint
# ---------------------------------------------------------------------------


def _patch_specifier_response(monkeypatch, *, content, model="test-model"):
    """Helper: install a fake auxiliary client so the specifier endpoint
    can run without hitting any real provider."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    # specify_task routes through call_llm now (#35566) — mock it directly.
    fake_call = MagicMock(return_value=resp)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call)
    return fake_call


def test_specify_happy_path(client, monkeypatch):
    import json as jsonlib

    # Create a triage task.
    t = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "one-liner", "triage": True},
    ).json()["task"]
    assert t["status"] == "triage"

    _patch_specifier_response(
        monkeypatch,
        content=jsonlib.dumps(
            {"title": "Polished", "body": "**Goal**\nDo the thing."}
        ),
    )

    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/specify",
        json={"author": "ui-tester"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["task_id"] == t["id"]
    assert body["new_title"] == "Polished"

    # Task should have moved off the triage column.
    detail = client.get(f"/api/plugins/kanban/tasks/{t['id']}").json()["task"]
    assert detail["status"] in {"todo", "ready"}
    assert detail["title"] == "Polished"
    assert "**Goal**" in (detail["body"] or "")


def test_specify_non_triage_returns_ok_false_not_http_error(client, monkeypatch):
    """The endpoint intentionally returns ``{ok: false, reason: ...}`` for
    "task not in triage" rather than a 4xx — the dashboard renders the
    reason inline so the user can fix it without a page reload."""
    # Create a normal (ready) task — not in triage.
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]

    _patch_specifier_response(monkeypatch, content="unused")

    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/specify",
        json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "not in triage" in body["reason"]


def test_specify_no_aux_client_surfaces_reason(client, monkeypatch):
    t = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "rough", "triage": True},
    ).json()["task"]

    # Simulate "no auxiliary client configured" — call_llm raises when
    # no provider resolves (#35566 routing).
    def _no_provider(**kwargs):
        raise RuntimeError("No LLM provider configured")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _no_provider)

    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/specify",
        json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    # call_llm's no-provider RuntimeError surfaces via the LLM-error branch.
    assert "LLM error" in body["reason"]

    # Task must stay in triage — nothing was touched.
    detail = client.get(f"/api/plugins/kanban/tasks/{t['id']}").json()["task"]
    assert detail["status"] == "triage"


def test_board_endpoint_accepts_explicit_board_default_param(client):
    """GET /board?board=default must not fall through to env/current-file resolution.

    The dashboard always sends ``?board=<slug>`` (including ``board=default``)
    so that the server-side ``current`` file can never override the dashboard's
    selected board.  This test asserts the endpoint accepts the parameter and
    returns the default board without falling back to environment variable or
    current-file resolution.
    Regression: #21819.
    """
    # Create a task on the default board.
    t = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "on-default-board"},
    ).json()["task"]
    assert t["status"] == "ready"

    # Request with explicit board=default — must succeed and include the task.
    r = client.get("/api/plugins/kanban/board?board=default")
    assert r.status_code == 200
    data = r.json()
    ready = next((c for c in data["columns"] if c["name"] == "ready"), None)
    assert ready is not None, "no 'ready' column in default board response"
    task_ids = [task["id"] for task in ready["tasks"]]
    assert t["id"] in task_ids, (
        f"task {t['id']} not found in ready column of default board "
        f"(got tasks: {task_ids}). The board=default param was likely ignored."
    )


def test_dashboard_requests_default_board_explicitly():
    """Dashboard REST calls must include board=default instead of relying on server current board."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()

    assert "SDK.fetchJSON(withBoard(`${API}/config`, board))" in dist
    assert "SDK.fetchJSON(withBoard(`${API}/boards`, board))" in dist
    assert "}, [loadBoardList, switchBoard, board]);" in dist


def test_dashboard_search_includes_body_and_result():
    """Client-side search must match body, result, latest_summary, and summary
    so full card contents are findable."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()

    assert "t.body || \"\"" in dist
    assert "t.result || \"\"" in dist
    assert "t.latest_summary || \"\"" in dist


def test_dashboard_bulk_actions_include_reclaim_first():
    """Bulk action bar must expose reclaim_first checkbox and expanded status buttons."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()

    assert "reclaim_first: reclaimFirst" in dist
    assert "hermes-kanban-bulk-reclaim-first" in dist
    assert '"→ todo"' in dist
    assert '"Block"' in dist
    assert '"Unblock"' in dist


def test_dashboard_shift_click_range_selection_exists():
    """Shift-click must trigger range selection via toggleRange."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()

    assert "function toggleRange" in dist or "const toggleRange =" in dist
    assert "props.toggleRange(t.id)" in dist or "props.toggleRange" in dist
    assert "e.shiftKey" in dist


def test_dashboard_multi_move_bulk_exists():
    """Dragging a selected card with other selections must use /tasks/bulk."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()

    assert "onMoveSelected" in dist
    assert "props.onMoveSelected" in dist
    assert "`${API}/tasks/bulk`" in dist


def test_dashboard_failed_card_highlight_class_exists():
    """Partial bulk failures must highlight failing cards."""
    repo_root = Path(__file__).resolve().parents[2]
    js = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    css = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "style.css").read_text()

    assert "hermes-kanban-card--failed" in js
    assert "hermes-kanban-card--failed" in css
    assert "failedIds" in js

# ---------------------------------------------------------------------------
# Final result visibility for Done cards
# ---------------------------------------------------------------------------


def test_task_detail_exposes_result_and_latest_summary_separately(client):
    """The drawer receives both source fields without a duplicate alias."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "Task with explicit result"},
    )
    task_id = r.json()["task"]["id"]
    client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "done", "result": "The final answer is 42.", "summary": "short handoff"},
    )
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200
    data = r.json()["task"]
    assert data["result"] == "The final answer is 42."
    assert data["latest_summary"] == "short handoff"
    assert "final_result" not in data


def test_task_detail_exposes_latest_summary_when_result_is_empty(client):
    """Summary-only completions remain available to the drawer fallback."""
    conn = kb.connect()
    task_id = kb.create_task(conn, title="Task with only run summary")
    kb.claim_task(conn, task_id)
    kb.complete_task(conn, task_id, summary="Report written to /output/report.md")
    conn.close()

    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200
    data = r.json()["task"]
    assert data["status"] == "done"
    assert not data["result"]
    assert data["latest_summary"] == "Report written to /output/report.md"


def test_task_detail_latest_summary_none_when_nothing_recorded(client):
    """When no run summary exists, the existing field remains None."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "Task with no result at all"},
    )
    task_id = r.json()["task"]["id"]
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json()["task"]["latest_summary"] is None


def test_board_tasks_include_latest_summary(client):
    """Board cards already expose the summary used by the drawer fallback."""
    conn = kb.connect()
    task_id = kb.create_task(conn, title="Board card with summary only")
    kb.claim_task(conn, task_id)
    kb.complete_task(conn, task_id, summary="Done: see attachment")
    conn.close()

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    done_col = next(c for c in r.json()["columns"] if c["name"] == "done")
    card = next((t for t in done_col["tasks"] if t["id"] == task_id), None)
    assert card is not None
    assert "Done: see attachment" in card["latest_summary"]


def test_dashboard_done_final_result_section_rendered_from_summary():
    """Frontend must render Final Result section from run summary when task.result is empty."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    assert "t.result || t.latest_summary" in dist
    assert "Final Result (run summary)" in dist
    assert "No final result was recorded" in dist
    assert "orchestrator" in dist or "parent task" in dist


def test_task_detail_includes_child_result_summaries(client):
    """Parent drawers should receive the child results they need to render."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="Research topic")
        child = kb.create_task(conn, title="Collect sources")
        kb.link_tasks(conn, parent, child)
        kb.complete_task(conn, parent, summary="Delegated research to child tasks.")
        kb.recompute_ready(conn)
        kb.complete_task(conn, child, summary="Collected five primary sources.")

    response = client.get(f"/api/plugins/kanban/tasks/{parent}")

    assert response.status_code == 200
    assert response.json()["child_results"] == [
        {
            "id": child,
            "title": "Collect sources",
            "status": "done",
            "latest_summary": "Collected five primary sources.",
            "result": None,
        }
    ]


def test_dashboard_final_result_uses_existing_fields_without_alias():
    """The drawer should not duplicate result/summary into another API field."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    api = (repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py").read_text()

    assert "var finalResult = t.result || t.latest_summary || null;" in dist
    assert "t.final_result" not in dist
    assert 'd["final_result"]' not in api


def test_dashboard_parent_notice_and_child_results_use_detail_links():
    """Parent detection must use links.children, which exists in task detail."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    detail = dist[dist.index("function TaskDetail"):]

    assert "links.children.length > 0" in detail
    assert "t.link_counts" not in detail
    assert "Child Results" in detail
    assert "props.data.child_results" in detail


# ============================================================================
# ============================================================================
# AION R4/I01 — bounded adversarial scanner for direct tasks.status SQL
# ============================================================================
CANONICAL_BOUNDARY = "hermes_cli/kanban_db.py"
DASHBOARD_PLUGIN = "plugins/kanban/dashboard/plugin_api.py"
DYNAMIC = "\x00DYNAMIC\x00"
MUTATION_METHODS = ("execute", "executemany", "executescript")
READ_LEAD = ("SELECT", "PRAGMA", "WITH", "EXPLAIN", "VACUUM", "ANALYZE", "ATTACH", "DETACH")

# --- bounded Kanban object provenance ------------------------------------
# A name only becomes a *flag-able* target when its provenance is proven to be
# a Kanban connection/cursor/bound execute method. Provenance is seeded from
# the kanban_db connection factory (hermes_cli.kanban_db.connect / the dashboard
# _conn helper) and, in explicit-entrypoint adversarial fixtures, from a
# connection-named parameter declared by the fixture. It propagates through
# plain aliases, ``.cursor()``, bound ``.execute``/``.executemany``/
# ``.executescript`` methods, opaque ``getattr(obj, 'execute')``, and
# interprocedural argument binding. Arbitrary non-Kanban ``.execute`` receivers
# (logger, sqlite3 state/projects/session connections, console/terminal
# engines) carry no tag and therefore stay non-violating.
PROV_CONN = "conn"
PROV_CURSOR = "cursor"
PROV_METHOD = "method"
PROV_KANBAN_MODULE = "kanban_module"
PROV_KANBAN_CONNECT = "kanban_connect"
_KANBAN_OBJECT_TAGS = {PROV_CONN, PROV_CURSOR, PROV_METHOD}
CONNECTION_PARAM_NAMES = {"conn", "connection", "db", "database"}

# Unknown / no-provenance sentinel for the return-summary lattice. Every
# reachable return arm that is not a supported known Kanban provenance tag
# (unknown, non-Kanban, cyclic, bare ``return``, or implicit ``None``
# fall-through) is classified as UNKNOWN_RETURN so it cannot silently
# disappear from the unanimity check in ``_function_return_summary``.
UNKNOWN_RETURN = "unknown_return"


class _Unresolved:
    def __repr__(self):  # pragma: no cover
        return "<UNRESOLVED>"


UNRESOLVED = _Unresolved()


def _norm_sql(s: str) -> str:
    s = re.sub(r"--[^\n]*", " ", s)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return " ".join(s.split())


def _is_read(s: str) -> bool:
    lead = _norm_sql(s).lstrip().upper().split(" ")[0]
    return lead in READ_LEAD


# ---------------------------------------------------------------------------
# String folding (bounded constant propagation)
# ---------------------------------------------------------------------------

def _fold_string(node, env, depth=0):
    if depth > 24 or node is None:
        return UNRESOLVED
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else UNRESOLVED
    if isinstance(node, ast.Name):
        v = env.get(node.id, UNRESOLVED)
        return v if isinstance(v, str) else UNRESOLVED
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                sub = _fold_string(v.value, env, depth + 1)
                parts.append(sub if isinstance(sub, str) else DYNAMIC)
            else:
                parts.append(DYNAMIC)
        return "".join(parts)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            l = _fold_string(node.left, env, depth + 1)
            r = _fold_string(node.right, env, depth + 1)
            if isinstance(l, str) and isinstance(r, str):
                return l + r
            if isinstance(l, str):
                return l + DYNAMIC
            if isinstance(r, str):
                return DYNAMIC + r
            return DYNAMIC
        if isinstance(node.op, ast.Mod):
            template = _fold_string(node.left, env, depth + 1)
            if not isinstance(template, str):
                return UNRESOLVED
            return _apply_percent(template, node.right, env, depth)
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "join":
            sep = _fold_string(fn.value, env, depth + 1)
            elems = _fold_list(node.args[0], env, depth + 1) if node.args else UNRESOLVED
            if isinstance(sep, str) and isinstance(elems, list):
                return sep.join(elems)
            return DYNAMIC
        if isinstance(fn, ast.Attribute) and fn.attr == "format":
            base = _fold_string(fn.value, env, depth + 1)
            if not isinstance(base, str):
                return UNRESOLVED
            return _apply_format(base, node, env, depth)
        return UNRESOLVED
    return UNRESOLVED


def _apply_percent(template, right, env, depth):
    args = []
    if isinstance(right, ast.Tuple):
        for elt in right.elts:
            v = _fold_string(elt, env, depth + 1)
            args.append(v if isinstance(v, str) else DYNAMIC)
    else:
        v = _fold_string(right, env, depth + 1)
        args.append(v if isinstance(v, str) else DYNAMIC)
    it = iter(args)
    out = []
    i = 0
    while i < len(template):
        if template[i] == "%" and i + 1 < len(template) and template[i + 1] in "srd":
            out.append(next(it, DYNAMIC))
            i += 2
            continue
        out.append(template[i])
        i += 1
    return "".join(out)


def _apply_format(base, node, env, depth):
    args = []
    for a in node.args:
        v = _fold_string(a, env, depth + 1)
        args.append(v if isinstance(v, str) else DYNAMIC)
    it = iter(args)
    out = []
    i = 0
    while i < len(base):
        if base[i] == "{" and i + 1 < len(base) and base[i + 1] == "}":
            out.append(next(it, DYNAMIC))
            i += 2
            continue
        out.append(base[i])
        i += 1
    return "".join(out)


def _fold_list(node, env, depth=0):
    if depth > 24 or node is None:
        return UNRESOLVED
    if isinstance(node, ast.List):
        out = []
        for elt in node.elts:
            v = _fold_string(elt, env, depth + 1)
            out.append(v if isinstance(v, str) else DYNAMIC)
        return out
    if isinstance(node, ast.Name):
        v = env.get(node.id, UNRESOLVED)
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [v]
        return UNRESOLVED
    return UNRESOLVED


# ---------------------------------------------------------------------------
# SQL mutation classification
# ---------------------------------------------------------------------------

def _parse_set_columns(raw: str):
    cols = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        eq = part.find("=")
        name = (part[:eq] if eq >= 0 else part).strip().lower()
        if DYNAMIC in name:
            cols.append(DYNAMIC)
        else:
            cols.append(name)
    return cols


def _classify_sql(shape: str):
    norm = _norm_sql(shape)
    up = norm.upper()
    if not up:
        return None
    if _is_read(up):
        return None
    m = re.match(r"^UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\b.*)?$", up, re.DOTALL)
    if m:
        return ("UPDATE", m.group(1).lower(), _parse_set_columns(m.group(2)))
    m = re.match(r"^INSERT\s+INTO\s+(\w+)\s*\(([^)]*)\)", up, re.DOTALL)
    if m:
        cols = [c.strip().lower() for c in m.group(2).split(",") if c.strip()]
        return ("INSERT", m.group(1).lower(), cols)
    m = re.match(r"^INSERT\s+INTO\s+(\w+)", up)
    if m:
        return ("INSERT", m.group(1).lower(), [DYNAMIC])
    m = re.match(r"^DELETE\s+FROM\s+(\w+)", up)
    if m:
        return ("DELETE", m.group(1).lower(), [])
    m = re.match(r"^REPLACE\s+INTO\s+(\w+)\s*\(([^)]*)\)", up, re.DOTALL)
    if m:
        cols = [c.strip().lower() for c in m.group(2).split(",") if c.strip()]
        return ("REPLACE", m.group(1).lower(), cols)
    if up.startswith(("UPDATE", "INSERT", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER")):
        return ("UNKNOWN_MUTATION", None, [DYNAMIC])
    return None


# ---------------------------------------------------------------------------
# Scanner core (entrypoint-scoped, bounded interprocedural)
# ---------------------------------------------------------------------------

def _module_to_path(module, relpath=""):
    if module.endswith(".py"):
        return module
    return "/".join(module.split(".")) + ".py"


def _split_entrypoint(ep):
    parts = ep.split(".")
    if len(parts) >= 2:
        name = parts[-1]
        module = ".".join(parts[:-1])
        return (_module_to_path(module), name)
    return (ep, ep)


def _build_import_map(files):
    import_map = {}
    for relpath, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                mod_rel = _module_to_path(node.module, relpath)
                for alias in node.names:
                    import_map[(relpath, alias.asname or alias.name)] = (mod_rel, alias.name)
    return import_map


def _receiver_prov(node, env, prov, import_map=None, relpath=None,
                   funcs=None, module_envs=None, canonical_boundary=None,
                   depth=0, ret_visited=None):
    """Return the Kanban object provenance tag for a receiver expression, else None.

    A receiver may be a name, a direct factory call (``kanban_db.connect()``),
    a cursor-derivation chain (``conn.cursor()``), or — when the interprocedural
    context is available — a helper call whose summarized return is a proven
    Kanban connection/cursor/bound method. Only proven Kanban
    connection/cursor/method provenance is returned; non-Kanban receivers
    (logger, sqlite3) carry no tag.
    """
    if isinstance(node, ast.Name):
        p = prov.get(node.id)
        return p if p in _KANBAN_OBJECT_TAGS else None
    if isinstance(node, ast.Call):
        if import_map is not None:
            if _is_kanban_factory(node, env, prov, import_map, relpath):
                return PROV_CONN
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "cursor":
            if _receiver_prov(fn.value, env, prov, import_map, relpath,
                              funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
                return PROV_CURSOR
        # Helper-call receiver: ``get_conn()`` / ``get_cursor()`` / ``get_run()``.
        if isinstance(fn, ast.Name) and funcs is not None:
            target = _resolve_callee(fn.id, relpath, funcs, import_map)
            if target is not None:
                trel, tfunc = target
                if trel != canonical_boundary:
                    tag, _bm = _function_return_summary(
                        trel, tfunc, node, env, prov, relpath, funcs, import_map,
                        module_envs, canonical_boundary, depth, ret_visited)
                    if tag in _KANBAN_OBJECT_TAGS:
                        return tag
    return None


def _arg_prov(arg, env, prov, import_map=None, relpath=None):
    """Return the Kanban object provenance tag carried by a call argument, else None."""
    if isinstance(arg, ast.Name):
        p = prov.get(arg.id)
        return p if p in _KANBAN_OBJECT_TAGS else None
    if isinstance(arg, ast.Attribute) and arg.attr in MUTATION_METHODS:
        if _receiver_prov(arg.value, env, prov, import_map, relpath) is not None:
            return PROV_METHOD
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "getattr":
        if len(arg.args) >= 2:
            attr = _fold_string(arg.args[1], env)
            if isinstance(attr, str) and attr in MUTATION_METHODS:
                if _receiver_prov(arg.args[0], env, prov, import_map, relpath):
                    return PROV_METHOD
    return None


def _bound_method_env(node, env):
    """Return the ``('__BOUND_METHOD__', attr)`` callable identity for an
    expression naming a bound mutation method, else None."""
    if isinstance(node, ast.Name):
        bm = env.get(node.id)
        if isinstance(bm, tuple) and len(bm) == 2 and bm[0] == "__BOUND_METHOD__":
            return bm
        return None
    if isinstance(node, ast.Attribute) and node.attr in MUTATION_METHODS:
        return ("__BOUND_METHOD__", node.attr)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if len(node.args) >= 2:
            attr = _fold_string(node.args[1], env)
            if isinstance(attr, str) and attr in MUTATION_METHODS:
                return ("__BOUND_METHOD__", attr)
    return None


def _record_import_prov(prov, alias):
    name = alias.asname or alias.name
    if alias.name in ("kanban_db", "hermes_cli.kanban_db"):
        prov[name] = PROV_KANBAN_MODULE


def _record_importfrom_prov(prov, module, alias):
    name = alias.asname or alias.name
    mod = module or ""
    if mod in ("hermes_cli", "hermes_cli.kanban_db"):
        if alias.name == "kanban_db":
            prov[name] = PROV_KANBAN_MODULE
        elif alias.name == "connect":
            prov[name] = PROV_KANBAN_CONNECT


def _is_kanban_factory(call, env, prov, import_map, relpath):
    """True if ``call`` is a Kanban connection factory invocation."""
    fn = call.func
    if isinstance(fn, ast.Name):
        if fn.id == "_conn":
            return True
        if prov.get(fn.id) == PROV_KANBAN_CONNECT:
            return True
        target = import_map.get((relpath, fn.id))
        if target and target[0] == CANONICAL_BOUNDARY and target[1] == "connect":
            return True
    if isinstance(fn, ast.Attribute) and fn.attr == "connect":
        base = fn.value
        if isinstance(base, ast.Name):
            if prov.get(base.id) == PROV_KANBAN_MODULE:
                return True
            target = import_map.get((relpath, base.id))
            if target and target[0] == CANONICAL_BOUNDARY:
                return True
    return False


def _factory_callable_prov(node, env, prov, import_map=None, relpath=None):
    """Return PROV_KANBAN_CONNECT if ``node`` names the Kanban connection
    factory *callable* without invoking it, else None.

    Covers the imported symbol (``from hermes_cli.kanban_db import connect`` ->
    ``connect``) and the attribute form (``kanban_db.connect``) where
    ``kanban_db`` is the proven factory module. This is the callable identity
    that must survive ``make = kanban_db.connect`` / ``make = connect`` and any
    plain/repeated alias of ``make``.
    """
    if isinstance(node, ast.Name):
        if prov.get(node.id) == PROV_KANBAN_CONNECT:
            return PROV_KANBAN_CONNECT
        if import_map is not None and relpath is not None:
            target = import_map.get((relpath, node.id))
            if target and target[0] == CANONICAL_BOUNDARY and target[1] == "connect":
                return PROV_KANBAN_CONNECT
        return None
    if isinstance(node, ast.Attribute) and node.attr == "connect":
        base = node.value
        if isinstance(base, ast.Name):
            if prov.get(base.id) == PROV_KANBAN_MODULE:
                return PROV_KANBAN_CONNECT
            if import_map is not None and relpath is not None:
                target = import_map.get((relpath, base.id))
                if target and target[0] == CANONICAL_BOUNDARY:
                    return PROV_KANBAN_CONNECT
    return None


def scan_status_sql(files, entrypoints=None, canonical_boundary=CANONICAL_BOUNDARY):
    """Scan a set of files for direct ``tasks.status`` lifecycle SQL.

    * ``entrypoints``: functions to begin analysis from (``"app.f"`` style).
      When None, every module-level function in every non-boundary file is an
      entrypoint (the real production-scan mode).
    * A mutation call (``.execute``/``.executemany``/``.executescript`` or the
      opaque ``getattr(obj, 'execute')``) is only a violation when its receiver
      carries a *proven* Kanban connection/cursor/bound-method provenance.
      Unresolved mutation-capable SQL on a proven-Kanban receiver fails closed
      across the whole tracked production universe (no dashboard-only scope).
    """
    trees = {}
    for relpath, src in files.items():
        try:
            trees[relpath] = ast.parse(src)
        except SyntaxError:
            trees[relpath] = None

    funcs = {}
    for relpath, tree in trees.items():
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                funcs[(relpath, node.name)] = node

    import_map = _build_import_map(files)

    module_envs = {}
    for relpath, tree in trees.items():
        if tree is not None:
            module_envs[relpath] = _module_env(tree, relpath, import_map)

    eps = []
    if entrypoints is None:
        for relpath, tree in trees.items():
            if tree is None or relpath == canonical_boundary:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    eps.append((relpath, node.name))
    else:
        for ep in entrypoints:
            relpath, name = _split_entrypoint(ep)
            if (relpath, name) in funcs:
                eps.append((relpath, name))

    raw = []
    for relpath, name in eps:
        func = funcs[(relpath, name)]
        env, prov = _module_env(trees[relpath], relpath, import_map)
        if entrypoints is not None:
            _seed_entrypoint_params(func, prov)
        _process_stmts(func.body, env, prov, relpath, funcs, import_map, raw,
                       canonical_boundary, 0, set(), module_envs)

    # Dedupe (a function reached via multiple entrypoints/inlines is one find).
    seen = set()
    out = []
    for v in raw:
        key = (v["file"], v["line"], v["expectation"], v["operation"], v["table"])
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _seed_entrypoint_params(func, prov):
    """In explicit-entrypoint fixture mode, connection-named parameters are the
    declared Kanban connection under test."""
    params = list(func.args.posonlyargs) + list(func.args.args) + list(func.args.kwonlyargs)
    for a in params:
        if a.arg in CONNECTION_PARAM_NAMES:
            prov[a.arg] = PROV_CONN


def _module_env(tree, relpath, import_map):
    env = {}
    prov = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                _assign_target(env, prov, tgt, stmt.value, relpath, import_map)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                _assign(env, prov, stmt.target.id, stmt.value, relpath, import_map)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                _record_import_prov(prov, alias)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                _record_importfrom_prov(prov, stmt.module, alias)
    return env, prov


def _assign_target(env, prov, tgt, value, relpath, import_map, funcs=None,
                   module_envs=None, canonical_boundary=None, depth=0, ret_visited=None):
    if isinstance(tgt, ast.Name):
        _assign(env, prov, tgt.id, value, relpath, import_map, funcs,
                module_envs, canonical_boundary, depth, ret_visited)
    elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        if len(tgt.elts) == len(value.elts):
            for te, ve in zip(tgt.elts, value.elts):
                _assign_target(env, prov, te, ve, relpath, import_map, funcs,
                               module_envs, canonical_boundary, depth, ret_visited)


def _assign(env, prov, name, value, relpath, import_map, funcs=None,
            module_envs=None, canonical_boundary=None, depth=0, ret_visited=None):
    # Bound mutation method alias: ``run = conn.execute``.
    if isinstance(value, ast.Attribute) and value.attr in MUTATION_METHODS:
        env[name] = ("__BOUND_METHOD__", value.attr)
        if _receiver_prov(value.value, env, prov, import_map, relpath,
                          funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
            prov[name] = PROV_METHOD
        return
    # Kanban connection factory *callable* alias: ``make = kanban_db.connect`` /
    # ``make = connect``. The callable identity must survive so ``conn = make()``
    # is still recognized as a factory invocation.
    if (isinstance(value, ast.Attribute) and value.attr == "connect"
            and _factory_callable_prov(value, env, prov, import_map, relpath)):
        prov[name] = PROV_KANBAN_CONNECT
        return
    # Opaque bound method alias: ``run = getattr(conn, 'execute')``.
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "getattr" and len(value.args) >= 2):
        attr_name = _fold_string(value.args[1], env)
        if isinstance(attr_name, str) and attr_name in MUTATION_METHODS:
            env[name] = ("__BOUND_METHOD__", attr_name)
            if _receiver_prov(value.args[0], env, prov, import_map, relpath,
                              funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
                prov[name] = PROV_METHOD
            return
    # Kanban cursor: ``cur = conn.cursor()``.
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "cursor"):
        if _receiver_prov(value.func.value, env, prov, import_map, relpath,
                          funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
            prov[name] = PROV_CURSOR
        return
    # Kanban connection factory: ``conn = kanban_db.connect(...)`` / ``_conn(...)``.
    if isinstance(value, ast.Call) and _is_kanban_factory(value, env, prov, import_map, relpath):
        prov[name] = PROV_CONN
        return
    # Helper-call return provenance: ``conn = get_conn()`` / ``run = get_run(conn)``
    # / ``cur = get_cursor(conn)`` (local or mapped cross-file helper).
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and funcs is not None):
        target = _resolve_callee(value.func.id, relpath, funcs, import_map)
        if target is not None:
            trel, tfunc = target
            if trel != canonical_boundary:
                rv = ret_visited if ret_visited is not None else set()
                tag, bm = _function_return_summary(
                    trel, tfunc, value, env, prov, relpath, funcs, import_map,
                    module_envs, canonical_boundary, depth, rv)
                if tag == PROV_METHOD:
                    env[name] = bm
                    prov[name] = PROV_METHOD
                    return
                if tag in (PROV_CONN, PROV_CURSOR):
                    prov[name] = tag
                    return
                if tag == PROV_KANBAN_CONNECT:
                    prov[name] = PROV_KANBAN_CONNECT
                    return
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        env[name] = value.value
        return
    if isinstance(value, ast.List):
        env[name] = _fold_list(value, env)
        return
    # Plain alias: ``db = conn`` (inherit Kanban provenance) or a factory-callable
    # alias (``make2 = make``).
    if isinstance(value, ast.Name):
        p = prov.get(value.id)
        if p in _KANBAN_OBJECT_TAGS:
            prov[name] = p
        if p == PROV_KANBAN_CONNECT:
            prov[name] = PROV_KANBAN_CONNECT
        # A bound mutation method also keeps its callable identity through a
        # plain alias: ``run = conn.execute; alias = run`` must stay callable.
        bm = env.get(value.id)
        if isinstance(bm, tuple) and len(bm) == 2 and bm[0] == "__BOUND_METHOD__":
            env[name] = bm
    v = _fold_string(value, env)
    if isinstance(v, str):
        env[name] = v


def _process_stmts(stmts, env, prov, relpath, funcs, import_map, violations,
                   canonical_boundary, depth, visited, module_envs):
    if depth > 24:
        return
    for stmt in stmts:
        _process_stmt(stmt, env, prov, relpath, funcs, import_map, violations,
                      canonical_boundary, depth, visited, module_envs)


def _process_stmt(stmt, env, prov, relpath, funcs, import_map, violations,
                  canonical_boundary, depth, visited, module_envs):
    if isinstance(stmt, ast.Assign):
        _scan_calls(stmt.value, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)
        for tgt in stmt.targets:
            _assign_target(env, prov, tgt, stmt.value, relpath, import_map,
                           funcs, module_envs, canonical_boundary, depth, None)
    elif isinstance(stmt, ast.AnnAssign):
        if stmt.value is not None:
            _scan_calls(stmt.value, env, prov, relpath, funcs, import_map, violations,
                        canonical_boundary, depth, visited, module_envs)
        if isinstance(stmt.target, ast.Name) and stmt.value is not None:
            _assign(env, prov, stmt.target.id, stmt.value, relpath, import_map,
                    funcs, module_envs, canonical_boundary, depth, None)
    elif isinstance(stmt, ast.AugAssign):
        _scan_calls(stmt.value, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)
        if isinstance(stmt.target, ast.Name):
            cur = env.get(stmt.target.id, UNRESOLVED)
            inc = _fold_string(stmt.value, env)
            if isinstance(cur, str) and isinstance(inc, str):
                env[stmt.target.id] = cur + inc
    elif isinstance(stmt, ast.Expr):
        _scan_calls(stmt.value, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)
    elif isinstance(stmt, ast.Return):
        if stmt.value is not None:
            _scan_calls(stmt.value, env, prov, relpath, funcs, import_map, violations,
                        canonical_boundary, depth, visited, module_envs)
    elif isinstance(stmt, ast.If):
        _scan_calls(stmt.test, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)
        _process_stmts(stmt.body, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
        _process_stmts(stmt.orelse, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        _scan_calls(stmt.iter, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)
        _process_stmts(stmt.body, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
        _process_stmts(stmt.orelse, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            _scan_calls(item.context_expr, env, prov, relpath, funcs, import_map,
                        violations, canonical_boundary, depth, visited, module_envs)
        _process_stmts(stmt.body, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
    elif isinstance(stmt, ast.Try):
        _process_stmts(stmt.body, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
        for h in stmt.handlers:
            _process_stmts(h.body, env, prov, relpath, funcs, import_map, violations,
                           canonical_boundary, depth + 1, visited, module_envs)
        _process_stmts(stmt.orelse, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
        _process_stmts(stmt.finalbody, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
    elif isinstance(stmt, ast.While):
        _scan_calls(stmt.test, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)
        _process_stmts(stmt.body, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)
        _process_stmts(stmt.orelse, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)


def _scan_calls(node, env, prov, relpath, funcs, import_map, violations,
                canonical_boundary, depth, visited, module_envs):
    if node is None:
        return
    if isinstance(node, ast.Call):
        _classify_call(node, env, prov, relpath, funcs, import_map, violations,
                       canonical_boundary, depth, visited, module_envs)
    for child in ast.iter_child_nodes(node):
        _scan_calls(child, env, prov, relpath, funcs, import_map, violations,
                    canonical_boundary, depth, visited, module_envs)


def _classify_call(call, env, prov, relpath, funcs, import_map, violations,
                   canonical_boundary, depth, visited, module_envs):
    if depth > 24:
        return
    fn = call.func
    if isinstance(fn, ast.Attribute):
        if fn.attr == "append":
            base = fn.value
            if isinstance(base, ast.Name) and isinstance(env.get(base.id), list):
                v = _fold_string(call.args[0], env) if call.args else UNRESOLVED
                env[base.id] = env[base.id] + ([v] if isinstance(v, str) else [DYNAMIC])
            return
        if fn.attr == "set_task_status":
            return
        if fn.attr in MUTATION_METHODS:
            # Flag only when the receiver is a *proven* Kanban object.
            if _receiver_prov(fn.value, env, prov, import_map, relpath,
                              funcs, module_envs, canonical_boundary, depth, set()) is None:
                return
            sql_arg = call.args[0] if call.args else None
            if sql_arg is None:
                return
            shape = _fold_string(sql_arg, env)
            _record_if_violation(shape, sql_arg, call, relpath, canonical_boundary, violations)
            return
        return
    # Opaque/reflection getattr: ``getattr(obj, 'execute')(sql)``.
    if isinstance(fn, ast.Call) and isinstance(fn.func, ast.Name) and fn.func.id == "getattr":
        if len(fn.args) >= 2:
            attr_name = _fold_string(fn.args[1], env)
            if isinstance(attr_name, str) and attr_name in MUTATION_METHODS:
                if _receiver_prov(fn.args[0], env, prov, import_map, relpath,
                                  funcs, module_envs, canonical_boundary, depth, set()) is None:
                    return
                sql_arg = call.args[0] if call.args else None
                if sql_arg is None:
                    return
                shape = _fold_string(sql_arg, env)
                _record_if_violation(shape, sql_arg, call, relpath, canonical_boundary, violations)
                return
        return
    if isinstance(fn, ast.Name):
        if fn.id == "set_task_status":
            return
        bound = env.get(fn.id)
        if isinstance(bound, tuple) and len(bound) == 2 and bound[0] == "__BOUND_METHOD__":
            # Flag only when the bound-method alias carries Kanban provenance.
            if prov.get(fn.id) != PROV_METHOD:
                return
            sql_arg = call.args[0] if call.args else None
            if sql_arg is not None:
                shape = _fold_string(sql_arg, env)
                _record_if_violation(shape, sql_arg, call, relpath, canonical_boundary, violations)
            return
        target = _resolve_callee(fn.id, relpath, funcs, import_map)
        if target is None:
            return
        trel, tfunc = target
        if trel == canonical_boundary:
            return
        key = (trel, tfunc.name)
        if key in visited:
            return
        visited.add(key)
        sub_env = _bind_params(tfunc, call, env)
        sub_prov = _bind_params_prov(tfunc, call, env, prov, import_map, relpath)
        _process_stmts(tfunc.body, sub_env, sub_prov, trel, funcs, import_map, violations,
                       canonical_boundary, depth + 1, visited, module_envs)


def _resolve_callee(name, relpath, funcs, import_map):
    if (relpath, name) in funcs:
        return (relpath, funcs[(relpath, name)])
    if (relpath, name) in import_map:
        trel, tname = import_map[(relpath, name)]
        if (trel, tname) in funcs:
            return (trel, funcs[(trel, tname)])
    return None


def _bind_params(tfunc, call, env):
    sub_env = dict(env)
    for i, p in enumerate(tfunc.args.args):
        if i < len(call.args):
            arg = call.args[i]
            v = _fold_string(arg, env)
            if isinstance(v, str):
                sub_env[p.arg] = v
            else:
                # A bound mutation method keeps its callable identity through a
                # parameter binding (``mutate(conn.execute, sql)``).
                bm = _bound_method_env(arg, env)
                sub_env[p.arg] = bm if bm is not None else UNRESOLVED
    return sub_env


def _bind_params_prov(tfunc, call, env, prov, import_map, relpath):
    sub_prov = dict(prov)
    for i, p in enumerate(tfunc.args.args):
        if i < len(call.args):
            pv = _arg_prov(call.args[i], env, prov, import_map, relpath)
            if pv is not None:
                sub_prov[p.arg] = pv
    return sub_prov


def _collect_returns(node_or_stmts, depth=0):
    """Yield ``Return`` nodes reachable from a body, skipping nested
    function/class/lambda definitions. Bounded by ``depth``."""
    if depth > 24:
        return
    stmts = node_or_stmts if isinstance(node_or_stmts, (list, tuple)) else [node_or_stmts]
    for node in stmts:
        if node is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return):
            yield node
            continue
        for child in ast.iter_child_nodes(node):
            yield from _collect_returns(child, depth + 1)


def _expr_prov(node, env, prov, relpath, funcs, import_map, module_envs,
               canonical_boundary, depth, ret_visited):
    """Summarize the Kanban provenance of an arbitrary expression.

    Returns ``(tag, bound_method)`` where ``tag`` is PROV_CONN / PROV_CURSOR /
    PROV_METHOD / PROV_KANBAN_CONNECT / None, and ``bound_method`` is a
    ``('__BOUND_METHOD__', attr)`` tuple for a bound mutation method (else
    None). Conservative: unknown / non-Kanban / conflicting expressions yield
    ``(None, None)``.
    """
    if depth > 24 or node is None:
        return (None, None)
    if isinstance(node, ast.Name):
        tag = prov.get(node.id)
        if tag in _KANBAN_OBJECT_TAGS:
            if tag == PROV_METHOD:
                bm = env.get(node.id)
                if isinstance(bm, tuple) and len(bm) == 2 and bm[0] == "__BOUND_METHOD__":
                    return (PROV_METHOD, bm)
            return (tag, None)
        if tag == PROV_KANBAN_CONNECT:
            return (PROV_KANBAN_CONNECT, None)
        return (None, None)
    if isinstance(node, ast.Attribute):
        if node.attr in MUTATION_METHODS:
            if _receiver_prov(node.value, env, prov, import_map, relpath,
                              funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
                return (PROV_METHOD, ("__BOUND_METHOD__", node.attr))
            return (None, None)
        if node.attr == "connect" and _factory_callable_prov(node, env, prov, import_map, relpath):
            return (PROV_KANBAN_CONNECT, None)
        return (None, None)
    if isinstance(node, ast.Call):
        # Opaque bound method value: ``getattr(conn, 'execute')``.
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
            attr = _fold_string(node.args[1], env)
            if isinstance(attr, str) and attr in MUTATION_METHODS:
                if _receiver_prov(node.args[0], env, prov, import_map, relpath,
                                  funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
                    return (PROV_METHOD, ("__BOUND_METHOD__", attr))
            return (None, None)
        if _is_kanban_factory(node, env, prov, import_map, relpath):
            return (PROV_CONN, None)
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "cursor":
            if _receiver_prov(fn.value, env, prov, import_map, relpath,
                              funcs, module_envs, canonical_boundary, depth, ret_visited) is not None:
                return (PROV_CURSOR, None)
            return (None, None)
        if isinstance(fn, ast.Name):
            target = _resolve_callee(fn.id, relpath, funcs, import_map)
            if target is not None:
                trel, tfunc = target
                if trel != canonical_boundary:
                    return _function_return_summary(
                        trel, tfunc, node, env, prov, relpath, funcs, import_map,
                        module_envs, canonical_boundary, depth, ret_visited)
        return (None, None)
    return (None, None)


def _definitely_returns(stmts):
    """Return True only if ``stmts`` is guaranteed to execute a ``return`` (or
    ``raise``) on every control-flow path.

    Used to detect an implicit ``None`` fall-through arm: if a function body
    can reach its end without returning, that path yields ``None`` and must
    participate in the return-summary lattice as UNKNOWN_RETURN rather than
    disappearing. The check is deliberately conservative — loops (which may run
    zero times) and ``try`` blocks (which may not return) never count as a
    guaranteed terminal, and a bare ``if`` without a guaranteed-returning
    ``else`` does not guarantee a return.
    """
    if stmts is None:
        return False
    for stmt in stmts:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If):
            if _definitely_returns(stmt.body) and _definitely_returns(stmt.orelse):
                return True
            continue
        # Assign/Expr/Pass/Import/For/While/Try/... do not guarantee a return
        # on this path; keep scanning (a loop may run zero times, a try may
        # not return, a plain statement falls through).
    return False


def _function_return_summary(trel, func, call, env, prov, caller_relpath, funcs,
                             import_map, module_envs, canonical_boundary, depth,
                             ret_visited):
    """Argument-sensitive, cycle-safe, depth-bounded return-provenance summary.

    Binds the callee's parameters to the actual call arguments, merges the
    callee module's import provenance, and merges every reachable ``return``
    expression's provenance into a total-domain lattice. Returns
    ``(tag, bound_method)``; conservative ``(None, None)`` for unknown /
    conflicting / boundary / cyclic functions, or when any reachable return arm
    is unknown / non-Kanban / bare / implicit-``None``.
    """
    if depth > 24:
        return (None, None)
    key = (trel, func.name)
    if key in ret_visited:
        return (None, None)
    # Path-local active recursion stack. The current function belongs to the
    # active call path only for THIS invocation's analysis: we copy the entry
    # stack, add the current function, and evaluate every sibling return arm
    # from its own fresh copy of that stack. Nested helper calls therefore
    # extend only their own path, and completing one sibling arm cannot mark a
    # helper as "visited" for a later sibling arm (the R4 false-positive
    # regression). True cycles and depth exhaustion still resolve to UNKNOWN
    # via the entry-stack membership check above.
    path_stack = set(ret_visited)
    path_stack.add(key)

    # Base the callee's environment on its own module globals (so it sees its
    # own ``from hermes_cli import kanban_db`` / ``from ... import connect``),
    # then bind each parameter to the caller-side argument expression.
    m_env, m_prov = (module_envs or {}).get(trel, ({}, {}))
    sub_env = dict(m_env)
    sub_prov = dict(m_prov)
    for i, p in enumerate(func.args.args):
        if i < len(call.args):
            arg = call.args[i]
            v = _fold_string(arg, env)
            if isinstance(v, str):
                sub_env[p.arg] = v
            else:
                bm = _bound_method_env(arg, env)
                if bm is not None:
                    sub_env[p.arg] = bm
            pv = _arg_prov(arg, env, prov, import_map, caller_relpath)
            if pv is not None:
                sub_prov[p.arg] = pv

    # Total-domain lattice: every reachable return arm participates. Unknown,
    # non-Kanban, cyclic, bare ``return``, and implicit ``None`` fall-through
    # arms contribute UNKNOWN_RETURN instead of being filtered out, so a
    # remaining Kanban arm can never be mistaken for unanimous.
    tags = []
    bms = []
    for ret in _collect_returns(func.body):
        if ret.value is None:
            tags.append(UNKNOWN_RETURN)
            bms.append(None)
            continue
        tag, bm = _expr_prov(ret.value, sub_env, sub_prov, trel, funcs, import_map,
                             module_envs, canonical_boundary, depth + 1, set(path_stack))
        tags.append(tag if tag is not None else UNKNOWN_RETURN)
        bms.append(bm)
    if not _definitely_returns(func.body):
        # A body that can fall off the end (or loop/try that may not return)
        # contributes an implicit ``None`` arm.
        tags.append(UNKNOWN_RETURN)
        bms.append(None)
    if not tags:
        return (None, None)
    if UNKNOWN_RETURN in tags:
        return (None, None)
    first_tag = tags[0]
    first_bm = bms[0]
    if all(t == first_tag for t in tags) and all(b == first_bm for b in bms):
        return (first_tag, first_bm)
    return (None, None)


def _record_if_violation(shape, sql_arg, call, relpath, canonical_boundary, violations):
    if relpath == canonical_boundary:
        return
    if shape is UNRESOLVED:
        violations.append(_v("FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL", sql_arg, call, relpath, None, None, [DYNAMIC]))
        return
    parsed = _classify_sql(shape)
    if parsed is None:
        return
    operation, table, columns = parsed
    if operation == "UNKNOWN_MUTATION":
        violations.append(_v("FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL", sql_arg, call, relpath, None, None, [DYNAMIC]))
        return
    if table != "tasks":
        return
    if any(DYNAMIC in (c or "") for c in columns):
        violations.append(_v("FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL", sql_arg, call, relpath, operation, table, columns))
        return
    if "status" in columns:
        violations.append(_v("FAIL_DIRECT_TASKS_STATUS_SQL", sql_arg, call, relpath, operation, table, columns))


def _v(expectation, sql_arg, call, relpath, operation, table, columns):
    line = getattr(sql_arg, "lineno", None) or getattr(call, "lineno", None)
    return {
        "expectation": expectation,
        "file": relpath,
        "line": line,
        "operation": operation,
        "table": table,
        "columns": columns,
    }


# AION R4/I01 — bounded adversarial scanner tests
# (direct tasks.status lifecycle SQL outside the canonical kanban_db boundary)
# ============================================================================

_NEG_FIXTURES = [
    {"id": "SNEG01_DIRECT_EXECUTE", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn, task_id, status):\n    conn.execute(\"UPDATE tasks SET status = ? WHERE id = ?\", (status, task_id))\n"}},
    {"id": "SNEG02_CONNECTION_ALIAS", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    db = conn\n    db.execute(\"UPDATE tasks SET status='todo'\")\n"}},
    {"id": "SNEG03_CURSOR_ALIAS", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    cur = conn.cursor()\n    other = cur\n    other.execute(\"UPDATE tasks SET status='todo'\")\n"}},
    {"id": "SNEG04_BOUND_METHOD_ALIAS", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    run = conn.execute\n    run(\"UPDATE tasks SET status='todo'\")\n"}},
    {"id": "SNEG05_MULTILINE_SQL", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    conn.execute(\"\"\"\n        UPDATE tasks\n        SET status = 'todo'\n        WHERE id = 't1'\n    \"\"\")\n"}},
    {"id": "SNEG06_ADJACENT_LITERAL_SQL", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    conn.execute(\"UPDATE tasks SET \" \"status='todo' WHERE id='t1'\")\n"}},
    {"id": "SNEG07_CONCATENATED_SQL", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    sql = \"UPDATE tasks SET \" + \"status='todo' WHERE id='t1'\"\n    conn.execute(sql)\n"}},
    {"id": "SNEG08_STATIC_FSTRING_SQL", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    column = 'status'\n    sql = f\"UPDATE tasks SET {column}='todo'\"\n    conn.execute(sql)\n"}},
    {"id": "SNEG09_STATIC_PERCENT_SQL", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    sql = \"UPDATE tasks SET %s='todo'\" % 'status'\n    conn.execute(sql)\n"}},
    {"id": "SNEG10_STATIC_FORMAT_SQL", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    sql = \"UPDATE tasks SET {}='todo'\".format('status')\n    conn.execute(sql)\n"}},
    {"id": "SNEG11_EXECUTEMANY", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn, rows):\n    conn.executemany(\"UPDATE tasks SET status=? WHERE id=?\", rows)\n"}},
    {"id": "SNEG12_EXECUTESCRIPT", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def f(conn):\n    conn.executescript(\"UPDATE tasks SET status='todo';\")\n"}},
    {"id": "SNEG13_LOCAL_WRAPPER", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "def mutate(db, sql):\n    db.execute(sql)\n\ndef f(conn):\n    mutate(conn, \"UPDATE tasks SET status='todo'\")\n"}},
    {"id": "SNEG14_IMPORTED_PROJECT_HELPER", "entrypoint": "app.f", "expectation": "FAIL_DIRECT_TASKS_STATUS_SQL",
     "virtual_files": {"app.py": "from helper import mutate\n\ndef f(conn):\n    mutate(conn, \"UPDATE tasks SET status='todo'\")\n",
                      "helper.py": "def mutate(db, sql):\n    db.execute(sql)\n"}},
    {"id": "SNEG15_UNRESOLVED_DYNAMIC_SQL", "entrypoint": "app.f", "expectation": "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL",
     "virtual_files": {"app.py": "def f(conn, sql_from_runtime):\n    conn.execute(sql_from_runtime)\n"}},
]

_POS_FIXTURES = [
    {"id": "SPOS01_MODULE_PUBLIC_WRITER", "entrypoint": "app.f", "expectation": "PASS_CONTROLLED_PUBLIC_WRITER_CALL",
     "virtual_files": {"app.py": "from hermes_cli import kanban_db\n\ndef f(conn, task_id):\n    return kanban_db.set_task_status(conn, task_id, 'todo')\n"}},
    {"id": "SPOS02_IMPORTED_PUBLIC_WRITER", "entrypoint": "app.f", "expectation": "PASS_CONTROLLED_PUBLIC_WRITER_CALL",
     "virtual_files": {"app.py": "from hermes_cli.kanban_db import set_task_status\n\ndef f(conn, task_id):\n    return set_task_status(conn, task_id, 'todo')\n"}},
    {"id": "SPOS03_CANONICAL_BOUNDARY_SQL", "entrypoint": "hermes_cli.kanban_db.set_task_status", "expectation": "PASS_EXACT_CANONICAL_BOUNDARY",
     "virtual_files": {"hermes_cli/kanban_db.py": "def set_task_status(conn, task_id, status):\n    conn.execute(\"UPDATE tasks SET status=? WHERE id=?\", (status, task_id))\n"}},
    {"id": "SPOS04_NON_STATUS_DASHBOARD_PRIORITY", "entrypoint": "app.f", "expectation": "PASS_OUTSIDE_I01_CLAIM_WITHOUT_CLOSURE",
     "virtual_files": {"app.py": "def f(conn, task_id):\n    conn.execute(\"UPDATE tasks SET priority=? WHERE id=?\", (1, task_id))\n"}},
    {"id": "SPOS05_NON_STATUS_DASHBOARD_TITLE_BODY", "entrypoint": "app.f", "expectation": "PASS_OUTSIDE_I01_CLAIM_WITHOUT_CLOSURE",
     "virtual_files": {"app.py": "def f(conn, task_id):\n    conn.execute(\"UPDATE tasks SET title=?, body=? WHERE id=?\", ('t', 'b', task_id))\n"}},
]


def _scan_fixture(fixture):
    files = dict(fixture["virtual_files"])
    return scan_status_sql(files, entrypoints=[fixture["entrypoint"]])


def test_i01_scanner_negative_fixtures():
    """All 15 named adversaries must be detected with the exact expectation."""
    for f in _NEG_FIXTURES:
        viols = _scan_fixture(f)
        assert viols, f"{f['id']} should be flagged"
        assert any(v["expectation"] == f["expectation"] for v in viols), \
            f"{f['id']}: expected {f['expectation']}, got {[v['expectation'] for v in viols]}"


def test_i01_scanner_positive_fixtures():
    """All 5 positive fixtures must pass clean (no violation)."""
    for f in _POS_FIXTURES:
        viols = _scan_fixture(f)
        assert viols == [], f"{f['id']} should be clean, got {viols}"


def test_i01_scanner_mutation_kills(tmp_path):
    """15/15 adversarial mutation kills: each negative fixture injected into a
    temporary production-path copy must fail the scanner with its fixture id."""
    for f in _NEG_FIXTURES:
        # Write virtual files under a production-like path (NOT tests/).
        for rel, src in f["virtual_files"].items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src, encoding="utf-8")
        # Scan the temporary production copy, entrypoint-scoped.
        files = {rel: (tmp_path / rel).read_text(encoding="utf-8") for rel in f["virtual_files"]}
        # Map the fixture entrypoint module -> the tmp_path relative path.
        mod = f["entrypoint"].split(".")[0]
        ep_rel = mod + ".py"
        viols = scan_status_sql(files, entrypoints=[f["entrypoint"].replace(mod, ep_rel)])
        assert viols, f"mutation {f['id']} survived: not flagged"
        assert any(v["expectation"] == f["expectation"] for v in viols), \
            f"mutation {f['id']}: expected {f['expectation']}"


def _production_py_files(repo_root):
    out = subprocess.check_output(["git", "-C", str(repo_root), "ls-files", "-z", "*.py"])
    paths = [p.decode("utf-8") for p in out.split(b"\x00") if p]
    return sorted(p for p in paths if not p.startswith("tests/"))


def test_i01_scanner_green_at_head():
    """PR-head GREEN: no direct tasks.status SQL outside the canonical boundary.

    Scans every tracked production .py file (excluding tests/). The unresolved
    mutation-capable-SQL fail-closed rule applies across the whole tracked
    universe (no dashboard-only scope), gated by proven Kanban connection
    provenance so non-Kanban sqlite3/console/terminal ``.execute`` receivers
    remain non-violating.
    """
    repo_root = Path(__file__).resolve().parents[2]
    files = {}
    for p in _production_py_files(repo_root):
        files[p] = (repo_root / p).read_text(encoding="utf-8", errors="replace")
    viols = scan_status_sql(files)
    assert viols == [], f"direct tasks.status SQL outside canonical boundary: {viols}"


def test_i01_scanner_flags_pinned_baseline_sql():
    """Pinned baseline RED: the exact two pre-relocation SQL statements are
    detected (the multiline CASE update and the child-demotion update)."""
    baseline = {
        "app.py": (
            "def f(conn, task_id, new_status):\n"
            "    cur = conn.execute(\n"
            "        \"UPDATE tasks SET status = ?, \"\n"
            "        \"  claim_lock = CASE WHEN ? = 'running' THEN claim_lock ELSE NULL END, \"\n"
            "        \"  claim_expires = CASE WHEN ? = 'running' THEN claim_expires ELSE NULL END, \"\n"
            "        \"  worker_pid = CASE WHEN ? = 'running' THEN worker_pid ELSE NULL END \"\n"
            "        \"WHERE id = ?\",\n"
            "        (new_status, new_status, new_status, new_status, task_id),\n"
            "    )\n"
            "    demoted = conn.execute(\n"
            "        \"UPDATE tasks SET status = 'todo' \"\n"
            "        \"WHERE id = ? AND status = 'ready'\",\n"
            "        (task_id,),\n"
            "    )\n"
        )
    }
    viols = scan_status_sql(baseline, entrypoints=["app.f"])
    direct = [v for v in viols if v["expectation"] == "FAIL_DIRECT_TASKS_STATUS_SQL"]
    assert len(direct) == 2, f"expected 2 direct status violations, got {viols}"
    for v in direct:
        assert v["operation"] == "UPDATE" and v["table"] == "tasks"
        assert "status" in v["columns"]


# ---------------------------------------------------------------------------
# AION R4/I01 — F1 repair: fail-closed universe + Kanban provenance regressions
# ---------------------------------------------------------------------------

def test_i01_scanner_getattr_reflection_fails_closed():
    """Opaque reflection ``getattr(conn, 'execute')(sql)`` must fail closed.

    Regression for audit counterexample OPAQUE_REFLECTION_GETATTR_EXECUTE.
    """
    files = {
        "app.py": "def f(conn, sql):\n    getattr(conn, 'execute')(sql)\n",
    }
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "getattr(conn, 'execute')(sql) must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_getattr_resolved_status_fails_direct():
    """Resolved ``getattr(conn, 'execute')('UPDATE tasks SET status=...')`` is FAIL_DIRECT."""
    files = {
        "app.py": "def f(conn):\n    getattr(conn, 'execute')(\"UPDATE tasks SET status='todo'\")\n",
    }
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert any(v["expectation"] == "FAIL_DIRECT_TASKS_STATUS_SQL" for v in viols), viols


def test_i01_scanner_unresolved_sql_fails_closed_outside_dashboard():
    """Unresolved Kanban SQL in a non-dashboard tracked file fails closed.

    Regression for audit counterexample UNRESOLVED_KANBAN_SQL_OUTSIDE_DASHBOARD:
    the previous dashboard-only dynamic scope discarded unresolved results.
    """
    files = {
        "other.py": "def f(conn, sql):\n    conn.execute(sql)\n",
    }
    viols = scan_status_sql(files, entrypoints=["other.f"])
    assert viols, "unresolved Kanban SQL outside the dashboard must fail closed"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_non_kanban_execute_nonviolating():
    """A non-Kanban ``.execute`` receiver is non-violating via proven provenance.

    Regression for audit counterexample NON_KANBAN_OBJECT_FALSE_PROVENANCE:
    ``logger.execute("UPDATE tasks SET status=...")`` is not Kanban SQL.
    """
    files = {
        "app.py": "def f(logger):\n    logger.execute(\"UPDATE tasks SET status='todo'\")\n",
    }
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols == [], f"non-Kanban logger.execute must be clean, got {viols}"


def test_i01_scanner_factory_kanban_conn_fails_closed():
    """A factory-proven Kanban connection fails closed on unresolved SQL.

    Proves the provenance model is fail-closed (not fail-open): a connection
    traced to the kanban_db factory is a real Kanban connection.
    """
    files = {
        "app.py": (
            "from hermes_cli import kanban_db\n"
            "\n"
            "def f(sql):\n"
            "    conn = kanban_db.connect()\n"
            "    conn.execute(sql)\n"
        ),
    }
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_generic_sqlite_connect_nonviolating():
    """A generic ``sqlite3.connect(...)`` connection is NOT Kanban and stays clean."""
    files = {
        "app.py": (
            "import sqlite3\n"
            "\n"
            "def f(sql):\n"
            "    conn = sqlite3.connect(':memory:')\n"
            "    conn.execute(sql)\n"
        ),
    }
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols == [], f"generic sqlite3.connect must be clean, got {viols}"


# ---------------------------------------------------------------------------
# AION R4/I01 — R2 repair: bound-method provenance survives alias/reflection/
# cursor/interprocedural/factory transport (structural, not name/pattern-based)
# ---------------------------------------------------------------------------

def _bound_method_closure_cases():
    """Bounded metamorphic family: ``(case_id, source, expect_violation)``.

    Crosses receiver source (connection parameter vs. kanban_db factory),
    method selection (attribute vs. getattr reflection), transport (direct,
    one alias, repeated alias, cursor alias, interprocedural argument) and
    mutation method (execute/executemany/executescript, cursor has no
    executescript). Kanban forms must fail closed; symmetric non-Kanban
    (logger) forms over equivalent transports stay clean.
    """
    cases = []
    for method in MUTATION_METHODS:
        cases.append((f"BM_DIRECT_{method.upper()}",
                      f"def f(conn, sql):\n    conn.{method}(sql)\n", True))
        cases.append((f"BM_ALIAS_{method.upper()}",
                      f"def f(conn, sql):\n    run = conn.{method}\n    run(sql)\n", True))
        cases.append((f"BM_ALIAS_TWICE_{method.upper()}",
                      f"def f(conn, sql):\n    run = conn.{method}\n    alias = run\n    alias(sql)\n", True))
        cases.append((f"BM_REFLECTED_{method.upper()}",
                      f"def f(conn, sql):\n    run = getattr(conn, '{method}')\n    run(sql)\n", True))
        cases.append((f"BM_REFLECTED_ALIAS_TWICE_{method.upper()}",
                      f"def f(conn, sql):\n    run = getattr(conn, '{method}')\n    alias = run\n    alias(sql)\n", True))
        cases.append((f"BM_INTERPROCEDURAL_{method.upper()}",
                      f"def mutate(run, sql):\n    run(sql)\n\ndef f(conn, sql):\n    mutate(conn.{method}, sql)\n", True))
        cases.append((f"BM_FACTORY_CHAIN_{method.upper()}",
                      f"from hermes_cli import kanban_db\n\ndef f(sql):\n    kanban_db.connect().{method}(sql)\n", True))
        if method != "executescript":
            cases.append((f"BM_CURSOR_ALIAS_TWICE_{method.upper()}",
                          f"def f(conn, sql):\n    cur = conn.cursor()\n    run = cur.{method}\n    alias = run\n    alias(sql)\n", True))
    # Symmetric non-Kanban positives (logger) over equivalent transports.
    for transport in ("ALIAS_TWICE", "REFLECTED_ALIAS_TWICE", "INTERPROCEDURAL"):
        if transport == "ALIAS_TWICE":
            src = "def f(logger, sql):\n    run = logger.execute\n    alias = run\n    alias(sql)\n"
        elif transport == "REFLECTED_ALIAS_TWICE":
            src = "def f(logger, sql):\n    run = getattr(logger, 'execute')\n    alias = run\n    alias(sql)\n"
        else:
            src = "def mutate(run, sql):\n    run(sql)\n\ndef f(logger, sql):\n    mutate(logger.execute, sql)\n"
        cases.append((f"BM_NONKANBAN_{transport}_EXECUTE", src, False))
    return cases


def test_i01_scanner_redteam_alias_of_bound_method():
    """REDTEAM_ALIAS_OF_BOUND_METHOD: ``run = conn.execute; alias = run; alias(sql)``."""
    files = {"app.py": "def f(conn, sql):\n    run = conn.execute\n    alias = run\n    alias(sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "aliased bound method must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_redteam_alias_of_reflected_bound_method():
    """REDTEAM_ALIAS_OF_REFLECTED_BOUND_METHOD: reflected bound method aliased twice."""
    files = {"app.py": "def f(conn, sql):\n    run = getattr(conn, 'execute')\n    alias = run\n    alias(sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "reflected aliased bound method must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_redteam_direct_factory_chain():
    """REDTEAM_DIRECT_FACTORY_CHAIN: ``kanban_db.connect().execute(sql)``."""
    files = {"app.py": "from hermes_cli import kanban_db\n\ndef f(sql):\n    kanban_db.connect().execute(sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "direct factory receiver chain must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_redteam_cursor_bound_alias_twice():
    """REDTEAM_CURSOR_BOUND_ALIAS_TWICE: cursor-derived bound method aliased twice."""
    files = {"app.py": "def f(conn, sql):\n    cur = conn.cursor()\n    run = cur.execute\n    alias = run\n    alias(sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "cursor-derived aliased bound method must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_redteam_interprocedural_bound_method():
    """REDTEAM_INTERPROCEDURAL_BOUND_METHOD: ``mutate(conn.execute, sql)``."""
    files = {"app.py": "def mutate(run, sql):\n    run(sql)\n\ndef f(conn, sql):\n    mutate(conn.execute, sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "interprocedural bound method must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_non_kanban_alias_of_bound_method_nonviolating():
    """Non-Kanban ``logger.execute`` aliased twice stays clean (provenance-based)."""
    files = {"app.py": "def f(logger, sql):\n    run = logger.execute\n    alias = run\n    alias(sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols == [], f"non-Kanban aliased bound method must be clean, got {viols}"


def test_i01_scanner_non_kanban_interprocedural_bound_method_nonviolating():
    """Non-Kanban ``logger.execute`` passed through a helper stays clean."""
    files = {"app.py": "def mutate(run, sql):\n    run(sql)\n\ndef f(logger, sql):\n    mutate(logger.execute, sql)\n"}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols == [], f"non-Kanban interprocedural bound method must be clean, got {viols}"


def test_i01_scanner_bound_method_transport_closure():
    """Generated/metamorphic bound-method provenance closure.

    Every Kanban bound-callable transport (direct, alias, repeated alias,
    reflection, cursor alias, interprocedural argument, factory receiver) must
    fail closed across execute/executemany/executescript; symmetric non-Kanban
    (logger) transports must remain non-violating.
    """
    for case_id, src, expect in _bound_method_closure_cases():
        viols = scan_status_sql({"app.py": src}, entrypoints=["app.f"])
        observed = bool(viols)
        assert observed == expect, (
            f"{case_id}: expected_violation={expect} observed={observed} "
            f"violations={viols}"
        )


# ============================================================================
# AION R4/I01 — R3 repair: factory callable + helper return provenance
# (structural, cycle-safe, depth-bounded, argument-sensitive)
# ============================================================================

def test_i01_scanner_factory_callable_alias_then_conn():
    """FRESH_FACTORY_CALLABLE_ALIAS_THEN_CONN: ``make = kanban_db.connect;
    conn = make(); conn.execute(sql)`` must fail closed (factory callable
    identity survives the alias)."""
    files = {"app.py": (
        "from hermes_cli import kanban_db\n\n"
        "def f(sql):\n"
        "    make = kanban_db.connect\n"
        "    conn = make()\n"
        "    conn.execute(sql)\n"
    )}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "factory callable alias must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_helper_returns_conn():
    """FRESH_HELPER_RETURNS_CONN: ``get_conn() -> kanban_db.connect();
    conn = get_conn(); conn.execute(sql)`` must fail closed."""
    files = {"app.py": (
        "from hermes_cli import kanban_db\n\n"
        "def get_conn():\n"
        "    return kanban_db.connect()\n\n"
        "def f(sql):\n"
        "    conn = get_conn()\n"
        "    conn.execute(sql)\n"
    )}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "helper-returned connection must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_helper_returns_bound_method():
    """FRESH_HELPER_RETURNS_BOUND_METHOD: ``get_run(conn) -> conn.execute;
    run = get_run(conn); run(sql)`` must fail closed."""
    files = {"app.py": (
        "def get_run(conn):\n"
        "    return conn.execute\n\n"
        "def f(conn, sql):\n"
        "    run = get_run(conn)\n"
        "    run(sql)\n"
    )}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "helper-returned bound method must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def test_i01_scanner_helper_returns_cursor():
    """FRESH_HELPER_RETURNS_CURSOR: ``get_cursor(conn) -> conn.cursor();
    cur = get_cursor(conn); cur.execute(sql)`` must fail closed."""
    files = {"app.py": (
        "def get_cursor(conn):\n"
        "    return conn.cursor()\n\n"
        "def f(conn, sql):\n"
        "    cur = get_cursor(conn)\n"
        "    cur.execute(sql)\n"
    )}
    viols = scan_status_sql(files, entrypoints=["app.f"])
    assert viols, "helper-returned cursor must be flagged"
    assert any(v["expectation"] == "FAIL_CLOSED_UNRESOLVED_DYNAMIC_SQL" for v in viols), viols


def _factory_helper_return_closure():
    """Bounded metamorphic family: ``(case_id, files, expect_violation)``.

    Crosses factory-callable aliases (one/two), helper return hops (one/two),
    direct/assigned helper results, attribute/getattr bound returns,
    conn/cursor/factory return forms, local/mapped cross-file helpers, and the
    execute family. Kanban forms fail closed; symmetric non-Kanban (sqlite /
    logger / engine) chains stay clean.
    """
    cases = []

    # --- factory callable aliases (one/two) --------------------------------
    cases.append(("FACTORY_ALIAS_ONE", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef f(sql):\n"
        "    make = kanban_db.connect\n    conn = make()\n    conn.execute(sql)\n")}, True))
    cases.append(("FACTORY_ALIAS_TWO", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef f(sql):\n"
        "    make = kanban_db.connect\n    make2 = make\n    conn = make2()\n    conn.execute(sql)\n")}, True))
    cases.append(("FACTORY_ALIAS_IMPORTED_CONNECT", {"app.py": (
        "from hermes_cli.kanban_db import connect\n\ndef f(sql):\n"
        "    make = connect\n    conn = make()\n    conn.execute(sql)\n")}, True))

    # --- helper return hops (one/two) --------------------------------------
    cases.append(("HELPER_RETURN_CONN_ONE_HOP", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef get_conn():\n"
        "    return kanban_db.connect()\n\ndef f(sql):\n"
        "    conn = get_conn()\n    conn.execute(sql)\n")}, True))
    cases.append(("HELPER_RETURN_CONN_TWO_HOP", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef inner():\n"
        "    return kanban_db.connect()\n\ndef outer():\n"
        "    return inner()\n\ndef f(sql):\n"
        "    conn = outer()\n    conn.execute(sql)\n")}, True))
    cases.append(("HELPER_RETURN_BOUND_METHOD_TWO_HOP", {"app.py": (
        "def get_run(conn):\n    return conn.execute\n\ndef wrap(conn):\n"
        "    return get_run(conn)\n\ndef f(conn, sql):\n"
        "    run = wrap(conn)\n    run(sql)\n")}, True))

    # --- direct vs assigned helper results --------------------------------
    cases.append(("HELPER_RESULT_ASSIGNED", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef get_conn():\n"
        "    return kanban_db.connect()\n\ndef f(sql):\n"
        "    conn = get_conn()\n    conn.execute(sql)\n")}, True))
    cases.append(("HELPER_RESULT_DIRECT_CHAINED", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef get_conn():\n"
        "    return kanban_db.connect()\n\ndef f(sql):\n"
        "    get_conn().execute(sql)\n")}, True))

    # --- attribute / getattr bound returns --------------------------------
    cases.append(("HELPER_RETURN_BOUND_METHOD_ATTR", {"app.py": (
        "def get_run(conn):\n    return conn.execute\n\ndef f(conn, sql):\n"
        "    run = get_run(conn)\n    run(sql)\n")}, True))
    cases.append(("HELPER_RETURN_BOUND_METHOD_GETATTR", {"app.py": (
        "def get_run(conn):\n    return getattr(conn, 'execute')\n\ndef f(conn, sql):\n"
        "    run = get_run(conn)\n    run(sql)\n")}, True))

    # --- conn / cursor / factory return forms -----------------------------
    cases.append(("HELPER_RETURN_CURSOR", {"app.py": (
        "def get_cursor(conn):\n    return conn.cursor()\n\ndef f(conn, sql):\n"
        "    cur = get_cursor(conn)\n    cur.execute(sql)\n")}, True))
    cases.append(("HELPER_RETURN_FACTORY", {"app.py": (
        "from hermes_cli import kanban_db\n\ndef get_factory():\n"
        "    return kanban_db.connect\n\ndef f(sql):\n"
        "    make = get_factory()\n    conn = make()\n    conn.execute(sql)\n")}, True))

    # --- mapped cross-file helper -----------------------------------------
    cases.append(("HELPER_CROSS_FILE_CONN", {
        "app.py": "from helper import get_conn\n\ndef f(sql):\n    conn = get_conn()\n    conn.execute(sql)\n",
        "helper.py": "from hermes_cli import kanban_db\n\ndef get_conn():\n    return kanban_db.connect()\n"}, True))
    cases.append(("HELPER_CROSS_FILE_CURSOR", {
        "app.py": "from helper import get_cursor\n\ndef f(conn, sql):\n    cur = get_cursor(conn)\n    cur.execute(sql)\n",
        "helper.py": "def get_cursor(conn):\n    return conn.cursor()\n"}, True))

    # --- execute family where applicable -----------------------------------
    for method in ("executemany", "executescript"):
        cases.append((f"HELPER_RETURN_BOUND_METHOD_{method.upper()}", {"app.py": (
            f"def get_run(conn):\n    return conn.{method}\n\ndef f(conn, sql, rows):\n"
            f"    run = get_run(conn)\n    run(sql, rows)\n")}, True))

    # --- symmetric non-Kanban (sqlite / logger / engine) -------------------
    cases.append(("NONKANBAN_SQLITE_FACTORY_ALIAS", {"app.py": (
        "import sqlite3\n\ndef f(sql):\n"
        "    make = sqlite3.connect\n    conn = make(':memory:')\n    conn.execute(sql)\n")}, False))
    cases.append(("NONKANBAN_SQLITE_HELPER_RETURN", {"app.py": (
        "import sqlite3\n\ndef get_conn():\n    return sqlite3.connect(':memory:')\n\ndef f(sql):\n"
        "    conn = get_conn()\n    conn.execute(sql)\n")}, False))
    cases.append(("NONKANBAN_LOGGER_HELPER_RETURN_BOUND_METHOD", {"app.py": (
        "def get_run(logger):\n    return logger.execute\n\ndef f(logger, sql):\n"
        "    run = get_run(logger)\n    run(sql)\n")}, False))
    cases.append(("NONKANBAN_ENGINE_HELPER_RETURN", {"app.py": (
        "def get_conn(engine):\n    return engine.connect()\n\ndef f(engine, sql):\n"
        "    conn = get_conn(engine)\n    conn.execute(sql)\n")}, False))

    return cases


def test_i01_scanner_factory_helper_return_closure():
    """Generated/metamorphic factory + helper-return provenance closure.

    Every Kanban factory/helper-return transport must fail closed across one/
    two aliases, one/two hops, direct/assigned results, attribute/getattr
    bound returns, conn/cursor/factory forms, local/mapped cross-file helpers,
    and the execute family; symmetric non-Kanban sqlite/logger/engine chains
    must remain non-violating.
    """
    for case_id, files, expect in _factory_helper_return_closure():
        viols = scan_status_sql(files, entrypoints=["app.f"])
        observed = bool(viols)
        assert observed == expect, (
            f"{case_id}: expected_violation={expect} observed={observed} "
            f"violations={viols}"
        )


def test_i01_scanner_helper_return_cycle_terminates():
    """Cycle-safe termination: a self-recursive and a mutually-recursive
    helper must terminate without infinite recursion and stay conservative
    (no false Kanban provenance from an unresolvable cycle)."""
    self_rec = {"app.py": (
        "def get_conn():\n    return get_conn()\n\ndef f(sql):\n"
        "    conn = get_conn()\n    conn.execute(sql)\n")}
    mut_rec = {"app.py": (
        "def a():\n    return b()\n\ndef b():\n    return a()\n\ndef f(sql):\n"
        "    conn = a()\n    conn.execute(sql)\n")}
    for files in (self_rec, mut_rec):
        viols = scan_status_sql(files, entrypoints=["app.f"])
        assert viols == [], f"cyclic helper return must stay clean, got {viols}"


def test_i01_scanner_helper_return_depth_bounded():
    """Depth-bounded termination: a 30-hop helper chain terminates cleanly
    (the depth guard truncates analysis; no infinite recursion)."""
    lines = ["def f(sql):\n    conn = h0()\n    conn.execute(sql)\n"]
    for i in range(30):
        nxt = f"h{i + 1}()" if i < 29 else "kanban_db.connect()"
        if i == 29:
            header = "from hermes_cli import kanban_db\n\n"
        else:
            header = ""
        lines.insert(0, f"{header}def h{i}():\n    return {nxt}\n")
    src = "\n".join(lines)
    viols = scan_status_sql({"app.py": src}, entrypoints=["app.f"])
    # Terminated (no exception); the >24-depth tail is conservatively unresolved.
    assert isinstance(viols, list)


# ============================================================================
# AION R4/I01 — R4 repair: total-domain return-summary lattice
# (unknown / non-Kanban / cyclic / bare / implicit-None arms participate)
# ============================================================================

def _return_lattice_closure():
    """Bounded return-lattice family: ``(case_id, files, expect_violation)``.

    Every reachable return arm of a summarized helper participates. A known
    Kanban provenance tag is returned only when the return-arm set is nonempty
    and every arm resolves to the exact same known tag; any unknown /
    non-Kanban / cyclic / bare / implicit-``None`` arm, or any tag
    disagreement, yields no proven Kanban provenance (clean). Two same-known
    Kanban arms still propagate (fail-closed).
    """
    cases = []

    # --- exact R3 audit reproductions (must stay clean) --------------------
    cases.append(("AUDIT_MIXED_KANBAN_OR_UNKNOWN_OBJECT_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return object()\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))
    cases.append(("AUDIT_MIXED_KANBAN_OR_SQLITE_MUST_STAY_CLEAN", {
        "app.py": (
            "import sqlite3\n"
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return sqlite3.connect(':memory:')\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))
    cases.append(("AUDIT_RECURSIVE_OR_KANBAN_MIXED_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return choose(flag)\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))

    # --- bounded controls: mixed with explicit / bare / implicit None -------
    cases.append(("MIXED_KANBAN_OR_EXPLICIT_NONE_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return None\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))
    cases.append(("MIXED_KANBAN_OR_BARE_RETURN_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))
    cases.append(("MIXED_KANBAN_OR_IMPLICIT_NONE_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))

    # --- bounded controls: mixed with scalar / conflicting known tags -------
    cases.append(("MIXED_KANBAN_OR_SCALAR_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return 0\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))
    cases.append(("CONFLICTING_KNOWN_TAGS_MUST_STAY_CLEAN", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag, conn):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return conn.cursor()\n\n"
            "def f(flag, conn, sql):\n"
            "    selected = choose(flag, conn)\n"
            "    selected.execute(sql)\n"
        ),
    }, False))

    # --- two same-known Kanban arms still propagate (fail-closed) -----------
    cases.append(("TWO_SAME_KNOWN_KANBAN_ARMS_MUST_STILL_PROPAGATE", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return kanban_db.connect()\n"
            "    return kanban_db.connect()\n\n"
            "def f(flag, sql):\n"
            "    selected = choose(flag)\n"
            "    selected.execute(sql)\n"
        ),
    }, True))

    # --- repeated same-wrapper arms must still propagate (path-local) -------
    # The R4 regression: a shared recursion stack let the first ``make()`` arm
    # mark ``make`` as visited, so the second sibling ``make()`` arm was
    # misclassified as a cycle and dropped to UNKNOWN. With path-local stacks
    # every sibling arm is evaluated from the same clean entry stack.
    cases.append(("TWO_SAME_WRAPPER_ARMS_MUST_STILL_PROPAGATE", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def make():\n"
            "    return kanban_db.connect()\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return make()\n"
            "    return make()\n\n"
            "def f(flag, sql):\n"
            "    choose(flag).execute(sql)\n"
        ),
    }, True))
    cases.append(("TWO_SAME_TWO_HOP_WRAPPER_ARMS_MUST_STILL_PROPAGATE", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def make():\n"
            "    return kanban_db.connect()\n\n"
            "def wrap():\n"
            "    return make()\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return wrap()\n"
            "    return wrap()\n\n"
            "def f(flag, sql):\n"
            "    choose(flag).execute(sql)\n"
        ),
    }, True))
    cases.append(("THREE_SAME_WRAPPER_ARMS_MUST_STILL_PROPAGATE", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def make():\n"
            "    return kanban_db.connect()\n\n"
            "def choose(flag):\n"
            "    if flag == 1:\n"
            "        return make()\n"
            "    if flag == 2:\n"
            "        return make()\n"
            "    return make()\n\n"
            "def f(flag, sql):\n"
            "    choose(flag).execute(sql)\n"
        ),
    }, True))
    cases.append(("SAME_HELPER_INDEPENDENT_BRANCHES_MUST_STILL_PROPAGATE", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def make():\n"
            "    return kanban_db.connect()\n\n"
            "def left():\n"
            "    return make()\n\n"
            "def right():\n"
            "    return make()\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return left()\n"
            "    return right()\n\n"
            "def f(flag, sql):\n"
            "    choose(flag).execute(sql)\n"
        ),
    }, True))
    cases.append(("DIFFERENT_ACYCLIC_WRAPPERS_SAME_TAG_MUST_STILL_PROPAGATE", {
        "app.py": (
            "from hermes_cli import kanban_db\n\n"
            "def make_a():\n"
            "    return kanban_db.connect()\n\n"
            "def make_b():\n"
            "    return kanban_db.connect()\n\n"
            "def choose(flag):\n"
            "    if flag:\n"
            "        return make_a()\n"
            "    return make_b()\n\n"
            "def f(flag, sql):\n"
            "    choose(flag).execute(sql)\n"
        ),
    }, True))

    return cases


def test_i01_scanner_return_lattice_closure():
    """Total-domain return-summary lattice: mixed / unknown / non-Kanban /
    cyclic / bare / implicit-None arms never yield proven Kanban provenance;
    two same-known Kanban arms still propagate."""
    for case_id, files, expect in _return_lattice_closure():
        viols = scan_status_sql(files, entrypoints=["app.f"])
        observed = bool(viols)
        assert observed == expect, (
            f"{case_id}: expected_violation={expect} observed={observed} "
            f"violations={viols}"
        )


# ============================================================================
# AION R4/I01 — 44-row transaction / parity / fault matrix
# ============================================================================

class _Fault(Exception):
    """Sentinel raised by the fault injector."""


def _snapshot(conn):
    def rows(t, o):
        return [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY {o}").fetchall()]
    return (
        rows("tasks", "id"),
        rows("task_runs", "id"),
        rows("task_events", "id"),
        rows("task_links", "parent_id, child_id"),
    )


def _inject(conn, when, substr, occ=1):
    """Monkeypatch conn.execute to raise _Fault before/after the Nth statement
    whose SQL text contains ``substr``. Returns the original execute."""
    original = conn.execute
    state = {"n": 0}

    def patched(sql, *a, **k):
        s = sql if isinstance(sql, str) else str(sql)
        matched = substr in s
        if when == "before" and matched:
            state["n"] += 1
            if state["n"] == occ:
                raise _Fault(f"{when}:{substr}:{occ}")
        r = original(sql, *a, **k)
        if when == "after" and matched:
            state["n"] += 1
            if state["n"] == occ:
                raise _Fault(f"{when}:{substr}:{occ}")
        return r

    conn.execute = patched
    return original


def _mk_task(conn, title="t", status="todo"):
    tid = kb.create_task(conn, title=title)
    if status != "ready":
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
    return tid


def _mk_run(conn, task_id):
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at) "
        "VALUES (?, ?, 'running', ?)",
        (task_id, "test", int(time.time())),
    )
    run_id = cur.lastrowid
    conn.execute(
        "UPDATE tasks SET status = 'running', current_run_id = ? WHERE id = ?",
        (run_id, task_id),
    )
    return run_id


def _task(conn, task_id):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def _events(conn, task_id):
    return conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
    ).fetchall()


def _new_conn(kanban_home):
    return kb.connect()


# ---------------------------------------------------------------------------
# Parity rows (writer-level semantics)
# ---------------------------------------------------------------------------

def test_P01_todo_to_ready_no_parent(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "todo")
        assert kb.set_task_status(conn, tid, "ready") is True
        t = _task(conn, tid)
        assert t["status"] == "ready"
        assert t["claim_lock"] is None and t["claim_expires"] is None and t["worker_pid"] is None
        evs = _events(conn, tid)
        status_evs = [e for e in evs if e["kind"] == "status"]
        assert len(status_evs) == 1
    finally:
        conn.close()


def test_P02_triage_to_todo(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "triage")
        assert kb.set_task_status(conn, tid, "todo") is True
        assert _task(conn, tid)["status"] == "todo"
    finally:
        conn.close()


def test_P03_ready_to_triage(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "ready")
        assert kb.set_task_status(conn, tid, "triage") is True
        assert _task(conn, tid)["status"] == "triage"
    finally:
        conn.close()


def test_P04_same_status_todo(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "todo")
        before = len([e for e in _events(conn, tid) if e["kind"] == "status"])
        assert kb.set_task_status(conn, tid, "todo") is True
        assert _task(conn, tid)["status"] == "todo"
        after = len([e for e in _events(conn, tid) if e["kind"] == "status"])
        assert after == before + 1  # same-status is not a no-op
    finally:
        conn.close()


def test_P05_same_status_ready(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "ready")
        before = len([e for e in _events(conn, tid) if e["kind"] == "status"])
        assert kb.set_task_status(conn, tid, "ready") is True
        assert _task(conn, tid)["status"] == "ready"
        after = len([e for e in _events(conn, tid) if e["kind"] == "status"])
        assert after == before + 1
    finally:
        conn.close()


def test_P06_running_to_ready_reclaim(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "running")
        run_id = _mk_run(conn, tid)
        assert kb.set_task_status(conn, tid, "ready") is True
        t = _task(conn, tid)
        assert t["status"] == "ready"
        assert t["current_run_id"] is None
        assert t["claim_lock"] is None and t["claim_expires"] is None and t["worker_pid"] is None
        run = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert run["status"] == "reclaimed" and run["outcome"] == "reclaimed"
        assert run["ended_at"] is not None
        evs = [e for e in _events(conn, tid) if e["kind"] == "status"]
        assert len(evs) == 1 and evs[0]["run_id"] == run_id
    finally:
        conn.close()


def test_P07_running_to_todo_reclaim(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "running")
        run_id = _mk_run(conn, tid)
        assert kb.set_task_status(conn, tid, "todo") is True
        assert _task(conn, tid)["status"] == "todo"
        run = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert run["status"] == "reclaimed"
    finally:
        conn.close()


def test_P08_ready_allowed_by_done_parent(kanban_home):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "done")
        child = _mk_task(conn, "c", "todo")
        kb.link_tasks(conn, parent, child)
        assert kb.set_task_status(conn, child, "ready") is True
        assert _task(conn, child)["status"] == "ready"
    finally:
        conn.close()


def test_P09_ready_refused_by_archived_parent(kanban_home):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "archived")
        child = _mk_task(conn, "c", "todo")
        kb.link_tasks(conn, parent, child)
        assert kb.set_task_status(conn, child, "ready") is False
        assert _task(conn, child)["status"] == "todo"
    finally:
        conn.close()


def test_P10_ready_refused_by_non_done_parent(kanban_home):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "running")
        child = _mk_task(conn, "c", "todo")
        kb.link_tasks(conn, parent, child)
        assert kb.set_task_status(conn, child, "ready") is False
        assert _task(conn, child)["status"] == "todo"
    finally:
        conn.close()


def test_P11_done_parent_reopen_to_todo(kanban_home):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "done")
        child_a = _mk_task(conn, "ca", "ready")
        child_b = _mk_task(conn, "cb", "todo")
        kb.link_tasks(conn, parent, child_a)
        kb.link_tasks(conn, parent, child_b)
        assert kb.set_task_status(conn, parent, "todo") is True
        assert _task(conn, parent)["status"] == "todo"
        assert _task(conn, child_a)["status"] == "todo"  # ready child demoted
        assert _task(conn, child_b)["status"] == "todo"  # non-ready unchanged
    finally:
        conn.close()


def test_P12_archived_parent_reopen_to_triage(kanban_home):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "archived")
        child_a = _mk_task(conn, "ca", "ready")
        child_b = _mk_task(conn, "cb", "ready")
        child_c = _mk_task(conn, "cc", "blocked")
        for c in (child_a, child_b, child_c):
            kb.link_tasks(conn, parent, c)
        assert kb.set_task_status(conn, parent, "triage") is True
        assert _task(conn, parent)["status"] == "triage"
        assert _task(conn, child_a)["status"] == "todo"
        assert _task(conn, child_b)["status"] == "todo"
        assert _task(conn, child_c)["status"] == "blocked"
    finally:
        conn.close()


def test_P13_done_parent_reopen_to_ready(kanban_home):
    conn = kb.connect()
    try:
        grandparent = _mk_task(conn, "gp", "done")
        parent = _mk_task(conn, "p", "done")
        kb.link_tasks(conn, grandparent, parent)
        child = _mk_task(conn, "c", "ready")
        kb.link_tasks(conn, parent, child)
        assert kb.set_task_status(conn, parent, "ready") is True
        assert _task(conn, parent)["status"] == "ready"
        assert _task(conn, child)["status"] == "todo"  # demoted, not re-promoted
    finally:
        conn.close()


def test_P14_missing_task(kanban_home):
    conn = kb.connect()
    try:
        assert kb.set_task_status(conn, "t_nonexistent", "todo") is False
    finally:
        conn.close()


def test_P15_archived_parent_blocks_ready_even_though_recompute_accepts(kanban_home):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "archived")
        child = _mk_task(conn, "c", "triage")
        kb.link_tasks(conn, parent, child)
        # Direct helper refuses ready on archived parent; recompute_ready would
        # accept archived, so this pins the divergence.
        assert kb.set_task_status(conn, child, "ready") is False
        assert _task(conn, child)["status"] == "triage"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pre-commit fault rows (22/22 full-snapshot rollback equality)
# ---------------------------------------------------------------------------

_RUNNING_READY_FAULTS = [
    ("F-RUNNING-READY-01", "before", "UPDATE tasks SET status = ?", 1),
    ("F-RUNNING-READY-02", "after", "UPDATE tasks SET status = ?", 1),
    ("F-RUNNING-READY-03", "before", "UPDATE task_runs", 1),
    ("F-RUNNING-READY-04", "after", "UPDATE task_runs", 1),
    ("F-RUNNING-READY-05", "before", "current_run_id = NULL", 1),
    ("F-RUNNING-READY-06", "after", "current_run_id = NULL", 1),
    ("F-RUNNING-READY-07", "before", "INSERT INTO task_events (task_id, run_id", 1),
    ("F-RUNNING-READY-08", "after", "INSERT INTO task_events (task_id, run_id", 1),
    ("F-RUNNING-READY-09", "before", "COMMIT", 1),
]

_REOPEN_FAULTS = [
    ("F-REOPEN-TWO-CHILDREN-01", "before", "UPDATE tasks SET status = ?", 1),
    ("F-REOPEN-TWO-CHILDREN-02", "after", "UPDATE tasks SET status = ?", 1),
    ("F-REOPEN-TWO-CHILDREN-03", "before", "INSERT INTO task_events (task_id, run_id", 1),
    ("F-REOPEN-TWO-CHILDREN-04", "after", "INSERT INTO task_events (task_id, run_id", 1),
    ("F-REOPEN-TWO-CHILDREN-05", "before", "UPDATE tasks SET status = 'todo'", 1),
    ("F-REOPEN-TWO-CHILDREN-06", "after", "UPDATE tasks SET status = 'todo'", 1),
    ("F-REOPEN-TWO-CHILDREN-07", "before", "INSERT INTO task_events (task_id, kind", 1),
    ("F-REOPEN-TWO-CHILDREN-08", "after", "INSERT INTO task_events (task_id, kind", 1),
    ("F-REOPEN-TWO-CHILDREN-09", "before", "UPDATE tasks SET status = 'todo'", 2),
    ("F-REOPEN-TWO-CHILDREN-10", "after", "UPDATE tasks SET status = 'todo'", 2),
    ("F-REOPEN-TWO-CHILDREN-11", "before", "INSERT INTO task_events (task_id, kind", 2),
    ("F-REOPEN-TWO-CHILDREN-12", "after", "INSERT INTO task_events (task_id, kind", 2),
    ("F-REOPEN-TWO-CHILDREN-13", "before", "COMMIT", 1),
]


@pytest.mark.parametrize("row_id,when,substr,occ", _RUNNING_READY_FAULTS)
def test_precommit_fault_running_ready(kanban_home, row_id, when, substr, occ):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "running")
        _mk_run(conn, tid)
        before = _snapshot(conn)
        original = _inject(conn, when, substr, occ)
        with pytest.raises(_Fault):
            kb.set_task_status(conn, tid, "ready")
        conn.execute = original
        after = _snapshot(conn)
        assert before == after, f"{row_id}: snapshot must be byte/row-value equal"
    finally:
        conn.close()


@pytest.mark.parametrize("row_id,when,substr,occ", _REOPEN_FAULTS)
def test_precommit_fault_reopen_two_children(kanban_home, row_id, when, substr, occ):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "done")
        child_a = _mk_task(conn, "ca", "ready")
        child_b = _mk_task(conn, "cb", "ready")
        kb.link_tasks(conn, parent, child_a)
        kb.link_tasks(conn, parent, child_b)
        before = _snapshot(conn)
        original = _inject(conn, when, substr, occ)
        with pytest.raises(_Fault):
            kb.set_task_status(conn, parent, "todo")
        conn.execute = original
        after = _snapshot(conn)
        assert before == after, f"{row_id}: snapshot must be byte/row-value equal"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Post-commit recompute failure rows (2/2 non-atomic boundary)
# ---------------------------------------------------------------------------

def test_PC01_update_ready_recompute_failure(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        parent = _mk_task(conn, "p", "done")
        tid = _mk_task(conn, "a", "todo")
        kb.link_tasks(conn, parent, tid)

        def boom(*a, **k):
            raise _Fault("recompute boom")

        monkeypatch.setattr(kb, "recompute_ready", boom)
        with pytest.raises(_Fault):
            kb.set_task_status(conn, tid, "ready")
        # Controlled-writer commit is NOT undone by recompute failure.
        assert _task(conn, tid)["status"] == "ready"
        evs = [e for e in _events(conn, tid) if e["kind"] == "status"]
        assert len(evs) == 1  # status event committed
    finally:
        conn.close()


def test_PC02_bulk_ready_recompute_failure_continues(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        p1 = _mk_task(conn, "p1", "done")
        p2 = _mk_task(conn, "p2", "done")
        first = _mk_task(conn, "first", "todo")
        second = _mk_task(conn, "second", "todo")
        kb.link_tasks(conn, p1, first)
        kb.link_tasks(conn, p2, second)

        calls = {"n": 0}
        real = kb.recompute_ready

        def flaky(conn_arg, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Fault("first recompute boom")
            return real(conn_arg, *a, **k)

        monkeypatch.setattr(kb, "recompute_ready", flaky)
        with pytest.raises(_Fault):
            kb.set_task_status(conn, first, "ready")
        assert _task(conn, first)["status"] == "ready"  # first committed despite recompute failure
        assert kb.set_task_status(conn, second, "ready") is True  # second independently succeeds
        assert _task(conn, second)["status"] == "ready"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bulk partial rows (5/5 ordered partial-result semantics)
# ---------------------------------------------------------------------------

def test_B01_ordered_partial_status_results(kanban_home):
    conn = kb.connect()
    try:
        ok_a = _mk_task(conn, "ok-a", "todo")
        refused = _mk_task(conn, "refused", "todo")
        blocker = _mk_task(conn, "blocker", "running")
        kb.link_tasks(conn, blocker, refused)
        ok_b = _mk_task(conn, "ok-b", "triage")

        results = []
        for tid in [ok_a, "t_missing", refused, ok_b]:
            results.append(kb.set_task_status(conn, tid, "ready"))
        assert results == [True, False, False, True]
        assert _task(conn, ok_a)["status"] == "ready"
        assert _task(conn, refused)["status"] == "todo"  # unchanged
        assert _task(conn, ok_b)["status"] == "ready"
    finally:
        conn.close()


def test_B02_precommit_failure_rolls_back_one_id_only(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        faulted = _mk_task(conn, "faulted", "running")
        _mk_run(conn, faulted)
        healthy = _mk_task(conn, "healthy", "todo")
        before_faulted = _snapshot(conn)

        real = kb.set_task_status

        def flaky(c, tid, status):
            if tid == faulted:
                # raise inside the controlled writer before commit
                original = c.execute
                def patched(sql, *a, **k):
                    if "UPDATE tasks SET status = ?" in (sql if isinstance(sql, str) else str(sql)):
                        raise _Fault("boom")
                    return original(sql, *a, **k)
                c.execute = patched
                try:
                    return real(c, tid, status)
                finally:
                    c.execute = original
            return real(c, tid, status)

        monkeypatch.setattr(kb, "set_task_status", flaky)
        with pytest.raises(_Fault):
            kb.set_task_status(conn, faulted, "ready")
        # faulted id rolled back, healthy id still succeeds independently
        assert kb.set_task_status(conn, healthy, "ready") is True
        assert _task(conn, healthy)["status"] == "ready"
        assert _task(conn, faulted)["status"] == "running"  # unchanged
    finally:
        conn.close()


def test_B03_status_success_later_priority_failure(kanban_home, monkeypatch):
    """Within one bulk id, a committed status survives a later priority fault."""
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "todo")
        real = kb.set_task_status

        def status_then_boom(c, task_id, status):
            ok = real(c, task_id, status)  # status commits
            original = c.execute
            def patched(sql, *a, **k):
                if "UPDATE tasks SET priority" in (sql if isinstance(sql, str) else str(sql)):
                    raise _Fault("priority boom")
                return original(sql, *a, **k)
            c.execute = patched
            try:
                with kb.write_txn(c):
                    c.execute("UPDATE tasks SET priority = ? WHERE id = ?", (5, task_id))
            finally:
                c.execute = original
            return ok

        monkeypatch.setattr(kb, "set_task_status", status_then_boom)
        with pytest.raises(_Fault):
            kb.set_task_status(conn, tid, "ready")
        assert _task(conn, tid)["status"] == "ready"  # status committed
        assert _task(conn, tid)["priority"] != 5  # priority rollback
    finally:
        conn.close()


def test_B04_status_refused_priority_still_applies(kanban_home):
    conn = kb.connect()
    try:
        blocker = _mk_task(conn, "blocker", "running")
        tid = _mk_task(conn, "a", "todo")
        kb.link_tasks(conn, blocker, tid)
        # status refused by non-done parent, but priority is independent
        assert kb.set_task_status(conn, tid, "ready") is False
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET priority = ? WHERE id = ?", (7, tid))
        assert _task(conn, tid)["status"] == "todo"  # unchanged
        assert _task(conn, tid)["priority"] == 7  # priority applied
    finally:
        conn.close()


def test_B05_duplicate_ids_are_separate_iterations(kanban_home):
    conn = kb.connect()
    try:
        tid = _mk_task(conn, "a", "todo")
        before = len([e for e in _events(conn, tid) if e["kind"] == "status"])
        results = [kb.set_task_status(conn, tid, "todo"), kb.set_task_status(conn, tid, "todo")]
        assert results == [True, True]
        after = len([e for e in _events(conn, tid) if e["kind"] == "status"])
        assert after == before + 2  # two status events, input not deduplicated
    finally:
        conn.close()
