"""Typed non-PR PASS evidence on the Native review-verdict path.

The commit-bound route (repository/PR/head/tree/base + GitHub APPROVED review)
stays the ONLY path for code/PR audits.  A host-configuration readback audit has
no repository or PR, so it attests the SAME role-separated lifecycle with a
closed, narrowly typed evidence shape:

    {"audit_type": "host_config_readback",
     "artifacts": [{"path": "<abs>", "sha256": "<64-hex>"}, ...]}

Every artifact must exist under the audit task's workspace and hash-match.
Malformed / mixed / missing / out-of-root / hash-drift / duplicate / wrong-role
/ code-PR-handoff shapes must all fail closed with zero mutation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


_APPROVED_EVIDENCE = {
    "repository": "kiddhu/hermes-agent", "pr": 98,
    "head": "a" * 40, "tree": "b" * 40, "base": "c" * 40,
    "github_review_id": 123,
    "github_review_url": "https://github.com/kiddhu/hermes-agent/pull/98#pullrequestreview-123",
    "github_review_state": "APPROVED",
}
_APPROVED_HANDOFF = json.dumps({
    "version": 1,
    "candidate": {
        key: _APPROVED_EVIDENCE[key] for key in ("repository", "pr", "head", "tree", "base")
    },
    "summary": "candidate frozen",
}, sort_keys=True, separators=(",", ":"))

_PROSE_HANDOFF = (
    "GPT-6 Astra three-profile binding applied and runtime-readback verified: "
    "gm/gm2=openai-codex/gpt-6-astra/low, elder-senate=medium; zero secret/"
    "restart/control-plane. Independent audit by bafuxunan."
)


def _write_artifact(ws, name, data):
    p = ws / name
    p.write_bytes(data)
    return str(p), hashlib.sha256(data).hexdigest()


def _non_pr_evidence(ws, **files):
    artifacts = []
    for name, data in files.items():
        path, sha = _write_artifact(ws, name, data)
        artifacts.append({"path": path, "sha256": sha})
    return {"audit_type": "host_config_readback", "artifacts": artifacts}


def _non_pr_review_pair(conn, tmp_path, *, reason=_PROSE_HANDOFF, same_assignee=False):
    author = kb.create_task(conn, title="host config binding", assignee="author")
    claimed = kb.claim_task(conn, author)
    assert claimed is not None
    review_task = kb.create_task(
        conn,
        title="independent readback audit",
        assignee="author" if same_assignee else "auditor",
        parents=[author],
    )
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?", (str(ws), review_task))
    conn.commit()
    run = kb.latest_run(conn, author)
    assert run is not None
    return author, int(run.id), review_task, ws


def _dump(conn) -> str:
    return "\n".join(conn.iterdump())


# ---------------------------------------------------------------------------
# Validator unit tests (RED/GREEN at the smallest boundary)
# ---------------------------------------------------------------------------

def test_pr_evidence_validator_still_rejects_non_pr_shape():
    """The commit-bound route is unchanged: non-PR shape is not PR evidence."""
    assert kb._canonical_audit_evidence({
        "audit_type": "host_config_readback", "artifacts": [],
    }) is None


def test_non_pr_validator_accepts_exact_hashed_artifacts(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    evidence = _non_pr_evidence(ws, a=b"hello", b=b"world")
    normalized = kb._canonical_non_pr_audit_evidence(evidence, artifact_root=str(ws))
    assert normalized is not None
    assert normalized["audit_type"] == "host_config_readback"
    paths = [a["path"] for a in normalized["artifacts"]]
    assert paths == sorted(paths)
    assert all(a["sha256"] for a in normalized["artifacts"])


@pytest.mark.parametrize("mutate", ["hash_drift", "missing", "out_of_root", "empty", "dup", "bad_type", "extra_key", "bad_sha", "not_abs"])
def test_non_pr_validator_rejects_malformed(tmp_path, mutate):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence = _non_pr_evidence(ws, a=b"hello")

    if mutate == "hash_drift":
        evidence["artifacts"][0]["sha256"] = "0" * 64
    elif mutate == "missing":
        evidence["artifacts"][0]["path"] = str(ws / "nope.txt")
    elif mutate == "out_of_root":
        p = outside / "x.txt"
        p.write_bytes(b"x")
        evidence["artifacts"][0]["path"] = str(p)
    elif mutate == "empty":
        evidence["artifacts"] = []
    elif mutate == "dup":
        evidence["artifacts"] = [evidence["artifacts"][0], dict(evidence["artifacts"][0])]
    elif mutate == "bad_type":
        evidence["audit_type"] = "code_audit"
    elif mutate == "extra_key":
        evidence["extra"] = "x"
    elif mutate == "bad_sha":
        evidence["artifacts"][0]["sha256"] = "xyz"
    elif mutate == "not_abs":
        evidence["artifacts"][0]["path"] = "relative/path.txt"

    assert kb._canonical_non_pr_audit_evidence(evidence, artifact_root=str(ws)) is None


def test_non_pr_validator_rejects_mixed_pr_and_non_pr_fields(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    evidence = _non_pr_evidence(ws, a=b"hello")
    mixed = dict(evidence)
    mixed["repository"] = "kiddhu/hermes-agent"
    assert kb._canonical_non_pr_audit_evidence(mixed, artifact_root=str(ws)) is None
    assert kb._canonical_audit_evidence(mixed) is None


# ---------------------------------------------------------------------------
# End-to-end review-verdict path
# ---------------------------------------------------------------------------

def test_non_pr_pass_terminalizes_and_authenticates(kanban_home, tmp_path):
    with kb.connect() as conn:
        author, run_id, review_task, ws = _non_pr_review_pair(conn, tmp_path)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_PROSE_HANDOFF,
        )
        claim = kb.claim_task(conn, review_task)
        assert claim is not None and claim.current_run_id is not None
        audit_run = claim.current_run_id
        evidence = _non_pr_evidence(ws, AUDIT_R1=b"report", readback=b"result")

        assert kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="independent readback PASS", evidence=evidence,
        )
        assert kb.get_task(conn, author).status == "review"
        assert kb.get_task(conn, review_task).status == "done"
        run = kb.latest_run(conn, review_task)
        assert (run.status, run.outcome) == ("done", "completed")
        outcome = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='canonical_audit_outcome'",
            (review_task,),
        ).fetchone()["payload"])
        assert outcome["evidence"]["audit_type"] == "host_config_readback"
        assert kb._canonical_audit_receipt(conn, author)["authenticated"] is True


def test_non_pr_pass_is_idempotent_on_exact_replay(kanban_home, tmp_path):
    with kb.connect() as conn:
        author, run_id, review_task, ws = _non_pr_review_pair(conn, tmp_path)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_PROSE_HANDOFF,
        )
        audit_run = kb.claim_task(conn, review_task).current_run_id
        evidence = _non_pr_evidence(ws, AUDIT_R1=b"report")
        assert kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="independent readback PASS", evidence=evidence,
        )
        committed = _dump(conn)
        assert kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="independent readback PASS", evidence=evidence,
        )
        assert _dump(conn) == committed


def test_non_pr_pass_uses_factory_kernel_terminal_path(kanban_home, tmp_path, monkeypatch):
    with kb.connect() as conn:
        author, run_id, review_task, ws = _non_pr_review_pair(conn, tmp_path)
        conn.execute("UPDATE tasks SET factory_build_gate=1 WHERE id=?", (review_task,))
        conn.commit()
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_PROSE_HANDOFF,
        )
        audit_run = kb.claim_task(conn, review_task).current_run_id
        evidence = _non_pr_evidence(ws, AUDIT_R1=b"report")
        called = []

        def fake_kernel(c, task_id, kernel_run_id, **kwargs):
            called.append((task_id, kernel_run_id))
            cur = kb._execute_factory_terminal_write(
                c, task_id,
                "UPDATE tasks SET status='done', result=?, completed_at=1, "
                "factory_terminal_receipt_sha256=? WHERE id=? AND current_run_id=?",
                (kwargs["result"], "d" * 64, task_id, int(kernel_run_id)),
            )
            assert cur.rowcount == 1
            return {"bound": True}

        monkeypatch.setattr(kb, "_run_kernel_finalizer", fake_kernel)
        assert kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="independent readback PASS", evidence=evidence,
        )
        assert called == [(review_task, str(audit_run))]
        assert kb._canonical_audit_receipt(conn, author)["authenticated"] is True


def test_code_pr_audit_cannot_select_non_pr_variant(kanban_home, tmp_path):
    """A structured PR candidate handoff must stay on the commit-bound route."""
    with kb.connect() as conn:
        author, run_id, review_task, ws = _non_pr_review_pair(conn, tmp_path)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_APPROVED_HANDOFF,  # structured PR candidate handoff
        )
        audit_run = kb.claim_task(conn, review_task).current_run_id
        evidence = _non_pr_evidence(ws, AUDIT_R1=b"report")
        before = _dump(conn)
        assert not kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="sneaky non-PR PASS", evidence=evidence,
        )
        assert _dump(conn) == before


@pytest.mark.parametrize("mutate", ["hash_drift", "missing", "out_of_root", "empty", "bad_type", "mixed"])
def test_non_pr_pass_rejects_bad_evidence_without_mutation(kanban_home, tmp_path, mutate):
    with kb.connect() as conn:
        author, run_id, review_task, ws = _non_pr_review_pair(conn, tmp_path)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_PROSE_HANDOFF,
        )
        audit_run = kb.claim_task(conn, review_task).current_run_id
        evidence = _non_pr_evidence(ws, AUDIT_R1=b"report")

        if mutate == "hash_drift":
            evidence["artifacts"][0]["sha256"] = "0" * 64
        elif mutate == "missing":
            evidence["artifacts"][0]["path"] = str(ws / "missing.txt")
        elif mutate == "out_of_root":
            outside = tmp_path / "outside"
            outside.mkdir()
            p = outside / "x.txt"
            p.write_bytes(b"x")
            evidence["artifacts"][0]["path"] = str(p)
        elif mutate == "empty":
            evidence["artifacts"] = []
        elif mutate == "bad_type":
            evidence["audit_type"] = "code_audit"
        elif mutate == "mixed":
            evidence["repository"] = "kiddhu/hermes-agent"

        before = _dump(conn)
        assert not kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="bad evidence", evidence=evidence,
        )
        assert _dump(conn) == before
        assert kb.get_task(conn, review_task).status == "running"


def test_non_pr_pass_requires_role_separation(kanban_home, tmp_path):
    with kb.connect() as conn:
        author, run_id, review_task, ws = _non_pr_review_pair(
            conn, tmp_path, same_assignee=True,
        )
        # Same-assignee child can't even reach a valid review handoff.
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_PROSE_HANDOFF,
        ) is None


def test_pr_evidence_route_unchanged(kanban_home):
    """Regression: the commit-bound PR route still terminalizes as before."""
    with kb.connect() as conn:
        author = kb.create_task(conn, title="implementation", assignee="author")
        kb.claim_task(conn, author)
        review_task = kb.create_task(
            conn, title="independent audit", assignee="auditor", parents=[author],
        )
        run_id = int(kb.latest_run(conn, author).id)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=run_id, review_task_id=review_task,
            reason=_APPROVED_HANDOFF,
        )
        audit_run = kb.claim_task(conn, review_task).current_run_id
        assert kb.record_review_verdict(
            conn, author, review_task_id=review_task,
            expected_review_run_id=audit_run, verdict="pass",
            reason="exact head approved", evidence=_APPROVED_EVIDENCE,
        )
        assert kb.get_task(conn, review_task).status == "done"
        assert kb._canonical_audit_receipt(conn, author)["authenticated"] is True
