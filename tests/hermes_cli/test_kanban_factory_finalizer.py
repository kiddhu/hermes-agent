"""AION-889 I1+I2 — kernel-owned atomic bind-and-terminalize finalizer tests.

These prove the Elder-approved architecture A (ATOMIC KERNEL-OWNED BIND-AND-
TERMINALIZE) end-to-end against the real ``hermes_cli.kanban_db`` runtime, in a
clean isolated board DB (never the live aion-factory board).

The finalizer SHA-pin-loads the frozen aion-governance kernel/adapters/binder
from ``AION_GOVERNANCE_SOURCE_DIR``; set that env (or the ``aion_gov_src``
fixture) to the aion-governance checkout whose modules match the pinned hashes
in ``kanban_db``. Tests that require the pinned source skip cleanly when it is
not configured or does not match.

T1 RED  gate=1 / no receipt / finalizer disabled -> FAIL_CLOSED zero mutation
T2 GREEN non-merge running task -> done + receipt bound + r1..r8 + C1..C10
        all true + child wakes, uploaded_by=aion_monarch_proof_kernel
T3 RED  worker-forged 'agent' uploader rejected (provenance)
T4 RED  stale run / CAS miss -> FAIL_CLOSED zero mutation
T5 RED  cross-task receipt replay rejected (r4 task/run mismatch)
T6 RED  fault injection at a write boundary -> rollback, retry idempotent
T7 GREEN merge-bearing pre-bound receipt path unchanged
T8 GREEN already-terminal path unchanged
T10     ALTERNATE_SUCCESS_PATHS stays 0 (no alternate success path)
T11 HOSTILE subprocess: worker/gateway kanban_db module-hash equality + guard
        present + finalizer path succeeds; no live service used
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest

from hermes_cli import kanban_db as kb


_PR947_AUTHORITY_COMMIT = "256bf3e56869d7219b5712d45066e96ce4ec2c6d"
_PR947_MODULES = {
    "scripts/aion_monarch_outcome_proof_gate.py": (
        "249a0fde7d5f9dfc4f3a2fc00d3f2c691041720f9215fac7b61269d4b55962f2"
    ),
    "scripts/aion_monarch_typed_adapters.py": (
        "dd411bb7151837b45271151e2190cd95c405831773b7f0c591f9d824d5ea0327"
    ),
    "scripts/aion_monarch_receipt_binder.py": (
        "b477aea2afe6eb3c23f778dc347be552e0d28b081b597559d9da1bb7841c004c"
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + fully rebound Native Kanban env pins.

    Rebinds ``HERMES_KANBAN_*`` path pins and clears the board/worker pins via
    :func:`kanban_db.isolated_kanban_env` so ``connect()`` and the
    attachment/event/board paths can never resolve to the live aion-factory
    board (the AION-RL2-CORE-01-R10 synthetic-residue class). Setting only
    ``HERMES_HOME`` is insufficient: the dispatcher injects
    ``HERMES_KANBAN_HOME`` / ``HERMES_KANBAN_BOARD`` with higher precedence.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with kb.isolated_kanban_env(tmp_path):
        kb.init_db()
        yield home


def _materialize_pr947_git_object(tmp_path: Path) -> Path | None:
    """Materialize PR #947's immutable merge object without reading its worktree."""
    repo = Path(os.environ.get("HOME", "")) / "aion-governance"
    if not repo.is_dir():
        return None
    dest = tmp_path / _PR947_AUTHORITY_COMMIT
    for rel_path, expected_sha in _PR947_MODULES.items():
        proc = subprocess.run(
            ["git", "-C", str(repo), "show", f"{_PR947_AUTHORITY_COMMIT}:{rel_path}"],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0 or hashlib.sha256(proc.stdout).hexdigest() != expected_sha:
            return None
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(proc.stdout)
    return dest


def _aion_gov_source_dir(tmp_path: Path | None = None) -> Path | None:
    """Resolve only a byte-exact PR #947 authority source for integration tests."""
    if tmp_path is not None:
        immutable = _materialize_pr947_git_object(tmp_path)
        if immutable is not None:
            return immutable
    raw = os.environ.get("AION_GOVERNANCE_SOURCE_DIR")
    if raw and Path(raw).is_dir():
        return Path(raw)
    # Fall back to a sibling checkout next to the hermes-agent clone.
    repo_root = Path(kb.__file__).resolve().parents[1]
    sibling = repo_root.parent / "aion-governance"
    if (sibling / "scripts" / "aion_monarch_outcome_proof_gate.py").is_file():
        return sibling
    return None


@pytest.fixture
def aion_gov_src(monkeypatch, tmp_path):
    """Point the finalizer at immutable PR #947 bytes, or skip if unavailable."""
    src = _aion_gov_source_dir(tmp_path)
    if src is None:
        pytest.skip("AION_GOVERNANCE_SOURCE_DIR not configured")
    monkeypatch.setenv("AION_GOVERNANCE_SOURCE_DIR", str(src))
    for rel_path, expected_sha in _PR947_MODULES.items():
        module_bytes = (src / rel_path).read_bytes()
        if hashlib.sha256(module_bytes).hexdigest() != expected_sha:
            pytest.skip(f"aion-governance module {rel_path} does not match PR #947")
    return src


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim_and_run_id(conn, task_id) -> int:
    kb.claim_task(conn, task_id)
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert row and row["current_run_id"] is not None
    return int(row["current_run_id"])


def _record_legacy_review_verdict_fixture(
    conn,
    task_id,
    *,
    review_task_id,
    expected_review_run_id,
    verdict,
    reason,
    **_superseded_recovery_fields,
):
    """Seed historical v1 PASS rows without invoking current write authority."""
    if verdict != "pass":
        return kb.record_review_verdict(
            conn,
            task_id,
            review_task_id=review_task_id,
            expected_review_run_id=expected_review_run_id,
            verdict=verdict,
            reason=reason,
        )
    payload = {
        "version": 1,
        "review_task_id": review_task_id,
        "review_run_id": expected_review_run_id,
        "verdict": verdict,
        "reason": reason,
    }
    with kb.write_txn(conn):
        kb._append_event(
            conn, task_id, "review_verdict", payload,
            run_id=expected_review_run_id,
        )
        kb._append_event(
            conn, review_task_id, "review_verdict", payload,
            run_id=expected_review_run_id,
        )
    return True


def _bound_receipt_doc(conn, task_id) -> dict:
    atts = kb.list_attachments(conn, task_id)
    receipts = [a for a in atts if a.filename == "aion_monarch_receipt.json"]
    assert receipts, "no bound receipt attachment"
    return json.loads(Path(receipts[-1].stored_path).read_bytes().decode("utf-8"))


_FROZEN_CONDITION_KEYS = tuple(f"C{i}" for i in range(1, 11))
_FROZEN_TRUSTED_FIELDS = (
    "adapter_type_and_version", "exact_source_refs", "target_identity",
    "before_digest", "action_receipt_ref", "after_digest",
    "head_epoch_or_run_binding", "acquired_at",
)


def _assert_receipt_passes_r1_r8(doc: dict, task_id: str, run_id: str) -> None:
    assert doc["schema"] == "aion.monarch.trusted_receipt.v1"          # r1
    assert doc["verdict"] == "OUTCOME_ACCEPTED"                         # r2
    assert doc["conditions"] == {k: True for k in _FROZEN_CONDITION_KEYS}  # r3
    tid = doc["target_identity"].get("task_id") or doc["target_identity"]["fields"]["task_id"]
    assert str(tid) == str(task_id)                                     # r4 (task)
    binding = doc["head_epoch_or_run_binding"]
    run_ref = binding.get("value") or binding.get("run_id")
    assert str(run_ref) == str(run_id)                                  # r4 (run)
    assert doc["contract_hash_sha256"] == kb.FACTORY_CONTRACT_HASH_SHA256  # r5
    assert isinstance(doc.get("kernel_version"), str) and doc["kernel_version"].strip()  # r6
    for f in _FROZEN_TRUSTED_FIELDS:                                    # r7
        assert f in doc, f"missing trusted_receipt_binding field {f}"
    action = doc["action_receipt_ref"]
    assert action["actor_identity_source"] not in {"", "self", "self_declared"}  # r8
    assert action["actor_role"] == "action_executor"


def _kernel_receipt_doc(task_id: str, run_id: str) -> dict:
    """Build a kernel-shaped receipt (all r1..r8 fields present)."""
    return {
        "schema": "aion.monarch.trusted_receipt.v1",
        "verdict": "OUTCOME_ACCEPTED",
        "kernel_version": "aion.monarch.proof_kernel.v2",
        "contract_hash_sha256": kb.FACTORY_CONTRACT_HASH_SHA256,
        "conditions": {f"C{i}": True for i in range(1, 11)},
        "adapter_type_and_version": "aion.monarch.typed_adapter.task_terminal.v1",
        "exact_source_refs": {"task_id": task_id, "run_id": run_id},
        "target_identity": {
            "object_type": "kanban_task_run",
            "object_ref_exact": f"{task_id}/{run_id}",
            "fields": {"task_id": task_id, "run_id": run_id},
        },
        "before_digest": "a" * 64,
        "action_receipt_ref": {
            "action_kind": "task_terminal",
            "actor": "aion_monarch_proof_kernel",
            "actor_role": "action_executor",
            "actor_identity_source": "native_task_run_authorization_binding",
            "executed_effect_ref": "task_events:status=done",
            "executed_at": "2026-08-18T06:00:01Z",
        },
        "after_digest": "b" * 64,
        "head_epoch_or_run_binding": {
            "bound_to": "task_run_id",
            "value": run_id,
            "authorization_source_ref": "native_task_run_authorization_binding",
            "authorization_epoch_or_version": 1,
        },
        "acquired_at": "2026-08-18T06:00:02Z",
    }


# ---------------------------------------------------------------------------
# PR #947 immutable authority and source-drift fence
# ---------------------------------------------------------------------------

def test_pr947_authority_is_exact_and_mutable_checkout_is_not_a_default():
    assert kb.AION_GOVERNANCE_AUTHORITY_PR == 947
    assert kb.AION_GOVERNANCE_AUTHORITY_HEAD == (
        "339312bcdc9794fe996f28aadb3309a1d46b3b4e"
    )
    assert kb.AION_GOVERNANCE_AUTHORITY_COMMIT == _PR947_AUTHORITY_COMMIT
    assert kb.AION_GOVERNANCE_KERNEL_SHA256 == _PR947_MODULES[
        "scripts/aion_monarch_outcome_proof_gate.py"
    ]
    assert kb.AION_GOVERNANCE_TYPED_ADAPTERS_SHA256 == _PR947_MODULES[
        "scripts/aion_monarch_typed_adapters.py"
    ]
    assert kb.AION_GOVERNANCE_RECEIPT_BINDER_SHA256 == _PR947_MODULES[
        "scripts/aion_monarch_receipt_binder.py"
    ]
    assert all(
        _PR947_AUTHORITY_COMMIT in source_dir
        for source_dir in kb.AION_GOVERNANCE_DEFAULT_SOURCE_DIRS
    )
    assert "/root/aion-governance" not in kb.AION_GOVERNANCE_DEFAULT_SOURCE_DIRS


def test_pr947_module_drift_fails_closed_with_zero_mutation(
    kanban_home, aion_gov_src, tmp_path, monkeypatch,
):
    drifted = tmp_path / "drifted-pr947"
    for rel_path in _PR947_MODULES:
        target = drifted / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((aion_gov_src / rel_path).read_bytes())
    binder_path = drifted / "scripts" / "aion_monarch_receipt_binder.py"
    binder_path.write_bytes(binder_path.read_bytes() + b"\n# injected drift\n")
    monkeypatch.setenv("AION_GOVERNANCE_SOURCE_DIR", str(drifted))

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="drift-fenced", factory_build_gate=1, assignee="agent007",
        )
        run_id = _claim_and_run_id(conn, task_id)
        status_before = kb.get_task(conn, task_id).status
        events_before = [event.kind for event in kb.list_events(conn, task_id)]

        with pytest.raises(
            kb.FactoryTerminalReceiptRequiredError,
            match="aion_monarch_receipt_binder.py sha256 mismatch",
        ):
            kb.complete_task(conn, task_id, result="must roll back", expected_run_id=run_id)

        row = conn.execute(
            "SELECT status, factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == status_before == "running"
        assert row["factory_terminal_receipt_sha256"] is None
        assert kb.list_attachments(conn, task_id) == []
        assert [event.kind for event in kb.list_events(conn, task_id)] == events_before
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_terminal_write_grants"
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Loader regression — pinned single-read loader must register the module in
# ``sys.modules`` BEFORE ``exec``.
#
# The PR #947 binder (``scripts/aion_monarch_receipt_binder.py``) is a
# ``@dataclass(frozen=True)`` module with ``from __future__ import
# annotations``. During ``exec``, ``dataclasses._is_type`` resolves string
# field annotations via ``sys.modules.get(cls.__module__).__dict__``. When the
# pinned loader exec's the module BEFORE registering it in ``sys.modules``,
# that lookup returns ``None`` and dataclass raises
# ``AttributeError: 'NoneType' object has no attribute '__dict__'``, blocking
# TASK_TERMINAL finalization. The loader must register the module under its
# pinned name before ``exec`` so self-referential dataclass annotations
# resolve. This is independent of the authority pin values: any
# ``@dataclass(frozen=True)`` + ``from __future__ import annotations`` module
# hits the same path.
# ---------------------------------------------------------------------------

def test_pinned_loader_registers_module_before_exec_for_dataclass_annotations(
    tmp_path,
):
    source = (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class CanonicalGitHubRepository:\n"
        "    owner: str\n"
        "    name: str\n"
        "\n"
        "    @property\n"
        "    def full_name(self) -> str:\n"
        "        return f'{self.owner}/{self.name}'\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class CanonicalImplementationIdentity:\n"
        "    repository: CanonicalGitHubRepository\n"
        "    pull_request_number: int\n"
        "    merge_commit_sha: str\n"
    )
    rel = "synth_receipt_binder.py"
    (tmp_path / rel).write_bytes(source.encode("utf-8"))
    expected_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    module_name = "scripts.aion_synth_receipt_binder_regression"

    module = kb._load_pinned_aion_module(tmp_path, rel, expected_sha, module_name)

    # The loader registered the module under its pinned name BEFORE exec, so
    # dataclass could resolve its own forward-reference annotations.
    assert sys.modules.get(module_name) is module
    repo_cls = getattr(module, "CanonicalGitHubRepository")
    identity_cls = getattr(module, "CanonicalImplementationIdentity")
    identity = identity_cls(
        repository=repo_cls(owner="kiddhu", name="hermes-agent"),
        pull_request_number=947,
        merge_commit_sha="0" * 40,
    )
    assert identity.repository.full_name == "kiddhu/hermes-agent"
    assert identity.pull_request_number == 947


# ---------------------------------------------------------------------------
# T1 RED — finalizer disabled -> FAIL_CLOSED zero mutation
# ---------------------------------------------------------------------------

def test_t1_finalizer_disabled_fail_closed_zero_mutation(kanban_home, aion_gov_src, monkeypatch):
    monkeypatch.setenv("AION_FACTORY_FINALIZER_ENABLED", "0")
    with kb.connect() as conn:
        t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
        run_id = _claim_and_run_id(conn, t)
        status_before = kb.get_task(conn, t).status
        events_before = [e.kind for e in kb.list_events(conn, t)]

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        # Zero mutation.
        assert kb.get_task(conn, t).status == status_before == "running"
        assert [e.kind for e in kb.list_events(conn, t)] == events_before
        row = conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["factory_terminal_receipt_sha256"] is None
        assert kb.list_attachments(conn, t) == []


# ---------------------------------------------------------------------------
# T2 GREEN — non-merge running task completes atomically, child wakes
# ---------------------------------------------------------------------------

def test_t2_finalizer_completes_atomically_child_wakes(kanban_home, aion_gov_src):
    with kb.connect() as conn:
        parent = kb.create_task(
            conn, title="factory parent", factory_build_gate=1, assignee="agent007",
        )
        child = kb.create_task(
            conn, title="child", assignee="agent007", parents=[parent],
        )
        assert kb.get_task(conn, child).status == "todo"
        run_id = _claim_and_run_id(conn, parent)

        assert kb.complete_task(conn, parent, result="kernel done", expected_run_id=run_id)

        # Terminal write + receipt binding all happened atomically.
        t = kb.get_task(conn, parent)
        assert t.status == "done"
        assert t.result == "kernel done"
        row = conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        sha = row["factory_terminal_receipt_sha256"]
        assert sha and len(sha) == 64

        # Receipt attachment stamped with the trusted kernel identity.
        atts = kb.list_attachments(conn, parent)
        receipt = [a for a in atts if a.filename == "aion_monarch_receipt.json"]
        assert receipt and receipt[0].uploaded_by == "aion_monarch_proof_kernel"

        # The bound sha == sha256 of the receipt attachment bytes (r5/attachment).
        doc_bytes = Path(receipt[0].stored_path).read_bytes()
        assert hashlib.sha256(doc_bytes).hexdigest() == sha

        # r1..r8 + C1..C10 all true.
        doc = json.loads(doc_bytes.decode("utf-8"))
        _assert_receipt_passes_r1_r8(doc, parent, str(run_id))
        assert doc["conditions"] == {k: True for k in _FROZEN_CONDITION_KEYS}

        # Child wakes (dependency promotion).
        assert kb.get_task(conn, child).status == "ready"


def _authorized_detached_controller_chain(
    conn, monkeypatch, *, child_parent_satisfied=True,
):
    """Build the machine-authenticated action shape from t_420d4177/run3500."""
    repair = kb.create_task(
        conn, title="authenticated controller repair", factory_build_gate=1,
        assignee="gm2",
    )
    repair_run = _claim_and_run_id(conn, repair)
    assert kb.complete_task(
        conn, repair, expected_run_id=repair_run,
        summary="controller repair authenticated",
    )
    upstream = kb.create_task(conn, title="satisfied upstream", assignee="gm2")
    upstream_run = _claim_and_run_id(conn, upstream)
    assert kb.complete_task(conn, upstream, expected_run_id=upstream_run)

    parent = kb.create_task(
        conn, title="authorized activation", factory_build_gate=1, assignee="gm",
        parents=[repair, upstream],
    )
    source_run = _claim_and_run_id(conn, parent)
    assert kb.block_task(
        conn, parent, reason="prior approval boundary", kind="needs_input",
        expected_run_id=source_run,
    )
    assert kb.unblock_task(conn, parent)
    conn.execute(
        "UPDATE tasks SET block_recurrences = 0 WHERE id = ?", (parent,),
    )
    conn.commit()
    action_run = _claim_and_run_id(conn, parent)

    other_parent = kb.create_task(conn, title="satisfied child parent", assignee="gm2")
    if child_parent_satisfied:
        other_run = _claim_and_run_id(conn, other_parent)
        assert kb.complete_task(conn, other_parent, expected_run_id=other_run)
    child = kb.create_task(
        conn, title="natural dependent", assignee="elder-senate",
        parents=[parent, other_parent],
    )

    monkeypatch.setattr(kb, "AION889_ATOMIC_FINALIZER_TASK_ID", parent)
    monkeypatch.setattr(kb, "AION889_ATOMIC_FINALIZER_CHILD_ID", child)
    monkeypatch.setattr(kb, "AION889_ATOMIC_FINALIZER_REPAIR_PARENT_ID", repair)
    monkeypatch.setattr(kb, "AION889_ATOMIC_FINALIZER_SOURCE_RUN_ID", source_run)
    monkeypatch.setattr(kb, "AION889_ATOMIC_FINALIZER_ACTION_RUN_ID", action_run)
    with kb.write_txn(conn):
        kb._append_event(
            conn, parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND,
            kb._aion889_atomic_finalizer_event_payload(), run_id=action_run,
        )
    assert kb.block_task(
        conn, parent, reason="goal judge circular postcondition", kind="needs_input",
        expected_run_id=action_run,
    )
    monkeypatch.setattr(
        kb,
        "_read_aion889_atomic_finalizer_runtime",
        lambda _service: {
            "ActiveState": "active", "SubState": "running", "Result": "success",
            "MainPID": "930317", "ExecMainStartTimestamp": "fresh", "NRestarts": "0",
        },
        raising=False,
    )
    return parent, action_run, child


def _controlled_no_product_blocked_closeout_chain(conn):
    from hermes_cli.aion_889_preflight import (
        ACCEPTANCE_CONTRACT_SHA256,
        CHECKER_VERSION,
        FACTORY_PREFLIGHT_RECEIPT_SCHEMA,
        REQUIREMENTS_SHA256,
        SOLUTION_CONTRACT_SHA256,
        compute_scope_identity_sha256,
    )

    task_id = kb.create_task(
        conn,
        title="generic controlled no-product fixture",
        assignee="agent007",
        factory_build_gate=1,
        factory_directive_id="AION-889-PREFLIGHT-V1-IMPLEMENT",
    )
    scope_hash = compute_scope_identity_sha256(
        directive_id="AION-889-PREFLIGHT-V1-IMPLEMENT",
        risk_tier="T2_CORE_OR_HIGH_RISK",
        requirements_sha256=REQUIREMENTS_SHA256,
        solution_sha256=SOLUTION_CONTRACT_SHA256,
        acceptance_sha256=ACCEPTANCE_CONTRACT_SHA256,
    )
    receipt = {
        "schema": FACTORY_PREFLIGHT_RECEIPT_SCHEMA,
        "task_id": task_id,
        "risk_tier": "T2_CORE_OR_HIGH_RISK",
        "outcome_requirement_sha256": REQUIREMENTS_SHA256,
        "solution_contract_sha256": SOLUTION_CONTRACT_SHA256,
        "solution_challenge_verdict": "PASS_SOLUTION_CHALLENGE",
        "acceptance_contract_sha256": ACCEPTANCE_CONTRACT_SHA256,
        "acceptance_challenge_verdict": "PASS_ACCEPTANCE_CHALLENGE",
        "test_first_evidence": "3" * 64,
        "preflight_checker_version": CHECKER_VERSION,
        "preflight_verdict": "PASS",
        "scope_identity_sha256": scope_hash,
        "checker_output_sha256": "4" * 64,
    }
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    attachment_id = kb.store_attachment_bytes(
        conn, task_id, "aion_factory_preflight_receipt.json", raw,
        content_type="application/json",
        uploaded_by="aion_monarch_proof_kernel",
    )
    kb.bind_factory_preflight_receipt(conn, task_id, attachment_id)
    claimed = kb.claim_task(conn, task_id, claimer="controlled-no-product")
    assert claimed is not None and claimed.current_run_id is not None
    run_id = int(claimed.current_run_id)
    assert kb.block_task(
        conn,
        task_id,
        reason=kb.FACTORY_CONTROLLED_NO_PRODUCT_CLOSEOUT_REASON,
        kind="capability",
        expected_run_id=run_id,
    )
    return task_id, run_id


def test_detached_controller_finalizes_and_recomputes_child_atomically(
    kanban_home, aion_gov_src, monkeypatch,
):
    """A consumed action receipt closes its parent before natural child wake."""
    with kb.connect() as conn:
        parent, action_run, child = _authorized_detached_controller_chain(conn, monkeypatch)
        parent_row = kb.get_task(conn, parent)
        child_row = kb.get_task(conn, child)
        assert parent_row is not None and parent_row.status == "blocked"
        assert child_row is not None and child_row.status == "todo"

        assert kb.complete_task(conn, parent, summary="authorized action completed")

        parent_row = kb.get_task(conn, parent)
        assert parent_row is not None and parent_row.status == "done"
        child_row = kb.get_task(conn, child)
        assert child_row is not None
        assert child_row.status == "ready"
        assert child_row.current_run_id is None
        completed = conn.execute(
            "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (parent,),
        ).fetchone()
        assert completed is not None and completed["run_id"] == action_run


def test_controlled_no_product_blocked_closeout_is_generic_and_read_only(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        task_id, run_id = _controlled_no_product_blocked_closeout_chain(conn)
        before = _native_state_snapshot(conn)

        assert kb._controlled_no_product_blocked_closeout_run_id(conn, task_id) == run_id
        assert kb._detached_controller_finalizer_run_id(conn, task_id) == run_id
        assert _native_state_snapshot(conn) == before


def test_controlled_no_product_blocked_closeout_finalizes_through_existing_kernel(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        task_id, run_id = _controlled_no_product_blocked_closeout_chain(conn)

        assert kb.complete_task(
            conn, task_id, summary="controlled no-product canary closeout",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "done"
        completed = conn.execute(
            "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchone()
        assert completed is not None and completed["run_id"] == run_id


@pytest.mark.parametrize(
    "drift",
    [
        "wrong_profile", "wrong_block_reason", "missing_preflight", "extra_edge",
        "extra_run", "malformed_decision", "duplicate_pass", "later_claim",
        "active_identity", "wrong_family_block_kind",
    ],
)
def test_controlled_no_product_blocked_closeout_hostile_drift_zero_mutation(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        task_id, run_id = _controlled_no_product_blocked_closeout_chain(conn)
        if drift == "wrong_profile":
            conn.execute("UPDATE tasks SET assignee = 'gm2' WHERE id = ?", (task_id,))
        elif drift == "wrong_block_reason":
            conn.execute(
                "UPDATE task_runs SET summary = 'ordinary capability block' WHERE id = ?",
                (run_id,),
            )
        elif drift == "missing_preflight":
            conn.execute(
                "DELETE FROM task_attachments WHERE task_id = ? "
                "AND filename = 'aion_factory_preflight_receipt.json'",
                (task_id,),
            )
        elif drift == "extra_edge":
            child = kb.create_task(conn, title="unexpected product", assignee="agent007")
            conn.execute(
                "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
                (task_id, child),
            )
        elif drift == "extra_run":
            conn.execute(
                "INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'agent007', 'blocked', 'blocked', 1, 1)",
                (task_id,),
            )
        elif drift == "malformed_decision":
            conn.execute(
                "UPDATE task_events SET payload = '{}' WHERE task_id = ? "
                "AND kind = 'preflight_decision'", (task_id,),
            )
        elif drift == "duplicate_pass":
            row = conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'preflight_decision'", (task_id,),
            ).fetchone()
            kb._append_event(conn, task_id, "preflight_decision", json.loads(row["payload"]))
        elif drift == "later_claim":
            kb._append_event(
                conn, task_id, "claimed", {"run_id": run_id}, run_id=run_id,
            )
        elif drift == "active_identity":
            conn.execute(
                "UPDATE tasks SET current_run_id = ?, claim_lock = 'live' WHERE id = ?",
                (run_id, task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (task_id,),
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._controlled_no_product_blocked_closeout_run_id(conn, task_id) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, task_id, summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


def test_detached_controller_requires_frozen_terminal_action_run(
    kanban_home, aion_gov_src, monkeypatch,
):
    """A valid receipt chain cannot authorize any run except the frozen action."""
    with kb.connect() as conn:
        parent, action_run, _child = _authorized_detached_controller_chain(
            conn, monkeypatch,
        )
        assert kb._aion889_atomic_finalizer_run_id(conn, parent) == action_run

        monkeypatch.setattr(
            kb, "AION889_ATOMIC_FINALIZER_ACTION_RUN_ID", action_run + 1,
        )
        assert kb._aion889_atomic_finalizer_run_id(conn, parent) is None


def test_detached_controller_rejects_later_run_receipt_substitution(
    kanban_home, aion_gov_src, monkeypatch,
):
    """A later blocked GM run cannot replace the frozen terminal action run."""
    with kb.connect() as conn:
        parent, action_run, _child = _authorized_detached_controller_chain(
            conn, monkeypatch,
        )
        assert kb.unblock_task(conn, parent)
        conn.execute(
            "UPDATE tasks SET block_recurrences = 0 WHERE id = ?", (parent,),
        )
        conn.commit()
        later_run = _claim_and_run_id(conn, parent)
        assert later_run > action_run
        conn.execute(
            "UPDATE task_events SET run_id = ? WHERE task_id = ? AND kind = ?",
            (later_run, parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND),
        )
        conn.commit()
        assert kb.block_task(
            conn, parent, reason="copied receipt on later run",
            kind="needs_input", expected_run_id=later_run,
        )
        before = _native_state_snapshot(conn)

        assert kb._aion889_atomic_finalizer_run_id(conn, parent) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, parent, summary="reject later substitution")
        assert _native_state_snapshot(conn) == before


def test_detached_controller_preserves_todo_when_another_child_parent_is_unsatisfied(
    kanban_home, aion_gov_src, monkeypatch,
):
    with kb.connect() as conn:
        parent, action_run, child = _authorized_detached_controller_chain(
            conn, monkeypatch, child_parent_satisfied=False,
        )

        assert kb.complete_task(conn, parent, summary="authorized action completed")

        parent_row = kb.get_task(conn, parent)
        assert parent_row is not None and parent_row.status == "done"
        child_row = kb.get_task(conn, child)
        assert child_row is not None
        assert child_row.status == "todo"
        assert child_row.current_run_id is None
        assert conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'promoted'",
            (child,),
        ).fetchone() is None
        completed = conn.execute(
            "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (parent,),
        ).fetchone()
        assert completed is not None and completed["run_id"] == action_run


def _native_state_snapshot(conn):
    tables = (
        "tasks", "task_links", "task_comments", "task_events",
        "task_runs", "task_attachments",
    )
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
        for table in tables
    }


@pytest.mark.parametrize(
    "drift",
    [
        "tampered_event", "duplicate_event", "active_run", "wrong_child",
        "child_status", "stale_source_run", "unauthenticated_parent", "stale_runtime",
        "null_event_run", "conflicting_event_run", "unsupported_status",
        "unsatisfied_upstream", "unauthorized_actor", "prose_only",
    ],
)
def test_detached_controller_drift_fails_closed_with_zero_mutation(
    kanban_home, aion_gov_src, monkeypatch, drift,
):
    with kb.connect() as conn:
        parent, action_run, child = _authorized_detached_controller_chain(conn, monkeypatch)
        if drift == "tampered_event":
            conn.execute(
                "UPDATE task_events SET payload = '{}' WHERE task_id = ? AND kind = ?",
                (parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND),
            )
        elif drift == "duplicate_event":
            kb._append_event(
                conn, parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND,
                kb._aion889_atomic_finalizer_event_payload(), run_id=action_run,
            )
        elif drift == "active_run":
            conn.execute(
                "UPDATE tasks SET current_run_id = ?, claim_lock = 'stale-owner' WHERE id = ?",
                (action_run, parent),
            )
        elif drift == "wrong_child":
            monkeypatch.setattr(kb, "AION889_ATOMIC_FINALIZER_CHILD_ID", "t_deadbeef")
        elif drift == "child_status":
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        elif drift == "stale_source_run":
            conn.execute(
                "UPDATE task_runs SET status = 'done', outcome = 'completed' "
                "WHERE id = ?", (kb.AION889_ATOMIC_FINALIZER_SOURCE_RUN_ID,),
            )
        elif drift == "unauthenticated_parent":
            conn.execute(
                "UPDATE tasks SET factory_terminal_receipt_sha256 = NULL WHERE id = ?",
                (kb.AION889_ATOMIC_FINALIZER_REPAIR_PARENT_ID,),
            )
        elif drift == "stale_runtime":
            monkeypatch.setattr(
                kb, "_read_aion889_atomic_finalizer_runtime",
                lambda _service: {
                    "ActiveState": "active", "SubState": "running", "Result": "success",
                    **kb.AION889_ATOMIC_FINALIZER_PRE_RUNTIME,
                },
            )
        elif drift == "null_event_run":
            conn.execute(
                "UPDATE task_events SET run_id = NULL WHERE task_id = ? AND kind = ?",
                (parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND),
            )
        elif drift == "conflicting_event_run":
            conn.execute(
                "UPDATE task_events SET run_id = ? WHERE task_id = ? AND kind = ?",
                (kb.AION889_ATOMIC_FINALIZER_SOURCE_RUN_ID, parent,
                 kb.AION889_ATOMIC_FINALIZER_EVENT_KIND),
            )
        elif drift == "unsupported_status":
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,))
        elif drift == "unsatisfied_upstream":
            upstream = conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ? "
                "AND parent_id != ? ORDER BY parent_id LIMIT 1",
                (parent, kb.AION889_ATOMIC_FINALIZER_REPAIR_PARENT_ID),
            ).fetchone()
            assert upstream is not None
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (upstream["parent_id"],),
            )
        elif drift == "unauthorized_actor":
            conn.execute("UPDATE tasks SET assignee = 'gm2' WHERE id = ?", (parent,))
        elif drift == "prose_only":
            payload = conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
                (parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND),
            ).fetchone()["payload"]
            conn.execute(
                "DELETE FROM task_events WHERE task_id = ? AND kind = ?",
                (parent, kb.AION889_ATOMIC_FINALIZER_EVENT_KIND),
            )
            conn.execute(
                "INSERT INTO task_comments(task_id, author, body, created_at) "
                "VALUES (?, 'gm', ?, ?)", (parent, payload, int(time.time())),
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, parent, summary="must reject drift")

        assert _native_state_snapshot(conn) == before


def test_detached_controller_recompute_fault_rolls_back_parent_child_and_receipt(
    kanban_home, aion_gov_src, monkeypatch,
):
    with kb.connect() as conn:
        parent, _action_run, child = _authorized_detached_controller_chain(conn, monkeypatch)
        before = _native_state_snapshot(conn)
        before_files = {
            path: path.read_bytes()
            for path in kanban_home.rglob("aion_monarch_receipt*.json")
        }
        original = kb._aion889_recompute_exact_child_in_txn

        def fail_after_recompute(c):
            original(c)
            raise RuntimeError("fault after dependent recompute")

        monkeypatch.setattr(kb, "_aion889_recompute_exact_child_in_txn", fail_after_recompute)
        with pytest.raises(RuntimeError, match="fault after dependent recompute"):
            kb.complete_task(conn, parent, summary="rollback everything")

        assert _native_state_snapshot(conn) == before
        parent_row = kb.get_task(conn, parent)
        child_row = kb.get_task(conn, child)
        assert parent_row is not None and parent_row.status == "blocked"
        assert child_row is not None and child_row.status == "todo"
        after_files = {
            path: path.read_bytes()
            for path in kanban_home.rglob("aion_monarch_receipt*.json")
        }
        assert after_files == before_files


def test_detached_controller_completion_is_not_replayable(
    kanban_home, aion_gov_src, monkeypatch,
):
    with kb.connect() as conn:
        parent, _action_run, child = _authorized_detached_controller_chain(conn, monkeypatch)
        assert kb.complete_task(conn, parent, summary="authorized action completed")
        before = _native_state_snapshot(conn)

        assert kb.complete_task(conn, parent, summary="replay") is False

        assert _native_state_snapshot(conn) == before
        parent_row = kb.get_task(conn, parent)
        child_row = kb.get_task(conn, child)
        assert parent_row is not None and parent_row.status == "done"
        assert child_row is not None and child_row.status == "ready"


def _terminal_run_metadata(conn, task_id):
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return json.loads(row["metadata"])


def _set_terminal_run_metadata(conn, task_id, metadata):
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = "
        "(SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1)",
        (json.dumps(metadata), task_id),
    )
    conn.commit()


def _reviewed_author_chain(
    conn, *, canonical_merger_receipt=False, source_pr=49, live_multi_child_shape=False,
    immutable_pr54_receipts=False, activation_cycle=False, terminal_runtime=True,
    runtime_assignee="gm", current_pr56_receipts=False,
    current_descendant_runtime=False, handoff_reason=None,
    terminal_typed_audit=False, terminal_audit_verdict="PASS_EXACT_HEAD",
    durable_terminal_gm_receipt=False,
):
    """Create one exact reviewed author -> auditor -> merger -> runtime chain."""
    if terminal_typed_audit or durable_terminal_gm_receipt:
        terminal_typed_audit = True
        current_pr56_receipts = True
    if current_pr56_receipts:
        source_pr = 56
        runtime_assignee = "merger"
    elif immutable_pr54_receipts:
        source_pr = 54
    head = (
        "6b2b21dbf7c986902f809199637d1a89c8333359"
        if current_pr56_receipts else
        "b4cf71fffbeeea2446fe3060b2ee283d09d12014"
        if immutable_pr54_receipts else "d622fcf38da613ce25bf4eaf37c54a94053d5e70"
        if canonical_merger_receipt else "1" * 40
    )
    tree = (
        "c02c37759a6b9da788f2c1141a4736841396ef31"
        if current_pr56_receipts else
        "00aca8ed67457e0b0bdccb7ea07343da1031bbc4"
        if immutable_pr54_receipts else "1e5e822f33c5fe227982ff9e4e1de320cb862c42"
        if canonical_merger_receipt else "2" * 40
    )
    base = (
        "9cae83f18cac5639e2ae48e667b231ec904f41ac"
        if current_pr56_receipts else
        "d153fadd9fac7a321d728254e3136cd3e717bbf6"
        if immutable_pr54_receipts else "f6dfcb6d50f512e0944cd8f56367ccb0eac6ace8"
        if canonical_merger_receipt else "3" * 40
    )
    merge_sha = (
        "e0db622bcdb87fa60297cdf2ee98d6b1e7e0fc48"
        if current_pr56_receipts else
        "6b521c8637d477a76451d0d029cc24026d01cf61"
        if immutable_pr54_receipts else "9920fd369b4b0c14bc292b7585431f16a4028f31"
        if canonical_merger_receipt else "4" * 40
    )
    review_id = (
        5053786034 if current_pr56_receipts else
        5050442013 if immutable_pr54_receipts else
        5037761833 if canonical_merger_receipt else 12345
    )
    changed_files = (
        ["hermes_cli/kanban_db.py", "tests/hermes_cli/test_kanban_factory_finalizer.py"]
        if canonical_merger_receipt or current_pr56_receipts else
        ["tools/approval.py", "tests/tools/test_aion889_prior_authorization.py"]
        if immutable_pr54_receipts else ["hermes_cli/kanban_db.py"]
    )
    author = kb.create_task(
        conn, title="reviewed author", factory_build_gate=1, assignee="agent007",
    )
    reviewer = kb.create_task(
        conn, title="exact-head audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    merger = kb.create_task(
        conn, title="role-separated merge", factory_build_gate=1,
        assignee="gm" if current_pr56_receipts else "merger", parents=[reviewer],
    )
    if activation_cycle:
        kb.create_task(
            conn, title="author-gated activation", factory_build_gate=1,
            assignee="gm", parents=[merger, author],
        )
    install_parent = None
    runtime_parents = [merger]
    if current_descendant_runtime:
        install_parent = kb.create_task(
            conn, title="current installed runtime", factory_build_gate=1,
            assignee="installer",
        )
        runtime_parents.append(install_parent)
    runtime = kb.create_task(
        conn, title="runtime install witness", factory_build_gate=1,
        assignee=runtime_assignee, parents=runtime_parents,
    )
    if install_parent is not None:
        install_run = _claim_and_run_id(conn, install_parent)
        assert kb.complete_task(
            conn, install_parent, expected_run_id=install_run,
            metadata={"kind": "authenticated-current-install"},
        )

    author_run = _claim_and_run_id(conn, author)
    handoff = kb.request_review_handoff(
        conn,
        author,
        expected_run_id=author_run,
        review_task_id=reviewer,
        reason=(
            f"PR #{source_pr} frozen at exact head {head}"
            if handoff_reason is None else handoff_reason
        ),
    )
    assert handoff is not None

    reviewer_run = _claim_and_run_id(conn, reviewer)
    if current_pr56_receipts:
        assert _record_legacy_review_verdict_fixture(
            conn, author, review_task_id=reviewer,
            expected_review_run_id=reviewer_run, verdict="request_changes",
            reason="REQUEST_CHANGES_EXACT_HEAD sanitized prior finding",
        )
        author_run = _claim_and_run_id(conn, author)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=reviewer,
            reason=(
                f"PR #{source_pr} repaired at exact head {head}"
                if handoff_reason is None else handoff_reason
            ),
        ) is not None
        reviewer_run = _claim_and_run_id(conn, reviewer)
    review_reason = (
        f"APPROVE_EXACT_HEAD head={head} tree={tree} review={review_id}"
    )
    assert _record_legacy_review_verdict_fixture(
        conn,
        author,
        review_task_id=reviewer,
        expected_review_run_id=reviewer_run,
        verdict="pass",
        reason=review_reason,
    )
    reviewer_metadata = {
        "author_task": author,
        "review_outcome": "APPROVE_EXACT_HEAD",
        "source_pr": source_pr,
        "head_sha": head,
        "tree_sha": tree,
        "base_sha": base,
        "github_review_id": review_id,
        "changed_files": changed_files,
    }
    if immutable_pr54_receipts:
        # Exact sanitized t_68fee4c0/run3434 immutable metadata shape.  The
        # historical receipt cannot be enriched with canonical aliases.
        reviewer_metadata = {
            "review_outcome": "PASS_EXACT_HEAD",
            "pr": "https://github.com/kiddhu/hermes-agent/pull/54",
            "head": head,
            "tree": tree,
            "base": base,
            "author_identity": "007AION",
            "auditor_identity": "GemAION",
            "review_comment_id": 4798,
            "forbidden_actions_performed": [],
            "secret_exposure": "none",
        }
    if current_pr56_receipts:
        reviewer_metadata = {
            "audit_outcome": "PASS_EXACT_HEAD",
            "audit_run_id": reviewer_run,
            "author_run_id": author_run,
            "author_task_id": author,
            "base": base,
            "changed_files": changed_files,
            "forbidden_actions_performed": [],
            "github_review": {"id": review_id, "state": "APPROVED", "url": "sanitized"},
            "head": head,
            "pr": source_pr,
            "repository": "kiddhu/hermes-agent",
            "secret_exposure": "none",
            "tree": tree,
        }
    if terminal_typed_audit:
        reviewer_metadata = {
            "verdict": terminal_audit_verdict,
            "native_review_run": reviewer_run,
            "commit_bound_review": (
                f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}"
                f"#pullrequestreview-{review_id}"
            ),
            "head": head,
            "tree": tree,
            "base": base,
            "changed_files": changed_files,
            "merge_allowed": True,
            "forbidden_actions_performed": [],
            "secret_exposure": "none",
        }
    if canonical_merger_receipt:
        # Exact immutable t_463814e3/run3382 shape: author identity is bound by
        # the typed handoff/direct edge/verdict, not duplicated in metadata.
        reviewer_metadata.pop("author_task")
        reviewer_metadata.update({
            "ci_run": 33150562165,
            "tests": {"factory_finalizer": "43 passed"},
            "worker_session_id": "sanitized-review-session",
        })
    assert kb.complete_task(
        conn,
        reviewer,
        expected_run_id=reviewer_run,
        summary="independent exact-head audit passed",
        metadata=reviewer_metadata,
    )

    merger_run = _claim_and_run_id(conn, merger)
    merger_metadata = {
        "repository": "kiddhu/hermes-agent",
        "pr_number": source_pr,
        "head_sha": head,
        "tree_sha": tree,
        "audited_base_sha": base,
        "merge_commit_sha": merge_sha,
        "merged_by": "kiddhu",
        "review_id": review_id,
        "author": "007AION",
        "auditor": "GemAION",
        "role_separation": {
            "author": "007AION",
            "auditor": "GemAION",
            "merger": "kiddhu",
            "distinct": True,
        },
        "forbidden_actions_performed": [],
        "secret_exposure": "none",
    }
    if canonical_merger_receipt:
        merger_metadata = {
            "verdict": "EXACT_HEAD_MERGED_MAIN_READBACK",
            "native_profile": "merger",
            "native_task_id": merger,
            "native_run_id": merger_run,
            "repository": "kiddhu/hermes-agent",
            "pr_number": source_pr,
            "expected_head": head,
            "audited_tree": tree,
            "audited_base": base,
            "merge_commit_sha": merge_sha,
            "merged_by": "kiddhu",
            "canonical_main_sha": merge_sha,
            "canonical_main_parents": ["6" * 40, head],
            "audited_head_is_main_parent": True,
            "main_equals_merge_commit": True,
            "implementation_task_id": author,
            "implementation_run_id": author_run,
            "implementation_profile": "agent007",
            "implementation_actor": "007AION",
            "audit_task_id": reviewer,
            "audit_run_id": reviewer_run,
            "audit_profile": "bafuxunan",
            "auditor_actor": "GemAION",
            "github_review_id": review_id,
            "gate_verdict": "PASS",
            "merge_performed": True,
            "production_or_runtime_mutation": False,
        }
    if immutable_pr54_receipts:
        # Exact sanitized t_e404f7ab/run3438 immutable metadata shape.  Keep
        # audited_head/audited_tree/base_at_audit and native_audit_* as-is.
        merger_metadata = {
            "verdict": "EXACT_HEAD_MERGED_MAIN_READBACK",
            "repo": "kiddhu/hermes-agent",
            "pr": 54,
            "audited_head": head,
            "audited_tree": tree,
            "base_at_audit": base,
            "merge_commit": merge_sha,
            "canonical_main": merge_sha,
            "canonical_main_tree": tree,
            "merge_parents": [base, head],
            "merger_identity": "kiddhu",
            "auditor_identity": "GemAION",
            "github_review_id": review_id,
            "native_audit_task": reviewer,
            "native_audit_run": reviewer_run,
            "native_audit_verdict": "PASS_EXACT_HEAD",
            "tools_approval_blob_candidate": "badb5ef99e71334fc7c0d38555c9f2f2011df7cb",
            "tools_approval_blob_main": "badb5ef99e71334fc7c0d38555c9f2f2011df7cb",
            "merge_performed": True,
            "production_or_runtime_mutation": False,
            "forbidden_actions_performed": [],
            "secret_exposure": "none",
        }
    if current_pr56_receipts:
        merger_metadata = {
            "outcome": "ROLE_SEPARATED_CAS_MERGE_AND_MAIN_READBACK_COMPLETE",
            "native_task_id": merger, "native_run_id": merger_run,
            "repository": "kiddhu/hermes-agent", "pr": source_pr,
            "merger_profile": "gm", "merger_actor": "kiddhu",
            "implementation_actor": "007AION", "auditor_actor": "GemAION",
            "audit_task_id": reviewer, "audit_run_id": reviewer_run,
            "audit_outcome": "PASS_EXACT_HEAD",
            "github_review_id": review_id, "github_review_state": "APPROVED",
            "head": head, "head_tree": tree, "base_main_before": base,
            "merged_files": changed_files, "merge_method": "merge",
            "cas_merge_attempt_count": 1, "merge_commit": merge_sha,
            "merge_tree": tree, "merge_parents": [base, head],
            "canonical_main_after": merge_sha,
            "runtime_install_performed": False, "runtime_witness_performed": False,
            "author_finalizer_performed": False,
            "forbidden_actions_performed": [], "secret_exposure": "none",
        }
    if terminal_typed_audit:
        merger_metadata = {
            "audit_verdict": terminal_audit_verdict,
            "audited_head": head,
            "audited_tree": tree,
            "base_main_before": base,
            "canonical_main_after": merge_sha,
            "cas_merge_count": 1,
            "changed_files": changed_files,
            "commit_bound_review": (
                f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}"
                f"#pullrequestreview-{review_id}"
            ),
            "exact_audit_run": reviewer_run,
            "exact_audit_task": reviewer,
            "implementation_author": "007AION",
            "implementation_run": author_run,
            "implementation_task": author,
            "independent_auditor": "GemAION/bafuxunan",
            "merge_commit": merge_sha,
            "merge_parents": [base, head],
            "merge_tree": tree,
            "merger_role": "AION-GM",
            "native_run_id": merger_run,
            "pr": f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}",
            "pr_state": "MERGED",
            "reviewed_author_finalizer_performed": False,
            "runtime_install_performed": False,
            "typed_runtime_witness_performed": False,
            "forbidden_actions_performed": [],
            "secret_exposure": "none",
        }
    if durable_terminal_gm_receipt:
        merger_metadata = {
            "outcome": "ROLE_SEPARATED_CAS_MERGE_AND_MAIN_READBACK_COMPLETE",
            "canonical_run_id": merger_run,
            "project_id": "AION-889 / AION-RL2-CORE-01",
            "source_pr": source_pr,
            "source_pr_url": f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}",
            "implementation_task": author,
            "implementation_run": author_run,
            "implementation_profile": "agent007",
            "implementation_github_actor": "007AION",
            "exact_audit_task": reviewer,
            "exact_audit_run": reviewer_run,
            "audit_profile": "bafuxunan",
            "audit_github_actor": "GemAION",
            "audit_verdict": terminal_audit_verdict,
            "github_review_id": review_id,
            "commit_bound_review": (
                f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}"
                f"#pullrequestreview-{review_id}"
            ),
            "audited_head": head,
            "audited_tree": tree,
            "audited_base": base,
            "base_ref": "main",
            "changed_files": changed_files,
            "native_collision_readback": {"other_nonterminal_exact_merge_owners": 0},
            "hosted_checks": {
                "total": 37, "terminal": 37, "pending": 0, "failing": 0,
                "required_aggregate": "All required checks pass",
                "required_aggregate_conclusion": "success",
                "required_aggregate_url": "https://github.com/example/check",
            },
            "cas_merge": {
                "attempts": 1, "method": "merge", "expected_head": head,
                "api_result": "Pull Request successfully merged",
            },
            "merge_profile": "gm",
            "merge_github_actor": "kiddhu",
            "roles_distinct": True,
            "pr_state": "MERGED",
            "merge_commit": merge_sha,
            "merge_tree": tree,
            "merge_parents": [base, head],
            "canonical_main": merge_sha,
            "audited_head_containment": {
                "status": "ahead", "ahead_by": 1, "behind_by": 0,
                "exact_second_parent": True,
            },
            "public_receipts": ["https://github.com/example/receipt"],
            "runtime_install_performed": False,
            "typed_runtime_witness_performed": False,
            "reviewed_author_finalizer_performed": False,
            "source_edit_performed": False,
            "forbidden_actions_performed": [],
            "secret_exposure": "none",
            "new_control_plane_count": 0,
            "not_true_done_for": ["reviewed-author finalization"],
            "next_machine_transition": "separate typed runtime witness",
            "artifacts": ["/tmp/sanitized-public-receipt.md"],
            "worker_session_id": "sanitized-merger-session",
        }
    assert kb.complete_task(
        conn,
        merger,
        expected_run_id=merger_run,
        summary="role-separated merge readback passed",
        metadata=merger_metadata,
    )

    if terminal_runtime:
        runtime_run = _claim_and_run_id(conn, runtime)
        runtime_metadata = {
            "canonical_run_id": runtime_run,
            "install": {
                "head": merge_sha,
                "tree": tree,
                "changed_paths": changed_files,
            },
            "forbidden_actions_performed": [],
            "secret_exposure": "none",
        }
        if immutable_pr54_receipts or terminal_typed_audit:
            runtime_metadata.update({
                "witness_type": "RUNTIME_INSTALL_READBACK",
                "source_pr": source_pr,
                "source_head": head,
                "source_tree": tree,
                "source_merge": merge_sha,
                "github_review_id": review_id,
                "source_changed_paths": changed_files,
            })
        if current_descendant_runtime:
            installed_paths = [
                "hermes_cli/kanban_db.py",
                "tests/hermes_cli/test_kanban_factory_finalizer.py",
            ]
            packet = {
                "canonical_run_id": runtime_run,
                "install": {
                    "head": "e0db622bcdb87fa60297cdf2ee98d6b1e7e0fc48",
                    "tree": "c02c37759a6b9da788f2c1141a4736841396ef31",
                    "changed_paths": installed_paths,
                },
                "witness_type": "RUNTIME_INSTALL_READBACK",
                "source_pr": source_pr, "source_head": head, "source_tree": tree,
                "source_merge": merge_sha, "github_review_id": review_id,
                "source_changed_paths": changed_files,
                "forbidden_actions_performed": [], "secret_exposure": "none",
            }
            runtime_metadata = {
                "canonical_run_id": runtime_run,
                "installed_runtime": packet["install"].copy(),
                "source_lineage": {
                    key: packet[key] for key in (
                        "source_pr", "source_head", "source_tree", "source_merge",
                        "github_review_id", "source_changed_paths",
                    )
                },
                "candidate_packet": packet,
                "role_binding": {
                    "direct_parents": [merger, install_parent], "parents_terminal": True,
                    "runtime_profile": runtime_assignee, "author_profile": "agent007",
                    "auditor_profile": "bafuxunan", "selected_merger_profile": "merger",
                    "roles_distinct": True,
                },
                "review_obligations": {
                    "unchanged": True, "author_finalizer_performed": False,
                },
                "forbidden_actions_performed": [], "secret_exposure": "none",
            }
        assert kb.complete_task(
            conn,
            runtime,
            expected_run_id=runtime_run,
            summary="exact runtime installed and read back",
            metadata=runtime_metadata,
        )
    if live_multi_child_shape:
        # The Monarch-named live author has historical/dependency children and
        # its handoff-selected reviewer has an unrelated historical child.
        # Neither is evidence for this exact reviewed-finalizer chain.
        for index in range(5):
            kb.create_task(
                conn,
                title=f"unrelated author child {index}",
                assignee="gm2",
                parents=[author],
            )
        historical = kb.create_task(
            conn,
            title="unrelated completed reviewer child",
            assignee="gm2",
            parents=[reviewer],
        )
        historical_run = _claim_and_run_id(conn, historical)
        assert kb.complete_task(
            conn,
            historical,
            expected_run_id=historical_run,
            summary="historical lane completed",
        )
    return author, author_run, reviewer, merger, runtime


def _non_pr_reviewed_evidence_chain(conn, *, handoff_reason="exact runtime evidence"):
    """Create the sanitized authenticated no-PR shape from t_cc5e6d1b/run3474."""
    author = kb.create_task(
        conn, title="runtime evidence author", factory_build_gate=1,
        assignee="agent007",
    )
    reviewer = kb.create_task(
        conn, title="independent runtime evidence audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    author_run = _claim_and_run_id(conn, author)
    assert kb.request_review_handoff(
        conn,
        author,
        expected_run_id=author_run,
        review_task_id=reviewer,
        reason=handoff_reason,
    ) is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn,
        author,
        review_task_id=reviewer,
        expected_review_run_id=reviewer_run,
        verdict="pass",
        reason="PASS for exact immutable runtime evidence only",
    )
    metadata = {
        "audit_outcome": "APPROVE_EXACT_NATURAL_RECOVERY_EVIDENCE",
        "source_task_id": author,
        "source_run_id": author_run,
        "canonical_run_id": "real-20260828T192101Z",
        "native_review_verdict": "pass",
        "evidence_sha256": "1" * 64,
        "timer_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "github_receipt": "https://github.com/example/governance/issues/790#issuecomment-12345",
        "github_receipt_body_sha256": "5" * 64,
        "checks": [
            "exact source task and run binding",
            "authenticated runtime receipt hash",
            "zero-secret and zero-mutation ledger",
        ],
        "scope_limit": "Exact independently audited runtime evidence only.",
        "artifacts": ["/tmp/sanitized-audit.md"],
        "worker_session_id": "sanitized-review-session",
    }
    assert kb.complete_task(
        conn,
        reviewer,
        expected_run_id=reviewer_run,
        summary="independent runtime evidence audit passed",
        metadata=metadata,
    )
    return author, author_run, reviewer


def _canonical_audit_receipt_chain(conn, *, author_assignee="gm2"):
    author = kb.create_task(
        conn, title="generic reviewed author", factory_build_gate=1,
        assignee=author_assignee,
    )
    reviewer = kb.create_task(
        conn, title="role-separated audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    child = kb.create_task(
        conn, title="downstream product transition", factory_build_gate=1,
        assignee="agent007", parents=[author],
    )
    author_run = _claim_and_run_id(conn, author)
    handoff = kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=reviewer,
        reason="exact Native receipt audit",
    )
    assert handoff is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run, verdict="pass",
        reason="PASS_EXACT_NATIVE_RECEIPT",
    )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=reviewer_run,
        summary="independent audit passed",
        metadata={"legacy_business_packet": {"ignored": True}},
    )
    return {
        "author": author, "author_run": author_run, "reviewer": reviewer,
        "reviewer_run": reviewer_run, "child": child, "handoff": handoff,
    }


def _canonical_multi_round_audit_receipt_chain(conn):
    author = kb.create_task(
        conn, title="multi-round reviewed author", factory_build_gate=1,
        assignee="agent007",
    )
    historical = [
        kb.create_task(
            conn, title=f"superseded audit {index}", factory_build_gate=1,
            assignee="bafuxunan", parents=[author],
        )
        for index in range(2)
    ]
    reviewer = kb.create_task(
        conn, title="latest exact audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    for index, prior in enumerate(historical):
        author_run = _claim_and_run_id(conn, author)
        assert kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=prior,
            reason=f"repair round {index}",
        )
        review_run = _claim_and_run_id(conn, prior)
        assert _record_legacy_review_verdict_fixture(
            conn, author, review_task_id=prior,
            expected_review_run_id=review_run, verdict="request_changes",
            reason=f"REQUEST_CHANGES_ROUND_{index}",
        )
    author_run = _claim_and_run_id(conn, author)
    handoff = kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=reviewer,
        reason="latest exact repair",
    )
    assert handoff is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run, verdict="pass",
        reason="PASS_LATEST_EXACT_REPAIR",
    )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=reviewer_run,
        summary="latest independent audit passed",
    )
    return {
        "author": author, "author_run": author_run, "historical": historical,
        "reviewer": reviewer, "reviewer_run": reviewer_run, "handoff": handoff,
    }


def _canonical_completed_recovery_history_chain(conn):
    author = kb.create_task(
        conn, title="recovered reviewed author", factory_build_gate=1,
        assignee="agent007",
    )
    historical = kb.create_task(
        conn, title="terminal recovered audit",
        assignee="bafuxunan", parents=[author],
    )
    reviewer = kb.create_task(
        conn, title="latest exact audit",
        assignee="bafuxunan", parents=[author],
    )
    author_run = _claim_and_run_id(conn, author)
    assert kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=historical,
        reason="initial exact head",
    )
    historical_run = _claim_and_run_id(conn, historical)
    recovery_reason = "REQUEST_CHANGES_EXACT_HEAD: exact terminal blockers"
    recovery_receipt = {
        "review_outcome": "REQUEST_CHANGES_EXACT_HEAD",
        "repository": "kiddhu/hermes-agent",
        "pr": 60,
        "head": "1" * 40,
        "tree": "2" * 40,
        "base": "3" * 40,
        "github_review_id": 12345,
        "github_review_url": (
            "https://github.com/kiddhu/hermes-agent/pull/60"
            "#pullrequestreview-12345"
        ),
        "github_review_state": "CHANGES_REQUESTED",
    }
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='completed', summary=?, "
            "metadata=?, ended_at=12345, claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL WHERE id=?",
            (
                recovery_reason,
                json.dumps({**recovery_receipt, "changed_files": ["exact.py"]}),
                historical_run,
            ),
        )
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (historical,),
        )
    controller = kb.create_task(conn, title="gm2 recovery controller", assignee="gm2")
    controller_run = _claim_and_run_id(conn, controller)
    assert _record_legacy_review_verdict_fixture(
        conn,
        author,
        review_task_id=historical,
        expected_review_run_id=historical_run,
        verdict="request_changes",
        reason=recovery_reason,
        recovery_receipt=recovery_receipt,
        controller_task_id=controller,
        controller_run_id=controller_run,
        controller_profile="gm2",
    )
    assert kb.complete_task(
        conn, controller, expected_run_id=controller_run,
        summary="recovery controller completed",
    )
    latest_author_run = _claim_and_run_id(conn, author)
    handoff = kb.request_review_handoff(
        conn, author, expected_run_id=latest_author_run, review_task_id=reviewer,
        reason="repaired exact head",
    )
    assert handoff is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run, verdict="pass",
        reason="PASS_LATEST_EXACT_REPAIR",
    )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=reviewer_run,
        summary="latest independent audit passed",
    )
    recovery_event = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_verdict' "
        "AND run_id = ?", (author, historical_run),
    ).fetchone()
    assert recovery_event is not None
    return {
        "author": author, "author_run": latest_author_run,
        "historical": historical, "historical_run": historical_run,
        "controller": controller, "controller_run": controller_run,
        "reviewer": reviewer, "reviewer_run": reviewer_run, "handoff": handoff,
        "recovery_event": int(recovery_event["id"]),
    }


def _canonical_reused_auditor_history_chain(conn):
    author = kb.create_task(
        conn, title="reused-child reviewed author", factory_build_gate=1,
        assignee="bafuxunan",
    )
    historical = kb.create_task(
        conn, title="reused historical audit",
        assignee="elder-senate", parents=[author],
    )
    reviewer = kb.create_task(
        conn, title="latest role-separated audit",
        assignee="elder-senate", parents=[author],
    )

    precursor_author_run = _claim_and_run_id(conn, author)
    precursor_handoff = kb.request_review_handoff(
        conn, author, expected_run_id=precursor_author_run,
        review_task_id=historical, reason="initial audit attempt",
    )
    assert precursor_handoff is not None
    precursor_run = _claim_and_run_id(conn, historical)
    assert kb.block_task(
        conn, historical, reason="transient provider failure", kind="transient",
        expected_run_id=precursor_run,
    )
    assert kb.unblock_task(conn, historical)
    with kb.write_txn(conn):
        assert conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=? AND status='todo'",
            (historical,),
        ).rowcount == 1
        kb._append_event(conn, historical, "promoted", None)

    initial_round_run = _claim_and_run_id(conn, historical)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=historical,
        expected_review_run_id=initial_round_run, verdict="request_changes",
        reason="REQUEST_CHANGES_ROUND_0",
    )
    rounds = [(precursor_handoff, initial_round_run)]
    for index in range(1, 3):
        round_author_run = _claim_and_run_id(conn, author)
        round_handoff = kb.request_review_handoff(
            conn, author, expected_run_id=round_author_run,
            review_task_id=historical, reason=f"repair round {index}",
        )
        assert round_handoff is not None
        round_run = _claim_and_run_id(conn, historical)
        assert _record_legacy_review_verdict_fixture(
            conn, author, review_task_id=historical,
            expected_review_run_id=round_run, verdict="request_changes",
            reason=f"REQUEST_CHANGES_ROUND_{index}",
        )
        rounds.append((round_handoff, round_run))

    latest_author_run = _claim_and_run_id(conn, author)
    latest_handoff = kb.request_review_handoff(
        conn, author, expected_run_id=latest_author_run, review_task_id=reviewer,
        reason="latest exact repair",
    )
    assert latest_handoff is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    legacy_pass = {
        "version": 1,
        "review_task_id": reviewer,
        "review_run_id": reviewer_run,
        "verdict": "pass",
        "reason": "PASS_LATEST_EXACT_REPAIR",
    }
    with kb.write_txn(conn):
        kb._append_event(
            conn, author, "review_verdict", legacy_pass, run_id=reviewer_run,
        )
        kb._append_event(
            conn, reviewer, "review_verdict", legacy_pass, run_id=reviewer_run,
        )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=reviewer_run,
        summary="latest independent audit passed",
    )
    assert kb.archive_task(
        conn, historical, reason="superseded after latest exact PASS",
        actor="kanban-orchestrator", source="kanban_archive",
        fail_if_active_run=True, expected_status="todo",
    )
    archive = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'archived'",
        (historical,),
    ).fetchone()
    assert archive is not None
    return {
        "author": author, "author_run": latest_author_run,
        "historical": historical, "precursor_handoff": precursor_handoff,
        "precursor_run": precursor_run, "rounds": rounds,
        "reviewer": reviewer, "reviewer_run": reviewer_run,
        "handoff": latest_handoff, "archive_event": int(archive["id"]),
    }


def _canonical_factory_packet_chain(
    conn, *, installed_source_shapes=False, legacy_installed_source_tail=False,
    emitted_review_shape=False, real_systemd_order=True, existing_author=None,
    actual_emitted_packets=False, resident_request_changes=False,
    request_changes_merger_shape=False,
    merge_mutate=None, install_mutate=None, resident_mutate=None,
):
    """Create the generic authenticated packet shapes emitted by the current lane."""
    head, tree, base, merge = "1" * 40, "2" * 40, "3" * 40, "4" * 40
    review_id, source_pr = 12345, 64
    paths = ["hermes_cli/kanban_db.py", "tests/hermes_cli/test_kanban_db.py"]
    author = existing_author or kb.create_task(
        conn, title="packet author", factory_build_gate=1, assignee="agent007",
    )
    reviewer = kb.create_task(conn, title="packet audit", factory_build_gate=1, assignee="bafuxunan", parents=[author])
    merger = kb.create_task(conn, title="packet merger", factory_build_gate=1, assignee="gm", parents=[reviewer])
    installer = kb.create_task(conn, title="packet install", factory_build_gate=1, assignee="merger", parents=[merger])
    activation = kb.create_task(conn, title="packet activation", factory_build_gate=1, assignee="agent007", parents=[installer])
    resident_audit = kb.create_task(conn, title="packet resident audit", factory_build_gate=1, assignee="bafuxunan", parents=[activation])

    author_run = _claim_and_run_id(conn, author)
    assert kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=reviewer,
        reason=f"PR #{source_pr} frozen for independent exact-head audit",
    )
    review_run = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=review_run, verdict="pass", reason="PASS_EXACT_HEAD",
    )
    review_url = f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}#pullrequestreview-{review_id}"
    review_metadata = {
        "approval_commit_id": head, "approved": True, "base": base,
        "github_review_id": review_id, "github_review_url": review_url,
        "head": head, "head_tree": tree, "review_outcome": "PASS_EXACT_HEAD",
        "tests_passed": 406, "tests_failed": 0,
        "verification": ["canonical tests", "live exact-head readback"],
        "worker_session_id": "canonical-review-session",
    }
    if emitted_review_shape:
        review_metadata = {
            "artifacts": ["/tmp/generic-review.yaml"],
            "base": base,
            "forbidden_actions_performed": [],
            "github_review": {
                "commit_id": head, "id": review_id, "state": "APPROVED",
                "url": review_url,
            },
            "head": head,
            "hosted_ci": {
                "all_required_checks_pass": "SUCCESS", "failures": 0,
                "run": 123456, "terminal": 37,
            },
            f"lean_pr{source_pr}_compare": {
                "decision": "MINIMAL_REPAIR_WITH_NATIVE_GAP_PROOF",
                "finalizer_schema_branch_count": 2,
                "mixed_family_state_space": "removed",
                "new_control_plane_count": 0,
                "new_long_lived_state_count": 0,
                "new_runtime_component_count": 0,
                "packet_family_count": 2,
            },
            "local_verification": {
                "diff_check": "PASS", "factory_finalizer": 314,
                "independent_hostile_zero_mutation": 12, "kanban_db": 380,
                "py_compile": "PASS", "ruff": "PASS", "total_failed": 0,
                "total_passed": 694,
            },
            "outcome": "PASS_EXACT_HEAD",
            "repository": "kiddhu/hermes-agent",
            "secret_exposure": "none",
            "tree": tree,
            "worker_session_id": "emitted-review-session",
        }
    assert kb.complete_task(
        conn, reviewer, expected_run_id=review_run, metadata=review_metadata,
    )

    merger_run = _claim_and_run_id(conn, merger)
    merger_metadata = {
        "actor": "kiddhu", "audited_head": head, "audited_tree": tree,
        "audit": {"github_review_id": review_id, "github_review_state": "APPROVED", "native_run_id": review_run, "native_task_id": reviewer, "verdict": "PASS_EXACT_HEAD"},
        "base": base, "canonical_checkout": {"clean": True, "installed": False, "preserved_head": base},
        "checks": {"bad_or_pending": 0, "neutral": 1, "skipped": 1, "success": 2, "total": 4},
        "changed_files": paths, "child_task_id": installer,
        "forbidden_actions_performed": [],
        "formal_receipts": ["https://github.com/example/governance/issues/833#issuecomment-1"],
        "issues_kept_open": [833, 790], "merge_commit": merge,
        "merge_parents": [base, head], "merge_tree": tree,
        "mutation_ledger": {"github_cas_merge": 1, "github_comments": 2, "native_child_creations": 1, "native_evidence_comments": 1},
        "new_control_plane_count": 0, "pr": source_pr, "remote_main": merge,
        "role_separation": {"author": "agent007/007AION", "auditor": "bafuxunan/GemAION", "merger": "gm/kiddhu"},
        "secret_exposure": "none", "worker_session_id": "canonical-merge-session",
    }
    if installed_source_shapes:
        merger_metadata = {
            "actor": {
                "github": "kiddhu", "profile": "gm",
                "role_separated_from": ["agent007/007AION", "bafuxunan/GemAION"],
            },
            "audit": {
                "approval_commit_id": head, "github_review_id": review_id,
                "native_run_id": review_run, "native_task_id": reviewer,
                "review_actor": "GemAION", "review_state": "APPROVED",
                "verdict": "PASS_EXACT_HEAD",
            },
            "base": base,
            "canonical_checkout": {"dirty": False, "head": base, "installed_in_task": False},
            "checks": "all terminal non-failure at exact audited head",
            "child": {
                "assignee": "merger", "id": installer,
                "purpose": "guarded canonical install plus typed source/installed/runtime witness",
                "status_at_creation": "todo",
            },
            "evidence_comments": ["https://github.com/example/governance/issues/833#issuecomment-1"],
            "forbidden_actions_performed": [], "head": head, "head_tree": tree,
            "merge": {
                "cas_calls": 1, "commit": merge, "method": "merge",
                "parents": [base, head], "remote_main": merge, "tree": tree,
            },
            "merged_blobs": {path: "6" * 40 for path in paths},
            "new_control_plane_count": 0,
            "not_true_done_for": ["reviewed-author finalization"],
            "pr": source_pr, "pr_state": "MERGED",
            "protected_issues": {"790": "open", "833": "open"},
            "secret_exposure": "none", "worker_session_id": "installed-source-merge-session",
        }
    if actual_emitted_packets:
        merger_metadata = {
            "actor": {
                "distinct_from": ["007AION", "GemAION"],
                "github": "kiddhu", "role": "AION-GM",
            },
            "audit": {
                "review_commit_id": head, "review_id": review_id,
                "review_state": "APPROVED", "review_url": review_url,
                "run_id": review_run, "task_id": reviewer,
            },
            "base": base,
            "canonical_checkout": {
                "branch": "main", "clean": True, "head": base,
                "installed_in_this_task": False,
            },
            "changed_paths": paths,
            "checks": {
                "bad": 0, "ci_run": 123456, "ci_status": "completed/success",
                "neutral": 1, "pending": 0, "skipped": 1, "success": 2,
            },
            "child": {
                "assignee": "merger", "id": installer,
                "purpose": "guarded clean fast-forward install plus typed witness",
                "status": "todo",
            },
            "forbidden_actions_performed": [], "head": head,
            "merge": {
                "cas_sha_guard": head, "commit": merge,
                "parents": [base, head], "remote_main": merge, "tree": tree,
            },
            "new_control_plane_count": 0, "new_runtime_module_count": 0,
            "outcome": "ROLE_SEPARATED_CAS_MERGE_AND_MAIN_READBACK_COMPLETE",
            "public_receipts": ["https://github.com/example/governance/issues/833#issuecomment-1"],
            "secret_exposure": "none", "tree": tree,
            "worker_session_id": "actual-merge-session",
        }
    if request_changes_merger_shape:
        merger_metadata = {
            "artifacts": ["/tmp/role-separated-merge-receipt.json"],
            "audited_head": head, "audited_tree": tree, "base": base,
            "canonical_checkout": {
                "clean": True, "head": base, "install_performed": False,
            },
            "changed_paths": paths,
            "checks": {
                "all_required_pass": True, "failing": 0, "pending": 0,
                "total": 37,
            },
            "created_child": {
                "assignee": "merger", "id": installer,
                "scope": "guarded source install plus typed witness",
            },
            "exact_audit": {
                "review_id": review_id, "run": review_run,
                "state": "APPROVED", "task": reviewer,
            },
            "forbidden_actions_performed": [],
            "github_receipts": {
                "issue_833": (
                    "https://github.com/kiddhu/aion-governance/issues/833"
                    "#issuecomment-1"
                ),
                "pr": (
                    f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}"
                    "#issuecomment-2"
                ),
            },
            "issue_833_state": "OPEN",
            "merge": {
                "actor": "kiddhu", "attempts": 1, "commit": merge,
                "method": "merge", "parents": [base, head],
                "sha_guarded": True, "tree": tree,
            },
            "new_control_plane_count": 0, "new_runtime_module_count": 0,
            "outcome": "ROLE_SEPARATED_CAS_MERGE_AND_MAIN_READBACK_COMPLETE",
            "pr": source_pr, "remote_main": merge, "secret_exposure": "none",
            "worker_session_id": "request-changes-merge-session",
        }
    if merge_mutate is not None:
        merge_mutate(merger_metadata)
    assert kb.complete_task(
        conn, merger, expected_run_id=merger_run, metadata=merger_metadata,
    )

    install_run = _claim_and_run_id(conn, installer)
    module_hash = "5" * 64
    install_metadata = {
        "artifacts": ["/tmp/canonical-install-receipt.json"],
        "author_finalizer_performed": False, "author_status_before_and_after": "review",
        "canonical_run_id": install_run, "forbidden_actions_performed": [],
        "fresh_runtime": {"bytes_match": True, "canonical_git_blob": "6" * 40, "module_path": "/repo/hermes_cli/kanban_db.py", "module_sha256": module_hash, "resolver_loaded": True, "working_git_blob": "6" * 40},
        "github_review_id": review_id,
        "install": {"changed_paths": paths, "head": merge, "method": "existing_clean_git_editable_guarded_fast_forward", "parents": [base, head], "preinstall_commit": base, "rollback_commit": base, "rollback_ref": "refs/aion/rollback/generic", "tree": tree, "worktree_clean": True},
        "native_binding": {"auditor_profile": "bafuxunan", "author_profile": "agent007", "direct_parent_only": merger, "merge_profile": "gm", "parent_run": merger_run, "roles_distinct": True, "runtime_profile": "merger"},
        "new_control_plane_count": 0, "not_true_done_for": ["reviewed-author finalization"],
        "public_receipts": ["https://github.com/example/governance/issues/833#issuecomment-2"],
        "receipt_sha256": "7" * 64, "review_obligations": {"count": 21, "review_statuses_unchanged": True},
        "secret_exposure": "none", "source_changed_paths": paths, "source_head": head,
        "source_merge": merge, "source_pr": source_pr, "source_tree": tree,
        "tests": {"focused_total": "950 passed, 0 failed"},
        "typed_witness": {
            "admission_call_sites": {"claim_review_task": True, "claim_task": True, "recompute_ready": True},
            "functional_smoke": {"absent_predecessor_releases": True, "live_matching_predecessor_blocks": True, "predecessor_exited_signal_emitted": True},
            "live_board_smoke_zero_mutation": True,
            "states": {"ACTIVATION_GATED": True, "INSTALLED_PRESENT": True, "RESIDENT_ACTIVE": True, "SOURCE_PRESENT": True},
        },
        "witness_type": f"EXACT_PR{source_pr}_INSTALLED_AND_TYPED_RUNTIME_WITNESS",
        "worker_session_id": "canonical-install-session",
    }
    if installed_source_shapes and not legacy_installed_source_tail:
        install_metadata = {
            "activation_performed": False, "audited_head": head,
            "author_finalizer_performed": False, "base": base,
            "blobs": {path: "6" * 40 for path in paths},
            "canonical_run_id": install_run, "forbidden_actions_performed": [],
            "fresh_runtime": {
                "bytes_match": True, "module_path": "/repo/hermes_cli/kanban_db.py",
                "module_sha256": module_hash, "resolver_loaded": True,
                "resolves_to_authoritative_root": True,
            },
            "install": {
                "changed_paths": paths, "head": merge,
                "method": "existing_clean_git_editable_guarded_fast_forward",
                "parents": [base, head], "preinstall_commit": base,
                "rollback_commit": base, "rollback_ref": "refs/aion/rollback/generic",
                "tree": tree, "worktree_clean": True,
            },
            "merge": merge, "new_control_plane_count": 0,
            "not_true_done_for": ["reviewed-author finalization"], "pr": source_pr,
            "public_receipts": ["https://github.com/example/governance/issues/833#issuecomment-2"],
            "receipt_sha256": "7" * 64, "resident_activated": False,
            "resident_runtime": {
                "all_running_gateways_predate_install": True,
                "hermes_gateway_gm2_NRestarts": "0",
                "hermes_gateway_gm2_active_state": "active",
                "hermes_gateway_gm2_sub_state": "running",
                "resident_kanban_db_blob_at_base": "8" * 40,
            },
            "secret_exposure": "none", "source_installed": True,
            "tests": {
                "diff_check": "pass", "focused_total": "659 passed, 0 failed",
                "py_compile": "pass", "ruff": "pass",
                "test_kanban_db": "380 passed", "test_kanban_factory_finalizer": "279 passed",
            },
            "tree": tree,
            "typed_symbol": {
                "approval_commit_id_branch_in_finalizer": True, "present_callable": True,
                "symbol": "_authenticated_canonical_factory_packet_chain",
            },
            "witness_type": f"EXACT_PR{source_pr}_INSTALLED_AND_TYPED_SOURCE_INSTALLED_RESIDENT_NOT_ACTIVATED_WITNESS",
            "worker_session_id": "installed-source-install-session",
        }
    if actual_emitted_packets:
        install_metadata.update({
            "artifacts": ["/tmp/runtime.json", "/tmp/typed-witness.json"],
            "changed_files": paths,
            "install_epoch": 123456789,
            "new_runtime_module_count": 0,
            "receipt_actor": "kiddhu", "receipt_actor_id": 12485573,
            "resident_runtime": {
                "all_running_gateways_predate_install": True,
                "hermes_gateway_gm2_NRestarts": "0",
                "hermes_gateway_gm2_active_state": "active",
                "hermes_gateway_gm2_main_pid": 2000,
                "hermes_gateway_gm2_sub_state": "running",
                "resident_kanban_db_blob_at_base": "8" * 40,
                "running_gateway_pids": [2000],
            },
            "typed_symbols": {
                "_canonical_factory_review_packet_present_callable": True,
                "_authenticated_canonical_factory_packet_chain_present_callable": True,
                "_reviewed_author_finalizer_run_id_wraps_canonical_packet": True,
                "monotonic_fix_active_enter_lt_exec_start": True,
            },
        })
        install_metadata.pop("typed_symbol")
    if install_mutate is not None:
        install_mutate(install_metadata)
    assert kb.complete_task(
        conn, installer, expected_run_id=install_run, metadata=install_metadata,
    )

    activation_run = _claim_and_run_id(conn, activation)
    external = {"compressed_sha256": "8" * 64, "exact_shell_pid_unique_attribution": False, "outside_target_cgroup_proven": True, "restart_count": 1, "second_restart": 0, "uncompressed_sha256": "9" * 64}
    source = {"audited_head": head, "clean": True, "head": merge, "kanban_db_blob": "6" * 40, "kanban_db_sha256": module_hash, "tree": tree}
    resident = {"active_state": "active", "barrier_loaded": True, "configured_import_exact": True, "deep_health_exit_code": 0, "exec_start_monotonic": 1000, "main_pid": 2000, "nrestarts": 0, "pids_events_max": 0, "pids_max": 120, "pids_peak": 42, "proc_starttime_ticks": 3000, "result": "success", "sub_state": "running", "tasks_max": 120}
    if installed_source_shapes and not legacy_installed_source_tail:
        source = {
            **source, "merge_commit": merge,
        }
        resident = {
            "active_enter_timestamp_monotonic": (
                1100 if real_systemd_order else 900
            ),
            "active_state": "active",
            "barrier_loaded": True, "configured_import_exact": True,
            "deep_health_exit_code": 0, "exec_start_monotonic": 1000,
            "main_pid": 2000, "memory_current": 1024, "memory_peak": 2048,
            "nrestarts": 0, "pids_peak": 42, "proc_starttime_ticks": 3000,
            "result": "success", "sub_state": "running", "tasks_max": 120,
        }
    assert kb.complete_task(conn, activation, expected_run_id=activation_run, metadata={
        "artifacts": ["/tmp/canonical-activation.json"], "audit_task": resident_audit,
        "external_activation_receipt": external, "focused_barrier_tests": {"failed": 0, "passed": 5},
        "forbidden_actions_performed": [], "formal_evidence": ["https://github.com/example/governance/issues/833#issuecomment-3"],
        "new_control_plane_count": 0, "not_true_done_for": ["fresh resident audit"],
        "outcome": "SAME_TASK_POST_ACTIVATION_READBACK_COMPLETE", "receipt_sha256": "a" * 64,
        "replay_restart_attempts": 0, "resident_runtime": resident, "secret_exposure": "none",
        "source": source, "worker_session_id": "canonical-activation-session",
    })

    resident_run = _claim_and_run_id(conn, resident_audit)
    resident_metadata = {
        "artifact_sha256": "b" * 64, "artifacts": ["/tmp/canonical-resident-audit.md"],
        "barrier_tests": {"failed": 0, "passed": 5}, "deep_health_exit_code": 0,
        "external_receipt": external, "forbidden_actions_performed": [],
        "formal_evidence": ["https://github.com/example/governance/issues/833#issuecomment-4"],
        "new_control_plane_count": 0, "next_supported_gate": "reviewed-author finalization",
        "not_true_done_for": ["production PASS"], "outcome": "PASS_EXACT_RESIDENT_RUNTIME",
        "parent_replay": {"manual_claim_or_dispatch_count": 0, "natural_claim_run": activation_run, "restart_attempts": 0},
        "resident": {"active_state": "active", "exec_start_monotonic": 1000, "main_pid": 2000, "nrestarts": 0, "proc_starttime_ticks": 3000, "result": "success", "sub_state": "running", "tasks_max": 120},
        "resource_readback": {"memory_events_high": 1, "memory_events_max": 0, "memory_peak_bytes": 1024, "oom": 0, "oom_group_kill": 0, "oom_kill": 0, "pids_current": 2, "pids_events_max": 0, "pids_max": 120, "pids_peak": 42},
        "secret_exposure": "none",
        "source": {"audited_head": head, "kanban_db_blob": "6" * 40, "kanban_db_sha256": module_hash, "merge_commit": merge, "tree": tree},
        "worker_session_id": "canonical-resident-audit-session",
    }
    if installed_source_shapes and not legacy_installed_source_tail:
        resident_metadata = {
            "artifact_sha256": "b" * 64,
            "artifacts": ["/tmp/canonical-resident-audit.md"],
            "external_receipt": {
                "compressed_sha256": external["compressed_sha256"],
                "exact_operator_cgroup_path_attributed": False,
                "outside_target_cgroup_proven": True,
                "restart_count": 1, "second_restart": 0,
                "uncompressed_sha256": external["uncompressed_sha256"],
            },
            "focused_tests": {"failed": 0, "passed": 22},
            "forbidden_actions_performed": [],
            "formal_evidence": ["https://github.com/example/governance/issues/833#issuecomment-4"],
            "native_replay": {
                "manual_claim_or_dispatch_count": 0, "restart_attempts": 0,
                "run_id": activation_run, "task": activation,
            },
            "new_control_plane_count": 0,
            "next_supported_gate": "SAME reviewed-author finalization",
            "not_true_done_for": ["production PASS"],
            "outcome": "PASS_EXACT_RESIDENT_RUNTIME",
            "resident_runtime": {
                key: resident[key]
                for key in (
                    "active_state", "exec_start_monotonic", "main_pid", "nrestarts",
                    "pids_peak", "proc_starttime_ticks", "result", "sub_state", "tasks_max",
                )
            } | {"pids_events_max": 0, "pids_max": 120},
            "secret_exposure": "none",
            "source": {
                key: source[key]
                for key in (
                    "audited_head", "clean", "head", "kanban_db_blob",
                    "kanban_db_sha256", "tree",
                )
            },
            "worker_session_id": "installed-source-resident-audit-session",
        }
    if resident_request_changes:
        resident_metadata = {
            "artifact_sha256": "b" * 64,
            "artifacts": ["/tmp/request-changes-audit.md"],
            "exact_candidate": {
                "audited_head": head, "base": base, "merge_commit": merge,
                "pr": f"https://github.com/kiddhu/hermes-agent/pull/{source_pr}",
                "tree": tree,
            },
            "focused_tests": {"failed": 0, "passed": 22},
            "forbidden_actions_performed": [],
            "formal_evidence": ["https://github.com/example/governance/issues/833#issuecomment-4"],
            "github_readback": {
                "actor": "GemAION", "body_sha256": "b" * 64,
                "byte_exact": True,
            },
            "new_control_plane_count": 0,
            "outcome": "REQUEST_CHANGES_EXACT_RESIDENT_PACKET",
            "packet_blockers": [
                {"evidence": "sanitized lifecycle mismatch", "id": "VALID_SYSTEMD_ORDER_REJECTED"},
                {"evidence": "sanitized packet mismatch", "id": "EXACT_REVIEW_PACKET_NOT_ROUTED_TO_INSTALLED_CHAIN"},
            ],
            "parent_replay": {
                "manual_claim_or_dispatch_count": 0, "natural_claim_run": activation_run,
                "restart_attempts": 0, "task": activation,
            },
            "post_test_resource_readback": {
                "memory_events_max": 0, "oom": 0, "oom_kill": 0,
                "pids_events_max": 0, "pids_max": 120, "pids_peak": 42,
                "tasks_max": 120,
            },
            "reviewed_author_probe": {
                "author_task": author, "connection_total_changes_delta": 0,
                "data_version_unchanged": True, "resolver_result": None,
                "status_before_after": "review/review",
            },
            "secret_exposure": "none",
            "source_identity": {
                "kanban_db_blob": source["kanban_db_blob"],
                "kanban_db_sha256": source["kanban_db_sha256"],
                "rollback_commit": base,
            },
            "static_checks": {
                "checkout_clean": True, "diff_check": "pass",
                "py_compile": "pass", "ruff": "pass",
            },
            "worker_session_id": "request-changes-resident-session",
        }
    if resident_mutate is not None:
        resident_mutate(resident_metadata)
    assert kb.complete_task(
        conn, resident_audit, expected_run_id=resident_run, metadata=resident_metadata,
    )
    return {"author": author, "author_run": author_run, "reviewer": reviewer, "review_run": review_run, "merger": merger, "merger_run": merger_run, "installer": installer, "install_run": install_run, "activation": activation, "activation_run": activation_run, "resident_audit": resident_audit, "resident_run": resident_run}


def _canonical_factory_repair_phase_chain(
    conn, *, resident_mutate=None, original_merge_mutate=None,
    original_install_mutate=None, merge_mutate=None, install_mutate=None,
):
    """Model the exact resident REQUEST_CHANGES -> repair-author phase boundary."""
    original = _canonical_factory_packet_chain(
        conn,
        installed_source_shapes=True,
        emitted_review_shape=True,
        resident_request_changes=True,
        request_changes_merger_shape=True,
        merge_mutate=original_merge_mutate,
        install_mutate=original_install_mutate,
        resident_mutate=resident_mutate,
    )
    repair_author = kb.create_task(
        conn,
        title="packet repair author",
        factory_build_gate=1,
        assignee="agent007",
        parents=[original["resident_audit"]],
    )
    repaired = _canonical_factory_packet_chain(
        conn,
        installed_source_shapes=True,
        emitted_review_shape=True,
        existing_author=repair_author,
        actual_emitted_packets=True,
        merge_mutate=merge_mutate,
        install_mutate=install_mutate,
    )
    return original, repaired


def _cold_reviewed_author_accepts_exact_repaired_phase_from_live_packet_families(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        original, repaired = _canonical_factory_repair_phase_chain(conn)
        before = _native_state_snapshot(conn)

        assert kb._reviewed_author_repair_phase_task_id(
            conn,
            task_id=original["author"],
            author_run_id=original["author_run"],
            reviewer_id=original["reviewer"],
            reviewer_run_id=original["review_run"],
            reviewer_profile="bafuxunan",
            review_md=kb._canonical_factory_review_packet(
                _terminal_run_metadata(conn, original["reviewer"]),
            ),
            handoff_reason="PR #64 frozen for independent exact-head audit",
        ) == repaired["author"]
        assert kb._reviewed_author_finalizer_run_id(
            conn, repaired["author"], _allow_repair_phase=False,
        ) == repaired["author_run"]
        assert kb._reviewed_author_finalizer_run_id(
            conn, original["author"], _allow_repair_phase=False,
        ) is None
        assert kb._reviewed_author_finalizer_run_id(
            conn, original["author"],
        ) == original["author_run"]
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("mutate", [
    lambda md: md.pop("outcome"),
    lambda md: md.__setitem__("unknown", True),
    lambda md: md["exact_candidate"].__setitem__("audited_head", "f" * 40),
    lambda md: md["exact_candidate"].__setitem__("tree", "malformed"),
    lambda md: md["exact_candidate"].__setitem__("base", "f" * 40),
    lambda md: md["exact_candidate"].__setitem__("merge_commit", True),
    lambda md: md["exact_candidate"].__setitem__(
        "pr", "https://github.com/kiddhu/hermes-agent/pull/65",
    ),
    lambda md: md["reviewed_author_probe"].__setitem__("author_task", "t_attacker"),
    lambda md: md["reviewed_author_probe"].__setitem__("resolver_result", 1),
    lambda md: md["reviewed_author_probe"].__setitem__(
        "connection_total_changes_delta", False,
    ),
    lambda md: md["parent_replay"].__setitem__("task", "t_attacker"),
    lambda md: md["parent_replay"].__setitem__("natural_claim_run", True),
    lambda md: md["parent_replay"].__setitem__("restart_attempts", 1),
    lambda md: md["github_readback"].__setitem__("actor", "007AION"),
    lambda md: md["github_readback"].__setitem__("byte_exact", 1),
    lambda md: md["source_identity"].__setitem__("rollback_commit", "f" * 40),
    lambda md: md["focused_tests"].__setitem__("failed", True),
    lambda md: md["post_test_resource_readback"].__setitem__("oom_kill", 1),
    lambda md: md["post_test_resource_readback"].__setitem__("pids_events_max", -1),
    lambda md: md["static_checks"].__setitem__("ruff", "FAIL"),
    lambda md: md["packet_blockers"].append(dict(md["packet_blockers"][0])),
    lambda md: md["packet_blockers"][0].__setitem__("id", "UNAUTHENTICATED"),
    lambda md: md.__setitem__("new_control_plane_count", False),
    lambda md: md.__setitem__("secret_exposure", "present"),
])
def _cold_reviewed_author_repair_phase_packet_drift_fails_closed_without_mutation(
    kanban_home, aion_gov_src, mutate,
):
    with kb.connect() as conn:
        original, _repaired = _canonical_factory_repair_phase_chain(
            conn, resident_mutate=mutate,
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, original["author"]) is None
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(("packet", "mutate"), [
    ("merge", lambda md: md.__setitem__("audited_head", "f" * 40)),
    ("merge", lambda md: md["exact_audit"].__setitem__("run", True)),
    ("install", lambda md: md.__setitem__("source_installed", False)),
    ("install", lambda md: md["install"].__setitem__("head", "f" * 40)),
])
def _cold_reviewed_author_original_failed_phase_drift_fails_closed_without_mutation(
    kanban_home, aion_gov_src, packet, mutate,
):
    with kb.connect() as conn:
        kwargs = {f"original_{packet}_mutate": mutate}
        original, _repaired = _canonical_factory_repair_phase_chain(conn, **kwargs)
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, original["author"]) is None
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("mutate", [
    lambda md: md.pop("outcome"),
    lambda md: md.__setitem__("unknown", True),
    lambda md: md["actor"].__setitem__("github", "007AION"),
    lambda md: md["actor"].__setitem__("distinct_from", ["GemAION", "007AION"]),
    lambda md: md["audit"].__setitem__("review_id", True),
    lambda md: md["audit"].__setitem__("task_id", "t_attacker"),
    lambda md: md["audit"].__setitem__("run_id", True),
    lambda md: md["canonical_checkout"].__setitem__("clean", 1),
    lambda md: md["checks"].__setitem__("bad", True),
    lambda md: md["checks"].__setitem__("pending", 1),
    lambda md: md["checks"].__setitem__("ci_status", "completed/failure"),
    lambda md: md["checks"].__setitem__("success", False),
    lambda md: md["child"].__setitem__("assignee", "agent007"),
    lambda md: md["child"].__setitem__("id", "t_attacker"),
    lambda md: md["child"].__setitem__("status", "ready"),
    lambda md: md["merge"].__setitem__("cas_sha_guard", "f" * 40),
    lambda md: md["merge"].__setitem__("parents", list(reversed(md["merge"]["parents"]))),
    lambda md: md["merge"].__setitem__("remote_main", "f" * 40),
    lambda md: md["changed_paths"].append(md["changed_paths"][0]),
    lambda md: md.__setitem__("new_runtime_module_count", False),
])
def _cold_actual_role_separated_merge_packet_drift_fails_closed_without_mutation(
    kanban_home, aion_gov_src, mutate,
):
    with kb.connect() as conn:
        original, _repaired = _canonical_factory_repair_phase_chain(
            conn, merge_mutate=mutate,
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, original["author"]) is None
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("mutate", [
    lambda md: md.pop("typed_symbols"),
    lambda md: md.__setitem__("unknown", True),
    lambda md: md.__setitem__("canonical_run_id", True),
    lambda md: md.__setitem__("activation_performed", 0),
    lambda md: md.__setitem__("source_installed", 1),
    lambda md: md.__setitem__("audited_head", "f" * 40),
    lambda md: md["changed_files"].append(md["changed_files"][0]),
    lambda md: md["blobs"].pop(next(iter(md["blobs"]))),
    lambda md: md["install"].__setitem__("head", "f" * 40),
    lambda md: md["install"].__setitem__("parents", list(reversed(md["install"]["parents"]))),
    lambda md: md["install"].__setitem__("method", "force"),
    lambda md: md["fresh_runtime"].__setitem__("bytes_match", 1),
    lambda md: md["fresh_runtime"].__setitem__("module_sha256", "malformed"),
    lambda md: md["typed_symbols"].pop("_canonical_factory_review_packet_present_callable"),
    lambda md: md["typed_symbols"].__setitem__("monotonic_fix_active_enter_lt_exec_start", 1),
    lambda md: md.__setitem__("receipt_actor", "007AION"),
    lambda md: md.__setitem__("receipt_actor_id", True),
    lambda md: md.__setitem__("install_epoch", False),
    lambda md: md["resident_runtime"].__setitem__("hermes_gateway_gm2_NRestarts", "1"),
    lambda md: md["resident_runtime"].__setitem__("hermes_gateway_gm2_main_pid", True),
    lambda md: md["tests"].__setitem__("ruff", "FAIL"),
    lambda md: md.__setitem__("new_runtime_module_count", False),
    lambda md: md["artifacts"].append(md["artifacts"][0]),
])
def _cold_actual_installed_source_packet_drift_fails_closed_without_mutation(
    kanban_home, aion_gov_src, mutate,
):
    with kb.connect() as conn:
        original, _repaired = _canonical_factory_repair_phase_chain(
            conn, install_mutate=mutate,
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, original["author"]) is None
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_repair_phase_ambiguous_edge_fails_closed_without_mutation(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        original, _repaired = _canonical_factory_repair_phase_chain(conn)
        kb.create_task(
            conn, title="ambiguous sibling", factory_build_gate=1,
            assignee="gm", parents=[original["reviewer"]],
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, original["author"]) is None
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("author_assignee", ["agent007", "gm", "gm2"])
def test_canonical_audit_receipt_finalizes_generic_author_and_recomputes_child(
    kanban_home, aion_gov_src, author_assignee,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(
            conn, author_assignee=author_assignee,
        )
        before = _native_state_snapshot(conn)

        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt == kb._canonical_audit_receipt(conn, chain["author"])
        assert receipt == {
            "task_id": chain["author"],
            "subject_id": f"{chain['author']}/{chain['author_run']}",
            "subject_version_or_exact_hash": chain["handoff"].receipt_sha256,
            "author_task_id": chain["author"],
            "author_run_id": chain["author_run"],
            "author_profile": author_assignee,
            "auditor_task_id": chain["reviewer"],
            "auditor_run_id": chain["reviewer_run"],
            "auditor_profile": "bafuxunan",
            "verdict": "PASS",
            "issued_at": receipt["issued_at"],
            "receipt_hash": receipt["receipt_hash"],
            "authenticated": True,
        }
        assert re.fullmatch(r"[0-9a-f]{64}", receipt["receipt_hash"])
        assert kb._reviewed_author_finalizer_run_id(
            conn, chain["author"],
        ) == chain["author_run"]
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="canonical receipt terminalized",
        )
        author_task = kb.get_task(conn, chain["author"])
        child_task = kb.get_task(conn, chain["child"])
        assert author_task is not None and author_task.status == "done"
        assert child_task is not None and child_task.status == "ready"
        completed = _native_state_snapshot(conn)
        assert not kb.complete_task(
            conn, chain["author"], summary="idempotent replay",
        )
        assert _native_state_snapshot(conn) == completed


def test_canonical_audit_receipt_selects_latest_handoff_after_terminal_request_changes(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_multi_round_audit_receipt_chain(conn)
        before = _native_state_snapshot(conn)

        assert all(
            kb._historical_auditor_child_is_non_authoritative(
                conn,
                author_task_id=chain["author"],
                author_profile="agent007",
                auditor_task_id=historical,
                auditor_profile="bafuxunan",
                latest_handoff_event_id=chain["handoff"].event_id,
            )
            for historical in chain["historical"]
        )
        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt["author_run_id"] == chain["author_run"]
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert receipt["auditor_run_id"] == chain["reviewer_run"]
        assert receipt["verdict"] == "PASS"
        assert _native_state_snapshot(conn) == before


def _canonical_reused_current_auditor_chain(conn):
    author = kb.create_task(
        conn, title="author with reused current auditor", factory_build_gate=1,
        assignee="agent007",
    )
    archived = kb.create_task(
        conn, title="authenticated superseded sibling", assignee="bafuxunan",
        parents=[author],
    )
    with kb._authenticated_strict_orchestrator_archive():
        assert kb.archive_task(
            conn, archived, reason="superseded before ordered final audit",
            actor="kanban-orchestrator", source="kanban_archive",
            fail_if_active_run=True, expected_status="todo",
        )
    reviewer = kb.create_task(
        conn, title="reused current auditor", assignee="bafuxunan",
        parents=[author],
    )
    rounds = []
    blocked_run = None
    for index in range(2):
        author_run = _claim_and_run_id(conn, author)
        handoff = kb.request_review_handoff(
            conn, author, expected_run_id=author_run,
            review_task_id=reviewer, reason=f"candidate {index}",
        )
        assert handoff is not None
        review_run = _claim_and_run_id(conn, reviewer)
        verdict = "request_changes" if index == 0 else "pass"
        assert _record_legacy_review_verdict_fixture(
            conn, author, review_task_id=reviewer,
            expected_review_run_id=review_run, verdict=verdict,
            reason=f"{verdict.upper()}_CANDIDATE_{index}",
        )
        rounds.append((author_run, handoff, review_run))
        if index == 0:
            blocked_run = _claim_and_run_id(conn, author)
            assert kb.block_task(
                conn, author, reason="intermediate non-review work paused",
                kind="needs_input", expected_run_id=blocked_run,
            )
            assert kb.unblock_task(conn, author)
    assert blocked_run is not None
    assert kb.complete_task(
        conn, reviewer, expected_run_id=rounds[-1][2],
        summary="latest reused-current audit passed",
    )
    child = kb.create_task(
        conn, title="ordered multi-round downstream", assignee="gm2",
        parents=[author],
    )
    return {
        "author": author,
        "reviewer": reviewer,
        "archived": archived,
        "blocked_run": blocked_run,
        "child": child,
        "rounds": rounds,
        "first_author_run": rounds[0][0],
        "first_handoff": rounds[0][1],
        "first_review_run": rounds[0][2],
        "latest_author_run": rounds[-1][0],
        "latest_handoff": rounds[-1][1],
        "latest_review_run": rounds[-1][2],
    }


def test_canonical_audit_receipt_authenticates_ordered_reused_current_auditor(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_reused_current_auditor_chain(conn)
        before = _native_state_snapshot(conn)

        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt["author_run_id"] == chain["latest_author_run"]
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert receipt["auditor_run_id"] == chain["latest_review_run"]
        assert receipt["verdict"] == "PASS"
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="ordered multi-round terminalized",
        )
        author_task = kb.get_task(conn, chain["author"])
        child_task = kb.get_task(conn, chain["child"])
        assert author_task is not None and author_task.status == "done"
        assert child_task is not None and child_task.status == "ready"


@pytest.mark.parametrize(
    "drift",
    [
        "missing_handoff", "duplicate_handoff", "wrong_handoff_task",
        "handoff_receipt_drift", "missing_verdict", "duplicate_verdict",
        "conflicting_verdict", "missing_mirror", "duplicate_mirror",
        "mirror_before_author", "mirror_after_terminal", "historical_pass",
        "final_request_changes", "skipped_round", "out_of_order_round",
        "wrong_review_run", "open_author_run", "open_nonreview_run",
        "open_review_run", "null_review_run",
        "post_final_verdict", "archive_provenance_drift", "missing_edge",
        "wrong_role", "stale_author_claim", "stale_auditor_claim",
        "ambiguous_sibling",
    ],
)
def test_ordered_reused_current_auditor_hostile_drift_is_zero_mutation(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        chain = _canonical_reused_current_auditor_chain(conn)
        author, reviewer = chain["author"], chain["reviewer"]
        first_handoff = chain["first_handoff"]
        first_run = chain["first_review_run"]
        latest_run = chain["latest_review_run"]
        if drift == "missing_handoff":
            conn.execute("DELETE FROM task_events WHERE id=?", (first_handoff.event_id,))
        elif drift == "duplicate_handoff":
            row = conn.execute(
                "SELECT task_id,kind,payload,run_id,created_at FROM task_events WHERE id=?",
                (first_handoff.event_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                "VALUES (?,?,?,?,?)", tuple(row),
            )
        elif drift in {"wrong_handoff_task", "handoff_receipt_drift"}:
            row = conn.execute(
                "SELECT id,payload FROM task_events WHERE id=?", (first_handoff.event_id,),
            ).fetchone()
            payload = json.loads(row["payload"])
            if drift == "wrong_handoff_task":
                payload["review_task_id"] = chain["child"]
            else:
                payload["receipt_sha256"] = "f" * 64
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload), row["id"]),
            )
        elif drift in {"missing_verdict", "duplicate_verdict", "conflicting_verdict"}:
            row = conn.execute(
                "SELECT id,task_id,kind,payload,run_id,created_at FROM task_events "
                "WHERE task_id=? AND kind='review_verdict' AND run_id=?",
                (author, first_run),
            ).fetchone()
            if drift == "missing_verdict":
                conn.execute("DELETE FROM task_events WHERE id=?", (row["id"],))
            else:
                payload = json.loads(row["payload"])
                if drift == "conflicting_verdict":
                    payload["verdict"] = "pass"
                conn.execute(
                    "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                    "VALUES (?,?,?,?,?)",
                    (row["task_id"], row["kind"], json.dumps(payload), row["run_id"],
                     row["created_at"]),
                )
        elif drift in {"missing_mirror", "duplicate_mirror"}:
            row = conn.execute(
                "SELECT id,task_id,kind,payload,run_id,created_at FROM task_events "
                "WHERE task_id=? AND kind='review_verdict' AND run_id=?",
                (reviewer, first_run),
            ).fetchone()
            if drift == "missing_mirror":
                conn.execute("DELETE FROM task_events WHERE id=?", (row["id"],))
            else:
                conn.execute(
                    "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                    "VALUES (?,?,?,?,?)", tuple(row)[1:],
                )
        elif drift in {"mirror_before_author", "mirror_after_terminal"}:
            author_event_id = int(conn.execute(
                "SELECT id FROM task_events WHERE task_id=? "
                "AND kind='review_verdict' AND run_id=?", (author, first_run),
            ).fetchone()["id"])
            mirror_event_id = int(conn.execute(
                "SELECT id FROM task_events WHERE task_id=? "
                "AND kind='review_verdict' AND run_id=?", (reviewer, first_run),
            ).fetchone()["id"])
            moved_id = (
                author_event_id - 100000
                if drift == "mirror_before_author"
                else int(conn.execute("SELECT MAX(id) FROM task_events").fetchone()[0])
                + 100000
            )
            conn.execute(
                "UPDATE task_events SET id=? WHERE id=?", (moved_id, mirror_event_id),
            )
        elif drift in {"historical_pass", "final_request_changes"}:
            run_id = first_run if drift == "historical_pass" else latest_run
            verdict = "pass" if drift == "historical_pass" else "request_changes"
            for task_id in (author, reviewer):
                row = conn.execute(
                    "SELECT id,payload FROM task_events WHERE task_id=? "
                    "AND kind='review_verdict' AND run_id=?", (task_id, run_id),
                ).fetchone()
                payload = json.loads(row["payload"])
                payload["verdict"] = verdict
                conn.execute(
                    "UPDATE task_events SET payload=? WHERE id=?",
                    (json.dumps(payload), row["id"]),
                )
        elif drift == "skipped_round":
            conn.execute(
                "DELETE FROM task_events WHERE task_id IN (?,?) "
                "AND kind='review_verdict' AND run_id=?",
                (author, reviewer, first_run),
            )
        elif drift == "out_of_order_round":
            conn.execute(
                "UPDATE task_events SET id=id+100000 WHERE id=?",
                (first_handoff.event_id,),
            )
        elif drift == "wrong_review_run":
            row = conn.execute(
                "SELECT id,payload FROM task_events WHERE task_id=? "
                "AND kind='review_verdict' AND run_id=?", (author, first_run),
            ).fetchone()
            payload = json.loads(row["payload"])
            payload["review_run_id"] = latest_run
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload), row["id"]),
            )
        elif drift in {"open_author_run", "open_nonreview_run"}:
            conn.execute(
                "UPDATE task_runs SET ended_at=NULL WHERE id=?",
                (
                    chain["first_author_run"]
                    if drift == "open_author_run"
                    else chain["blocked_run"],
                ),
            )
        elif drift in {"open_review_run", "null_review_run"}:
            conn.execute(
                "UPDATE task_runs SET ended_at=NULL WHERE id=?",
                (first_run if drift == "open_review_run" else latest_run,),
            )
        elif drift == "post_final_verdict":
            payload = conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? "
                "AND kind='review_verdict' AND run_id=?", (author, latest_run),
            ).fetchone()["payload"]
            kb._append_event(
                conn, author, "review_verdict", json.loads(payload), run_id=latest_run,
            )
        elif drift == "archive_provenance_drift":
            conn.execute(
                "DELETE FROM task_events WHERE task_id=? "
                "AND kind='strict_orchestrator_archive_authenticated'",
                (chain["archived"],),
            )
        elif drift == "missing_edge":
            conn.execute(
                "DELETE FROM task_links WHERE parent_id=? AND child_id=?",
                (author, reviewer),
            )
        elif drift == "wrong_role":
            conn.execute("UPDATE tasks SET assignee='agent007' WHERE id=?", (reviewer,))
        elif drift in {"stale_author_claim", "stale_auditor_claim"}:
            task_id = author if drift == "stale_author_claim" else reviewer
            conn.execute("UPDATE tasks SET claim_lock='stale' WHERE id=?", (task_id,))
        else:
            kb.create_task(
                conn, title="ambiguous audit sibling", assignee="bafuxunan",
                parents=[author],
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, author) is None
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    ("fixture", "author_profile", "auditor_profile"),
    [(_canonical_reused_auditor_history_chain, "bafuxunan", "elder-senate")],
)
def test_canonical_audit_receipt_authenticates_residual_historical_variants(
    kanban_home, aion_gov_src, fixture, author_profile, auditor_profile,
):
    with kb.connect() as conn:
        chain = fixture(conn)
        before = _native_state_snapshot(conn)

        assert kb._historical_auditor_child_is_non_authoritative(
            conn,
            author_task_id=chain["author"],
            author_profile=author_profile,
            auditor_task_id=chain["historical"],
            auditor_profile=auditor_profile,
            latest_handoff_event_id=chain["handoff"].event_id,
        )
        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt["author_run_id"] == chain["author_run"]
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert receipt["auditor_run_id"] == chain["reviewer_run"]
        assert receipt["verdict"] == "PASS"
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="residual history terminalized on copy",
        )
        author = kb.get_task(conn, chain["author"])
        assert author is not None and author.status == "done"


def _superseded_completed_recovery_history_hostile_drift_zero_mutation(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        chain = _canonical_completed_recovery_history_chain(conn)
        event = conn.execute(
            "SELECT payload FROM task_events WHERE id = ?",
            (chain["recovery_event"],),
        ).fetchone()
        payload = json.loads(event["payload"])
        if drift == "missing_field":
            payload.pop("controller")
        elif drift == "extra_field":
            payload["caller_claim"] = True
        elif drift == "type_drift":
            payload["version"] = True
        elif drift == "reason_mismatch":
            payload["reason"] = "controller prose"
        elif drift == "receipt_mismatch":
            payload["recovery_receipt"]["head"] = "f" * 40
        elif drift == "metadata_mismatch":
            run = conn.execute(
                "SELECT metadata FROM task_runs WHERE id = ?",
                (chain["historical_run"],),
            ).fetchone()
            metadata = json.loads(run["metadata"])
            metadata["head"] = "f" * 40
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), chain["historical_run"]),
            )
        elif drift == "controller_nonterminal":
            conn.execute(
                "UPDATE tasks SET status='running', current_run_id=? WHERE id=?",
                (chain["controller_run"], chain["controller"]),
            )
            conn.execute(
                "UPDATE task_runs SET status='running', outcome=NULL, ended_at=NULL "
                "WHERE id=?", (chain["controller_run"],),
            )
        elif drift == "controller_wrong_profile":
            conn.execute(
                "UPDATE tasks SET assignee='gm' WHERE id=?", (chain["controller"],),
            )
        elif drift == "controller_nonlatest":
            conn.execute(
                "INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'gm2', 'done', 'completed', 1, 2)",
                (chain["controller"],),
            )
        elif drift == "cross_controller_run":
            payload["controller"]["run_id"] = chain["reviewer_run"]
        elif drift == "child_nonlatest":
            conn.execute(
                "INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'bafuxunan', 'done', 'completed', 1, 2)",
                (chain["historical"],),
            )
        elif drift == "duplicate_recovery":
            row = conn.execute(
                "SELECT task_id, kind, payload, run_id, created_at FROM task_events "
                "WHERE id=?", (chain["recovery_event"],),
            ).fetchone()
            conn.execute(
                "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)", tuple(row),
            )
        elif drift == "ordinary_done_no_recovery":
            conn.execute(
                "DELETE FROM task_events WHERE id=?", (chain["recovery_event"],),
            )
        elif drift == "unexpected_mirror":
            conn.execute(
                "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                "VALUES (?, 'review_verdict', ?, ?, 1)",
                (chain["historical"], json.dumps(payload), chain["historical_run"]),
            )
        elif drift == "wrong_role":
            conn.execute(
                "UPDATE tasks SET assignee='gm' WHERE id=?", (chain["historical"],),
            )
        elif drift == "missing_edge":
            conn.execute(
                "DELETE FROM task_links WHERE parent_id=? AND child_id=?",
                (chain["author"], chain["historical"]),
            )
        elif drift == "post_latest_event":
            kb._append_event(conn, chain["historical"], "status", {"status": "done"})
        if drift in {
            "missing_field", "extra_field", "type_drift", "reason_mismatch",
            "receipt_mismatch", "cross_controller_run",
        }:
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload), chain["recovery_event"]),
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    "drift",
    [
        "missing_handoff", "duplicate_handoff", "missing_mirror",
        "duplicate_mirror", "skipped_round", "mismatched_round",
        "out_of_order_round", "run_drift", "historical_pass", "open_run",
        "nonterminal_child", "archive_payload_drift", "duplicate_archive",
        "unexpected_post_latest_event",
    ],
)
def test_reused_auditor_history_hostile_drift_zero_mutation(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        chain = _canonical_reused_auditor_history_chain(conn)
        first_handoff, first_run = chain["rounds"][0]
        second_handoff, second_run = chain["rounds"][1]
        if drift == "missing_handoff":
            conn.execute("DELETE FROM task_events WHERE id=?", (first_handoff.event_id,))
        elif drift == "duplicate_handoff":
            row = conn.execute(
                "SELECT task_id,kind,payload,run_id,created_at FROM task_events WHERE id=?",
                (first_handoff.event_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                "VALUES (?,?,?,?,?)", tuple(row),
            )
        elif drift in {"missing_mirror", "duplicate_mirror"}:
            row = conn.execute(
                "SELECT id,task_id,kind,payload,run_id,created_at FROM task_events "
                "WHERE task_id=? AND kind='review_verdict' AND run_id=?",
                (chain["historical"], first_run),
            ).fetchone()
            if drift == "missing_mirror":
                conn.execute("DELETE FROM task_events WHERE id=?", (row["id"],))
            else:
                conn.execute(
                    "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                    "VALUES (?,?,?,?,?)",
                    (row["task_id"], row["kind"], row["payload"], row["run_id"],
                     row["created_at"]),
                )
        elif drift == "skipped_round":
            conn.execute(
                "DELETE FROM task_events WHERE task_id=? AND kind='review_verdict' "
                "AND run_id=?", (chain["author"], second_run),
            )
        elif drift == "mismatched_round":
            row = conn.execute(
                "SELECT id,payload FROM task_events WHERE task_id=? "
                "AND kind='review_verdict' AND run_id=?",
                (chain["author"], second_run),
            ).fetchone()
            payload = json.loads(row["payload"])
            payload["review_task_id"] = chain["reviewer"]
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=?",
                (json.dumps(payload), row["id"]),
            )
        elif drift == "out_of_order_round":
            conn.execute(
                "UPDATE task_events SET id=id+1000000 WHERE id=?",
                (first_handoff.event_id,),
            )
        elif drift == "run_drift":
            conn.execute(
                "UPDATE task_runs SET status='done', outcome='completed' WHERE id=?",
                (first_run,),
            )
        elif drift == "historical_pass":
            for task_id in (chain["author"], chain["historical"]):
                row = conn.execute(
                    "SELECT id,payload FROM task_events WHERE task_id=? "
                    "AND kind='review_verdict' AND run_id=?",
                    (task_id, first_run),
                ).fetchone()
                payload = json.loads(row["payload"])
                payload["verdict"] = "pass"
                conn.execute(
                    "UPDATE task_events SET payload=? WHERE id=?",
                    (json.dumps(payload), row["id"]),
                )
        elif drift == "open_run":
            conn.execute(
                "UPDATE task_runs SET ended_at=NULL WHERE id=?", (first_run,),
            )
        elif drift == "nonterminal_child":
            conn.execute(
                "UPDATE tasks SET status='todo' WHERE id=?", (chain["historical"],),
            )
        elif drift == "archive_payload_drift":
            conn.execute(
                "UPDATE task_events SET payload='{}' WHERE id=?",
                (chain["archive_event"],),
            )
        elif drift == "duplicate_archive":
            row = conn.execute(
                "SELECT task_id,kind,payload,run_id,created_at FROM task_events WHERE id=?",
                (chain["archive_event"],),
            ).fetchone()
            conn.execute(
                "INSERT INTO task_events(task_id,kind,payload,run_id,created_at) "
                "VALUES (?,?,?,?,?)", tuple(row),
            )
        else:
            kb._append_event(conn, chain["historical"], "status", {"status": "archived"})
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


def test_canonical_audit_receipt_ignores_pre_handoff_archived_same_profile_sibling(
    kanban_home, aion_gov_src, monkeypatch,
):
    with kb.connect() as conn:
        author = kb.create_task(
            conn, title="generic reviewed author", factory_build_gate=1,
            assignee="agent007",
        )
        archived = kb.create_task(
            conn, title="explicitly superseded audit", assignee="bafuxunan",
            parents=[author],
        )
        # The strict factory write consumes two earlier clock samples. Put the
        # second boundary exactly between the archived and authentication
        # events that used to sample time independently.
        boundary_times = iter((900, 900, 1000, 1001))
        with monkeypatch.context() as patch_context:
            patch_context.setattr(kb.time, "time", lambda: next(boundary_times))
            with kb._authenticated_strict_orchestrator_archive():
                assert kb.archive_task(
                    conn, archived, reason="superseded before exact audit selection",
                    actor="kanban-orchestrator", source="kanban_archive",
                    fail_if_active_run=True, expected_status="todo",
                )
        archive_events = conn.execute(
            "SELECT kind, created_at FROM task_events WHERE task_id = ? "
            "AND kind IN ('archived', 'strict_orchestrator_archive_authenticated') "
            "ORDER BY id",
            (archived,),
        ).fetchall()
        assert [(row["kind"], row["created_at"]) for row in archive_events] == [
            ("archived", 1000),
            ("strict_orchestrator_archive_authenticated", 1000),
        ]
        auth_payload = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'strict_orchestrator_archive_authenticated'",
            (archived,),
        ).fetchone()["payload"])
        assert auth_payload == {"version": 1}
        reviewer = kb.create_task(
            conn, title="role-separated audit", factory_build_gate=1,
            assignee="bafuxunan", parents=[author],
        )
        author_run = _claim_and_run_id(conn, author)
        handoff = kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=reviewer,
            reason="exact Native receipt audit",
        )
        assert handoff is not None
        reviewer_run = _claim_and_run_id(conn, reviewer)
        assert _record_legacy_review_verdict_fixture(
            conn, author, review_task_id=reviewer,
            expected_review_run_id=reviewer_run, verdict="pass",
            reason="PASS_EXACT_NATIVE_RECEIPT",
        )
        assert kb.complete_task(
            conn, reviewer, expected_run_id=reviewer_run,
            summary="independent audit passed",
        )
        before = _native_state_snapshot(conn)

        assert kb._historical_auditor_child_is_non_authoritative(
            conn,
            author_task_id=author,
            author_profile="agent007",
            auditor_task_id=archived,
            auditor_profile="bafuxunan",
            latest_handoff_event_id=handoff.event_id,
        )
        receipt = kb._canonical_audit_receipt(conn, author)

        assert receipt is not None
        assert receipt["auditor_task_id"] == reviewer
        assert _native_state_snapshot(conn) == before


def test_canonical_audit_receipt_rejects_forged_pre_handoff_archive_labels(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author = kb.create_task(
            conn, title="generic reviewed author", factory_build_gate=1,
            assignee="agent007",
        )
        archived = kb.create_task(
            conn, title="forged superseded audit", assignee="bafuxunan",
            parents=[author],
        )
        # Direct callers can forge actor/source strings, but cannot make
        # archive_task persist the authenticated strict-orchestrator marker.
        assert kb.archive_task(
            conn, archived, reason="caller-controlled labels",
            actor="kanban-orchestrator", source="kanban_archive",
        )
        assert conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? "
            "AND kind = 'strict_orchestrator_archive_authenticated'",
            (archived,),
        ).fetchone() is None
        reviewer = kb.create_task(
            conn, title="role-separated audit", factory_build_gate=1,
            assignee="bafuxunan", parents=[author],
        )
        author_run = _claim_and_run_id(conn, author)
        handoff = kb.request_review_handoff(
            conn, author, expected_run_id=author_run, review_task_id=reviewer,
            reason="exact Native receipt audit",
        )
        assert handoff is not None
        reviewer_run = _claim_and_run_id(conn, reviewer)
        assert _record_legacy_review_verdict_fixture(
            conn, author, review_task_id=reviewer,
            expected_review_run_id=reviewer_run, verdict="pass",
            reason="PASS_EXACT_NATIVE_RECEIPT",
        )
        assert kb.complete_task(
            conn, reviewer, expected_run_id=reviewer_run,
            summary="independent audit passed",
        )
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, author) is None
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary="reject forged archive labels")
        assert _native_state_snapshot(conn) == before


def test_canonical_audit_receipt_rejects_post_handoff_archive_without_prior_verdict(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(conn, author_assignee="agent007")
        competitor = kb.create_task(
            conn, title="post-handoff competing audit", assignee="bafuxunan",
            parents=[chain["author"]],
        )
        assert kb.archive_task(
            conn, competitor, reason="archive cannot manufacture supersession",
            actor="kanban-orchestrator", source="kanban_archive",
        )
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(
                conn, chain["author"], summary="reject post-handoff archive",
            )
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("archive_drift", ["missing", "malformed", "duplicate"])
def test_canonical_audit_receipt_rejects_bad_archive_provenance_zero_mutation(
    kanban_home, aion_gov_src, archive_drift,
):
    with kb.connect() as conn:
        chain = _canonical_multi_round_audit_receipt_chain(conn)
        historical = chain["historical"][0]
        archive_token = kb._STRICT_ORCHESTRATOR_ARCHIVE_AUTH.set(True)
        try:
            assert kb.archive_task(
                conn, historical, reason="authenticated earlier request changes",
                actor="kanban-orchestrator", source="kanban_archive",
                fail_if_active_run=True, expected_status="todo",
            )
        finally:
            kb._STRICT_ORCHESTRATOR_ARCHIVE_AUTH.reset(archive_token)
        archive_row = conn.execute(
            "SELECT id, payload, created_at FROM task_events WHERE task_id = ? "
            "AND kind = 'archived'", (historical,),
        ).fetchone()
        if archive_drift == "missing":
            conn.execute("DELETE FROM task_events WHERE id = ?", (archive_row["id"],))
        elif archive_drift == "malformed":
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (json.dumps({"source": "kanban_archive"}), archive_row["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO task_events(task_id, kind, payload, created_at) "
                "VALUES (?, 'archived', ?, ?)",
                (historical, archive_row["payload"], archive_row["created_at"]),
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(
                conn, chain["author"], summary="reject bad archive provenance",
            )
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("competitor_status", ["todo", "ready", "running", "review"])
def test_canonical_audit_receipt_rejects_live_same_profile_competitor_zero_mutation(
    kanban_home, aion_gov_src, competitor_status,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(conn, author_assignee="agent007")
        competitor = kb.create_task(
            conn, title="competing audit", assignee="bafuxunan",
            parents=[chain["author"]],
        )
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (competitor_status, competitor),
        )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject live competitor")
        assert _native_state_snapshot(conn) == before


def test_canonical_audit_receipt_rejects_malformed_historical_verdict_zero_mutation(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_multi_round_audit_receipt_chain(conn)
        historical = chain["historical"][0]
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? "
            "AND kind = 'review_verdict'", (historical,),
        ).fetchone()
        payload = json.loads(row["payload"])
        payload["review_run_id"] = "malformed"
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), row["id"]),
        )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject malformed history")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    "drift",
    [
        "author_profile", "auditor_profile", "self_audit", "missing_edge",
        "duplicate_auditor", "duplicate_handoff", "duplicate_verdict",
        "mixed_verdict", "missing_mirror", "null_author_run", "null_auditor_run",
        "extra_handoff_field", "handoff_version_bool", "handoff_run_bool",
        "handoff_task_nonstring", "handoff_reason_nonstring",
        "handoff_recovery_nonbool", "handoff_hash_malformed", "handoff_run_mismatch",
    ],
)
def test_canonical_audit_receipt_hostile_drift_is_zero_mutation(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(conn)
        author, reviewer = chain["author"], chain["reviewer"]
        if drift == "author_profile":
            conn.execute("UPDATE tasks SET assignee = 'gm' WHERE id = ?", (author,))
        elif drift in {"auditor_profile", "self_audit"}:
            profile = "gm" if drift == "auditor_profile" else "gm2"
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, reviewer))
        elif drift == "missing_edge":
            conn.execute(
                "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
                (author, reviewer),
            )
        elif drift == "duplicate_auditor":
            kb.create_task(
                conn, title="ambiguous auditor", assignee="bafuxunan",
                parents=[author],
            )
        elif drift == "duplicate_handoff":
            row = conn.execute(
                "SELECT payload, run_id FROM task_events WHERE task_id = ? "
                "AND kind = 'review_handoff'", (author,),
            ).fetchone()
            conn.execute(
                "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
                "VALUES (?, 'review_handoff', ?, ?, ?)",
                (author, row["payload"], row["run_id"], int(time.time())),
            )
        elif drift in {"duplicate_verdict", "mixed_verdict"}:
            row = conn.execute(
                "SELECT payload, run_id FROM task_events WHERE task_id = ? "
                "AND kind = 'review_verdict'", (author,),
            ).fetchone()
            payload = json.loads(row["payload"])
            if drift == "mixed_verdict":
                payload["approval_commit_id"] = "1" * 40
                conn.execute(
                    "UPDATE task_events SET payload = ? WHERE task_id = ? "
                    "AND kind = 'review_verdict'",
                    (json.dumps(payload), author),
                )
            else:
                conn.execute(
                    "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
                    "VALUES (?, 'review_verdict', ?, ?, ?)",
                    (author, row["payload"], row["run_id"], int(time.time())),
                )
        elif drift == "missing_mirror":
            conn.execute(
                "DELETE FROM task_events WHERE task_id = ? AND kind = 'review_verdict'",
                (reviewer,),
            )
        elif drift.startswith("handoff_") or drift == "extra_handoff_field":
            row = conn.execute(
                "SELECT id, payload FROM task_events WHERE task_id = ? "
                "AND kind = 'review_handoff'", (author,),
            ).fetchone()
            payload = json.loads(row["payload"])
            if drift == "extra_handoff_field":
                payload["approval_commit_id"] = "1" * 40
            elif drift == "handoff_version_bool":
                payload["version"] = True
            elif drift == "handoff_run_bool":
                payload["expected_run_id"] = True
            elif drift == "handoff_task_nonstring":
                payload["review_task_id"] = [reviewer]
            elif drift == "handoff_reason_nonstring":
                payload["reason"] = {"text": payload["reason"]}
            elif drift == "handoff_recovery_nonbool":
                payload["recovery"] = 0
            elif drift == "handoff_hash_malformed":
                payload["receipt_sha256"] = "not-a-sha256"
            else:
                conn.execute(
                    "UPDATE task_events SET run_id = ? WHERE id = ?",
                    (chain["reviewer_run"], row["id"]),
                )
            if drift != "handoff_run_mismatch":
                conn.execute(
                    "UPDATE task_events SET payload = ? WHERE id = ?",
                    (json.dumps(payload), row["id"]),
                )
        elif drift == "null_author_run":
            conn.execute(
                "UPDATE task_runs SET ended_at = NULL WHERE id = ?",
                (chain["author_run"],),
            )
        else:
            conn.execute(
                "UPDATE task_runs SET ended_at = NULL WHERE id = ?",
                (chain["reviewer_run"],),
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, author) is None
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize("scope", ["parent", "mirror", "paired"])
@pytest.mark.parametrize(("field", "invalid"), [
    ("version", True),
    ("review_task_id", None),
    ("review_run_id", "not-an-integer"),
    ("verdict", {"value": "pass"}),
    ("reason", False),
])
def test_canonical_audit_receipt_rejects_untyped_verdict_payloads_zero_mutation(
    kanban_home, aion_gov_src, scope, field, invalid,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(conn)
        targets = {
            "parent": [chain["author"]],
            "mirror": [chain["reviewer"]],
            "paired": [chain["author"], chain["reviewer"]],
        }[scope]
        for task_id in targets:
            row = conn.execute(
                "SELECT id, payload FROM task_events WHERE task_id = ? "
                "AND kind = 'review_verdict'", (task_id,),
            ).fetchone()
            payload = json.loads(row["payload"])
            payload[field] = invalid
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (json.dumps(payload), row["id"]),
            )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject untyped verdict")
        assert _native_state_snapshot(conn) == before


def test_canonical_audit_receipt_ignores_cold_business_packet_metadata(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(conn)
        receipt = kb._canonical_audit_receipt(conn, chain["author"])
        assert receipt is not None
        _rewrite_latest_run_metadata(
            conn,
            chain["reviewer"],
            lambda metadata: metadata.update({
                "approval_commit_id": "1" * 40,
                "canonical_factory_review_packet": {"verdict": "REJECTED"},
                "authenticated_non_pr_review_evidence": {"trusted": False},
                "merger_runtime_witness": {"resident": False},
            }),
        )
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) == receipt
        assert kb._reviewed_author_finalizer_run_id(
            conn, chain["author"],
        ) == chain["author_run"]
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="cold metadata ignored",
        )
        author_task = kb.get_task(conn, chain["author"])
        child_task = kb.get_task(conn, chain["child"])
        assert author_task is not None and author_task.status == "done"
        assert child_task is not None and child_task.status == "ready"


def test_cold_business_packet_metadata_cannot_replace_native_handoff(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_audit_receipt_chain(conn)
        conn.execute(
            "DELETE FROM task_events WHERE task_id = ? AND kind = 'review_handoff'",
            (chain["author"],),
        )
        _rewrite_latest_run_metadata(
            conn,
            chain["reviewer"],
            lambda metadata: metadata.update({
                "approval_commit_id": "1" * 40,
                "canonical_factory_review_packet": {"verdict": "PASS"},
                "authenticated_non_pr_review_evidence": {"trusted": True},
                "merger_runtime_witness": {"resident": True},
            }),
        )
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="legacy metadata refused")
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_accepts_current_canonical_factory_packet_chain(kanban_home, aion_gov_src):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn)
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) == chain["author_run"]
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(conn, chain["author"], summary="canonical chain terminalized")
        author = kb.get_task(conn, chain["author"])
        assert author is not None
        assert author.status == "done"


def _cold_reviewed_author_accepts_installed_source_factory_packet_chain(
    kanban_home, aion_gov_src,
):
    """The exact closed packet shapes emitted after the adapter remain consumable."""
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) == chain["author_run"]
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="installed-source chain terminalized",
        )
        author = kb.get_task(conn, chain["author"])
        assert author is not None
        assert author.status == "done"


def _cold_reviewed_author_accepts_real_systemd_lifecycle_order(
    kanban_home, aion_gov_src,
):
    """Active entry normally follows the main process start timestamp."""
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(
            conn, installed_source_shapes=True, real_systemd_order=True,
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(
            conn, chain["author"],
        ) == chain["author_run"]
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_accepts_exact_emitted_review_packet_at_canonical_boundary(
    kanban_home, aion_gov_src,
):
    """The current audit completion packet feeds the installed-source chain."""
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(
            conn, installed_source_shapes=True, emitted_review_shape=True,
        )
        assert kb._canonical_factory_review_packet(
            _terminal_run_metadata(conn, chain["reviewer"]),
        ) is not None
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(
            conn, chain["author"],
        ) == chain["author_run"]
        assert _native_state_snapshot(conn) == before


def _emitted_review_lean_compare(metadata):
    key = next(
        key for key in metadata
        if re.fullmatch(r"lean_pr[1-9][0-9]*_compare", key)
    )
    return metadata[key]


@pytest.mark.parametrize("mutate", [
    lambda md: md.pop("artifacts"),
    lambda md: md["artifacts"].append(md["artifacts"][0]),
    lambda md: md.__setitem__("pr", 64),
    lambda md: md.__setitem__("lean_pr65_compare", dict(_emitted_review_lean_compare(md))),
    lambda md: md.__setitem__("outcome", "APPROVE_EXACT_HEAD"),
    lambda md: md.__setitem__("repository", "attacker/repo"),
    lambda md: md.__setitem__("head", "f" * 40),
    lambda md: md.__setitem__("tree", "malformed"),
    lambda md: md.__setitem__("base", None),
    lambda md: md["github_review"].__setitem__("commit_id", "f" * 40),
    lambda md: md["github_review"].__setitem__("id", True),
    lambda md: md["github_review"].__setitem__("state", "CHANGES_REQUESTED"),
    lambda md: md["github_review"].__setitem__(
        "url", "https://github.com/kiddhu/hermes-agent/pull/65#pullrequestreview-12345",
    ),
    lambda md: md["hosted_ci"].__setitem__("failures", 1),
    lambda md: md["hosted_ci"].__setitem__("run", True),
    lambda md: md["hosted_ci"].__setitem__("terminal", 0),
    lambda md: _emitted_review_lean_compare(md).__setitem__(
        "finalizer_schema_branch_count", True,
    ),
    lambda md: _emitted_review_lean_compare(md).__setitem__("packet_family_count", 3),
    lambda md: _emitted_review_lean_compare(md).__setitem__(
        "new_control_plane_count", 1,
    ),
    lambda md: md["local_verification"].__setitem__("total_failed", True),
    lambda md: md["local_verification"].__setitem__("total_passed", 695),
    lambda md: md["local_verification"].__setitem__("ruff", "FAIL"),
    lambda md: md.__setitem__("forbidden_actions_performed", ["rewrite"]),
    lambda md: md.__setitem__("secret_exposure", "unknown"),
])
def _cold_reviewed_author_rejects_emitted_review_packet_drift_zero_mutation(
    kanban_home, aion_gov_src, mutate,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(
            conn, installed_source_shapes=True, emitted_review_shape=True,
        )
        _rewrite_latest_run_metadata(conn, chain["reviewer"], mutate)
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject emitted review drift")
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_emitted_review_edge_and_role_drift(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(
            conn, installed_source_shapes=True, emitted_review_shape=True,
        )
        conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (chain["author"], chain["reviewer"]),
        )
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(
            conn, installed_source_shapes=True, emitted_review_shape=True,
        )
        conn.execute(
            "UPDATE tasks SET assignee = 'agent007' WHERE id = ?",
            (chain["reviewer"],),
        )
        conn.execute(
            "UPDATE task_runs SET profile = 'agent007' WHERE task_id = ?",
            (chain["reviewer"],),
        )
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_mixed_installed_source_and_legacy_packet_families(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(
            conn,
            installed_source_shapes=True,
            legacy_installed_source_tail=True,
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject mixed packet families")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(("target", "mutate"), [
    ("merger", lambda md: md.__setitem__("head", "f" * 40)),
    ("merger", lambda md: md["audit"].__setitem__("native_run_id", -1)),
    ("merger", lambda md: md["actor"].__setitem__("github", "007AION")),
    ("merger", lambda md: md["merge"].__setitem__("parents", ["f" * 40, "1" * 40])),
    ("merger", lambda md: md["child"].__setitem__("id", "t_wrong")),
    ("merger", lambda md: md["child"].__setitem__("approved", True)),
    ("merger", lambda md: md.__setitem__("merge_commit", "f" * 40)),
    ("installer", lambda md: md.__setitem__("pr", 65)),
    ("installer", lambda md: md.__setitem__("merge", "f" * 40)),
    ("installer", lambda md: md["blobs"].__setitem__(next(iter(md["blobs"])), "f" * 40)),
    ("installer", lambda md: md.__setitem__("source_installed", False)),
    ("installer", lambda md: md["install"].__setitem__("rollback_ref", "")),
    ("installer", lambda md: md["fresh_runtime"].__setitem__("approved", True)),
    ("installer", lambda md: md.__setitem__("source_head", "f" * 40)),
    ("activation", lambda md: md["source"].__setitem__("head", "f" * 40)),
    ("activation", lambda md: md["source"].__setitem__("merge_commit", "f" * 40)),
    ("activation", lambda md: md["resident_runtime"].__setitem__(
        "active_enter_timestamp_monotonic", 900,
    )),
    ("activation", lambda md: md["resident_runtime"].__setitem__("main_pid", True)),
    ("activation", lambda md: md["source"].__setitem__("approved", True)),
    ("activation", lambda md: md.__setitem__("replay_restart_attempts", 1)),
    ("resident_audit", lambda md: md["native_replay"].__setitem__("run_id", -1)),
    ("resident_audit", lambda md: md["native_replay"].__setitem__("task", "t_wrong")),
    ("resident_audit", lambda md: md["external_receipt"].__setitem__("restart_count", True)),
    ("resident_audit", lambda md: md["source"].__setitem__("audited_head", "f" * 40)),
    ("resident_audit", lambda md: md["resident_runtime"].__setitem__("main_pid", -1)),
    ("resident_audit", lambda md: md["native_replay"].__setitem__("approved", True)),
    ("resident_audit", lambda md: md.__setitem__("barrier_tests", {"failed": 0, "passed": 1})),
])
def _cold_reviewed_author_rejects_installed_source_packet_drift_zero_mutation(
    kanban_home, aion_gov_src, target, mutate,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        _rewrite_latest_run_metadata(conn, chain[target], mutate)
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject installed-source drift")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("kanban_db_blob", "f" * 40),
        ("kanban_db_sha256", "f" * 64),
    ],
)
def _cold_reviewed_author_rejects_coordinated_installed_source_identity_drift(
    kanban_home, aion_gov_src, field, drift,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        _rewrite_latest_run_metadata(
            conn, chain["activation"],
            lambda md: md["source"].__setitem__(field, drift),
        )
        _rewrite_latest_run_metadata(
            conn, chain["resident_audit"],
            lambda md: md["source"].__setitem__(field, drift),
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject coordinated source drift")
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_coordinated_bool_runtime_identity_drift(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        for target in ("activation", "resident_audit"):
            _rewrite_latest_run_metadata(
                conn, chain[target],
                lambda md: md["resident_runtime"].__setitem__("main_pid", True),
            )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject bool runtime identity")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    ("target", "parent", "profile"),
    [
        ("merger", "reviewer", "gm"),
        ("installer", "merger", "merger"),
        ("resident_audit", "activation", "bafuxunan"),
    ],
)
def _cold_reviewed_author_rejects_duplicate_installed_source_packet_family(
    kanban_home, aion_gov_src, target, parent, profile,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        original = kb._authenticated_factory_run_metadata(conn, chain[target])
        assert original is not None
        duplicate = kb.create_task(
            conn, title="duplicate installed-source packet", factory_build_gate=1,
            assignee=profile, parents=[chain[parent]],
        )
        duplicate_run = _claim_and_run_id(conn, duplicate)
        assert kb.complete_task(
            conn, duplicate, expected_run_id=duplicate_run, metadata=original[2],
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_installed_source_missing_edge_and_self_audit(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (chain["activation"], chain["resident_audit"]),
        )
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn, installed_source_shapes=True)
        conn.execute("UPDATE tasks SET assignee = 'agent007' WHERE id = ?", (chain["reviewer"],))
        conn.execute("UPDATE task_runs SET profile = 'agent007' WHERE task_id = ?", (chain["reviewer"],))
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(("target", "mutate"), [
    ("reviewer", lambda md: md.pop("head")),
    ("reviewer", lambda md: md.__setitem__("audit_outcome", "PASS_EXACT_HEAD")),
    ("reviewer", lambda md: md.__setitem__("approved", False)),
    ("reviewer", lambda md: md.__setitem__("tests_passed", True)),
    ("reviewer", lambda md: md.__setitem__("github_review_url", "https://example.invalid")),
    ("merger", lambda md: md.__setitem__("audited_head", "f" * 40)),
    ("merger", lambda md: md.__setitem__("audited_tree", "f" * 40)),
    ("merger", lambda md: md.__setitem__("base", "f" * 40)),
    ("merger", lambda md: md["audit"].__setitem__("native_run_id", -1)),
    ("merger", lambda md: md["role_separation"].__setitem__("auditor", md["role_separation"]["author"])),
    ("installer", lambda md: md.__setitem__("source_pr", 65)),
    ("installer", lambda md: md.__setitem__("source_merge", "f" * 40)),
    ("activation", lambda md: md.__setitem__("replay_restart_attempts", 1)),
    ("activation", lambda md: md["external_activation_receipt"].__setitem__("restart_count", True)),
    ("activation", lambda md: md["source"].__setitem__("head", "f" * 40)),
    ("resident_audit", lambda md: md["parent_replay"].__setitem__("natural_claim_run", -1)),
    ("resident_audit", lambda md: md.__setitem__("outcome", "PASS")),
    ("resident_audit", lambda md: md.__setitem__("deep_health_exit_code", False)),
])
def _cold_reviewed_author_rejects_current_canonical_packet_drift_zero_mutation(kanban_home, aion_gov_src, target, mutate):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn)
        _rewrite_latest_run_metadata(conn, chain[target], mutate)
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject canonical drift")
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_current_canonical_missing_edge_and_self_audit(kanban_home, aion_gov_src):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn)
        conn.execute("DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (chain["activation"], chain["resident_audit"]))
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn)
        conn.execute("UPDATE tasks SET assignee = 'agent007' WHERE id = ?", (chain["reviewer"],))
        conn.execute("UPDATE task_runs SET profile = 'agent007' WHERE task_id = ?", (chain["reviewer"],))
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_current_canonical_ambiguous_runtime_audit(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn)
        original = kb._authenticated_factory_run_metadata(conn, chain["resident_audit"])
        assert original is not None
        duplicate = kb.create_task(
            conn, title="duplicate packet resident audit", factory_build_gate=1,
            assignee="bafuxunan", parents=[chain["activation"]],
        )
        duplicate_run = _claim_and_run_id(conn, duplicate)
        assert kb.complete_task(
            conn, duplicate, expected_run_id=duplicate_run,
            metadata=original[2],
        )
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_current_canonical_unauthenticated_packet_mutation(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _canonical_factory_packet_chain(conn)
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?",
            (chain["activation_run"],),
        ).fetchone()
        metadata = json.loads(row["metadata"])
        metadata["source"]["head"] = "f" * 40
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps(metadata, sort_keys=True), chain["activation_run"]),
        )
        conn.commit()
        before = _native_state_snapshot(conn)
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_accepts_authenticated_non_pr_reviewed_evidence_receipt(
    kanban_home, aion_gov_src,
):
    """The audited runtime/evidence path converges without inventing a code PR."""
    with kb.connect() as conn:
        author, author_run, _reviewer = _non_pr_reviewed_evidence_chain(conn)
        before = _native_state_snapshot(conn)

        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    ("drift", "mutate"),
    [
        ("missing_source_task", lambda md: md.pop("source_task_id")),
        ("cross_task", lambda md: md.__setitem__("source_task_id", "t_wrong")),
        ("null_source_run", lambda md: md.__setitem__("source_run_id", None)),
        ("stale_source_run", lambda md: md.__setitem__("source_run_id", -1)),
        ("acceptance_identity", lambda md: md.__setitem__("audit_outcome", "PASS")),
        ("evidence_hash", lambda md: md.__setitem__("evidence_sha256", "f" * 63)),
        ("missing_timer_hash", lambda md: md.pop("timer_sha256")),
        ("artifact_hash", lambda md: md.__setitem__("artifact_sha256", md["evidence_sha256"])),
        ("manifest_hash", lambda md: md.__setitem__("manifest_sha256", None)),
        ("receipt_hash", lambda md: md.__setitem__("github_receipt_body_sha256", "f" * 63)),
        ("receipt_url", lambda md: md.__setitem__("github_receipt", "prose-only receipt")),
        ("empty_checks", lambda md: md.__setitem__("checks", [])),
        ("duplicate_checks", lambda md: md["checks"].append(md["checks"][0])),
        ("mixed_pr_family", lambda md: md.__setitem__("head", "a" * 40)),
        ("unknown_alias", lambda md: md.__setitem__("caller_evidence", "trusted")),
    ],
)
def _cold_reviewed_author_rejects_non_pr_evidence_metadata_drift_zero_mutation(
    kanban_home, aion_gov_src, drift, mutate,
):
    with kb.connect() as conn:
        author, _author_run, reviewer = _non_pr_reviewed_evidence_chain(conn)
        _rewrite_latest_run_metadata(conn, reviewer, mutate)
        before = _native_state_snapshot(conn)

        assert kb._reviewed_author_finalizer_run_id(conn, author) is None, drift
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    "drift",
    [
        "missing_pass", "duplicate_pass", "pass_then_request_changes",
        "null_review_run", "stale_review_run", "cross_task_verdict",
        "missing_direct_edge", "self_audit", "wrong_reviewer_profile",
        "unauthenticated_reviewer_receipt", "prose_only",
    ],
)
def _cold_reviewed_author_rejects_non_pr_event_identity_and_receipt_drift(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        author, _author_run, reviewer = _non_pr_reviewed_evidence_chain(conn)
        verdict = conn.execute(
            "SELECT id, run_id, payload FROM task_events WHERE task_id = ? "
            "AND kind = 'review_verdict' ORDER BY id DESC LIMIT 1",
            (author,),
        ).fetchone()
        if drift == "missing_pass":
            conn.execute("DELETE FROM task_events WHERE id = ?", (verdict["id"],))
        elif drift == "duplicate_pass":
            conn.execute(
                "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
                "VALUES (?, 'review_verdict', ?, ?, ?)",
                (author, verdict["payload"], verdict["run_id"], int(time.time())),
            )
        elif drift == "pass_then_request_changes":
            payload = json.loads(verdict["payload"])
            payload["verdict"] = "request_changes"
            conn.execute(
                "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
                "VALUES (?, 'review_verdict', ?, ?, ?)",
                (author, json.dumps(payload), verdict["run_id"], int(time.time())),
            )
        elif drift in {"null_review_run", "stale_review_run", "cross_task_verdict"}:
            payload = json.loads(verdict["payload"])
            if drift == "null_review_run":
                payload["review_run_id"] = None
            elif drift == "stale_review_run":
                payload["review_run_id"] = -1
            else:
                payload["review_task_id"] = "t_wrong"
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (json.dumps(payload), verdict["id"]),
            )
        elif drift == "missing_direct_edge":
            conn.execute(
                "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
                (author, reviewer),
            )
        elif drift in {"self_audit", "wrong_reviewer_profile"}:
            profile = "agent007" if drift == "self_audit" else "gm2"
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, reviewer))
            conn.execute("UPDATE task_runs SET profile = ? WHERE task_id = ?", (profile, reviewer))
        elif drift == "unauthenticated_reviewer_receipt":
            conn.execute(
                "UPDATE task_attachments SET uploaded_by = 'agent' WHERE task_id = ?",
                (reviewer,),
            )
        elif drift == "prose_only":
            metadata = _terminal_run_metadata(conn, reviewer)
            metadata.pop("evidence_sha256")
            _set_terminal_run_metadata(conn, reviewer, metadata)
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._reviewed_author_finalizer_run_id(conn, author) is None, drift
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary=f"reject {drift}")
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_rejects_non_pr_evidence_with_pr_handoff_prose(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, *_ = _non_pr_reviewed_evidence_chain(
            conn, handoff_reason="PR #999 allegedly authorizes runtime evidence",
        )
        before = _native_state_snapshot(conn)

        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary="reject mixed PR/non-PR families")
        assert _native_state_snapshot(conn) == before


def _cold_reviewed_author_accepts_canonical_immutable_merger_receipt(
    kanban_home, aion_gov_src,
):
    """The real merger lane's immutable canonical schema authenticates exactly."""
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(
            conn,
            canonical_merger_receipt=True,
            source_pr=50,
            live_multi_child_shape=True,
        )

        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run
        assert kb.complete_task(conn, author, summary="canonical chain terminalized")
        assert kb.get_task(conn, author).status == "done"


def _cold_reviewed_author_accepts_immutable_pr54_receipts_and_runtime_witness(
    kanban_home, aion_gov_src,
):
    """Exact live shape resolves without rewriting immutable receipts."""
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(
            conn,
            immutable_pr54_receipts=True,
            activation_cycle=True,
            handoff_reason="exact immutable audit frozen without prose authority",
        )

        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


def _cold_reviewed_author_accepts_repaired_same_reviewer_request_changes_then_pass_current_pr56_chain(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, author_run, reviewer, *_ = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        verdicts = [
            json.loads(row["payload"])["verdict"]
            for row in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'review_verdict' ORDER BY id",
                (author,),
            ).fetchall()
            if json.loads(row["payload"])["review_task_id"] == reviewer
        ]
        assert verdicts == ["request_changes", "pass"]
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


@pytest.mark.parametrize(
    "handoff_reason",
    [
        "exact head frozen for independent audit",
        "PR #56 exact head frozen for independent audit",
    ],
)
def _cold_reviewed_author_uses_typed_audit_pr_with_optional_matching_handoff_corroboration(
    kanban_home, aion_gov_src, handoff_reason,
):
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            handoff_reason=handoff_reason,
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


def _cold_reviewed_author_accepts_authenticated_terminal_approve_exact_head_verdict(
    kanban_home, aion_gov_src,
):
    """Model the sanitized affected terminal evidence family without live IDs."""
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            terminal_audit_verdict="APPROVE_EXACT_HEAD",
            durable_terminal_gm_receipt=True,
            handoff_reason="PR #56 frozen for independent exact-head audit",
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


@pytest.mark.parametrize(
    "verdict",
    [
        "APPROVED_EXACT_HEAD",
        "approve_exact_head",
        " APPROVE_EXACT_HEAD",
        "APPROVE_EXACT_HEAD ",
        "APPROVE EXACT HEAD",
        "APPROVE_EXACT_HEAD_WITH_NOTES",
        "audit verdict: APPROVE_EXACT_HEAD",
        "PASS_EXACT_HEAD|APPROVE_EXACT_HEAD",
        "REQUEST_CHANGES_EXACT_HEAD",
        "APPROVE",
        "pass",
        "",
        None,
        True,
    ],
)
def _cold_reviewed_author_rejects_non_enumerated_terminal_audit_verdicts(
    kanban_home, aion_gov_src, verdict,
):
    with kb.connect() as conn:
        author, *_ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            terminal_audit_verdict=verdict,
            durable_terminal_gm_receipt=True,
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


def _cold_reviewed_author_rejects_conflicting_enumerated_terminal_audit_verdicts(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, _, _, merger, _ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            terminal_audit_verdict="PASS_EXACT_HEAD",
        )
        _rewrite_latest_run_metadata(
            conn,
            merger,
            lambda md: md.__setitem__("audit_verdict", "APPROVE_EXACT_HEAD"),
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda md: md.pop("canonical_run_id"),
        lambda md: md.__setitem__("implementation_task", "t_wrong"),
        lambda md: md.__setitem__("implementation_run", -1),
        lambda md: md.__setitem__("implementation_profile", "gm"),
        lambda md: md.__setitem__("implementation_github_actor", "kiddhu"),
        lambda md: md.__setitem__("exact_audit_task", "t_wrong"),
        lambda md: md.__setitem__("exact_audit_run", -1),
        lambda md: md.__setitem__("audit_profile", "agent007"),
        lambda md: md.__setitem__("audit_github_actor", "007AION"),
        lambda md: md.__setitem__("github_review_id", -1),
        lambda md: md.__setitem__("source_pr", 57),
        lambda md: md.__setitem__("source_pr_url", "https://github.com/attacker/repo/pull/56"),
        lambda md: md.__setitem__("audited_head", "f" * 40),
        lambda md: md.__setitem__("audited_tree", "f" * 40),
        lambda md: md.__setitem__("audited_base", "f" * 40),
        lambda md: md.__setitem__("changed_files", ["unexpected.py"]),
        lambda md: md["native_collision_readback"].__setitem__(
            "other_nonterminal_exact_merge_owners", 1,
        ),
        lambda md: md["native_collision_readback"].__setitem__(
            "other_nonterminal_exact_merge_owners", False,
        ),
        lambda md: md["hosted_checks"].__setitem__("failing", 1),
        lambda md: md["hosted_checks"].__setitem__("pending", False),
        lambda md: md["cas_merge"].__setitem__("attempts", 2),
        lambda md: md["cas_merge"].__setitem__("attempts", True),
        lambda md: md.__setitem__("merge_profile", "merger"),
        lambda md: md.__setitem__("merge_github_actor", "007AION"),
        lambda md: md.__setitem__("roles_distinct", False),
        lambda md: md.__setitem__("pr_state", "OPEN"),
        lambda md: md.__setitem__("merge_commit", "f" * 40),
        lambda md: md.__setitem__("merge_tree", "f" * 40),
        lambda md: md.__setitem__("merge_parents", list(reversed(md["merge_parents"]))),
        lambda md: md.__setitem__("canonical_main", "f" * 40),
        lambda md: md["audited_head_containment"].__setitem__("exact_second_parent", False),
        lambda md: md["audited_head_containment"].__setitem__("ahead_by", True),
        lambda md: md.__setitem__("runtime_install_performed", True),
        lambda md: md.__setitem__("typed_runtime_witness_performed", True),
        lambda md: md.__setitem__("reviewed_author_finalizer_performed", True),
        lambda md: md.__setitem__("source_edit_performed", True),
        lambda md: md.__setitem__("new_control_plane_count", 1),
        lambda md: md.__setitem__("new_control_plane_count", False),
        lambda md: md.__setitem__("forbidden_actions_performed", ["rewrite"]),
        lambda md: md.__setitem__("secret_exposure", "unknown"),
        lambda md: md.__setitem__("expected_head", md["audited_head"]),
    ],
)
def _cold_reviewed_author_rejects_durable_terminal_gm_receipt_drift_and_mixed_schema(
    kanban_home, aion_gov_src, mutate,
):
    with kb.connect() as conn:
        author, _, _, merger, _ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            terminal_audit_verdict="APPROVE_EXACT_HEAD",
            durable_terminal_gm_receipt=True,
        )
        _rewrite_latest_run_metadata(conn, merger, mutate)
        before = conn.total_changes
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        assert conn.total_changes == before


@pytest.mark.parametrize(
    "handoff_reason",
    [
        "PR #55 exact head frozen for independent audit",
        "PR #56 supersedes PR #55 at the exact audited head",
    ],
)
def _cold_reviewed_author_rejects_mismatched_or_multiple_handoff_pr_tokens(
    kanban_home, aion_gov_src, handoff_reason,
):
    with kb.connect() as conn:
        author, *_ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            handoff_reason=handoff_reason,
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


@pytest.mark.parametrize("typed_pr", [None, "56", [56], 0, -1])
def _cold_reviewed_author_rejects_missing_or_malformed_typed_audit_source_pr(
    kanban_home, aion_gov_src, typed_pr,
):
    with kb.connect() as conn:
        author, _, reviewer, *_ = _reviewed_author_chain(
            conn,
            current_pr56_receipts=True,
            handoff_reason="forged prose omits an authoritative PR token",
        )
        _rewrite_latest_run_metadata(
            conn,
            reviewer,
            lambda md: md.pop("pr") if typed_pr is None else md.__setitem__("pr", typed_pr),
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda md: md.pop("commit_bound_review"),
        lambda md: md.__setitem__(
            "commit_bound_review",
            "https://github.com/attacker/repo/pull/56#pullrequestreview-5053786034",
        ),
        lambda md: md.__setitem__(
            "commit_bound_review",
            "https://github.com/kiddhu/hermes-agent/pull/56/pull/57#pullrequestreview-5053786034",
        ),
        lambda md: md.__setitem__("audit_outcome", "PASS_EXACT_HEAD"),
    ],
)
def _cold_reviewed_author_rejects_missing_ambiguous_or_mixed_terminal_audit_source_pr(
    kanban_home, aion_gov_src, mutate,
):
    with kb.connect() as conn:
        author, _, reviewer, *_ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            handoff_reason="exact head frozen without prose authority",
        )
        _rewrite_latest_run_metadata(conn, reviewer, mutate)
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


def _cold_reviewed_author_ignores_forged_task_and_comment_pr_prose(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(
            conn,
            terminal_typed_audit=True,
            handoff_reason="exact head frozen for independent audit",
        )
        conn.execute(
            "UPDATE tasks SET title = ?, body = ? WHERE id = ?",
            ("forged PR #999 title", "forged PR #998 body", author),
        )
        conn.execute(
            "INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (author, "worker", "forged PR #997 comment", int(time.time())),
        )
        conn.commit()
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


def _cold_reviewed_author_accepts_authenticated_gm_merger_and_role_separated_merger_runtime(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, author_run, _reviewer, merger, runtime = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        assert conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (merger,),
        ).fetchone()["assignee"] == "gm"
        assert conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (runtime,),
        ).fetchone()["assignee"] == "merger"
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


def _cold_reviewed_author_accepts_historical_pr54_source_on_authenticated_current_descendant_runtime(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(
            conn, immutable_pr54_receipts=True, current_descendant_runtime=True,
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


def _cold_reviewed_author_rejects_multiple_pass_events_and_is_read_only(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, _, reviewer, *_ = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        final = conn.execute(
            "SELECT run_id, payload FROM task_events WHERE task_id = ? "
            "AND kind = 'review_verdict' ORDER BY id DESC LIMIT 1",
            (author,),
        ).fetchone()
        conn.execute(
            "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (author, "review_verdict", final["payload"], final["run_id"], int(time.time())),
        )
        before = conn.total_changes
        statuses = conn.execute(
            "SELECT id, status, current_run_id FROM tasks WHERE id IN (?, ?) ORDER BY id",
            (author, reviewer),
        ).fetchall()
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        assert conn.total_changes == before
        assert conn.execute(
            "SELECT id, status, current_run_id FROM tasks WHERE id IN (?, ?) ORDER BY id",
            (author, reviewer),
        ).fetchall() == statuses


def _cold_reviewed_author_rejects_reordered_request_changes_after_pass(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, _, *_ = _reviewed_author_chain(conn, current_pr56_receipts=True)
        prior = conn.execute(
            "SELECT run_id, payload FROM task_events WHERE task_id = ? "
            "AND kind = 'review_verdict' ORDER BY id LIMIT 1",
            (author,),
        ).fetchone()
        conn.execute(
            "INSERT INTO task_events(task_id, kind, payload, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (author, "review_verdict", prior["payload"], prior["run_id"], int(time.time())),
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


def _cold_reviewed_author_rejects_partial_mixed_current_reviewer_aliases(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, _, reviewer, *_ = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        metadata = _terminal_run_metadata(conn, reviewer)
        metadata["head_sha"] = metadata["head"]
        _set_terminal_run_metadata(conn, reviewer, metadata)
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


def _cold_reviewed_author_rejects_gm_profile_without_exact_current_receipt_binding(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, _, _, merger, _ = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        metadata = _terminal_run_metadata(conn, merger)
        metadata["merger_profile"] = "merger"
        _set_terminal_run_metadata(conn, merger, metadata)
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None


@pytest.mark.parametrize(
    ("target", "mutate"),
    [
        ("reviewer", lambda md: md.pop("tree")),
        ("reviewer", lambda md: md.__setitem__("head_sha", md["head"])),
        ("merger", lambda md: md.pop("merge_tree")),
        ("merger", lambda md: md.__setitem__("expected_head", md["head"])),
        ("merger", lambda md: md.__setitem__("cas_merge_attempt_count", 2)),
        ("merger", lambda md: md.__setitem__("auditor_actor", md["implementation_actor"])),
        ("merger", lambda md: md.__setitem__("audit_run_id", -1)),
    ],
)
def _cold_reviewed_author_rejects_partial_mixed_or_unbound_current_receipts_zero_mutation(
    kanban_home, aion_gov_src, target, mutate,
):
    with kb.connect() as conn:
        author, _, reviewer, merger, _ = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        _rewrite_latest_run_metadata(
            conn, reviewer if target == "reviewer" else merger, mutate,
        )
        before = conn.total_changes
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        assert conn.total_changes == before


@pytest.mark.parametrize(
    "drift",
    [
        "nested_review_id", "nested_source_tree", "runtime_run",
        "installed_head", "direct_parents", "git_proof",
    ],
)
def _cold_reviewed_author_rejects_descendant_wrapper_drift_and_unprovable_git(
    kanban_home, aion_gov_src, monkeypatch, drift,
):
    monkeypatch.setattr(
        kb, "_historical_source_preserved_in_installed_git",
        lambda **_kw: drift != "git_proof",
    )
    with kb.connect() as conn:
        author, _, _, _, runtime = _reviewed_author_chain(
            conn, immutable_pr54_receipts=True, current_descendant_runtime=True,
        )
        metadata = _terminal_run_metadata(conn, runtime)
        if drift == "nested_review_id":
            metadata["source_lineage"]["github_review_id"] += 1
        elif drift == "nested_source_tree":
            metadata["candidate_packet"]["source_tree"] = "f" * 40
        elif drift == "runtime_run":
            metadata["candidate_packet"]["canonical_run_id"] += 1
        elif drift == "installed_head":
            metadata["installed_runtime"]["head"] = "f" * 40
        elif drift == "direct_parents":
            metadata["role_binding"]["direct_parents"] = metadata["role_binding"]["direct_parents"][:1]
        _set_terminal_run_metadata(conn, runtime, metadata)
        before = conn.total_changes
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        assert conn.total_changes == before


def _cold_reviewed_author_rejects_multiple_authenticated_runtime_witness_candidates(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, _, _, merger, runtime = _reviewed_author_chain(
            conn, current_pr56_receipts=True,
        )
        competing = kb.create_task(
            conn, title="competing runtime witness", factory_build_gate=1,
            assignee="installer", parents=[merger],
        )
        competing_run = _claim_and_run_id(conn, competing)
        metadata = _terminal_run_metadata(conn, runtime)
        metadata["canonical_run_id"] = competing_run
        assert kb.complete_task(
            conn, competing, expected_run_id=competing_run, metadata=metadata,
        )
        before = conn.total_changes
        assert kb._reviewed_author_finalizer_run_id(conn, author) is None
        assert conn.total_changes == before


@pytest.mark.parametrize(
    "drift",
    [
        None, "invalid_repo_head", "dirty_repo", "installed_head", "missing_commit", "missing_tree", "source_tree",
        "merge_tree", "first_parent", "second_parent", "third_parent",
        "source_not_installed_ancestor", "installed_not_current_ancestor",
        "path_set", "installed_blob", "current_blob",
    ],
)
def _cold_historical_source_git_proof_gates_every_immutable_relationship(monkeypatch, drift):
    current_head, installed_head, installed_tree = "0" * 40, "a" * 40, "b" * 40
    source_head, source_tree = "c" * 40, "d" * 40
    source_base, source_merge = "e" * 40, "f" * 40
    paths = ["alpha.py", "tests/test_alpha.py"]

    def fake_git(_repo, *args):
        if args == ("rev-parse", "HEAD"):
            return "invalid" if drift == "invalid_repo_head" else current_head
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return " M alpha.py" if drift == "dirty_repo" else ""
        if args[0:2] == ("cat-file", "-e"):
            if drift == "missing_commit" and args[2] == f"{source_head}^{{commit}}":
                return None
            if drift == "missing_tree" and args[2] == f"{source_tree}^{{tree}}":
                return None
            return ""
        if args == ("rev-parse", f"{installed_head}^{{tree}}"):
            return "0" * 40 if drift == "installed_head" else installed_tree
        if args == ("rev-parse", f"{source_head}^{{tree}}"):
            return "0" * 40 if drift == "source_tree" else source_tree
        if args == ("rev-parse", f"{source_merge}^{{tree}}"):
            return "0" * 40 if drift == "merge_tree" else source_tree
        if args == ("rev-list", "--parents", "-n", "1", source_merge):
            first = "0" * 40 if drift == "first_parent" else source_base
            second = "0" * 40 if drift == "second_parent" else source_head
            third = f" {'0' * 40}" if drift == "third_parent" else ""
            return f"{source_merge} {first} {second}{third}"
        if args == ("merge-base", "--is-ancestor", source_merge, installed_head):
            return None if drift == "source_not_installed_ancestor" else ""
        if args == ("merge-base", "--is-ancestor", installed_head, current_head):
            return None if drift == "installed_not_current_ancestor" else ""
        if args == ("diff", "--name-only", "--no-renames", source_base, source_head):
            return "alpha.py\nextra.py" if drift == "path_set" else "\n".join(paths)
        if args[0] == "rev-parse" and ":" in args[1]:
            commit, path = args[1].split(":", 1)
            if drift == "installed_blob" and commit == installed_head and path == paths[0]:
                return "2" * 40
            if drift == "current_blob" and commit == current_head and path == paths[0]:
                return "2" * 40
            return "1" * 40
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(kb, "_git_output", fake_git)
    result = kb._historical_source_preserved_in_installed_git(
        install={
            "head": installed_head,
            "tree": installed_tree,
            "changed_paths": ["alpha.py", "tests/test_alpha.py", "other.py"],
        },
        source_head=source_head,
        source_tree=source_tree,
        source_base=source_base,
        source_merge=source_merge,
        source_paths=paths,
    )
    assert result is (drift is None)


def _cold_reviewed_author_ignores_unrelated_terminal_runtime_child(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, author_run, _reviewer, merger, _runtime = _reviewed_author_chain(
            conn, immutable_pr54_receipts=True, activation_cycle=True,
        )
        unrelated = kb.create_task(
            conn, title="historical non-runtime child", factory_build_gate=1,
            assignee="gm2", parents=[merger],
        )
        unrelated_run = _claim_and_run_id(conn, unrelated)
        assert kb.complete_task(
            conn, unrelated, expected_run_id=unrelated_run,
            summary="unrelated historical receipt", metadata={"kind": "history"},
        )
        assert kb._reviewed_author_finalizer_run_id(conn, author) == author_run


@pytest.mark.parametrize(
    "drift",
    [
        "review_missing_head", "review_mixed_schema", "review_outcome",
        "review_pr", "review_role_collision", "merger_missing_head",
        "merger_mixed_schema", "merger_mixed_canonical_alias",
        "merger_mixed_legacy_alias", "merger_mixed_shared_alias",
        "merger_audit_task", "merger_audit_run",
        "merger_review", "merger_ancestry", "merger_tree", "merger_blob",
        "runtime_zero", "runtime_two", "runtime_partial_competitor",
        "runtime_self_role", "runtime_tree", "runtime_path",
        "runtime_source_head", "runtime_source_review", "runtime_witness_type",
        "runtime_wrong_ancestry", "runtime_unauthenticated",
    ],
)
def _cold_immutable_pr54_drift_fails_closed_zero_author_mutation(
    kanban_home, aion_gov_src, drift,
):
    terminal_runtime = drift not in {"runtime_zero", "runtime_unauthenticated"}
    runtime_assignee = "agent007" if drift == "runtime_self_role" else "gm"
    with kb.connect() as conn:
        author, _author_run, reviewer, merger, runtime = _reviewed_author_chain(
            conn, immutable_pr54_receipts=True, activation_cycle=True,
            terminal_runtime=terminal_runtime, runtime_assignee=runtime_assignee,
        )
        if drift == "review_missing_head":
            _rewrite_latest_run_metadata(conn, reviewer, lambda md: md.pop("head"))
        elif drift == "review_mixed_schema":
            _rewrite_latest_run_metadata(
                conn, reviewer, lambda md: md.__setitem__("head_sha", md["head"]),
            )
        elif drift == "review_outcome":
            _rewrite_latest_run_metadata(
                conn, reviewer,
                lambda md: md.__setitem__("review_outcome", "APPROVE_EXACT_HEAD"),
            )
        elif drift == "review_pr":
            _rewrite_latest_run_metadata(
                conn, reviewer,
                lambda md: md.__setitem__(
                    "pr", "https://github.com/attacker/repo/pull/54"
                ),
            )
        elif drift == "review_role_collision":
            _rewrite_latest_run_metadata(
                conn, reviewer,
                lambda md: md.__setitem__("auditor_identity", "007AION"),
            )
        elif drift == "merger_missing_head":
            _rewrite_latest_run_metadata(conn, merger, lambda md: md.pop("audited_head"))
        elif drift == "merger_mixed_schema":
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("expected_head", md["audited_head"]),
            )
        elif drift == "merger_mixed_canonical_alias":
            # audit_run_id belongs only to the canonical receipt family but
            # was omitted from the original discriminator subset.
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("audit_run_id", 999),
            )
        elif drift == "merger_mixed_legacy_alias":
            # author belongs only to the legacy receipt family but is not one
            # of that family's head/tree/base discriminator fields.
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("author", "007AION"),
            )
        elif drift == "merger_mixed_shared_alias":
            # repository is shared by canonical and legacy receipts, so it
            # cannot identify either family by itself but is still foreign to
            # the immutable family and must fail closed when present.
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("repository", "kiddhu/hermes-agent"),
            )
        elif drift == "merger_audit_task":
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("native_audit_task", "t_forged"),
            )
        elif drift == "merger_audit_run":
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("native_audit_run", -1),
            )
        elif drift == "merger_review":
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("github_review_id", 999),
            )
        elif drift == "merger_ancestry":
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("merge_parents", [md["base_at_audit"]]),
            )
        elif drift == "merger_tree":
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("canonical_main_tree", "9" * 40),
            )
        elif drift == "merger_blob":
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("tools_approval_blob_main", "9" * 40),
            )
        elif drift in {"runtime_two", "runtime_partial_competitor"}:
            competitor = kb.create_task(
                conn, title="competing runtime witness", factory_build_gate=1,
                assignee="gm2", parents=[merger],
            )
            competitor_run = _claim_and_run_id(conn, competitor)
            install = {} if drift == "runtime_partial_competitor" else {
                "head": "6b521c8637d477a76451d0d029cc24026d01cf61",
                "tree": "00aca8ed67457e0b0bdccb7ea07343da1031bbc4",
                "changed_paths": [
                    "tools/approval.py",
                    "tests/tools/test_aion889_prior_authorization.py",
                ],
            }
            assert kb.complete_task(
                conn, competitor, expected_run_id=competitor_run,
                metadata={
                    "canonical_run_id": competitor_run, "install": install,
                    "forbidden_actions_performed": [], "secret_exposure": "none",
                },
            )
        elif drift == "runtime_tree":
            _rewrite_latest_run_metadata(
                conn, runtime,
                lambda md: md["install"].__setitem__("tree", "9" * 40),
            )
        elif drift == "runtime_path":
            _rewrite_latest_run_metadata(
                conn, runtime,
                lambda md: md["install"].__setitem__(
                    "changed_paths", ["attacker.py"]
                ),
            )
        elif drift == "runtime_source_head":
            _rewrite_latest_run_metadata(
                conn, runtime, lambda md: md.__setitem__("source_head", "9" * 40),
            )
        elif drift == "runtime_source_review":
            _rewrite_latest_run_metadata(
                conn, runtime, lambda md: md.__setitem__("github_review_id", 999),
            )
        elif drift == "runtime_witness_type":
            _rewrite_latest_run_metadata(
                conn, runtime,
                lambda md: md.__setitem__("witness_type", "ACTIVATION"),
            )
        elif drift == "runtime_wrong_ancestry":
            unrelated_parent = kb.create_task(conn, title="unrelated parent", assignee="gm2")
            conn.execute(
                "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
                (unrelated_parent, runtime),
            )
            conn.commit()
        elif drift == "runtime_unauthenticated":
            unauthenticated = kb.create_task(
                conn, title="unauthenticated runtime claim", factory_build_gate=0,
                assignee="gm2", parents=[merger],
            )
            unauthenticated_run = _claim_and_run_id(conn, unauthenticated)
            assert kb.complete_task(
                conn, unauthenticated, expected_run_id=unauthenticated_run,
                metadata={
                    "canonical_run_id": unauthenticated_run,
                    "install": {
                        "head": "6b521c8637d477a76451d0d029cc24026d01cf61",
                        "tree": "00aca8ed67457e0b0bdccb7ea07343da1031bbc4",
                        "changed_paths": ["tools/approval.py"],
                    },
                    "forbidden_actions_performed": [], "secret_exposure": "none",
                },
            )

        before_task = dict(conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?", (author,),
        ).fetchone())
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0]
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary=f"reject {drift}")
        assert dict(conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?", (author,),
        ).fetchone()) == before_task
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0] == before_events
        assert kb.list_attachments(conn, author) == []


@pytest.mark.parametrize(
    ("drift", "mutate"),
    [
        ("stale_head", lambda md: md.__setitem__("expected_head", "9" * 40)),
        ("stale_tree", lambda md: md.__setitem__("audited_tree", "9" * 40)),
        ("stale_base", lambda md: md.__setitem__("audited_base", "9" * 40)),
        ("mixed_schema", lambda md: md.__setitem__("head_sha", md["expected_head"])),
        ("partial_schema", lambda md: md.pop("audited_base")),
        ("implementation_task", lambda md: md.__setitem__("implementation_task_id", "t_forged")),
        ("implementation_run", lambda md: md.__setitem__("implementation_run_id", -1)),
        ("implementation_profile", lambda md: md.__setitem__("implementation_profile", "intruder")),
        ("implementation_actor", lambda md: md.__setitem__("implementation_actor", "GemAION")),
        ("audit_task", lambda md: md.__setitem__("audit_task_id", "t_forged")),
        ("audit_run", lambda md: md.__setitem__("audit_run_id", -1)),
        ("audit_profile", lambda md: md.__setitem__("audit_profile", "intruder")),
        ("auditor_actor", lambda md: md.__setitem__("auditor_actor", "007AION")),
        ("native_task", lambda md: md.__setitem__("native_task_id", "t_forged")),
        ("native_run", lambda md: md.__setitem__("native_run_id", -1)),
        ("native_profile", lambda md: md.__setitem__("native_profile", "gm")),
        ("repository", lambda md: md.__setitem__("repository", "attacker/repo")),
        ("pr_number", lambda md: md.__setitem__("pr_number", 999)),
        ("review", lambda md: md.__setitem__("github_review_id", 99999)),
        ("canonical_main", lambda md: md.__setitem__("canonical_main_sha", "9" * 40)),
        ("main_parent", lambda md: md.__setitem__("canonical_main_parents", ["6" * 40])),
        ("gate_verdict", lambda md: md.__setitem__("gate_verdict", "FAIL")),
        ("merge_performed", lambda md: md.__setitem__("merge_performed", False)),
        (
            "runtime_mutation",
            lambda md: md.__setitem__("production_or_runtime_mutation", True),
        ),
    ],
)
def _cold_canonical_merger_receipt_drift_fails_closed_zero_author_mutation(
    kanban_home, aion_gov_src, drift, mutate,
):
    with kb.connect() as conn:
        author, _author_run, _reviewer, merger, _runtime = _reviewed_author_chain(
            conn, canonical_merger_receipt=True, source_pr=50,
        )
        _rewrite_latest_run_metadata(conn, merger, mutate)
        before_task = dict(conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (author,),
        ).fetchone())
        before_runs = [tuple(row) for row in conn.execute(
            "SELECT id, status, outcome, ended_at FROM task_runs "
            "WHERE task_id = ? ORDER BY id",
            (author,),
        ).fetchall()]
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0]

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary=f"reject {drift}")

        assert dict(conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (author,),
        ).fetchone()) == before_task
        assert [tuple(row) for row in conn.execute(
            "SELECT id, status, outcome, ended_at FROM task_runs "
            "WHERE task_id = ? ORDER BY id",
            (author,),
        ).fetchall()] == before_runs
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0] == before_events
        assert kb.list_attachments(conn, author) == []


def _cold_reviewed_author_controller_completion_uses_exact_ended_run(
    kanban_home, aion_gov_src,
):
    """RED: reviewed authors currently cannot enter the trusted finalizer."""
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(conn)
        run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (author,),
        ).fetchone()[0]

        assert kb.complete_task(
            conn,
            author,
            summary="reviewed candidate terminalized",
        )
        assert kb.get_task(conn, author).status == "done"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (author,),
        ).fetchone()[0] == run_count
        completed = conn.execute(
            "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (author,),
        ).fetchone()
        assert completed is not None and completed["run_id"] == author_run


def _rewrite_latest_run_metadata(conn, task_id, mutate):
    row = conn.execute(
        "SELECT id, metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    metadata = json.loads(row["metadata"])
    mutate(metadata)
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps(metadata), row["id"]),
    )
    conn.commit()


@pytest.mark.parametrize(
    "drift",
    [
        "stale_author_run",
        "active_author_run",
        "wrong_reviewer_identity",
        "nonmatching_reviewer_identity",
        "nonmatching_merger_identity",
        "multiple_reviewers",
        "review_author_task",
        "review_head",
        "merge_review",
        "merge_actor",
        "merge_repository",
        "merge_pr_number",
        "runtime_tree",
        "runtime_receipt_authenticator",
        "ambiguous_canonical_merger",
    ],
)
def _cold_reviewed_author_evidence_drift_fails_closed_zero_mutation(
    kanban_home, aion_gov_src, drift,
):
    with kb.connect() as conn:
        author, author_run, reviewer, merger, runtime = _reviewed_author_chain(conn)

        if drift == "stale_author_run":
            conn.execute(
                "UPDATE task_runs SET outcome = 'completed' WHERE id = ?",
                (author_run,),
            )
            conn.commit()
        elif drift == "active_author_run":
            conn.execute(
                "UPDATE task_runs SET ended_at = NULL WHERE id = ?", (author_run,),
            )
            conn.commit()
        elif drift == "wrong_reviewer_identity":
            conn.execute(
                "UPDATE task_runs SET profile = 'agent007' WHERE task_id = ?",
                (reviewer,),
            )
            conn.commit()
        elif drift == "nonmatching_reviewer_identity":
            conn.execute(
                "UPDATE tasks SET assignee = 'intruder-reviewer' WHERE id = ?",
                (reviewer,),
            )
            conn.execute(
                "UPDATE task_runs SET profile = 'intruder-reviewer' WHERE task_id = ?",
                (reviewer,),
            )
            conn.commit()
        elif drift == "nonmatching_merger_identity":
            conn.execute(
                "UPDATE tasks SET assignee = 'intruder-merger' WHERE id = ?",
                (merger,),
            )
            conn.execute(
                "UPDATE task_runs SET profile = 'intruder-merger' WHERE task_id = ?",
                (merger,),
            )
            conn.commit()
        elif drift == "multiple_reviewers":
            event = conn.execute(
                "SELECT run_id, payload FROM task_events "
                "WHERE task_id = ? AND kind = 'review_verdict'",
                (author,),
            ).fetchone()
            conn.execute(
                "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'review_verdict', ?, ?)",
                (author, event["run_id"], event["payload"], int(time.time())),
            )
            conn.commit()
        elif drift == "review_author_task":
            _rewrite_latest_run_metadata(
                conn,
                reviewer,
                lambda md: md.__setitem__("author_task", "t_forged"),
            )
        elif drift == "review_head":
            _rewrite_latest_run_metadata(
                conn, reviewer, lambda md: md.__setitem__("head_sha", "9" * 40),
            )
        elif drift == "merge_review":
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("review_id", 99999),
            )
        elif drift == "merge_actor":
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("merged_by", "attacker"),
            )
        elif drift == "merge_repository":
            _rewrite_latest_run_metadata(
                conn, merger,
                lambda md: md.__setitem__("repository", "attacker/other-repo"),
            )
        elif drift == "merge_pr_number":
            _rewrite_latest_run_metadata(
                conn, merger, lambda md: md.__setitem__("pr_number", 999),
            )
        elif drift == "runtime_tree":
            _rewrite_latest_run_metadata(
                conn, runtime,
                lambda md: md["install"].__setitem__("tree", "9" * 40),
            )
        elif drift == "runtime_receipt_authenticator":
            conn.execute(
                "UPDATE task_attachments SET uploaded_by = 'agent' WHERE task_id = ?",
                (runtime,),
            )
            conn.commit()
        elif drift == "ambiguous_canonical_merger":
            conn.execute(
                "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
                (reviewer, runtime),
            )
            conn.execute(
                "UPDATE tasks SET assignee = ? WHERE id = ?",
                (kb.FACTORY_REVIEW_MERGER_PROFILE, runtime),
            )
            conn.execute(
                "UPDATE task_runs SET profile = ?, metadata = ? WHERE task_id = ?",
                (
                    kb.FACTORY_REVIEW_MERGER_PROFILE,
                    json.dumps({"audit_task_id": reviewer}),
                    runtime,
                ),
            )
            conn.commit()

        before_task = dict(conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (author,),
        ).fetchone())
        before_runs = [tuple(row) for row in conn.execute(
            "SELECT id, status, outcome, ended_at FROM task_runs "
            "WHERE task_id = ? ORDER BY id",
            (author,),
        ).fetchall()]
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0]

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary="must not terminalize")

        assert dict(conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (author,),
        ).fetchone()) == before_task
        assert [tuple(row) for row in conn.execute(
            "SELECT id, status, outcome, ended_at FROM task_runs "
            "WHERE task_id = ? ORDER BY id",
            (author,),
        ).fetchall()] == before_runs
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0] == before_events
        assert kb.list_attachments(conn, author) == []


@pytest.mark.parametrize("sabotage", ["false_verdict", "signer_failure", "cas_miss"])
def _cold_reviewed_author_finalizer_sabotage_rolls_back_zero_mutation(
    kanban_home, aion_gov_src, monkeypatch, sabotage,
):
    with kb.connect() as conn:
        author, author_run, *_ = _reviewed_author_chain(conn)
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0]
        before_run = tuple(conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (author_run,),
        ).fetchone())

        if sabotage in {"false_verdict", "signer_failure"}:
            real_binder, real_kernel, real_adapters = kb._load_pinned_aion_modules(
                author
            )

            class _SabotagedBinder:
                def bind_task_terminal_in_txn(self, **kwargs):
                    kwargs["terminal_write"](
                        kwargs["conn"], kwargs["task_id"], kwargs["run_id"]
                    )
                    if sabotage == "signer_failure":
                        raise RuntimeError("simulated signer/authenticator failure")
                    return {
                        "bound": False,
                        "verdict": "FAIL_CLOSED",
                        "failed_conditions": ["C8"],
                    }

            monkeypatch.setattr(
                kb,
                "_load_pinned_aion_modules",
                lambda _task_id: (_SabotagedBinder(), real_kernel, real_adapters),
            )
        else:
            real_resolver = getattr(kb, "_reviewed_author_finalizer_run_id")
            calls = {"count": 0}

            def _cas_miss(c, task_id):
                calls["count"] += 1
                if calls["count"] == 1:
                    return real_resolver(c, task_id)
                return None

            monkeypatch.setattr(kb, "_reviewed_author_finalizer_run_id", _cas_miss)

        with pytest.raises((kb.FactoryTerminalReceiptRequiredError, RuntimeError)):
            kb.complete_task(conn, author, summary="must roll back")

        row = conn.execute(
            "SELECT status, current_run_id, completed_at, "
            "factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (author,),
        ).fetchone()
        assert tuple(row) == ("review", None, None, None)
        assert tuple(conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (author_run,),
        ).fetchone()) == before_run
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0] == before_events
        assert kb.list_attachments(conn, author) == []
        assert _receipt_residue_on_disk(conn, author) == []


def _cold_reviewed_author_prebound_receipt_cannot_bypass_evidence_drift(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        author, author_run, _reviewer, _merger, runtime = _reviewed_author_chain(conn)
        _rewrite_latest_run_metadata(
            conn, runtime,
            lambda md: md["install"].__setitem__("head", "9" * 40),
        )
        receipt = _kernel_receipt_doc(author, str(author_run))
        raw = json.dumps(receipt, sort_keys=True).encode("utf-8")
        kb.store_attachment_bytes(
            conn,
            author,
            "prebound.json",
            raw,
            uploaded_by="aion_monarch_proof_kernel",
        )
        conn.execute(
            "UPDATE tasks SET factory_terminal_receipt_sha256 = ? WHERE id = ?",
            (hashlib.sha256(raw).hexdigest(), author),
        )
        conn.commit()
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0]

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, author, summary="must not bypass evidence")

        row = conn.execute(
            "SELECT status, current_run_id, completed_at FROM tasks WHERE id = ?",
            (author,),
        ).fetchone()
        assert tuple(row) == ("review", None, None)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (author,),
        ).fetchone()[0] == before_events


# ---------------------------------------------------------------------------
# T3 RED — worker-forged 'agent' uploader rejected (provenance)
# ---------------------------------------------------------------------------

def test_t3_worker_forged_uploader_rejected(kanban_home, aion_gov_src, monkeypatch):
    # The finalizer stamps aion_monarch_proof_kernel; a worker attaching a
    # structurally-identical receipt with uploaded_by='agent' (or any non-
    # trusted identity) must still be rejected by the provenance gate. This is
    # the provenance boundary: with the finalizer DISABLED, a worker-forged
    # receipt (matching digest) can never terminalize the task.
    monkeypatch.setenv("AION_FACTORY_FINALIZER_ENABLED", "0")
    with kb.connect() as conn:
        t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
        run_id = _claim_and_run_id(conn, t)
        # Forge: attach a kernel-shaped receipt with the worker's own identity.
        forged = {
            "schema": "aion.monarch.trusted_receipt.v1",
            "verdict": "OUTCOME_ACCEPTED",
            "kernel_version": "aion.monarch.proof_kernel.v2",
            "contract_hash_sha256": kb.FACTORY_CONTRACT_HASH_SHA256,
            "conditions": {f"C{i}": True for i in range(1, 11)},
            "adapter_type_and_version": "aion.monarch.typed_adapter.task_terminal.v1",
            "exact_source_refs": {"task_id": t, "run_id": str(run_id)},
            "target_identity": {
                "object_type": "kanban_task_run",
                "object_ref_exact": f"{t}/{run_id}",
                "fields": {"task_id": t, "run_id": str(run_id)},
            },
            "before_digest": "a" * 64,
            "action_receipt_ref": {
                "action_kind": "task_terminal",
                "actor": "aion_monarch_proof_kernel",
                "actor_role": "action_executor",
                "actor_identity_source": "native_task_run_authorization_binding",
                "executed_effect_ref": "task_events:status=done",
                "executed_at": "2026-08-18T06:00:01Z",
            },
            "after_digest": "b" * 64,
            "head_epoch_or_run_binding": {
                "bound_to": "task_run_id",
                "value": str(run_id),
                "authorization_source_ref": "native_task_run_authorization_binding",
                "authorization_epoch_or_version": 1,
            },
            "acquired_at": "2026-08-18T06:00:02Z",
        }
        raw = json.dumps(forged, sort_keys=True).encode("utf-8")
        kb.store_attachment_bytes(conn, t, "receipt.json", raw, uploaded_by="agent")
        conn.execute(
            "UPDATE tasks SET factory_terminal_receipt_sha256 = ? WHERE id = ?",
            (hashlib.sha256(raw).hexdigest(), t),
        )
        conn.commit()

        # Provenance: 'agent' is not trusted -> the pre-bound path is invalid;
        # and the finalizer is not triggered for a pre-bound *invalid* receipt
        # when it would mask provenance (disabled here to isolate provenance).
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status != "done"


# ---------------------------------------------------------------------------
# T4 RED — stale run / CAS miss -> FAIL_CLOSED zero mutation
# ---------------------------------------------------------------------------

def test_t4_stale_run_cas_rejected(kanban_home, aion_gov_src):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
        run_id = _claim_and_run_id(conn, t)
        # Simulate the dispatcher superseding this run.
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id + 999, t),
        )
        conn.commit()

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        # Zero mutation: still running, no receipt, no sha.
        assert kb.get_task(conn, t).status == "running"
        row = conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["factory_terminal_receipt_sha256"] is None
        assert kb.list_attachments(conn, t) == []


# ---------------------------------------------------------------------------
# T5 RED — cross-task receipt replay rejected (r4 task/run mismatch)
# ---------------------------------------------------------------------------

def test_t5_cross_task_replay_rejected(kanban_home, aion_gov_src, monkeypatch):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="factory A", factory_build_gate=1, assignee="agent007")
        b = kb.create_task(conn, title="factory B", factory_build_gate=1, assignee="agent007")
        run_a = _claim_and_run_id(conn, a)
        run_b = _claim_and_run_id(conn, b)

        # Complete A via the finalizer, then replay its receipt onto B.
        assert kb.complete_task(conn, a, result="done", expected_run_id=run_a)
        doc_a = _bound_receipt_doc(conn, a)

        # Disable the finalizer so the replayed (cross-task) receipt is tested
        # on the pre-bound provenance path alone.
        monkeypatch.setenv("AION_FACTORY_FINALIZER_ENABLED", "0")

        # Replay A's exact receipt bytes onto B with the trusted uploader.
        raw = json.dumps(doc_a, sort_keys=True).encode("utf-8")
        kb.store_attachment_bytes(
            conn, b, "receipt.json", raw, uploaded_by="aion_monarch_proof_kernel",
        )
        conn.execute(
            "UPDATE tasks SET factory_terminal_receipt_sha256 = ? WHERE id = ?",
            (hashlib.sha256(raw).hexdigest(), b),
        )
        conn.commit()

        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, b, result="done", expected_run_id=run_b)
        assert kb.get_task(conn, b).status != "done"


# ---------------------------------------------------------------------------
# T6 RED/IDEMPOTENT — fault injection at a write boundary -> rollback + retry
# ---------------------------------------------------------------------------

def test_t6_fault_injection_rolls_back_then_retry_succeeds(kanban_home, aion_gov_src, monkeypatch):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
        run_id = _claim_and_run_id(conn, t)

        # Fault: the attachment-name helper raises AFTER the terminal write has
        # already been performed inside the txn, so the receipt bind fails and
        # the whole transaction must roll back (status, event, receipt).
        def _boom(_raw):
            raise RuntimeError("simulated on-disk attachment failure")

        _original = kb._safe_attachment_name
        monkeypatch.setattr(kb, "_safe_attachment_name", _boom)
        with pytest.raises(RuntimeError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        # Zero durable mutation: still running, no receipt, no sha, no event.
        assert kb.get_task(conn, t).status == "running"
        row = conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["factory_terminal_receipt_sha256"] is None
        assert kb.list_attachments(conn, t) == []
        assert "completed" not in [e.kind for e in kb.list_events(conn, t)]

        # Retry (fault cleared) is idempotent and succeeds.
        monkeypatch.setattr(kb, "_safe_attachment_name", _original)
        assert kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status == "done"
        assert _bound_receipt_doc(conn, t)["verdict"] == "OUTCOME_ACCEPTED"


# ---------------------------------------------------------------------------
# T7 GREEN — merge-bearing pre-bound receipt path unchanged
# ---------------------------------------------------------------------------

def test_t7_prebound_receipt_path_unchanged(kanban_home, aion_gov_src):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
        run_id = _claim_and_run_id(conn, t)
        # Pre-bind a valid receipt directly (merge/already-terminal style).
        doc = _kernel_receipt_doc(t, str(run_id))
        raw = json.dumps(doc, sort_keys=True).encode("utf-8")
        kb.store_attachment_bytes(
            conn, t, "receipt.json", raw, uploaded_by="aion_monarch_proof_kernel",
        )
        conn.execute(
            "UPDATE tasks SET factory_terminal_receipt_sha256 = ? WHERE id = ?",
            (hashlib.sha256(raw).hexdigest(), t),
        )
        conn.commit()

        assert kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status == "done"


# ---------------------------------------------------------------------------
# T8 GREEN — already-terminal path unchanged
# ---------------------------------------------------------------------------

def test_t8_already_terminal_path_unchanged(kanban_home, aion_gov_src):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
        run_id = _claim_and_run_id(conn, t)
        assert kb.complete_task(conn, t, result="first", expected_run_id=run_id)
        # A second completion of an already-done task must not double-complete.
        assert kb.complete_task(conn, t, result="second", expected_run_id=run_id) is False
        assert kb.get_task(conn, t).status == "done"


# ---------------------------------------------------------------------------
# T10 — ALTERNATE_SUCCESS_PATHS stays 0 (no alternate success path)
# ---------------------------------------------------------------------------

def test_t10_no_alternate_success_path():
    from hermes_cli.kanban_db import FACTORY_VERDICT_ACCEPTED
    assert FACTORY_VERDICT_ACCEPTED == "OUTCOME_ACCEPTED"
    # The aion-governance kernel's alternate_success_paths must remain 0; the
    # finalizer path reuses the SAME kernel (single semantic authority).
    src = _aion_gov_source_dir()
    if src is not None:
        from scripts.aion_monarch_outcome_proof_gate import ALTERNATE_SUCCESS_PATHS  # noqa
        assert ALTERNATE_SUCCESS_PATHS == 0


# ---------------------------------------------------------------------------
# T11 HOSTILE — real worker subprocess: module-hash equality + guard + finalizer
# ---------------------------------------------------------------------------

_T11_WORKER_SCRIPT = r"""
import hashlib, json, os, sys
from pathlib import Path
os.environ["HERMES_HOME"] = {hermes_home!r}
os.environ["AION_GOVERNANCE_SOURCE_DIR"] = {aion_src!r}
# F2 repair: bind to the audited CANDIDATE bytes despite editable/user-site
# import hooks. Strip every editable MetaPathFinder (setuptools _EditableFinder
# et al.) so it cannot shadow the candidate package, then pin the candidate
# repo to the head of sys.path BEFORE importing hermes_cli.kanban_db.
sys.meta_path[:] = [
    f for f in sys.meta_path
    if "_Editable" not in repr(f) and "Editable" not in f.__class__.__name__
]
sys.path.insert(0, {candidate_repo!r})
from hermes_cli import kanban_db as kb
module_file = Path(kb.__file__).resolve()
module_sha = hashlib.sha256(module_file.read_bytes()).hexdigest()
candidate_file = Path({candidate_file!r}).resolve()
path_matches_candidate = (module_file == candidate_file)
# F2 containment: fully isolate the Native Kanban env so connect()/attachments/
# events/claims resolve to the isolated root, never the live aion-factory board.
with kb.isolated_kanban_env(Path({worker_home!r})):
    conn = kb.connect(db_path=Path({db_path!r}))
    t = kb.create_task(conn, title="factory task", factory_build_gate=1, assignee="agent007")
    kb.claim_task(conn, t)
    row = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (t,)).fetchone()
    ok = kb.complete_task(conn, t, result="done", expected_run_id=int(row["current_run_id"]))
    final = conn.execute("SELECT status, factory_terminal_receipt_sha256 FROM tasks WHERE id=?", (t,)).fetchone()
    atts = kb.list_attachments(conn, t)
    conn.close()
print(json.dumps({{
    "module_file": str(module_file),
    "candidate_file": str(candidate_file),
    "path_matches_candidate": bool(path_matches_candidate),
    "module_sha256": module_sha,
    "has_guard": hasattr(kb, "FactoryTerminalReceiptRequiredError"),
    "complete_ok": bool(ok),
    "status": final["status"],
    "receipt_sha_present": bool(final["factory_terminal_receipt_sha256"]),
    "uploaded_by": [a.uploaded_by for a in atts],
}}))
"""


def test_t11_real_worker_subprocess_module_hash_and_finalizer(kanban_home, aion_gov_src, tmp_path):
    """Launch a REAL worker subprocess: assert (a) its loaded kanban_db module
    sha256 equals the gateway's (this process's) module sha256 — the I1
    containment proof — (b) the receipt guard is present, and (c) the finalizer
    path completes a gate=1 task. The child is pinned to the audited CANDIDATE
    bytes despite editable/user-site import hooks (F2 repair). No live service
    is used (fresh tmp DB)."""
    candidate_file = Path(kb.__file__).resolve()
    candidate_repo = candidate_file.parents[1]  # the hermes-agent checkout root
    gateway_module_sha = hashlib.sha256(candidate_file.read_bytes()).hexdigest()

    db_path = tmp_path / "worker.db"
    worker_home = tmp_path / "worker_home"
    script = _T11_WORKER_SCRIPT.format(
        hermes_home=str(Path(os.environ["HERMES_HOME"])),
        aion_src=str(_aion_gov_source_dir()),
        db_path=str(db_path),
        worker_home=str(worker_home),
        candidate_repo=str(candidate_repo),
        candidate_file=str(candidate_file),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(candidate_repo) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert out.returncode == 0, f"worker subprocess failed:\n{out.stderr}"
    result = json.loads(out.stdout.strip().splitlines()[-1])

    # F2 repair: the child must have loaded the exact CANDIDATE bytes despite
    # editable/user-site import hooks — assert path AND module hash equality
    # BEFORE trusting the behavioral results.
    assert result["path_matches_candidate"] is True, (
        f"child imported {result['module_file']!r}, expected candidate "
        f"{result['candidate_file']!r} (editable/user-site hook not neutralised)"
    )
    # I1 containment: worker and gateway load the SAME kanban_db bytes.
    assert result["module_sha256"] == gateway_module_sha
    assert result["has_guard"] is True
    # Finalizer path succeeded in the real worker subprocess.
    assert result["complete_ok"] is True
    assert result["status"] == "done"
    assert result["receipt_sha_present"] is True
    assert result["uploaded_by"] == ["aion_monarch_proof_kernel"]


# ---------------------------------------------------------------------------
# F3 — complete fault matrix: every transaction/file boundary + idempotent retry
# ---------------------------------------------------------------------------
# The R1 audit found only a pre-file-write fault was covered (T6); the omitted
# post-file boundary (fault AFTER receipt file + DB attachment INSERT, BEFORE
# receipt-sha update) left an orphan receipt file on SQLite rollback. These
# tests close that gap: each boundary is faulted, the transaction rolls back to
# zero mutation with NO on-disk residue, and a deterministic retry succeeds.
# Boundaries: M1 terminal write (status CAS), M2 before file write,
# M3 after file+DB insert before sha (the F1 boundary), M4 after DB insert
# before the 'attached' event, M5 after sha before commit.


def _receipt_residue_on_disk(conn, task_id) -> list[str]:
    """Receipt + staging files under the task's attachments dir (must be empty)."""
    att_dir = kb.task_attachments_dir(task_id)
    if not att_dir.exists():
        return []
    out = []
    for p in att_dir.rglob("*"):
        if p.is_file() and (
            "aion_monarch_receipt" in p.name or ".staging." in p.name
        ):
            out.append(str(p))
    return out


def _assert_zero_mutation_no_residue(conn, task_id, run_id) -> None:
    assert kb.get_task(conn, task_id).status == "running"
    row = conn.execute(
        "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["factory_terminal_receipt_sha256"] is None
    assert kb.list_attachments(conn, task_id) == []
    kinds = [e.kind for e in kb.list_events(conn, task_id)]
    assert "attached" not in kinds
    assert "completed" not in kinds
    # F1: no orphaned receipt or staging file may survive the rollback.
    assert _receipt_residue_on_disk(conn, task_id) == []


def _install_one_shot_binder_fault(monkeypatch, fault_kind):
    """Monkeypatch the finalizer's pinned-module loader so the REAL binder
    raises once at the named boundary, then delegates normally on retry."""
    real_binder, real_kernel, real_adapters = kb._load_pinned_aion_modules("__fault__")
    state = {"fired": False}

    class _FaultyBinder:
        def bind_task_terminal_in_txn(self, **kwargs):
            if not state["fired"]:
                state["fired"] = True
                if fault_kind == "terminal_write":
                    def _boom(*a, **k):
                        raise RuntimeError("FAULT_AT_TERMINAL_WRITE")
                    kwargs["terminal_write"] = _boom
                elif fault_kind == "before_file":
                    def _boom(*a, **k):
                        raise RuntimeError("FAULT_BEFORE_FILE_WRITE")
                    kwargs["store_attachment"] = _boom
                elif fault_kind == "before_sha":
                    def _boom(conn, task_id, sha):
                        raise RuntimeError(
                            "FAULT_AFTER_FILE_AND_DB_INSERT_BEFORE_SHA"
                        )
                    kwargs["set_factory_terminal_receipt_sha"] = _boom
                elif fault_kind == "after_sha":
                    real_binder.bind_task_terminal_in_txn(**kwargs)
                    raise RuntimeError("FAULT_AFTER_SHA_BEFORE_COMMIT")
            return real_binder.bind_task_terminal_in_txn(**kwargs)

    monkeypatch.setattr(
        kb, "_load_pinned_aion_modules",
        lambda task_id: (_FaultyBinder(), real_kernel, real_adapters),
    )


@pytest.mark.parametrize(
    "fault_kind",
    ["terminal_write", "before_file", "before_sha", "after_sha"],
)
def test_f3_binder_boundary_fault_rolls_back_no_residue_then_retries(
    kanban_home, aion_gov_src, monkeypatch, fault_kind,
):
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="factory task", factory_build_gate=1, assignee="agent007",
        )
        run_id = _claim_and_run_id(conn, t)

        _install_one_shot_binder_fault(monkeypatch, fault_kind)
        with pytest.raises(RuntimeError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        _assert_zero_mutation_no_residue(conn, t, run_id)

        # Deterministic idempotent retry (fault already fired once) succeeds.
        assert kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status == "done"
        assert _bound_receipt_doc(conn, t)["verdict"] == "OUTCOME_ACCEPTED"
        att_dir = kb.task_attachments_dir(t)
        receipt_files = [
            p for p in att_dir.rglob("aion_monarch_receipt*.json") if p.is_file()
        ]
        assert len(receipt_files) == 1
        assert [p for p in att_dir.rglob("*.staging.*")] == []


def test_f3_fault_after_db_insert_before_event_no_residue_then_retries(
    kanban_home, aion_gov_src, monkeypatch,
):
    """M4: fault after the attachment row INSERT but before the 'attached' event.

    The finalizer's _store_attachment writes the staged file, INSERTs the row,
    then emits the 'attached' event. Faulting _append_event for that one call
    proves the staged file is discarded and no row survives the rollback."""
    orig_append = kb._append_event
    state = {"fired": False}

    def _one_shot_append(conn, task_id, kind, payload=None, **kwargs):
        if kind == "attached" and not state["fired"]:
            state["fired"] = True
            raise RuntimeError("FAULT_AFTER_DB_INSERT_BEFORE_EVENT")
        return orig_append(conn, task_id, kind, payload, **kwargs)

    monkeypatch.setattr(kb, "_append_event", _one_shot_append)
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="factory task", factory_build_gate=1, assignee="agent007",
        )
        run_id = _claim_and_run_id(conn, t)

        with pytest.raises(RuntimeError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        _assert_zero_mutation_no_residue(conn, t, run_id)

        assert kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status == "done"
        assert _bound_receipt_doc(conn, t)["verdict"] == "OUTCOME_ACCEPTED"
        att_dir = kb.task_attachments_dir(t)
        assert len([p for p in att_dir.rglob("aion_monarch_receipt*.json") if p.is_file()]) == 1


# ---------------------------------------------------------------------------
# F4 — post-COMMIT/pre-promote promotion failure + hard-crash/restart (R3)
# ---------------------------------------------------------------------------
# The R2 audit reproduced that promoting the staged receipt only AFTER SQLite
# COMMIT left authoritative done/sha/attachment/event state pointing at a
# missing receipt (os.replace injected to fail), and ordinary retry died on a
# terminal CAS miss. These tests prove the reorder: promotion now happens
# BEFORE COMMIT, so a promotion failure rolls back to zero mutation with no
# residue and a deterministic retry succeeds, and a hard crash in the narrow
# promote->commit window leaves only an unreachable orphan (no DB reference)
# that a restart recovers from cleanly.


def test_f4_promotion_failure_rolls_back_no_residue_then_retries(
    kanban_home, aion_gov_src, monkeypatch,
):
    """Promotion failure (os.replace raises on the staged file) must NOT commit
    terminal state; the txn rolls back to zero mutation and retry is clean."""
    real_replace = os.replace

    def _fail_staged(src, dst):
        if ".staging." in Path(src).name:
            raise OSError("INJECTED_POST_COMMIT_PROMOTE_FAILURE")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _fail_staged)
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="factory task", factory_build_gate=1, assignee="agent007",
        )
        child = kb.create_task(conn, title="child", assignee="agent007", parents=[t])
        run_id = _claim_and_run_id(conn, t)

        with pytest.raises(OSError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        _assert_zero_mutation_no_residue(conn, t, run_id)
        # Child must not wake on a rolled-back completion.
        assert kb.get_task(conn, child).status == "todo"

        # Deterministic idempotent retry (fault cleared) succeeds.
        monkeypatch.setattr(os, "replace", real_replace)
        assert kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status == "done"
        assert _bound_receipt_doc(conn, t)["verdict"] == "OUTCOME_ACCEPTED"
        assert kb.get_task(conn, child).status == "ready"
        att_dir = kb.task_attachments_dir(t)
        assert len([p for p in att_dir.rglob("aion_monarch_receipt*.json") if p.is_file()]) == 1
        assert [p for p in att_dir.rglob("*.staging.*")] == []


_CRASH_CHILD_SCRIPT = r"""
import os, sys
from pathlib import Path
sys.meta_path[:] = [
    f for f in sys.meta_path
    if "_Editable" not in repr(f) and "Editable" not in f.__class__.__name__
]
sys.path.insert(0, {candidate_repo!r})
os.environ["HERMES_HOME"] = {hermes_home!r}
os.environ["AION_GOVERNANCE_SOURCE_DIR"] = {aion_src!r}
from hermes_cli import kanban_db as kb
with kb.isolated_kanban_env(Path({worker_home!r})):
    kb.init_db()
    conn = kb.connect()
    t = kb.create_task(conn, title="crash-window-task", assignee="agent007", factory_build_gate=1)
    child = kb.create_task(conn, title="crash-child", assignee="agent007", parents=[t])
    kb.claim_task(conn, t)
    run_id = int(conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (t,)).fetchone()[0])
    real_replace = kb.os.replace
    def crash_after_promote(src, dst):
        if ".staging." in Path(src).name:
            real_replace(src, dst)  # promotion succeeds ...
            os._exit(99)            # ... then hard crash BEFORE COMMIT
        return real_replace(src, dst)
    kb.os.replace = crash_after_promote
    kb.complete_task(conn, t, result="done", expected_run_id=run_id)
    os._exit(0)
"""


def test_f4_hard_crash_restart_no_broken_terminal_state_recovers(
    kanban_home, aion_gov_src, tmp_path,
):
    """A hard crash (os._exit) after promotion but before COMMIT must leave the
    task running with no authoritative receipt/sha/attachment/event; only an
    unreachable orphan final file (no DB reference). A restart then completes
    deterministically and wakes the dependent child."""
    candidate_file = Path(kb.__file__).resolve()
    candidate_repo = candidate_file.parents[1]
    worker_home = tmp_path / "crash_home"
    script = _CRASH_CHILD_SCRIPT.format(
        candidate_repo=str(candidate_repo),
        hermes_home=str(Path(os.environ["HERMES_HOME"])),
        aion_src=str(_aion_gov_source_dir()),
        worker_home=str(worker_home),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(candidate_repo) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert out.returncode == 99, f"expected crash exit 99, got {out.returncode}: {out.stderr}"

    # Re-open the SAME isolated DB and assert zero authoritative terminal state.
    with kb.isolated_kanban_env(worker_home):
        conn = kb.connect(db_path=worker_home / "kanban.db")
        t = conn.execute(
            "SELECT id FROM tasks WHERE title = 'crash-window-task'"
        ).fetchone()
        child = conn.execute(
            "SELECT id FROM tasks WHERE title = 'crash-child'"
        ).fetchone()
        assert t is not None and child is not None
        tid, cid = t["id"], child["id"]

        row = conn.execute(
            "SELECT status, factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["factory_terminal_receipt_sha256"] is None
        assert kb.list_attachments(conn, tid) == []
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "attached" not in kinds and "completed" not in kinds
        att_dir = kb.task_attachments_dir(tid)
        assert [p for p in att_dir.rglob("*.staging.*")] == []
        # The only residue is an unreachable orphan final file with NO DB row.
        orphan_files = [p for p in att_dir.rglob("aion_monarch_receipt*.json") if p.is_file()]
        assert len(orphan_files) == 1
        assert orphan_files[0].exists()

        # Deterministic recovery: retry completes and wakes the child.
        run_id = int(
            conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        )
        assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, tid).status == "done"
        assert _bound_receipt_doc(conn, tid)["verdict"] == "OUTCOME_ACCEPTED"
        assert kb.get_task(conn, cid).status == "ready"
        # Exactly one bound receipt attachment row, file present, no staging.
        assert len(kb.list_attachments(conn, tid)) == 1
        assert Path(kb.list_attachments(conn, tid)[0].stored_path).exists()
        assert [p for p in att_dir.rglob("*.staging.*")] == []
        conn.close()


# ---------------------------------------------------------------------------
# F5 — ambiguous COMMIT durability reconciliation (R4, R3-F5 repair)
# ---------------------------------------------------------------------------
# The R3 audit reproduced a P0 defect at the COMMIT boundary: when the real
# SQLite COMMIT durably lands and only then the boundary raises, the old
# exception path issued ROLLBACK (a no-op after a landed commit) and then
# discarded the promoted receipt files — deleting the trusted receipt that the
# now-durable done/sha/attachment/event rows reference, losing the dependent
# wake, and leaving ordinary retry on a terminal CAS dead-end. These tests
# prove the R4 reconciliation: the ambiguous COMMIT outcome is resolved
# against the connection's own transaction state (``conn.in_transaction``), so
# a landed commit preserves the receipt and reconciles the dependent wake,
# while a not-landed commit leaves zero residue and retries cleanly.


def test_f5_landed_commit_then_error_preserves_receipt_and_reconciles(
    kanban_home, aion_gov_src, monkeypatch,
):
    """A COMMIT that durably lands and then raises must NOT discard the receipt.

    The ambiguous outcome is reconciled via ``conn.in_transaction``: the commit
    landed (transaction closed), so the promoted receipt is preserved, the
    terminal rows stay consistent, the dependent child wakes, and a retry is
    idempotent (no terminal CAS dead-end)."""
    with kb.connect() as conn:
        parent = kb.create_task(
            conn, title="factory parent", factory_build_gate=1, assignee="agent007",
        )
        child = kb.create_task(
            conn, title="child", assignee="agent007", parents=[parent],
        )
        run_id = _claim_and_run_id(conn, parent)

        real = kb._execute_boundary_with_retry
        injected = {"done": False}

        def commit_then_raise(c, sql):
            result = real(c, sql)
            if (
                sql.strip().upper() == "COMMIT"
                and kb._AION_PROMOTED_RECEIPT_FILES.get(id(c))
                and not injected["done"]
            ):
                injected["done"] = True
                raise RuntimeError("INJECTED_AMBIGUOUS_COMMIT_AFTER_REAL_COMMIT")
            return result

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", commit_then_raise)

        # The ambiguous COMMIT is reconciled: no exception escapes, the durable
        # terminal state is preserved, and the dependent wake still runs.
        assert kb.complete_task(conn, parent, result="done", expected_run_id=run_id)

        assert kb.get_task(conn, parent).status == "done"
        row = conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (parent,),
        ).fetchone()
        assert row["factory_terminal_receipt_sha256"]
        atts = kb.list_attachments(conn, parent)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).exists()  # trusted receipt preserved
        assert "completed" in [e.kind for e in kb.list_events(conn, parent)]
        assert kb.get_task(conn, child).status == "ready"  # dependent wake

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", real)

        # Retry is idempotent: already-done with a valid receipt -> no exception
        # and no terminal CAS dead-end; the receipt remains present.
        assert kb.complete_task(conn, parent, result="retry", expected_run_id=run_id) is False
        assert Path(kb.list_attachments(conn, parent)[0].stored_path).exists()


def test_f5_no_land_commit_error_zero_residue_then_retries(
    kanban_home, aion_gov_src, monkeypatch,
):
    """A COMMIT that fails WITHOUT landing must leave zero mutation/residue.

    ``conn.in_transaction`` is still true, so ROLLBACK truly undoes the writes
    and the staged+promoted receipt files are discarded; a clean retry then
    regenerates them and wakes the dependent."""
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="factory task", factory_build_gate=1, assignee="agent007",
        )
        child = kb.create_task(conn, title="child", assignee="agent007", parents=[t])
        run_id = _claim_and_run_id(conn, t)

        real = kb._execute_boundary_with_retry
        injected = {"done": False}

        def commit_fails_without_landing(c, sql):
            if sql.strip().upper() == "COMMIT" and not injected["done"]:
                injected["done"] = True
                # Model a commit that fails BEFORE landing: a non-busy
                # OperationalError (no retry) raised without executing the
                # commit, so the transaction stays open (in_transaction True).
                raise sqlite3.OperationalError(
                    "disk I/O error during commit (not landed)"
                )
            return real(c, sql)

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", commit_fails_without_landing)

        with pytest.raises(sqlite3.OperationalError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        _assert_zero_mutation_no_residue(conn, t, run_id)
        assert kb.get_task(conn, child).status == "todo"

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", real)

        assert kb.complete_task(conn, t, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, t).status == "done"
        assert _bound_receipt_doc(conn, t)["verdict"] == "OUTCOME_ACCEPTED"
        assert kb.get_task(conn, child).status == "ready"
        att_dir = kb.task_attachments_dir(t)
        assert len([p for p in att_dir.rglob("aion_monarch_receipt*.json") if p.is_file()]) == 1
        assert [p for p in att_dir.rglob("*.staging.*")] == []


def test_f5_post_commit_invariant_error_preserves_receipt_and_retries(
    kanban_home, aion_gov_src, monkeypatch,
):
    """The post-COMMIT file-length invariant raising must NOT discard the receipt.

    This is the contrast-safe boundary: the rows are already committed and the
    promoted receipt must remain present, so a retry is idempotent rather than
    a terminal CAS dead-end."""
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="factory task", factory_build_gate=1, assignee="agent007",
        )
        run_id = _claim_and_run_id(conn, t)

        real_invariant = kb._check_file_length_invariant
        injected = {"done": False}

        def invariant_raises_once(conn_arg):
            if not injected["done"]:
                injected["done"] = True
                raise sqlite3.DatabaseError("INJECTED_POST_COMMIT_TORN_EXTEND")
            return real_invariant(conn_arg)

        monkeypatch.setattr(kb, "_check_file_length_invariant", invariant_raises_once)

        with pytest.raises(sqlite3.DatabaseError):
            kb.complete_task(conn, t, result="done", expected_run_id=run_id)

        # Rows committed, receipt preserved (no discard on the invariant path).
        row = conn.execute(
            "SELECT status, factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (t,),
        ).fetchone()
        assert row["status"] == "done"
        assert row["factory_terminal_receipt_sha256"]
        atts = kb.list_attachments(conn, t)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).exists()
        assert "completed" in [e.kind for e in kb.list_events(conn, t)]

        monkeypatch.setattr(kb, "_check_file_length_invariant", real_invariant)

        # Retry is idempotent: no exception, receipt remains present.
        assert kb.complete_task(conn, t, result="retry", expected_run_id=run_id) is False
        assert Path(kb.list_attachments(conn, t)[0].stored_path).exists()


_F5_CRASH_AFTER_COMMIT_SCRIPT = r"""
import os, sys
from pathlib import Path
sys.meta_path[:] = [
    f for f in sys.meta_path
    if "_Editable" not in repr(f) and "Editable" not in f.__class__.__name__
]
sys.path.insert(0, {candidate_repo!r})
os.environ["HERMES_HOME"] = {hermes_home!r}
os.environ["AION_GOVERNANCE_SOURCE_DIR"] = {aion_src!r}
from hermes_cli import kanban_db as kb
with kb.isolated_kanban_env(Path({worker_home!r})):
    kb.init_db()
    conn = kb.connect()
    t = kb.create_task(conn, title="crash-after-commit-task", assignee="agent007", factory_build_gate=1)
    child = kb.create_task(conn, title="crash-after-commit-child", assignee="agent007", parents=[t])
    kb.claim_task(conn, t)
    run_id = int(conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (t,)).fetchone()[0])
    # Crash AFTER the completion transaction commits (durable done/receipt) but
    # BEFORE the dependent wake (recompute_ready) runs.
    def crash_after_commit(*args, **kwargs):
        os._exit(99)
    kb.recompute_ready = crash_after_commit
    kb.complete_task(conn, t, result="done", expected_run_id=run_id)
    os._exit(0)
"""


def test_f5_hard_crash_after_commit_restart_reconciles_dependent_wake(
    kanban_home, aion_gov_src, tmp_path,
):
    """A hard crash after the COMMIT lands (before the dependent wake) must leave
    a consistent terminal state — done + receipt present — and a restart must
    reconcile the dependent wake via recompute_ready, with an idempotent retry."""
    candidate_file = Path(kb.__file__).resolve()
    candidate_repo = candidate_file.parents[1]
    worker_home = tmp_path / "crash_after_commit_home"
    script = _F5_CRASH_AFTER_COMMIT_SCRIPT.format(
        candidate_repo=str(candidate_repo),
        hermes_home=str(Path(os.environ["HERMES_HOME"])),
        aion_src=str(_aion_gov_source_dir()),
        worker_home=str(worker_home),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(candidate_repo) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert out.returncode == 99, f"expected crash exit 99, got {out.returncode}: {out.stderr}"

    with kb.isolated_kanban_env(worker_home):
        conn = kb.connect(db_path=worker_home / "kanban.db")
        t = conn.execute(
            "SELECT id FROM tasks WHERE title = 'crash-after-commit-task'"
        ).fetchone()
        child = conn.execute(
            "SELECT id FROM tasks WHERE title = 'crash-after-commit-child'"
        ).fetchone()
        assert t is not None and child is not None
        tid, cid = t["id"], child["id"]

        # The completion durably landed before the crash: consistent terminal
        # state with the trusted receipt present.
        row = conn.execute(
            "SELECT status, factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["status"] == "done"
        assert row["factory_terminal_receipt_sha256"]
        atts = kb.list_attachments(conn, tid)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).exists()
        assert "completed" in [e.kind for e in kb.list_events(conn, tid)]

        # The dependent wake was lost by the crash (recompute_ready never ran) ...
        assert kb.get_task(conn, cid).status == "todo"
        # ... but is reconciled by the dispatcher's recompute_ready on restart.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, cid).status == "ready"

        # A restart retry of the completion is idempotent: no exception and no
        # terminal CAS dead-end (the original defect raised
        # FactoryTerminalReceiptRequiredError here). The receipt stays present.
        assert kb.complete_task(conn, tid, result="retry") is False
        assert Path(kb.list_attachments(conn, tid)[0].stored_path).exists()
        conn.close()


# ---------------------------------------------------------------------------
# F5-R6 — opaque (missing in_transaction) landed/non-landed COMMIT boundary
# ---------------------------------------------------------------------------
# The R5 audit (GemAION 4987252999) reproduced ``OPAQUE_LANDED_COMMIT_DELETES_
# DURABLE_RECEIPT_AND_LOSES_IMMEDIATE_WAKE``: a connection proxy that HIDES
# ``in_transaction`` but durably lands the COMMIT was collapsed to NOT-landed,
# so the promoted receipt was deleted, the immediate dependent wake was lost,
# and retry hit a terminal CAS miss. These tests prove the R6 tri-state
# reconciliation (``_ambiguous_commit_landed``) handles BOTH opaque outcomes
# authoritatively: landed preserves the receipt and reconciles terminal success;
# non-landed leaves zero residue and re-raises the original OperationalError.


class _OpaqueLandedProxy:
    """Delegates to a real sqlite3.Connection but HIDES ``in_transaction`` and
    durably lands the COMMIT before raising an OperationalError."""

    def __init__(self, real):
        self._real = real
        self.armed = False
        self.landed_then_raised = False

    def __getattr__(self, name):
        if name == "in_transaction":
            raise AttributeError(name)
        return getattr(self._real, name)

    def execute(self, sql, *args):
        normalized = " ".join(sql.strip().upper().split())
        if normalized == "COMMIT" and self.armed and not self.landed_then_raised:
            self._real.execute(sql, *args)  # durably land the transaction
            self.landed_then_raised = True
            raise sqlite3.OperationalError(
                "injected opaque boundary: COMMIT landed before error"
            )
        return self._real.execute(sql, *args)


class _OpaqueNonLandedProxy:
    """Delegates to a real sqlite3.Connection but HIDES ``in_transaction`` and
    raises on COMMIT WITHOUT landing (transaction stays open)."""

    def __init__(self, real):
        self._real = real
        self.armed = False

    def __getattr__(self, name):
        if name == "in_transaction":
            raise AttributeError(name)
        return getattr(self._real, name)

    def execute(self, sql, *args):
        normalized = " ".join(sql.strip().upper().split())
        if normalized == "COMMIT" and self.armed:
            raise sqlite3.OperationalError(
                "disk I/O error during commit (not landed)"
            )
        return self._real.execute(sql, *args)


def test_f5_opaque_landed_commit_preserves_receipt_and_wakes_dependent(
    kanban_home, aion_gov_src,
):
    """An opaque proxy hiding ``in_transaction`` that LANDS the COMMIT then
    raises must be reconciled as landed: complete returns True, the promoted
    receipt survives, and the child wakes immediately (no restart-only wake)."""
    with kb.connect() as real:
        conn = _OpaqueLandedProxy(real)
        parent = kb.create_task(
            conn, title="opaque-landed-parent", factory_build_gate=1,
            assignee="agent007",
        )
        child = kb.create_task(
            conn, title="opaque-landed-child", assignee="agent007",
            parents=[parent],
        )
        run_id = _claim_and_run_id(conn, parent)
        conn.armed = True

        # Reconcile terminal success: no exception escapes, the durable state is
        # preserved, and the dependent wake still runs (no restart).
        assert kb.complete_task(conn, parent, result="done", expected_run_id=run_id)

        row = conn.execute(
            "SELECT status, factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (parent,),
        ).fetchone()
        assert row["status"] == "done"
        assert row["factory_terminal_receipt_sha256"]
        atts = kb.list_attachments(conn, parent)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).exists()  # trusted receipt preserved
        assert "completed" in [e.kind for e in kb.list_events(conn, parent)]
        assert kb.get_task(conn, child).status == "ready"  # immediate wake

        # Idempotent retry/readback: no exception, no terminal CAS dead-end,
        # receipt remains present.
        assert kb.complete_task(conn, parent, result="retry", expected_run_id=run_id) is False
        assert Path(kb.list_attachments(conn, parent)[0].stored_path).exists()


def test_f5_opaque_nonlanded_commit_reraises_zero_residue_then_retries(
    kanban_home, aion_gov_src,
):
    """An opaque proxy hiding ``in_transaction`` whose COMMIT did NOT land must
    re-raise the original OperationalError, leave zero residue, and permit a
    clean idempotent retry."""
    with kb.connect() as real:
        conn = _OpaqueNonLandedProxy(real)
        parent = kb.create_task(
            conn, title="opaque-noland-parent", factory_build_gate=1,
            assignee="agent007",
        )
        child = kb.create_task(
            conn, title="opaque-noland-child", assignee="agent007",
            parents=[parent],
        )
        run_id = _claim_and_run_id(conn, parent)
        conn.armed = True

        with pytest.raises(sqlite3.OperationalError, match="not landed"):
            kb.complete_task(conn, parent, result="done", expected_run_id=run_id)

        # Zero mutation / zero residue.
        assert kb.get_task(conn, parent).status == "running"
        row = conn.execute(
            "SELECT factory_terminal_receipt_sha256 FROM tasks WHERE id = ?",
            (parent,),
        ).fetchone()
        assert row["factory_terminal_receipt_sha256"] is None
        assert kb.list_attachments(conn, parent) == []
        kinds = [e.kind for e in kb.list_events(conn, parent)]
        assert "completed" not in kinds
        assert kb.get_task(conn, child).status == "todo"

        # Clean retry succeeds and wakes the dependent.
        conn.armed = False
        assert kb.complete_task(conn, parent, result="done", expected_run_id=run_id)
        assert kb.get_task(conn, parent).status == "done"
        assert kb.get_task(conn, child).status == "ready"
        assert _bound_receipt_doc(conn, parent)["verdict"] == "OUTCOME_ACCEPTED"


def test_detached_controller_opaque_landed_commit_finalizes_and_recomputes(
    kanban_home, aion_gov_src, monkeypatch,
):
    """Opaque nesting participates in the outer landed controller transaction."""
    with kb.connect() as real:
        parent, action_run, child = _authorized_detached_controller_chain(
            real, monkeypatch,
        )
        proxy = _OpaqueLandedProxy(real)
        proxy.armed = True
        conn = cast(sqlite3.Connection, proxy)

        assert kb.complete_task(conn, parent, summary="opaque landed controller")

        parent_row = kb.get_task(conn, parent)
        assert parent_row is not None and parent_row.status == "done"
        child_row = kb.get_task(conn, child)
        assert child_row is not None and child_row.status == "ready"
        assert child_row.current_run_id is None
        completed = conn.execute(
            "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (parent,),
        ).fetchone()
        assert completed is not None and completed["run_id"] == action_run
        assert Path(kb.list_attachments(conn, parent)[0].stored_path).exists()


def test_detached_controller_opaque_nonlanded_commit_rolls_back_and_retries(
    kanban_home, aion_gov_src, monkeypatch,
):
    """Opaque non-landed failure preserves zero mutation and outer ownership."""
    with kb.connect() as real:
        parent, _action_run, child = _authorized_detached_controller_chain(
            real, monkeypatch,
        )
        before = _native_state_snapshot(real)
        before_files = {
            path: path.read_bytes()
            for path in kanban_home.rglob("aion_monarch_receipt*.json")
        }
        proxy = _OpaqueNonLandedProxy(real)
        proxy.armed = True
        conn = cast(sqlite3.Connection, proxy)

        with pytest.raises(sqlite3.OperationalError, match="not landed"):
            kb.complete_task(conn, parent, summary="opaque non-landed controller")

        assert _native_state_snapshot(conn) == before
        assert {
            path: path.read_bytes()
            for path in kanban_home.rglob("aion_monarch_receipt*.json")
        } == before_files
        parent_row = kb.get_task(conn, parent)
        child_row = kb.get_task(conn, child)
        assert parent_row is not None and parent_row.status == "blocked"
        assert child_row is not None and child_row.status == "todo"

        proxy.armed = False
        assert kb.complete_task(conn, parent, summary="clean controller retry")
        parent_row = kb.get_task(conn, parent)
        child_row = kb.get_task(conn, child)
        assert parent_row is not None and parent_row.status == "done"
        assert child_row is not None and child_row.status == "ready"


# ---------------------------------------------------------------------------
# AION-CORE-LEGACY-REVIEWED-AUTHOR-TERMINAL-RESOLVER-V1
# Legacy reviewed-author chains produced by the pre-strict-resolution machinery
# carry terminal-evidence-complete but chain-shape "debris" (mid-chain PASS that
# was self-corrected, infra-blocked provider_failure rounds, run-less
# superseded-duplicate archives) that the strict ordered-chain validation
# rejects. These fixtures reproduce each legacy shape natively and must
# terminalize WITHOUT weakening the fail-closed strict validation for clean
# new chains (covered by the hostile-drift suites above).
# ---------------------------------------------------------------------------


def _legacy_reused_auditor_mid_chain_pass_chain(conn):
    """t_ff0f4582-shaped: reused final auditor with a mid-chain erroneous PASS
    that the auditor self-corrected (blocked -> corrective REQUEST_CHANGES on a
    fresh run) before the final PASS."""
    author = kb.create_task(
        conn, title="legacy reused author", factory_build_gate=1, assignee="agent007",
    )
    reviewer = kb.create_task(
        conn, title="legacy reused auditor", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    child = kb.create_task(
        conn, title="legacy downstream", factory_build_gate=1, assignee="gm2",
        parents=[author],
    )
    # Round 0: handoff -> erroneous PASS -> self-correct -> corrective RC.
    author_run_0 = _claim_and_run_id(conn, author)
    handoff_0 = kb.request_review_handoff(
        conn, author, expected_run_id=author_run_0, review_task_id=reviewer,
        reason="candidate 0",
    )
    assert handoff_0 is not None
    reviewer_run_0 = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run_0, verdict="pass",
        reason="ERRONEOUS_PASS_0",
    )
    assert kb.block_task(
        conn, reviewer, reason="erroneous pass self-corrected", kind="needs_input",
        expected_run_id=reviewer_run_0,
    )
    assert kb.unblock_task(conn, reviewer)
    with kb.write_txn(conn):
        assert conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=? AND status='todo'",
            (reviewer,),
        ).rowcount == 1
        kb._append_event(conn, reviewer, "promoted", None)
    reviewer_run_1 = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run_1, verdict="request_changes",
        reason="CORRECTIVE_RC_0",
    )
    # Round 1: handoff -> final PASS.
    author_run_1 = _claim_and_run_id(conn, author)
    handoff_1 = kb.request_review_handoff(
        conn, author, expected_run_id=author_run_1, review_task_id=reviewer,
        reason="candidate 1",
    )
    assert handoff_1 is not None
    reviewer_run_2 = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run_2, verdict="pass",
        reason="FINAL_PASS",
    )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=reviewer_run_2,
        summary="final independent audit passed",
    )
    return {
        "author": author, "reviewer": reviewer, "child": child,
        "author_run_0": author_run_0, "author_run_1": author_run_1,
        "handoff_0": handoff_0, "handoff_1": handoff_1,
        "reviewer_run_0": reviewer_run_0, "reviewer_run_1": reviewer_run_1,
        "reviewer_run_2": reviewer_run_2,
    }


def test_canonical_audit_receipt_authenticates_legacy_mid_chain_pass(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _legacy_reused_auditor_mid_chain_pass_chain(conn)
        before = _native_state_snapshot(conn)

        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt["verdict"] == "PASS"
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert receipt["auditor_run_id"] == chain["reviewer_run_2"]
        assert kb._reviewed_author_finalizer_run_id(
            conn, chain["author"],
        ) == chain["author_run_1"]
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="legacy reused-auditor terminalized",
        )
        author_task = kb.get_task(conn, chain["author"])
        child_task = kb.get_task(conn, chain["child"])
        assert author_task is not None and author_task.status == "done"
        assert child_task is not None and child_task.status == "ready"


def test_canonical_audit_receipt_rejects_unsuperseded_mid_chain_pass(
    kanban_home, aion_gov_src,
):
    """A mid-chain PASS with no later REQUEST_CHANGES must stay fail-closed."""
    with kb.connect() as conn:
        chain = _legacy_reused_auditor_mid_chain_pass_chain(conn)
        # Remove the corrective RC round entirely, leaving the mid-chain PASS
        # unsuperseded before the final PASS.
        for task_id in (chain["author"], chain["reviewer"]):
            conn.execute(
                "DELETE FROM task_events WHERE task_id=? AND kind='review_verdict' "
                "AND run_id=?",
                (task_id, chain["reviewer_run_1"]),
            )
        conn.execute(
            "DELETE FROM task_runs WHERE id=? AND task_id=?",
            (chain["reviewer_run_1"], chain["reviewer"]),
        )
        conn.commit()
        before = _native_state_snapshot(conn)

        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
        with pytest.raises(kb.FactoryTerminalReceiptRequiredError):
            kb.complete_task(conn, chain["author"], summary="reject unsuperseded pass")
        assert _native_state_snapshot(conn) == before


def _legacy_superseded_duplicate_post_handoff_archive_chain(conn):
    """t_9168252c/t_4eac3fb7-shaped: clean single-round PASS plus a run-less
    duplicate audit child created and authenticated-archived AFTER the handoff."""
    author = kb.create_task(
        conn, title="clean reviewed author", factory_build_gate=1, assignee="agent007",
    )
    reviewer = kb.create_task(
        conn, title="role-separated audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    child = kb.create_task(
        conn, title="downstream", factory_build_gate=1, assignee="gm2",
        parents=[author],
    )
    author_run = _claim_and_run_id(conn, author)
    handoff = kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=reviewer,
        reason="exact native receipt audit",
    )
    assert handoff is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=reviewer_run, verdict="pass", reason="PASS_EXACT",
    )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=reviewer_run, summary="passed",
    )
    duplicate = kb.create_task(
        conn, title="superseded duplicate audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    with kb._authenticated_strict_orchestrator_archive():
        assert kb.archive_task(
            conn, duplicate,
            reason="superseded duplicate: author already handed off to running child",
            actor="kanban-orchestrator", source="kanban_archive",
            fail_if_active_run=True, expected_status="todo",
        )
    return {
        "author": author, "reviewer": reviewer, "child": child,
        "duplicate": duplicate, "author_run": author_run,
        "handoff": handoff, "reviewer_run": reviewer_run,
    }


def test_canonical_audit_receipt_ignores_post_handoff_superseded_duplicate(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _legacy_superseded_duplicate_post_handoff_archive_chain(conn)
        before = _native_state_snapshot(conn)

        assert kb._historical_auditor_child_is_non_authoritative(
            conn,
            author_task_id=chain["author"],
            author_profile="agent007",
            auditor_task_id=chain["duplicate"],
            auditor_profile="bafuxunan",
            latest_handoff_event_id=chain["handoff"].event_id,
        )
        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="clean author with superseded duplicate",
        )
        author_task = kb.get_task(conn, chain["author"])
        assert author_task is not None and author_task.status == "done"


def _legacy_historical_auditor_blocked_provider_failure_chain(conn):
    """t_e93991d8-shaped: final auditor differs from a historical auditor whose
    chain contains a mid-chain provider_failure blocked run (no verdict)."""
    author = kb.create_task(
        conn, title="legacy multi-auditor author", factory_build_gate=1,
        assignee="agent007",
    )
    historical = kb.create_task(
        conn, title="historical auditor", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    reviewer = kb.create_task(
        conn, title="final auditor", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    child = kb.create_task(
        conn, title="downstream", factory_build_gate=1, assignee="gm2",
        parents=[author],
    )
    # Round 0 (historical): handoff -> blocked provider_failure -> unblock -> RC.
    author_run_0 = _claim_and_run_id(conn, author)
    h0 = kb.request_review_handoff(
        conn, author, expected_run_id=author_run_0, review_task_id=historical,
        reason="round 0",
    )
    assert h0 is not None
    hr0 = _claim_and_run_id(conn, historical)
    assert kb.block_task(
        conn, historical, reason="provider failure", kind="transient",
        expected_run_id=hr0,
    )
    assert kb.unblock_task(conn, historical)
    with kb.write_txn(conn):
        assert conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=? AND status='todo'",
            (historical,),
        ).rowcount == 1
        kb._append_event(conn, historical, "promoted", None)
    hr1 = _claim_and_run_id(conn, historical)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=historical,
        expected_review_run_id=hr1, verdict="request_changes", reason="RC_0",
    )
    # Final round: handoff to a fresh final auditor -> PASS.
    author_run_1 = _claim_and_run_id(conn, author)
    h1 = kb.request_review_handoff(
        conn, author, expected_run_id=author_run_1, review_task_id=reviewer,
        reason="final",
    )
    assert h1 is not None
    fr = _claim_and_run_id(conn, reviewer)
    assert _record_legacy_review_verdict_fixture(
        conn, author, review_task_id=reviewer,
        expected_review_run_id=fr, verdict="pass", reason="FINAL_PASS",
    )
    assert kb.complete_task(
        conn, reviewer, expected_run_id=fr, summary="final independent pass",
    )
    return {
        "author": author, "historical": historical, "reviewer": reviewer,
        "child": child, "author_run_0": author_run_0, "author_run_1": author_run_1,
        "handoff_0": h0, "handoff_1": h1, "hr0": hr0, "hr1": hr1, "fr": fr,
    }


def test_canonical_audit_receipt_authenticates_legacy_blocked_provider_failure(
    kanban_home, aion_gov_src,
):
    with kb.connect() as conn:
        chain = _legacy_historical_auditor_blocked_provider_failure_chain(conn)
        before = _native_state_snapshot(conn)

        assert kb._historical_auditor_child_is_non_authoritative(
            conn,
            author_task_id=chain["author"],
            author_profile="agent007",
            auditor_task_id=chain["historical"],
            auditor_profile="bafuxunan",
            latest_handoff_event_id=chain["handoff_1"].event_id,
        )
        receipt = kb._canonical_audit_receipt(conn, chain["author"])

        assert receipt is not None
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert receipt["verdict"] == "PASS"
        assert _native_state_snapshot(conn) == before
        assert kb.complete_task(
            conn, chain["author"], summary="legacy multi-auditor terminalized",
        )
        author_task = kb.get_task(conn, chain["author"])
        assert author_task is not None and author_task.status == "done"


# ---------------------------------------------------------------------------
# PR #98 earlier-v3 envelope compatibility ingress (pre-merge resident 95392cf8)
# ---------------------------------------------------------------------------
# The pre-merge resident persisted the canonical audit-outcome envelope in an
# earlier v3 key set; the final head accepts only the later v3 key set. These
# tests prove the read-only in-memory ingress resolves the authentic earlier-v3
# shape and fails closed on every hostile variant, without mutating history.


_EARLIER_V3_REASON = "APPROVED_EXACT_HEAD: earlier-v3 authentic envelope"


def _earlier_v3_audit_outcome_chain(conn, *, author_assignee="agent007"):
    """Seed a role-separated chain whose audit-outcome envelope is the earlier-v3
    key set (the live compatibility incident, minus any sensitive value)."""
    author = kb.create_task(
        conn, title="earlier-v3 reviewed author", factory_build_gate=1,
        assignee=author_assignee,
    )
    reviewer = kb.create_task(
        conn, title="role-separated audit", factory_build_gate=1,
        assignee="bafuxunan", parents=[author],
    )
    child = kb.create_task(
        conn, title="downstream product transition", factory_build_gate=1,
        assignee="gm2", parents=[author],
    )
    author_run = _claim_and_run_id(conn, author)
    handoff = kb.request_review_handoff(
        conn, author, expected_run_id=author_run, review_task_id=reviewer,
        reason="earlier-v3 exact receipt audit",
    )
    assert handoff is not None
    reviewer_run = _claim_and_run_id(conn, reviewer)
    # Directly terminalize the audit (no kernel finalizer): the role separation
    # is already bound and the factory terminal receipt is pre-seeded so the
    # compatibility ingress can validate the audit's terminal-receipt gate.
    with kb.write_txn(conn):
        kb._execute_factory_terminal_write(
            conn, reviewer,
            "UPDATE tasks SET status='done', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, factory_terminal_receipt_sha256=? "
            "WHERE id=?",
            ("f" * 64, reviewer),
        )
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='completed', ended_at=1, "
            "summary='independent audit passed' WHERE id=?",
            (reviewer_run,),
        )
    evidence = {
        "repository": "kiddhu/hermes-agent",
        "pr": 98,
        "head": "3e70a4a91562f60532c742404b126f4cb369d062",
        "tree": "e112b9b72910a39972b37229de7eeef7cc0a4d92",
        "base": "25fcb86314ea81b152406c42e5389f3f9d9849f3",
        "github_review_id": 5164970120,
        "github_review_url": "https://github.com/kiddhu/hermes-agent/pull/98#pullrequestreview-5164970120",
        "github_review_state": "APPROVED",
    }
    evidence_sha256 = kb._canonical_audit_outcome_evidence_sha256(evidence)
    chain = {
        "author": author, "author_run": author_run, "reviewer": reviewer,
        "reviewer_run": reviewer_run, "child": child, "handoff": handoff,
        "evidence": evidence, "evidence_sha256": evidence_sha256,
        "author_assignee": author_assignee,
    }
    _bind_earlier_v3_pass_verdict(conn, chain)
    chain["outcome_event_id"] = _seed_earlier_v3_outcome(conn, chain)
    return chain


def _bind_earlier_v3_pass_verdict(conn, chain):
    """Mirror the PASS review_verdict rows byte-bound to the envelope's
    reason/evidence, exactly as the pre-merge resident persisted them."""
    payload = {
        "version": 2,
        "review_task_id": chain["reviewer"],
        "review_run_id": chain["reviewer_run"],
        "verdict": "pass",
        "reason": _EARLIER_V3_REASON,
        "evidence": chain["evidence"],
        "evidence_sha256": chain["evidence_sha256"],
    }
    with kb.write_txn(conn):
        kb._append_event(
            conn, chain["reviewer"], "review_verdict", payload,
            run_id=chain["reviewer_run"],
        )
        kb._append_event(
            conn, chain["author"], "review_verdict", payload,
            run_id=chain["reviewer_run"],
        )


def _rebind_earlier_v3_verdict(conn, chain, *, verdict_payload=None):
    """Replace both bound v2 PASS verdict rows with ``verdict_payload``.

    Used to seed a tampered digest or an unknown/mixed key set so the ingress
    is proven to reject a malformed authority record rather than trust it.
    """
    if verdict_payload is None:
        verdict_payload = {
            "version": 2,
            "review_task_id": chain["reviewer"],
            "review_run_id": chain["reviewer_run"],
            "verdict": "pass",
            "reason": _EARLIER_V3_REASON,
            "evidence": chain["evidence"],
            "evidence_sha256": chain["evidence_sha256"],
        }
    with kb.write_txn(conn):
        conn.execute(
            "DELETE FROM task_events WHERE kind='review_verdict' "
            "AND task_id IN (?, ?) AND run_id = ?",
            (chain["author"], chain["reviewer"], chain["reviewer_run"]),
        )
        kb._append_event(
            conn, chain["reviewer"], "review_verdict", verdict_payload,
            run_id=chain["reviewer_run"],
        )
        kb._append_event(
            conn, chain["author"], "review_verdict", verdict_payload,
            run_id=chain["reviewer_run"],
        )


def _earlier_v3_envelope_for(chain):
    return {
        "version": 3,
        "author_task_id": chain["author"],
        "author_run_id": chain["author_run"],
        "audit_task_id": chain["reviewer"],
        "audit_run_id": chain["reviewer_run"],
        "handoff_event_id": chain["handoff"].event_id,
        "verdict": "PASS",
        "reason": _EARLIER_V3_REASON,
        "evidence": chain["evidence"],
        "role_separation": {
            "author_profile": chain["author_assignee"],
            "auditor_profile": "bafuxunan",
        },
        "evidence_sha256": chain["evidence_sha256"],
        "created_at": 1,
    }


def _earlier_v3_fact_for(chain, outcome_event_id, evidence_sha256=None):
    return {
        "task_id": chain["reviewer"],
        "run_id": chain["reviewer_run"],
        "prior_status": "running",
        "new_status": "done",
        "event_id": outcome_event_id,
        "audit_outcome_sha256": evidence_sha256 or chain["evidence_sha256"],
        "disposition": "CONTINUATION_COMMITTED",
        "continuation_task_ids": [chain["child"]],
    }


def _seed_earlier_v3_outcome(
    conn, chain, *, envelope=None, fact=None, author_mirror=True,
    author_envelope=None, dual=False, dual_author_mirror=False,
    conflicting_fact=False,
):
    """Seed the earlier-v3 canonical_audit_outcome + changed_fact events.

    Returns the audit outcome event id. ``author_envelope`` (when set) writes a
    different author mirror payload than the audit event; ``dual`` writes a
    second audit canonical_audit_outcome event (conflicting dual receipt);
    ``dual_author_mirror`` writes a second conflicting author mirror;
    ``conflicting_fact`` writes a conflicting changed_fact before the matching
    one on each side (hidden duplicate).
    """
    if envelope is None:
        envelope = _earlier_v3_envelope_for(chain)
    with kb.write_txn(conn):
        kb._append_event(
            conn, chain["reviewer"], "canonical_audit_outcome", envelope,
            run_id=chain["reviewer_run"], created_at=1,
        )
        outcome_event_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        if dual:
            kb._append_event(
                conn, chain["reviewer"], "canonical_audit_outcome", envelope,
                run_id=chain["reviewer_run"], created_at=2,
            )
        if author_mirror:
            kb._append_event(
                conn, chain["author"], "canonical_audit_outcome",
                envelope if author_envelope is None else author_envelope,
                run_id=chain["author_run"], created_at=1,
            )
            if dual_author_mirror:
                conflicting = dict(envelope)
                conflicting["reason"] = "CONFLICTING_AUTHOR_MIRROR"
                kb._append_event(
                    conn, chain["author"], "canonical_audit_outcome", conflicting,
                    run_id=chain["author_run"], created_at=2,
                )
        if fact is None:
            fact = _earlier_v3_fact_for(chain, outcome_event_id)
        if conflicting_fact:
            conflicting = dict(fact)
            conflicting["prior_status"] = "blocked"
            kb._append_event(
                conn, chain["reviewer"], "changed_fact", conflicting,
                run_id=chain["reviewer_run"], created_at=1,
            )
            kb._append_event(
                conn, chain["author"], "changed_fact", conflicting,
                run_id=chain["author_run"], created_at=1,
            )
        kb._append_event(
            conn, chain["reviewer"], "changed_fact", fact,
            run_id=chain["reviewer_run"], created_at=2 if conflicting_fact else 1,
        )
        kb._append_event(
            conn, chain["author"], "changed_fact", fact,
            run_id=chain["author_run"], created_at=2 if conflicting_fact else 1,
        )
    return outcome_event_id


def _clear_earlier_v3_outcome(conn, chain):
    with kb.write_txn(conn):
        conn.execute(
            "DELETE FROM task_events WHERE task_id IN (?, ?) AND kind IN "
            "('canonical_audit_outcome', 'changed_fact')",
            (chain["reviewer"], chain["author"]),
        )


def test_earlier_v3_audit_outcome_authentic_envelope_resolves_same_author(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        before = _native_state_snapshot(conn)

        present, receipt = kb._canonical_current_audit_outcome(conn, chain["author"])
        assert present is True
        assert receipt is not None
        assert receipt["author_task_id"] == chain["author"]
        assert receipt["author_run_id"] == chain["author_run"]
        assert receipt["author_profile"] == "agent007"
        assert receipt["auditor_task_id"] == chain["reviewer"]
        assert receipt["auditor_run_id"] == chain["reviewer_run"]
        assert receipt["auditor_profile"] == "bafuxunan"
        assert receipt["verdict"] == "PASS"
        assert receipt["subject_version_or_exact_hash"] == chain["evidence_sha256"]
        assert receipt["issued_at"] == chain["outcome_event_id"]
        assert re.fullmatch(r"[0-9a-f]{64}", receipt["receipt_hash"])
        assert receipt["authenticated"] is True

        # The full receipt resolver and finalizer return the exact same author
        # continuation identity, with zero mutation.
        resolved = kb._canonical_audit_receipt(conn, chain["author"])
        assert resolved == receipt
        assert kb._reviewed_author_finalizer_run_id(
            conn, chain["author"],
        ) == chain["author_run"]
        assert _native_state_snapshot(conn) == before


@pytest.mark.parametrize(
    "label, envelope_overrides",
    [
        ("tampered_evidence_digest", {"evidence_sha256": "0" * 64}),
        ("wrong_verdict", {"verdict": "REQUEST_CHANGES"}),
        ("wrong_author_task", {"author_task_id": "t_foreign_author"}),
        ("wrong_audit_task", {"audit_task_id": "t_foreign_audit"}),
        ("wrong_audit_run", {"audit_run_id": 999999}),
        ("wrong_author_run", {"author_run_id": 999998}),
        ("wrong_handoff_event", {"handoff_event_id": 987654}),
        ("collapsed_role_separation", {
            "role_separation": {
                "author_profile": "agent007", "auditor_profile": "agent007",
            },
        }),
        ("missing_role_profile", {
            "role_separation": {"author_profile": "agent007"},
        }),
        ("malformed_evidence", {
            "evidence": {
                "repository": "kiddhu/hermes-agent",
                "pr": 98,
                "head": "3e70a4a91562f60532c742404b126f4cb369d062",
                "tree": "e112b9b72910a39972b37229de7eeef7cc0a4d92",
                "base": "25fcb86314ea81b152406c42e5389f3f9d9849f3",
                "github_review_id": 5164970120,
                "github_review_url": "https://github.com/kiddhu/hermes-agent/pull/98#pullrequestreview-5164970120",
                "github_review_state": "CHANGES_REQUESTED",
            },
        }),
    ],
)
def test_earlier_v3_audit_outcome_hostile_envelope_fails_closed(
    kanban_home, label, envelope_overrides,
):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        envelope = _earlier_v3_envelope_for(chain)
        envelope.update(envelope_overrides)
        # A tampered evidence block must still carry a self-consistent digest
        # so the failure is attributable to the evidence identity, not the
        # digest recompute alone.
        if "evidence" in envelope_overrides:
            envelope["evidence_sha256"] = kb._canonical_audit_outcome_evidence_sha256(
                envelope["evidence"]
            )
        _seed_earlier_v3_outcome(conn, chain, envelope=envelope)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_mixed_key_set_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        envelope = _earlier_v3_envelope_for(chain)
        # Swap the earlier handoff key for the later review_handoff key: a
        # mixed-key envelope is neither earlier-v3 nor later-v3 and must fail
        # closed (no legacy fallback over a present canonical record).
        envelope["review_handoff_event_id"] = envelope.pop("handoff_event_id")
        _seed_earlier_v3_outcome(conn, chain, envelope=envelope)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_missing_author_mirror_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        _seed_earlier_v3_outcome(conn, chain, author_mirror=False)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_tampered_author_mirror_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        envelope = _earlier_v3_envelope_for(chain)
        tampered = dict(envelope)
        tampered["reason"] = "TAMPERED_MIRROR"
        _seed_earlier_v3_outcome(conn, chain, envelope=envelope, author_envelope=tampered)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_dual_receipt_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        _seed_earlier_v3_outcome(conn, chain, dual=True)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_duplicate_author_mirror_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        # A second conflicting author mirror must fail closed even when the
        # first (matching) mirror is still present.
        _seed_earlier_v3_outcome(conn, chain, dual_author_mirror=True)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_hidden_conflicting_fact_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        # A conflicting changed_fact followed by the matching one must fail
        # closed on non-singleton history, not resolve to the latest row.
        _seed_earlier_v3_outcome(conn, chain, conflicting_fact=True)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_missing_bound_verdict_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        # Removing the mirrored PASS review_verdict rows must revoke the ingress
        # authority: it may not act as an alternate authority over a present v3
        # envelope that no longer corroborates the grant.
        with kb.write_txn(conn):
            conn.execute(
                "DELETE FROM task_events WHERE kind='review_verdict' "
                "AND task_id IN (?, ?) AND run_id = ?",
                (chain["author"], chain["reviewer"], chain["reviewer_run"]),
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_drifted_evidence_identity_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        # A syntactically valid, self-consistent but different APPROVED identity
        # (recomputed digest + mirrors + facts) must fail closed: it no longer
        # byte-matches the bound PASS review_verdict evidence.
        drifted = {
            "repository": "kiddhu/hermes-agent",
            "pr": 98,
            "head": "4" * 40,
            "tree": "5" * 40,
            "base": "6" * 40,
            "github_review_id": 999999999,
            "github_review_url": (
                "https://github.com/kiddhu/hermes-agent/pull/98"
                "#pullrequestreview-999999999"
            ),
            "github_review_state": "APPROVED",
        }
        drifted_sha = kb._canonical_audit_outcome_evidence_sha256(drifted)
        envelope = _earlier_v3_envelope_for(chain)
        envelope["evidence"] = drifted
        envelope["evidence_sha256"] = drifted_sha
        chain["evidence_sha256"] = drifted_sha
        _seed_earlier_v3_outcome(conn, chain, envelope=envelope)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_ambiguous_second_parent_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        extra = kb.create_task(
            conn, title="foreign second parent", factory_build_gate=1,
            assignee="gm2",
        )
        kb.link_tasks(conn, extra, chain["reviewer"])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_tampered_bound_verdict_digest_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        # A bound PASS verdict whose evidence_sha256 was replaced with 64
        # zeroes must not authenticate: the digest no longer recomputes over
        # the closed evidence block.
        tampered = {
            "version": 2,
            "review_task_id": chain["reviewer"],
            "review_run_id": chain["reviewer_run"],
            "verdict": "pass",
            "reason": _EARLIER_V3_REASON,
            "evidence": chain["evidence"],
            "evidence_sha256": "0" * 64,
        }
        _rebind_earlier_v3_verdict(conn, chain, verdict_payload=tampered)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_unknown_bound_verdict_key_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        # An unknown foreign key makes the verdict neither the exact seven-key
        # v2 schema nor any supported shape; it must fail closed, never act as
        # an alternate authority over the present v3 record.
        tampered = {
            "version": 2,
            "review_task_id": chain["reviewer"],
            "review_run_id": chain["reviewer_run"],
            "verdict": "pass",
            "reason": _EARLIER_V3_REASON,
            "evidence": chain["evidence"],
            "evidence_sha256": chain["evidence_sha256"],
            "foreign_authority": "evil",
        }
        _rebind_earlier_v3_verdict(conn, chain, verdict_payload=tampered)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_byte_different_author_mirror_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        envelope = _earlier_v3_envelope_for(chain)
        # Semantically equal object, byte-different serialization: reordering
        # the top-level keys changes the json.dumps bytes without changing the
        # parsed value. The ingress requires byte-identity with the audit event.
        reordered = {key: envelope[key] for key in reversed(list(envelope.keys()))}
        assert json.loads(json.dumps(reordered)) == json.loads(json.dumps(envelope))
        assert json.dumps(reordered) != json.dumps(envelope)
        _seed_earlier_v3_outcome(conn, chain, envelope=envelope, author_envelope=reordered)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


# ---------------------------------------------------------------------------
# Round-4 hostile regressions — duplicate JSON member names fail closed.
#
# Python ``json.loads`` keeps only the last duplicate member, so a raw authority
# record that repeats a member name (a wrong value first, the authentic value
# last) collapses into the expected key set and authenticates. The ingress must
# decode every raw record in the earlier-v3 chain with recursive
# duplicate-member rejection so such tampering fails closed before any exact-key
# or digest check can trust the collapsed result.
# ---------------------------------------------------------------------------

def _raw_duplicate_member_json(obj, member_key, wrong_value) -> str:
    """Serialize ``obj`` with a duplicate ``member_key`` (wrong first, real last).

    The wrong value is injected immediately before the authentic member, so
    ``json.loads`` (last-wins) collapses to the authentic value while strict
    duplicate-member rejection raises on the repeated name.
    """
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    needle = f'"{member_key}":'
    idx = text.index(needle)
    return text[:idx] + f'"{member_key}":{json.dumps(wrong_value)},' + text[idx:]


def _overwrite_event_payloads(conn, specs) -> None:
    """Overwrite the payload of each singleton event row with raw JSON text."""
    with kb.write_txn(conn):
        for task_id, run_id, kind, raw_text in specs:
            rows = conn.execute(
                "SELECT id FROM task_events WHERE task_id = ? AND run_id = ? "
                "AND kind = ?",
                (task_id, run_id, kind),
            ).fetchall()
            assert len(rows) == 1, (task_id, run_id, kind, len(rows))
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (raw_text, rows[0]["id"]),
            )


def test_earlier_v3_audit_outcome_duplicate_envelope_member_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        raw = _raw_duplicate_member_json(_earlier_v3_envelope_for(chain), "version", 999)
        # Both mirrors carry the byte-identical duplicate so only the strict
        # decode can reject them (the byte-identity and collapsed key-set checks
        # would otherwise pass).
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "canonical_audit_outcome", raw),
            (chain["author"], chain["author_run"], "canonical_audit_outcome", raw),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_duplicate_evidence_member_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        # A wrong first ``head`` followed by the authentic ``head`` inside the
        # nested evidence block collapses to the authentic value under plain
        # json.loads but must fail closed under recursive duplicate rejection.
        raw = _raw_duplicate_member_json(
            _earlier_v3_envelope_for(chain), "head", "0" * 40,
        )
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "canonical_audit_outcome", raw),
            (chain["author"], chain["author_run"], "canonical_audit_outcome", raw),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_duplicate_bound_verdict_member_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        verdict = {
            "version": 2,
            "review_task_id": chain["reviewer"],
            "review_run_id": chain["reviewer_run"],
            "verdict": "pass",
            "reason": _EARLIER_V3_REASON,
            "evidence": chain["evidence"],
            "evidence_sha256": chain["evidence_sha256"],
        }
        raw = _raw_duplicate_member_json(verdict, "version", 999)
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "review_verdict", raw),
            (chain["author"], chain["reviewer_run"], "review_verdict", raw),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_duplicate_bound_verdict_evidence_member_fails_closed(
    kanban_home,
):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        verdict = {
            "version": 2,
            "review_task_id": chain["reviewer"],
            "review_run_id": chain["reviewer_run"],
            "verdict": "pass",
            "reason": _EARLIER_V3_REASON,
            "evidence": chain["evidence"],
            "evidence_sha256": chain["evidence_sha256"],
        }
        raw = _raw_duplicate_member_json(verdict, "head", "0" * 40)
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "review_verdict", raw),
            (chain["author"], chain["reviewer_run"], "review_verdict", raw),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_duplicate_handoff_member_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        handoff_row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ?",
            (chain["author"], "review_handoff"),
        ).fetchone()
        assert handoff_row is not None
        raw = _raw_duplicate_member_json(
            json.loads(handoff_row["payload"]), "version", 999,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (raw, handoff_row["id"]),
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_duplicate_changed_fact_member_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        fact = _earlier_v3_fact_for(chain, chain["outcome_event_id"])
        raw = _raw_duplicate_member_json(fact, "new_status", "blocked")
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "changed_fact", raw),
            (chain["author"], chain["author_run"], "changed_fact", raw),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


# ---------------------------------------------------------------------------
# Round-5 hostile regressions — deeply nested malformed JSON fails closed.
#
# ``json.loads`` uses a recursive C decoder, so an unboundedly nested record
# (a 1200-level array placed in the envelope, a bound v2 verdict, the handoff,
# or a changed fact) raises ``RecursionError`` — a ``RuntimeError`` that the
# call sites' ``except (TypeError, ValueError)`` handlers do not catch. The
# strict decoder must normalize that to the rejected-record result so a
# malformed authority record can never crash reviewed-author resolution.
# ---------------------------------------------------------------------------

def _deeply_nested_json(depth: int = 1200) -> str:
    """A ``depth``-level nested JSON array, deep enough to exceed the C
    decoder's recursion limit (RecursionError before the fix, fail-closed
    ValueError after)."""
    return "[" * depth + "]" * depth


def test_earlier_v3_audit_outcome_deeply_nested_envelope_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "canonical_audit_outcome",
             _deeply_nested_json()),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_deeply_nested_bound_verdict_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "review_verdict",
             _deeply_nested_json()),
            (chain["author"], chain["reviewer_run"], "review_verdict",
             _deeply_nested_json()),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_deeply_nested_handoff_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        handoff_row = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = ?",
            (chain["author"], "review_handoff"),
        ).fetchone()
        assert handoff_row is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (_deeply_nested_json(), handoff_row["id"]),
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_deeply_nested_changed_fact_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _overwrite_event_payloads(conn, [
            (chain["reviewer"], chain["reviewer_run"], "changed_fact",
             _deeply_nested_json()),
            (chain["author"], chain["author_run"], "changed_fact",
             _deeply_nested_json()),
        ])

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


# ---------------------------------------------------------------------------
# Round-9 regressions — lifecycle freshness and producer-shape gates.
#
# The earlier-v3 ingress must bind the envelope to the LATEST author/audit runs
# and the LATEST review handoff (freshness), and require byte-identical paired
# verdict/fact mirrors with exact integer id types and a created_at bound to the
# event timestamp (producer shape). A stale or drifted receipt must fail closed
# instead of authenticating a superseded or non-authentic authority record.
# ---------------------------------------------------------------------------


def test_earlier_v3_audit_outcome_later_duplicate_handoff_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        row = conn.execute(
            "SELECT payload FROM task_events WHERE id = ?",
            (chain["handoff"].event_id,),
        ).fetchone()
        # A second, later review_handoff (re-issued review) supersedes the bound
        # one and must fail closed.
        with kb.write_txn(conn):
            kb._append_event(
                conn, chain["author"], "review_handoff", json.loads(row["payload"]),
                run_id=chain["author_run"], created_at=2,
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_newer_author_run_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        # A newer terminal review_required author run supersedes the envelope's
        # author run and must fail closed.
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,"
                "outcome,summary) VALUES (?,?, 'review_required', 2, 2, "
                "'review_required', 'newer author review')",
                (chain["author"], "agent007"),
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_newer_audit_run_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        # A newer completed audit run supersedes the envelope's audit run and
        # must fail closed.
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,"
                "outcome,summary) VALUES (?,?, 'done', 2, 2, 'completed', "
                "'newer audit pass')",
                (chain["reviewer"], "bafuxunan"),
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_created_at_drift_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        envelope = _earlier_v3_envelope_for(chain)
        # The envelope's created_at no longer matches the seeded event timestamp.
        envelope["created_at"] = 999
        _seed_earlier_v3_outcome(conn, chain, envelope=envelope)

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_float_fact_ids_fail_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        _clear_earlier_v3_outcome(conn, chain)
        envelope = _earlier_v3_envelope_for(chain)
        # Float run_id/event_id in the changed_fact are not exact integers and
        # must fail closed.
        with kb.write_txn(conn):
            kb._append_event(
                conn, chain["reviewer"], "canonical_audit_outcome", envelope,
                run_id=chain["reviewer_run"], created_at=1,
            )
            event_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            kb._append_event(
                conn, chain["author"], "canonical_audit_outcome", envelope,
                run_id=chain["author_run"], created_at=1,
            )
            fact = _earlier_v3_fact_for(chain, event_id)
            fact["run_id"] = float(fact["run_id"])
            fact["event_id"] = float(fact["event_id"])
            kb._append_event(
                conn, chain["reviewer"], "changed_fact", fact,
                run_id=chain["reviewer_run"], created_at=1,
            )
            kb._append_event(
                conn, chain["author"], "changed_fact", fact,
                run_id=chain["author_run"], created_at=1,
            )

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_byte_different_fact_mirror_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'changed_fact'",
            (chain["author"], chain["author_run"]),
        ).fetchone()
        obj = json.loads(row["payload"])
        raw = json.dumps({key: obj[key] for key in reversed(list(obj))})
        assert raw != row["payload"] and json.loads(raw) == obj
        # A key-reordered (semantically equal) author fact mirror must fail closed.
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload = ? WHERE id = ?", (raw, row["id"]))

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None


def test_earlier_v3_audit_outcome_byte_different_verdict_mirror_fails_closed(kanban_home):
    with kb.connect() as conn:
        chain = _earlier_v3_audit_outcome_chain(conn)
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'review_verdict'",
            (chain["author"], chain["reviewer_run"]),
        ).fetchone()
        obj = json.loads(row["payload"])
        raw = json.dumps({key: obj[key] for key in reversed(list(obj))})
        assert raw != row["payload"] and json.loads(raw) == obj
        # A key-reordered (semantically equal) author verdict mirror must fail closed.
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload = ? WHERE id = ?", (raw, row["id"]))

        assert kb._canonical_current_audit_outcome(conn, chain["author"]) == (True, None)
        assert kb._canonical_audit_receipt(conn, chain["author"]) is None
        assert kb._reviewed_author_finalizer_run_id(conn, chain["author"]) is None
