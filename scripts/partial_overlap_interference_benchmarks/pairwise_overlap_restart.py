# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Case-boundary restart and idempotence support for overlap benchmarks.

GPU actor state is deliberately *not* checkpointed.  A case is the smallest
recoverable unit: its checkpoint is published only after every rank result has
been gathered, aggregated, and reconciled.  An interrupted case is rerun in
full on resume, while completed cases are skipped byte-for-byte.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid


CHECKPOINT_SCHEMA_VERSION = 1


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    """Hash an input file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_config_id(configuration: Mapping[str, Any]) -> str:
    """Identify all inputs that may change the measurements or trial plan."""

    return f"pairwise-config-{canonical_digest(configuration)}"


def new_run_id(config_id: str) -> str:
    """Create a unique run identity which remains stable through resume."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pairwise-{config_id.removeprefix('pairwise-config-')[:12]}-{timestamp}-{uuid.uuid4().hex[:12]}"


def case_execution_id(config_id: str, case_key: str) -> str:
    """Return the deterministic execution identity of one configured case."""

    digest = canonical_digest({"config_id": config_id, "case_key": case_key})
    return f"pairwise-case-{digest[:24]}"


def trial_id(case_id: str, sample: Mapping[str, Any]) -> str:
    """Return a deterministic treatment-attempt identity within a case."""

    identity = {
        "case_execution_id": case_id,
        "seq_len": int(sample["seq_len"]),
        "stage_mode": str(sample["stage_mode"]),
        "dispatch_mode": str(sample["dispatch_mode"]),
        "victim_index": int(sample["victim_index"]),
        "sample_index": int(sample["sample_index"]),
    }
    return f"trial-{canonical_digest(identity)[:32]}"


def annotate_and_reconcile_case(
    result: Mapping[str, Any],
    *,
    run_id: str,
    config_id: str,
) -> dict[str, Any]:
    """Attach stable trial IDs and prove case-boundary attempt reconciliation."""

    annotated = deepcopy(dict(result))
    assignment = annotated.get("assignment")
    if not isinstance(assignment, Mapping) or not assignment.get("case_key"):
        raise ValueError("case result is missing assignment.case_key")
    case_key = str(assignment["case_key"])
    execution_id = case_execution_id(config_id, case_key)
    samples = annotated.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"case {case_key} is missing a sample list")

    completed_ids: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"case {case_key} contains a malformed sample")
        identity = trial_id(execution_id, sample)
        existing = sample.get("trial_id")
        if existing not in (None, identity):
            raise ValueError(
                f"case {case_key} sample has conflicting trial_id {existing!r}"
            )
        sample["trial_id"] = identity
        completed_ids.append(identity)
    duplicate_ids = sorted(
        identity
        for identity, count in Counter(completed_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"case {case_key} produced duplicate trial IDs: {duplicate_ids}"
        )

    # All issued work must complete before benchmark() returns.  Adaptive mode
    # records that invariant explicitly; fixed mode has the same case-boundary
    # contract because ray.get() gathers every actor before aggregation.
    adaptive = annotated.get("adaptive_sampling")
    if adaptive is not None:
        reconciliation = adaptive.get("attempt_reconciliation", {})
        if reconciliation.get("passed") is not True:
            raise ValueError(
                f"case {case_key} adaptive attempts are not reconciled"
            )
        issued_count = int(reconciliation.get("issued", -1))
        completed_count = int(reconciliation.get("completed", -1))
        if issued_count != completed_count or completed_count != len(completed_ids):
            raise ValueError(
                f"case {case_key} adaptive count mismatch: issued={issued_count}, "
                f"completed={completed_count}, samples={len(completed_ids)}"
            )
    else:
        issued_count = completed_count = len(completed_ids)

    # At a committed case boundary, issued and completed trial identities are
    # the same set.  Keeping both makes interrupted/replayed merges auditable.
    annotated["run_id"] = run_id
    annotated["source_run_id"] = run_id
    annotated["execution_provenance"] = {
        "config_id": config_id,
        "case_execution_id": execution_id,
        "recovery_granularity": "case_boundary",
        "mid_case_actor_state_recoverable": False,
        "issued_trial_count": issued_count,
        "completed_trial_count": completed_count,
        "issued_trial_ids": completed_ids,
        "completed_trial_ids": completed_ids,
        "unique_trial_count": len(set(completed_ids)),
        "reconciled": issued_count == completed_count == len(set(completed_ids)),
    }
    if not annotated["execution_provenance"]["reconciled"]:
        raise ValueError(f"case {case_key} trial reconciliation failed")
    return annotated


def new_checkpoint(
    *,
    config_id: str,
    configuration_identity: Mapping[str, Any],
    requested_case_keys: Sequence[str],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create empty persisted state for a new invocation."""

    case_keys = [str(item) for item in requested_case_keys]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("requested case keys must be unique")
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id or new_run_id(config_id),
        "config_id": config_id,
        "configuration_identity": deepcopy(dict(configuration_identity)),
        "requested_case_keys": case_keys,
        "completed_cases": {},
        "execution_sessions": [],
        "checkpoint_generation": 0,
        "state": "running",
        "recovery_granularity": "case_boundary",
        "mid_case_actor_state_recoverable": False,
    }


def _validate_completed_case(
    case_key: str,
    record: Mapping[str, Any],
    *,
    run_id: str,
    config_id: str,
) -> None:
    result = record.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"checkpoint case {case_key} has no result")
    assignment = result.get("assignment")
    if not isinstance(assignment, Mapping) or assignment.get("case_key") != case_key:
        raise ValueError(f"checkpoint case key mismatch for {case_key}")
    expected_execution_id = case_execution_id(config_id, case_key)
    provenance = result.get("execution_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"checkpoint case {case_key} lacks execution provenance")
    if provenance.get("case_execution_id") != expected_execution_id:
        raise ValueError(f"checkpoint case execution ID mismatch for {case_key}")
    if result.get("run_id") != run_id or result.get("source_run_id") != run_id:
        raise ValueError(f"checkpoint run ID mismatch for case {case_key}")
    completed = list(provenance.get("completed_trial_ids", []))
    issued = list(provenance.get("issued_trial_ids", []))
    samples = result.get("samples", [])
    observed = [sample.get("trial_id") for sample in samples]
    recomputed = [trial_id(expected_execution_id, sample) for sample in samples]
    if (
        issued != completed
        or completed != observed
        or observed != recomputed
        or len(completed) != len(set(completed))
        or provenance.get("reconciled") is not True
    ):
        raise ValueError(f"checkpoint trial reconciliation failed for {case_key}")
    expected_digest = canonical_digest(result)
    if record.get("result_digest") != expected_digest:
        raise ValueError(f"checkpoint result digest mismatch for {case_key}")


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_config_id: str,
    expected_configuration_identity: Mapping[str, Any],
    expected_case_keys: Sequence[str],
) -> dict[str, Any]:
    """Validate all resume state, failing closed on any mismatch."""

    state = deepcopy(dict(checkpoint))
    if state.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported pairwise checkpoint schema")
    if state.get("config_id") != expected_config_id:
        raise ValueError(
            "checkpoint config/hash mismatch: "
            f"found {state.get('config_id')}, expected {expected_config_id}"
        )
    if canonical_digest(state.get("configuration_identity")) != canonical_digest(
        expected_configuration_identity
    ):
        raise ValueError("checkpoint configuration payload mismatch")
    expected = [str(item) for item in expected_case_keys]
    if state.get("requested_case_keys") != expected:
        raise ValueError("checkpoint requested case order mismatch")
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("checkpoint is missing run_id")
    completed = state.get("completed_cases")
    if not isinstance(completed, dict):
        raise ValueError("checkpoint completed_cases must be an object")
    unknown = sorted(set(completed) - set(expected))
    if unknown:
        raise ValueError(f"checkpoint contains unrequested cases: {unknown}")
    for case_key, record in completed.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"checkpoint case {case_key} is malformed")
        _validate_completed_case(
            case_key, record, run_id=run_id, config_id=expected_config_id
        )
    return state


def commit_case(
    checkpoint: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Return checkpoint state with one reconciled case committed once."""

    state = deepcopy(dict(checkpoint))
    case_key = str(result["assignment"]["case_key"])
    if case_key not in state["requested_case_keys"]:
        raise ValueError(f"cannot commit unrequested case {case_key}")
    digest = canonical_digest(result)
    existing = state["completed_cases"].get(case_key)
    if existing is not None:
        if existing.get("result_digest") != digest:
            raise ValueError(f"non-idempotent replay for completed case {case_key}")
        return state
    _validate_completed_case(
        case_key,
        {"result": result, "result_digest": digest},
        run_id=str(state["run_id"]),
        config_id=str(state["config_id"]),
    )
    state["completed_cases"][case_key] = {
        "result_digest": digest,
        "result": deepcopy(dict(result)),
    }
    state["checkpoint_generation"] = int(state["checkpoint_generation"]) + 1
    state["state"] = (
        "complete"
        if len(state["completed_cases"]) == len(state["requested_case_keys"])
        else "running"
    )
    return state


def ordered_results(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return completed case results in the original request order."""

    completed = checkpoint["completed_cases"]
    return [
        deepcopy(completed[key]["result"])
        for key in checkpoint["requested_case_keys"]
        if key in completed
    ]


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Durably publish one complete JSON document with atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint document, rejecting incomplete/non-object JSON."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {path} is not a JSON object")
    return value


def checkpoint_audit(checkpoint: Mapping[str, Any], *, resumed: bool) -> dict[str, Any]:
    """Summarize restart provenance for the final raw result document."""

    requested = list(checkpoint["requested_case_keys"])
    completed = list(checkpoint["completed_cases"])
    trial_count = sum(
        int(record["result"]["execution_provenance"]["completed_trial_count"])
        for record in checkpoint["completed_cases"].values()
    )
    return {
        "run_id": checkpoint["run_id"],
        "config_id": checkpoint["config_id"],
        "resumed": bool(resumed),
        "recovery_granularity": "case_boundary",
        "mid_case_actor_state_recoverable": False,
        "requested_case_count": len(requested),
        "completed_case_count": len(completed),
        "completed_trial_count": trial_count,
        "checkpoint_generation": int(checkpoint["checkpoint_generation"]),
        "all_requested_cases_reconciled": set(requested) == set(completed),
    }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "annotate_and_reconcile_case",
    "atomic_write_json",
    "canonical_digest",
    "case_execution_id",
    "checkpoint_audit",
    "commit_case",
    "file_digest",
    "load_checkpoint",
    "new_checkpoint",
    "new_run_id",
    "ordered_results",
    "run_config_id",
    "trial_id",
    "validate_checkpoint",
]
