#!/usr/bin/env python3
"""Validate or sequentially run the fixed-20 12K core-trend suite.

The default action is CPU-only. GPU work is possible only with ``--execute``.
Every chunk is checkpointed at case boundaries and an existing, compatible
checkpoint is resumed automatically. A nonblocking lock in the shared output
directory prevents two launchers from driving this suite concurrently.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_restart import (
    canonical_digest,
    load_checkpoint,
    run_config_id,
    validate_checkpoint,
)


SUITE_ID = "pairwise-authoritative-12k-coretrend-fixed20-v3"
BENCHMARK_RUNNER = (
    "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py"
)
LEGACY_BENCHMARK_RUNNER = "scripts/benchmark_pairwise_overlap_cases.py"
CORE_TREND_CASE_KEYS = [
    "A01:C-DENSE", "A01:C-GROUPED", "A02:C-DENSE",
    "A03:C-DENSE", "A03:C-GROUPED", "A03:C-BW",
    "A04:C-DENSE", "A04:C-GROUPED", "A05:C-DENSE",
    "A05:C-GROUPED", "A06", "A07", "A08", "A09", "A12",
    "A14", "A15", "A16", "A18", "A20", "M01", "M02",
    "M03", "M04", "M05b",
]
CHUNK_IDS = [
    "coretrend_fixed20_v3_01_pairwise_compute",
    "coretrend_fixed20_v3_02_pairwise_aux_comm",
    "coretrend_fixed20_v3_03_multi_aggressor",
]
EXPECTED_ARTIFACT_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_dlcompiler_agent/users/"
    "mfrisella/piper/out/20260809_pairwise_overlap_benchmark/"
    "authoritative_12k_coretrend_fixed20_v3"
)
DEFAULT_MANIFEST = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_dlcompiler_agent/users/"
    "mfrisella/piper/out/20260809_pairwise_overlap_benchmark/"
    "AUTHORITATIVE_12K_CORETREND_FIXED20_SUITE_MANIFEST_V3.json"
)
EXPECTED_SUITE_LOCK = EXPECTED_ARTIFACT_ROOT / f".{SUITE_ID}.lock"
REQUIRED_RUNNER_FLAGS = {
    "--iterations", "--baseline-repetitions", "--checkpoint", "--resume",
    "--run-id", "--dry-run",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--chunk", action="append",
        help="Run/validate only this chunk ID; repeat to select several",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="After CPU validation, execute selected chunks sequentially",
    )
    parser.add_argument(
        "--print-commands", action="store_true",
        help="Print exact new-run or resume commands",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def assignment_case_key(
    assignment: dict[str, Any], case_counts: dict[str, int]
) -> str:
    case_id = str(assignment["case_id"])
    if case_counts[case_id] == 1:
        return case_id
    variant = assignment.get("variant")
    if not variant:
        raise ValueError(f"matrix case {case_id} requires a variant")
    return f"{case_id}:{variant}"


def matrix_core_case_keys(matrix: dict[str, Any]) -> list[str]:
    assignments = [
        *matrix.get("pair_assignments", []),
        *matrix.get("multi_assignments", []),
    ]
    assignments = [
        row for row in assignments
        if str(row.get("case_id", "")).startswith(("A", "M"))
    ]
    counts: dict[str, int] = {}
    for row in assignments:
        case_id = str(row["case_id"])
        counts[case_id] = counts.get(case_id, 0) + 1
    return [assignment_case_key(row, counts) for row in assignments]


def selected_chunks(
    manifest: dict[str, Any], requested: Iterable[str] | None
) -> list[dict[str, Any]]:
    chunks = list(manifest["chunks"])
    wanted = set(requested or ())
    known = {str(row["chunk_id"]) for row in chunks}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown chunk IDs: {unknown}; known={sorted(known)}")
    return [row for row in chunks if not wanted or row["chunk_id"] in wanted]


def _require_absolute_artifact(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute shared path")
    resolved = path.resolve(strict=False)
    if resolved.parent != EXPECTED_ARTIFACT_ROOT.resolve(strict=False):
        raise ValueError(
            f"{label} must live directly under {EXPECTED_ARTIFACT_ROOT}"
        )
    return resolved


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("the coretrend fixed20 V3 manifest requires schema version 2")
    if manifest.get("suite_id") != SUITE_ID:
        raise ValueError(
            "the suite launcher accepts only the coretrend fixed20 V3 "
            "authoritative manifest; adaptive V1/V2 manifests are read-only "
            "predecessors"
        )
    if manifest.get("runner") not in {
        BENCHMARK_RUNNER,
        LEGACY_BENCHMARK_RUNNER,
    }:
        raise ValueError("unexpected benchmark runner")
    if manifest.get("python") != "/opt/piper-venv/bin/python":
        raise ValueError("authoritative Python interpreter changed")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "matrix", "a2a_profiles", "trace_events"
    }:
        raise ValueError("manifest immutable-input inventory changed")
    for name, payload in inputs.items():
        path = repo_path(payload["path"])
        if not path.is_file():
            raise ValueError(f"missing suite input {path}")
        observed = sha256(path)
        if observed != payload["sha256"]:
            raise ValueError(
                f"suite input hash mismatch for {name}: "
                f"{observed} != {payload['sha256']}"
            )

    matrix = load_json(repo_path(inputs["matrix"]["path"]))
    available = matrix_core_case_keys(matrix)
    observed_keys = list(manifest.get("core_case_keys", []))
    if observed_keys != CORE_TREND_CASE_KEYS or len(set(observed_keys)) != 25:
        raise ValueError("the ordered 25-case core-trend inventory changed")
    if [key for key in available if key in set(observed_keys)] != observed_keys:
        raise ValueError("core-trend cases are absent or reordered in the matrix")
    controls = [
        str(row["case_key"]).partition(":")[0]
        for row in matrix.get("component_controls", [])
    ]
    if manifest.get("excluded_component_control_keys") != controls or len(controls) != 13:
        raise ValueError("excluded component-control inventory changed")

    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 3:
        raise ValueError("the coretrend fixed20 suite must contain three chunks")
    if [row.get("chunk_id") for row in chunks] != CHUNK_IDS:
        raise ValueError("chunk identities or order changed")
    flattened = [key for row in chunks for key in row["case_keys"]]
    if flattened != CORE_TREND_CASE_KEYS:
        raise ValueError("chunks are not an ordered exact core-case partition")
    if [len(row["case_keys"]) for row in chunks] != [10, 10, 5]:
        raise ValueError("chunk case counts changed")
    for field in ("chunk_id", "run_id", "output", "checkpoint"):
        values = [str(row[field]) for row in chunks]
        if len(values) != len(set(values)):
            raise ValueError(f"chunk {field} values must be unique")
    artifact_paths: list[Path] = []
    for chunk in chunks:
        artifact_paths.extend([
            _require_absolute_artifact(chunk["output"], "chunk output"),
            _require_absolute_artifact(chunk["checkpoint"], "chunk checkpoint"),
        ])
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("chunk output/checkpoint paths overlap")
    expected_run_ids = [
        f"{SUITE_ID}-chunk{index:02d}" for index in range(1, 4)
    ]
    if [row["run_id"] for row in chunks] != expected_run_ids:
        raise ValueError("coretrend fixed20 V3 chunk run IDs changed")
    if Path(str(manifest.get("suite_lock", ""))).resolve(strict=False) != (
        EXPECTED_SUITE_LOCK.resolve(strict=False)
    ):
        raise ValueError("suite lock path changed")

    topology = manifest.get("topology", {})
    if topology != {
        "ray_address": "10.65.8.91",
        "ray_port": 6379,
        "ray_version": "2.44.1",
        "node_ips": ["10.65.8.91", "10.65.10.25"],
        "gpus_per_node": 8,
        "world_size": 16,
        "physical_pp": 2,
        "dp": 8,
        "ep": 8,
        "placement": "pp_outer",
        "placement_contract": (
            "physical PP rank selects the node; PP pairs cross nodes, while "
            "each DP/EP group stays within one node"
        ),
    }:
        raise ValueError("manifest does not describe the requested PP-outer topology")

    experiment = manifest.get("experiment", {})
    if (
        experiment.get("sequence_length") != 12288
        or experiment.get("warmups") != 2
        or experiment.get("coverage_targets_pct") != [0, 5, 25, 50, 75, 100]
        or experiment.get("dispatch_modes") != [
            "victim_first", "aggressor_first"
        ]
        or experiment.get("pair_orientations") != (
            "both participants separately act as victim"
        )
    ):
        raise ValueError("fixed20 experiment matrix changed")
    if experiment.get("fixed_sampling") != {
        "records_per_nominal_stratum": 20,
        "included_dispositions": [
            "confirmed",
            "runtime_start_geometry_unresolved",
            "not_applicable_static",
        ],
        "stopping_rule": (
            "exactly_20_issued_records_per_nominal_stratum_including_"
            "confirmed_unresolved_and_explicit_infeasible_dispositions"
        ),
        "ci_convergence_required": False,
        "confidence_intervals_role": (
            "descriptive_only_not_an_execution_or_acceptance_gate"
        ),
    }:
        raise ValueError("coretrend fixed20 sampling policy is not pinned")
    scheduling = experiment.get("scheduling", {})
    expected_scheduling = {
        "stage_lead_ms": 5.0,
        "baseline_measurement": (
            "per_treatment_primed_focal_adjacent_odd_k_median_v1"
        ),
        "baseline_measurement_repetitions": 3,
        "sample_priming": "one_unmeasured_launch_per_operation",
        "bracket_scope": "adjacent_per_treatment_cell",
        "bracket_operation_order": (
            "pre_nonvictims_then_victim__post_victim_then_nonvictims"
        ),
        "geometry_feedback_candidate_shifts_max": 8,
        "geometry_confirmation_trials_per_candidate": 3,
        "geometry_confirmation_minimum_passing": 2,
        "geometry_confirmation_baseline_measurement": (
            "per_treatment_primed_focal_adjacent_odd_k_median_v1"
        ),
        "geometry_confirmation_baseline_repetitions": 3,
        "geometry_confirmation_joint_gate": (
            "abs_scheduled_geometry_error_le_10_and_victim_abs_matched_"
            "baseline_drift_le_5_and_pre_post_counterfactual_geometry_"
            "errors_le_10_and_abs_candidate_median_error_le_5"
        ),
        "geometry_confirmation_drift_tolerance_pct": 5.0,
        "geometry_candidate_median_centering_tolerance_pct": 5.0,
        "geometry_confirmation_launches_max_per_feasible_stage": 24,
        "geometry_validity_basis": (
            "counterfactual_intervals_from_actual_global_gpu_starts_and_"
            "matched_isolated_pre_post_durations"
        ),
        "geometry_baseline_stability_basis": (
            "maximum_absolute_pre_post_drift_across_all_operations"
        ),
        "geometry_baseline_max_abs_drift_role": "diagnostic_only",
        "baseline_stability_policy": (
            "victim_relative_drift_and_pre_post_geometry_robust_v1"
        ),
        "concurrent_overlap_role": "reported_outcome_not_validity_gate",
        "legacy_realized_overlap_fields_are_concurrent_outcomes": True,
        "geometry_max_relative_shift_ms": 100.0,
        "device_delay_kernels": False,
    }
    if scheduling != expected_scheduling:
        raise ValueError("manifest scheduling methodology is not pinned")

    expected_chunk_cells = [240, 240, 60]
    expected_chunk_iterations = [4800, 4800, 1200]
    if [row.get("nominal_cells") for row in chunks] != expected_chunk_cells:
        raise ValueError("chunk nominal-cell counts changed")
    if [row.get("nominal_issued_records") for row in chunks] != (
        expected_chunk_iterations
    ):
        raise ValueError("chunk fixed20 iteration counts do not reconcile")
    if manifest.get("totals") != {
        "core_cases": 25,
        "pairwise_cases": 20,
        "multi_aggressor_cases": 5,
        "coverage_targets": 6,
        "nominal_cells": 540,
        "issued_records_per_nominal_stratum": 20,
        "nominal_issued_records": 10800,
        "maximum_start_anchor_unscored_stages_per_scored_record": 6,
        "maximum_nominal_start_anchor_unscored_stages": 64800,
    }:
        raise ValueError("core-trend fixed20 total accounting changed")
    return manifest


def runner_command(
    manifest: dict[str, Any], chunk: dict[str, Any], *,
    dry_run: bool, resume: bool,
) -> list[str]:
    topology = manifest["topology"]
    experiment = manifest["experiment"]
    scheduling = experiment["scheduling"]
    inputs = manifest["inputs"]
    command = [
        manifest["python"], BENCHMARK_RUNNER,
        "--address", str(topology["ray_address"]),
        "--port", str(topology["ray_port"]),
        "--node-ips", ",".join(topology["node_ips"]),
        "--matrix", inputs["matrix"]["path"],
        "--a2a-profiles", inputs["a2a_profiles"]["path"],
        "--trace-events", inputs["trace_events"]["path"],
        "--cases", ",".join(chunk["case_keys"]),
        "--seq-len", str(experiment["sequence_length"]),
        "--warmups", str(experiment["warmups"]),
        "--coverage-targets", ",".join(
            str(value) for value in experiment["coverage_targets_pct"]
        ),
        "--iterations", str(
            experiment["fixed_sampling"]
            ["records_per_nominal_stratum"]
        ),
        "--stage-lead-ms", str(scheduling["stage_lead_ms"]),
        "--baseline-repetitions", str(
            scheduling["baseline_measurement_repetitions"]
        ),
        "--geometry-calibration-trials", str(
            scheduling["geometry_feedback_candidate_shifts_max"]
        ),
        "--geometry-max-relative-shift-ms", str(
            scheduling["geometry_max_relative_shift_ms"]
        ),
        "--output", chunk["output"],
        "--checkpoint", chunk["checkpoint"],
        "--run-id", chunk["run_id"],
    ]
    if resume:
        command.append("--resume")
    if dry_run:
        command.append("--dry-run")
    return command


def check_runner_interface(manifest: dict[str, Any]) -> None:
    result = subprocess.run(
        [manifest["python"], BENCHMARK_RUNNER, "--help"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    )
    missing = sorted(flag for flag in REQUIRED_RUNNER_FLAGS if flag not in result.stdout)
    if missing:
        raise ValueError(f"runner is missing required flags: {missing}")


def validate_preview(
    manifest: dict[str, Any], chunk: dict[str, Any], preview: dict[str, Any]
) -> None:
    if [row["case_key"] for row in preview["cases"]] != chunk["case_keys"]:
        raise ValueError(f"{chunk['chunk_id']} dry-run case mismatch")
    if preview.get("sequence_length") != 12288:
        raise ValueError("dry-run sequence length is not 12,288")
    if preview.get("coverage_targets_pct") != [0.0, 5.0, 25.0, 50.0, 75.0, 100.0]:
        raise ValueError("dry-run coverage targets changed")
    if preview.get("sampling") != {
        "mode": "fixed", "fixed_iterations": 20, "adaptive": None
    }:
        raise ValueError("dry-run fixed sampling parameters changed")
    topology = preview.get("topology", {})
    if (
        topology.get("node_ips") != manifest["topology"]["node_ips"]
        or topology.get("world_size") != 16
    ):
        raise ValueError("dry-run topology changed")
    expected = manifest["experiment"]["scheduling"]
    scheduling = preview.get("scheduling", {})
    for field in (
        "baseline_measurement", "baseline_measurement_repetitions",
        "sample_priming", "bracket_scope", "bracket_operation_order",
        "stage_lead_ms", "device_delay_kernels",
    ):
        if scheduling.get(field) != expected[field]:
            raise ValueError(f"dry-run scheduling field changed: {field}")
    geometry = scheduling.get("geometry_feedback", {})
    geometry_fields = {
        "candidate_shifts_max": "geometry_feedback_candidate_shifts_max",
        "confirmation_trials_per_candidate": (
            "geometry_confirmation_trials_per_candidate"
        ),
        "confirmation_minimum_passing": "geometry_confirmation_minimum_passing",
        "confirmation_baseline_measurement": (
            "geometry_confirmation_baseline_measurement"
        ),
        "confirmation_baseline_repetitions": (
            "geometry_confirmation_baseline_repetitions"
        ),
        "confirmation_joint_gate": "geometry_confirmation_joint_gate",
        "confirmation_drift_tolerance_pct": (
            "geometry_confirmation_drift_tolerance_pct"
        ),
        "candidate_median_centering_tolerance_pct": (
            "geometry_candidate_median_centering_tolerance_pct"
        ),
        "confirmation_launches_max_per_feasible_stage": (
            "geometry_confirmation_launches_max_per_feasible_stage"
        ),
        "validity_basis": "geometry_validity_basis",
        "geometry_baseline_stability_basis": (
            "geometry_baseline_stability_basis"
        ),
        "concurrent_overlap_role": "concurrent_overlap_role",
        "legacy_realized_overlap_fields_are_concurrent_outcomes": (
            "legacy_realized_overlap_fields_are_concurrent_outcomes"
        ),
        "max_relative_shift_ms": "geometry_max_relative_shift_ms",
    }
    for preview_field, manifest_field in geometry_fields.items():
        if geometry.get(preview_field) != expected[manifest_field]:
            raise ValueError(
                f"dry-run geometry methodology changed: {preview_field}"
            )
    if geometry.get("tolerance_pct") != 10.0:
        raise ValueError("dry-run geometry tolerance changed")
    audit = preview.get("nccl_environment_audit", {})
    if (
        audit.get("uses_nccl_library_behavior_defaults") is not True
        or audit.get("effective_overrides")
        or audit.get("launch_order_implicit_enabled") is not False
    ):
        raise ValueError("dry-run did not preserve NCCL behavior defaults")
    restart = preview.get("restart", {})
    if restart.get("recovery_granularity") != "case_boundary":
        raise ValueError("runner checkpoint is not case-boundary recoverable")
    if Path(restart.get("checkpoint_path", "")).resolve(strict=False) != (
        Path(chunk["checkpoint"]).resolve(strict=False)
    ):
        raise ValueError("dry-run checkpoint path changed")


def dry_run_chunk(
    manifest: dict[str, Any], chunk: dict[str, Any]
) -> dict[str, Any]:
    result = subprocess.run(
        runner_command(manifest, chunk, dry_run=True, resume=False),
        cwd=REPO_ROOT, env=os.environ.copy(), text=True,
        capture_output=True, check=True,
    )
    preview = json.loads(result.stdout)
    validate_preview(manifest, chunk, preview)
    return preview


def validated_existing_checkpoint(
    chunk: dict[str, Any], preview: dict[str, Any]
) -> dict[str, Any] | None:
    path = Path(chunk["checkpoint"])
    if not path.exists():
        return None
    expected_config_id = str(preview["restart"]["config_id"])
    state = load_checkpoint(path)
    identity = state.get("configuration_identity")
    if not isinstance(identity, dict) or run_config_id(identity) != expected_config_id:
        raise ValueError(f"checkpoint {path} configuration hash changed")
    validated = validate_checkpoint(
        state, expected_config_id=expected_config_id,
        expected_configuration_identity=identity,
        expected_case_keys=chunk["case_keys"],
    )
    if validated.get("run_id") != chunk["run_id"]:
        raise ValueError(f"checkpoint {path} run ID changed")
    return validated


def completed_output(
    chunk: dict[str, Any], expected_config_id: str,
    checkpoint_state: dict[str, Any] | None = None,
) -> bool:
    path = Path(chunk["output"])
    if not path.exists():
        return False
    try:
        document = load_json(path)
        run_id = str(chunk["run_id"])
        if (
            document.get("run_id") != run_id
            or document.get("source_run_id") != run_id
            or document.get("config_id") != expected_config_id
        ):
            raise ValueError("top-level output identity mismatch")
        provenance = document.get("restart_provenance", {})
        expected_count = len(chunk["case_keys"])
        if (
            provenance.get("run_id") != run_id
            or provenance.get("config_id") != expected_config_id
            or provenance.get("all_requested_cases_reconciled") is not True
            or provenance.get("recovery_granularity") != "case_boundary"
            or provenance.get("requested_case_count") != expected_count
            or provenance.get("completed_case_count") != expected_count
        ):
            raise ValueError("output restart provenance is incomplete")
        results = document.get("results")
        if not isinstance(results, list) or [
            row["assignment"]["case_key"] for row in results
        ] != chunk["case_keys"]:
            raise ValueError("output result cases are incomplete or reordered")
        for result in results:
            execution = result.get("execution_provenance", {})
            issued = list(execution.get("issued_trial_ids", []))
            completed = list(execution.get("completed_trial_ids", []))
            samples = result.get("samples")
            observed = [sample.get("trial_id") for sample in samples or []]
            if (
                result.get("run_id") != run_id
                or result.get("source_run_id") != run_id
                or execution.get("config_id") != expected_config_id
                or execution.get("reconciled") is not True
                or issued != completed or completed != observed
                or len(completed) != len(set(completed))
            ):
                raise ValueError("output trial reconciliation failed")
        environment = document.get("environment", {})
        if environment.get("all_workers_default_nccl_behavior") is not True:
            raise ValueError("output does not prove default NCCL behavior")
        if checkpoint_state is not None:
            completed_cases = checkpoint_state.get("completed_cases", {})
            if list(completed_cases) != chunk["case_keys"]:
                raise ValueError("checkpoint case order is incomplete")
            if provenance.get("checkpoint_generation") != checkpoint_state.get(
                "checkpoint_generation"
            ):
                raise ValueError("checkpoint generation mismatch")
            for result in results:
                record = completed_cases[result["assignment"]["case_key"]]
                if (
                    record.get("result_digest") != canonical_digest(result)
                    or record.get("result") != result
                ):
                    raise ValueError("checkpoint/output result mismatch")
        return True
    except (KeyError, TypeError, ValueError) as exc:
        if checkpoint_state is not None:
            print(
                f"existing output {path} is reconstructible from checkpoint: {exc}",
                file=sys.stderr,
            )
            return False
        raise ValueError(f"existing output {path} is invalid: {exc}") from exc


@contextmanager
def suite_launch_lock(manifest: dict[str, Any]):
    """Atomically provision and exclusively lock the shared suite inode."""
    path = Path(manifest["suite_lock"])
    if path.resolve(strict=False) != EXPECTED_SUITE_LOCK.resolve(strict=False):
        raise ValueError("suite lock escaped the fixed artifact root")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A pre-existing lock needs only read permission for flock(2). O_CREAT is
    # retained so the first launcher atomically provisions the inode when its
    # shared output directory is writable.
    flags = os.O_RDONLY | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o644)
    stream = os.fdopen(fd, "r", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another authoritative suite launcher holds {path}"
            ) from exc
        yield path
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def shell_render(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def main() -> None:
    args = parse_args()
    manifest = validate_manifest(repo_path(args.manifest))
    chunks = selected_chunks(manifest, args.chunk)
    check_runner_interface(manifest)
    with suite_launch_lock(manifest) as lock_path:
        previews: dict[str, dict[str, Any]] = {}
        checkpoint_states: dict[str, dict[str, Any] | None] = {}
        for chunk in chunks:
            preview = dry_run_chunk(manifest, chunk)
            previews[chunk["chunk_id"]] = preview
            checkpoint_states[chunk["chunk_id"]] = validated_existing_checkpoint(
                chunk, preview
            )
            print(
                f"validated {chunk['chunk_id']}: {len(chunk['case_keys'])} cases, "
                f"{chunk['nominal_cells']} nominal cells"
            )
        if args.print_commands:
            for chunk in chunks:
                resume = checkpoint_states[chunk["chunk_id"]] is not None
                print(
                    f"{chunk['chunk_id']}: "
                    + shell_render(runner_command(
                        manifest, chunk, dry_run=False, resume=resume
                    ))
                )
        if not args.execute:
            print(
                "CPU validation passed: "
                f"{sum(len(row['case_keys']) for row in chunks)} cases; "
                f"single-launch lock {lock_path}; no GPU work was requested"
            )
            return
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            preview = previews[chunk_id]
            expected_config_id = str(preview["restart"]["config_id"])
            checkpoint_state = validated_existing_checkpoint(chunk, preview)
            if completed_output(chunk, expected_config_id, checkpoint_state):
                print(f"skipping completed {chunk_id}")
                continue
            resume = checkpoint_state is not None
            command = runner_command(
                manifest, chunk, dry_run=False, resume=resume
            )
            print(
                f"{'resuming' if resume else 'starting'} {chunk_id}: "
                f"{shell_render(command)}", flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), check=True)
            final_checkpoint = validated_existing_checkpoint(chunk, preview)
            if final_checkpoint is None or not completed_output(
                chunk, expected_config_id, final_checkpoint
            ):
                raise RuntimeError(f"runner returned without completing {chunk_id}")
        print("authoritative core-trend fixed20 12K suite completed")


if __name__ == "__main__":
    main()
