#!/usr/bin/env python3
"""Run trace-exact pairwise overlap cases on the PP2 x DP/EP8 topology.

The runner consumes the exact kernel execution matrix and extracted ragged-A2A
split matrices.  Every distributed operation launches all required endpoints,
uses a distinct trace-role CUDA stream, and retains per-rank completion data.
It is intentionally separate from the report generator: its output is the raw
timing/correctness corpus used to build the suite CSV tables and figures.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_benchmark import (
    MODEL_CONFIG,
    MODEL_NAME,
    PPOuterTopology,
    assert_default_nccl_environment,
    nccl_behavior_environment,
    nccl_debug_environment,
    sanitize_nccl_environment,
    validate_observed_placement,
)
from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_adaptive_controller import (
    AdaptiveBenchmarkCellKey,
    AdaptiveBenchmarkController,
    AdaptiveBenchmarkControllerConfig,
    classify_aggregate_trial,
)
from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_sampling import AdaptiveSamplingConfig
from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_restart import (
    annotate_and_reconcile_case,
    atomic_write_json,
    checkpoint_audit,
    commit_case,
    file_digest,
    load_checkpoint,
    new_checkpoint,
    ordered_results,
    run_config_id,
    validate_checkpoint,
)
from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_trace_kernels import (
    make_bf16_add,
    make_swiglu_backward,
)


BASE_SEQ = 12_288
KERNEL_TO_COMPUTE = {
    "K-ff832373130e": "vocab_projection_gemm",
    "K-28e33c0c27e1": "expert_grouped_gemm",
    "K-5f81a28e1cf0": "expert_activation_pointwise",
    "K-49abe359818e": "expert_grouped_gemm_forward",
    "K-9a11d32b92b3": "routing_permutation",
    "K-756e36a388ef": "sdpa_backward",
    "K-7e2ed2672834": "expert_grouped_gemm_backward_input",
    "K-320523f1b915": "elementwise_add",
    "K-c3ef3bbe1a3e": "grouped_gemm_prepare",
}
COMM_CATEGORIES = {"a2a", "ag", "rs", "p2p_send", "p2p_recv"}
GEOMETRY_TOLERANCE_PCT = 10.0
GEOMETRY_CENTERING_TOLERANCE_PCT = 5.0
BASELINE_DRIFT_TOLERANCE_PCT = 5.0
DEFAULT_GEOMETRY_CALIBRATION_TRIALS = 3
DEFAULT_GEOMETRY_MAX_RELATIVE_SHIFT_MS = 100.0
GEOMETRY_CONFIRMATION_TRIALS = 3
GEOMETRY_CONFIRMATION_MINIMUM_PASSING = 2
BASELINE_MEASUREMENT_METHOD = (
    "per_treatment_primed_focal_adjacent_odd_k_median_v1"
)
BASELINE_MEASUREMENT_REPETITIONS = 3
OUTPUT_SCHEMA_VERSION = 2
CLOCK_CALIBRATION_SCHEMA_VERSION = 2
CLOCK_REFRESH_TRIGGER_AGE_NS = 45_000_000_000
CLOCK_REFRESH_HEADROOM_NS = 15_000_000_000
CLOCK_REFRESH_MAX_AGE_NS = (
    CLOCK_REFRESH_TRIGGER_AGE_NS + CLOCK_REFRESH_HEADROOM_NS
)
CLOCK_CALIBRATION_PROBES = 32
CLOCK_REFRESH_BOUNDARIES = (
    "benchmark_start",
    "geometry_confirmation",
    "fixed_treatment",
    "adaptive_treatment",
)
START_ANCHOR_REPETITIONS = 3
START_ANCHOR_MINIMUM_PASSING = 2
START_ANCHOR_POLICY = (
    "same_dispatch_k3_preliminary_residual_plus_k3_corrected_confirmation_v2"
)
START_ANCHOR_CONFIRMATION_POLICY = (
    "corrected_delay_common_focal_geometry_k3_median_two_of_three_v1"
)
RUNTIME_START_GEOMETRY_CONFIRMED = "confirmed"
RUNTIME_START_GEOMETRY_UNRESOLVED = "runtime_start_geometry_unresolved"
RUNTIME_START_GEOMETRY_NOT_APPLICABLE_STATIC = "not_applicable_static"
GEOMETRY_VALIDITY_BASIS = (
    "counterfactual_intervals_from_actual_global_gpu_starts_and_"
    "matched_isolated_pre_post_durations"
)
CONCURRENT_OVERLAP_ROLE = "reported_outcome_not_validity_gate"


def implementation_paths() -> list[Path]:
    """Return every source file bound into the driver/worker provenance."""

    return [
        Path(__file__).resolve(),
        _PACKAGE_ROOT / "pairwise_overlap_adaptive_controller.py",
        _PACKAGE_ROOT / "pairwise_overlap_sampling.py",
        _PACKAGE_ROOT / "pairwise_overlap_benchmark.py",
        _PACKAGE_ROOT / "pairwise_overlap_trace_kernels.py",
        _PACKAGE_ROOT / "pairwise_overlap_restart.py",
        _PACKAGE_ROOT / "runtime_health.py",
    ]


def implementation_sha256() -> dict[str, str]:
    """Hash the exact imported implementation using repo-relative names."""

    return {
        str(path.relative_to(_REPO_ROOT)): file_digest(path)
        for path in implementation_paths()
    }


def validate_worker_implementation_provenance(
    placements: list[dict[str, Any]],
    expected: dict[str, str],
) -> dict[str, Any]:
    """Fail closed unless every worker imported the exact driver sources."""

    if not placements:
        raise RuntimeError("worker implementation provenance is empty")
    observed = [
        row.get("imported_worker_implementation_sha256")
        for row in placements
    ]
    if any(value != expected for value in observed):
        raise RuntimeError(
            "driver/worker implementation hashes differ across Ray actors"
        )
    return {
        "worker_implementation_sha256": dict(expected),
        "worker_implementation_source_match": True,
    }


def structured_clock_calibration_history(
    records: list[dict[str, Any]],
    *,
    world_size: int,
    session_index: int,
) -> list[dict[str, Any]]:
    """Validate and group flat v13 clock records by calibration generation."""

    by_generation: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        generation = int(record["clock_calibration_generation"])
        by_generation.setdefault(generation, []).append(dict(record))
    history = []
    for generation, generation_rows in sorted(by_generation.items()):
        rows = sorted(generation_rows, key=lambda row: int(row["global_rank"]))
        ranks = [int(row["global_rank"]) for row in rows]
        if ranks != list(range(world_size)):
            raise RuntimeError(
                f"clock generation {generation} has ranks {ranks}; "
                f"expected 0..{world_size - 1}"
            )
        labels = {str(row["clock_calibration_label"]) for row in rows}
        if len(labels) != 1:
            raise RuntimeError(
                f"clock generation {generation} has labels {sorted(labels)}"
            )
        history.append({
            "schema_version": CLOCK_CALIBRATION_SCHEMA_VERSION,
            "generation": generation,
            "session_index": int(session_index),
            "label": next(iter(labels)),
            "rows": rows,
            "rows_sha256": canonical_sha256(rows),
        })
    return history


def common_clock_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed unless retained rows share one calibrated generation."""

    if not rows:
        raise ValueError("clock provenance requires retained rows")
    generations = {int(row["clock_calibration_generation"]) for row in rows}
    labels = {str(row["clock_calibration_label"]) for row in rows}
    if len(generations) != 1 or len(labels) != 1:
        raise RuntimeError(
            "compound measurement crossed clock calibration generations: "
            f"generations={sorted(generations)}, labels={sorted(labels)}"
        )
    return {
        "clock_calibration_generation": next(iter(generations)),
        "clock_calibration_label": next(iter(labels)),
        "clock_calibration_max_half_rtt_uncertainty_ns": max(
            int(row["clock_calibration_half_rtt_uncertainty_ns"])
            for row in rows
        ),
    }


def clock_refresh_required(age_ns: int, *, force: bool = False) -> bool:
    """Return whether a safe boundary must advance the clock generation."""

    if age_ns < 0:
        raise ValueError("clock calibration age must be nonnegative")
    return bool(force or age_ns >= CLOCK_REFRESH_TRIGGER_AGE_NS)


def gpu_timebase_metadata(
    *,
    anchor_before_local_ns: int,
    anchor_after_local_ns: int,
    clock_offset_ns: int,
    calibration_local_monotonic_ns: int,
    gate_elapsed_ms: float,
) -> dict[str, Any]:
    """Describe the host-bracketed CUDA-event mapping used by timed rows."""

    if anchor_after_local_ns < anchor_before_local_ns:
        raise ValueError("GPU timebase host bracket is reversed")
    if calibration_local_monotonic_ns > anchor_after_local_ns:
        raise ValueError("GPU timebase predates its clock calibration")
    age_ns = anchor_after_local_ns - calibration_local_monotonic_ns
    if age_ns > CLOCK_REFRESH_MAX_AGE_NS:
        raise RuntimeError(
            "GPU timebase exceeded the advertised clock calibration age bound: "
            f"age_ns={age_ns}, max_age_ns={CLOCK_REFRESH_MAX_AGE_NS}"
        )
    anchor_global_ns = (
        (anchor_before_local_ns + anchor_after_local_ns) // 2
        + clock_offset_ns
    )
    return {
        "gpu_timebase_host_bracket_width_ns": int(
            anchor_after_local_ns - anchor_before_local_ns
        ),
        "gpu_timebase_host_bracket_before_global_ns": int(
            anchor_before_local_ns + clock_offset_ns
        ),
        "gpu_timebase_host_bracket_after_global_ns": int(
            anchor_after_local_ns + clock_offset_ns
        ),
        "gpu_timebase_gate_global_ns": int(
            anchor_global_ns + gate_elapsed_ms * 1e6
        ),
        "gpu_timebase_mapping_method": (
            "synchronized_gate_host_bracket_midpoint_plus_cuda_elapsed_time"
        ),
        "clock_calibration_age_ns": int(age_ns),
    }


def strict_scheduling_provenance() -> dict[str, Any]:
    """Return truthful schema-v2 clock and dynamic-delay declarations."""

    return {
        "ray_worker_python_source": "shared_filesystem_exact_source_hash_bound",
        "gpu_global_time_mapping": (
            "synchronized_gate_host_bracket_midpoint_plus_cuda_elapsed_time"
        ),
        "gpu_global_time_uncertainty": (
            "clock_calibration_half_rtt_plus_synchronized_gate_host_"
            "bracket_half_width"
        ),
        "gpu_global_time_refresh": (
            "gloo_all_reduce_any_actor_age_ge_45s_with_15s_headroom_at_"
            "treatment_safe_deterministic_boundaries"
        ),
        "gpu_global_time_refresh_trigger_age_ns": (
            CLOCK_REFRESH_TRIGGER_AGE_NS
        ),
        "gpu_global_time_refresh_headroom_ns": CLOCK_REFRESH_HEADROOM_NS,
        "gpu_global_time_refresh_max_age_ns": CLOCK_REFRESH_MAX_AGE_NS,
        "gpu_global_time_refresh_boundaries": list(CLOCK_REFRESH_BOUNDARIES),
        "gpu_global_time_refresh_probe_order": (
            "concurrent_rank_probes_global_clock_max_concurrency_32"
        ),
        "gpu_global_time_refresh_same_generation_scope": (
            "adjacent_pre_first_decision_preliminary_anchor_confirmation_"
            "anchor_second_decision_treatment_and_post"
        ),
        "primary_treatment_delay_policy_schema_version": 2,
        "primary_treatment_delay_policy": (
            "adjacent_pre_runtime_start_rebase_v2"
        ),
        "primary_treatment_delay_control_exchange": (
            "cpu_gloo_default_process_group_no_cuda_work"
        ),
        "primary_treatment_delay_dynamic_input": (
            "adjacent_isolated_pre_durations_plus_unscored_same_dispatch_"
            "k3_common_focal_gpu_start_residual"
        ),
        "primary_treatment_delay_dynamic_scope": (
            "every_duration_feasible_treatment_cell"
        ),
        "primary_treatment_delay_excludes_inputs": [
            "current_adjacent_isolated_post",
            "current_treatment_gpu_timings",
            "concurrent_outcomes",
            "controller_validity_or_slowdown",
        ],
        "primary_treatment_delay_frozen_anchor": (
            "corrected_delay_k3_confirmation_before_treatment"
        ),
        "same_dispatch_start_anchor_dynamic_scope": (
            "every_duration_feasible_treatment_cell"
        ),
        "same_dispatch_start_anchor_correction_inputs": (
            "three_unscored_preliminary_same_dispatch_common_focal_gpu_"
            "start_gaps_plus_configured_gap_and_adjacent_pre_durations"
        ),
        "same_dispatch_start_anchor_acceptance_inputs": (
            "three_corrected_delay_common_focal_counterfactual_geometry_"
            "measurements_plus_adjacent_pre_isolated_durations"
        ),
    }


def geometry_calibration_mode(treatment_mode: str) -> str:
    return f"geometry_calibration::{treatment_mode}"


def geometry_confirmation_sample_index(
    spec_index: int,
    candidate_index: int,
    confirmation_index: int,
    candidate_count: int,
) -> int:
    """Return a unique negative sample ID for one confirmation launch."""

    if spec_index < 0 or candidate_index < 0 or confirmation_index < 0:
        raise ValueError("geometry confirmation indices must be nonnegative")
    if candidate_count <= 0 or candidate_index >= candidate_count:
        raise ValueError("geometry candidate index is outside its configured count")
    if confirmation_index >= GEOMETRY_CONFIRMATION_TRIALS:
        raise ValueError("geometry confirmation index is outside its trial count")
    return -3000 - (
        (spec_index * candidate_count + candidate_index)
        * GEOMETRY_CONFIRMATION_TRIALS
        + confirmation_index
    )


def geometry_treatment_rows(
    rows: list[dict[str, Any]], treatment_mode: str, sample_index: int
) -> list[dict[str, Any]]:
    """Exclude matched bracket rows before interval geometry reduction."""

    mode = geometry_calibration_mode(treatment_mode)
    selected = [
        row for row in rows
        if row["mode"] == mode and int(row["sample_index"]) == sample_index
    ]
    if not selected:
        raise ValueError(
            f"missing geometry treatment rows for {treatment_mode}, "
            f"sample {sample_index}"
        )
    return selected


def validate_coverage_targets(coverage_targets: list[float]) -> None:
    """Reject target values that cannot have unique stage/bracket identities."""

    if not coverage_targets:
        raise ValueError("at least one coverage target is required")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 100.0
        for value in coverage_targets
    ):
        raise ValueError("coverage targets must be finite and in [0, 100]")
    if len(set(coverage_targets)) != len(coverage_targets):
        raise ValueError("coverage targets must be unique")
    tags = [f"{value:g}" for value in coverage_targets]
    if len(set(tags)) != len(tags):
        raise ValueError(
            "coverage targets must have unique formatted stage-mode tags"
        )


def focal_adjacent_bracket_orders(
    spec: dict[str, Any], operation_count: int
) -> tuple[list[int], list[int]]:
    """Place the focal victim literally next to treatment on both sides."""

    treatment_order = [int(index) for index in spec["order"]]
    expected = set(range(operation_count))
    if len(treatment_order) != operation_count or set(treatment_order) != expected:
        raise ValueError("treatment dispatch order must contain every operation once")
    victim = int(spec["victim_index"])
    if victim not in expected:
        raise ValueError("victim index is outside the operation range")
    non_victims = [index for index in treatment_order if index != victim]
    return [*non_victims, victim], [victim, *non_victims]


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible decision payload deterministically."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collapse_odd_k_stage_measurements(
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collapse repeated isolated stages to a deterministic odd-K median.

    The representative is the duration order statistic, with the lowest
    repetition index breaking ties. Completion remains the latest completion
    of the complete series so a following decision cannot race another repeat.
    """

    repetitions = len(stages)
    if repetitions < 3 or repetitions % 2 == 0:
        raise ValueError("baseline measurement count must be odd and at least 3")
    if any(stage.get("compute_output_signatures") for stage in stages):
        raise ValueError("isolated baseline repetitions must not capture outputs")
    if any(stage.get("communication_correctness") for stage in stages):
        raise ValueError("isolated baseline repetitions must not capture correctness")
    grouped: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
    for repetition_index, stage in enumerate(stages):
        for row in stage["rows"]:
            identity = (
                int(row["operation_index"]),
                str(row["participant_id"]),
                int(row["global_rank"]),
                int(row["sample_index"]),
                str(row["mode"]),
                row.get("bracket_for_mode"),
            )
            grouped.setdefault(identity, []).append((repetition_index, row))
    if not grouped:
        return {
            "rows": [],
            "compute_output_signatures": [],
            "communication_correctness": [],
        }
    collapsed = []
    for identity, records in sorted(grouped.items()):
        if len(records) != repetitions:
            raise ValueError(
                f"baseline endpoint {identity} has {len(records)} of "
                f"{repetitions} repetitions"
            )
        ordered = sorted(
            records,
            key=lambda item: (float(item[1]["duration_ms"]), item[0]),
        )
        representative_repeat, representative_source = ordered[repetitions // 2]
        representative = dict(representative_source)
        repeat_rows = [
            {
                "repeat_index": repeat_index,
                "duration_ms": float(row["duration_ms"]),
                "gate_to_start_ms": float(row["gate_to_start_ms"]),
                "gate_to_end_ms": float(row["gate_to_end_ms"]),
                "host_launch_global_ns": int(row["host_launch_global_ns"]),
                "gpu_start_global_ns": int(row["gpu_start_global_ns"]),
                "gpu_end_global_ns": int(row["gpu_end_global_ns"]),
                "host_complete_global_ns": int(row["host_complete_global_ns"]),
            }
            for repeat_index, row in sorted(records)
        ]
        durations = [row["duration_ms"] for row in repeat_rows]
        representative.update({
            "baseline_measurement_method": BASELINE_MEASUREMENT_METHOD,
            "measurement_repetition_count": repetitions,
            "measurement_representative_repeat_index": representative_repeat,
            "measurement_representative_policy": (
                "median_duration_order_statistic_then_lowest_repeat_index_v1"
            ),
            "measurement_duration_statistics_ms": {
                "count": repetitions,
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
            },
            "measurement_repeat_records": repeat_rows,
            "measurement_repeat_records_sha256": canonical_sha256(repeat_rows),
            "measurement_series_host_complete_global_ns": max(
                row["host_complete_global_ns"] for row in repeat_rows
            ),
            "host_complete_semantics": "measurement_series_latest",
        })
        representative["host_complete_global_ns"] = representative[
            "measurement_series_host_complete_global_ns"
        ]
        collapsed.append(representative)
    return {
        "rows": collapsed,
        "compute_output_signatures": [],
        "communication_correctness": [],
    }


def focal_start_gap_ms(
    rows: list[dict[str, Any]],
    participants: list[dict[str, Any]],
    spec: dict[str, Any],
) -> float:
    """Return the common-focal GPU-start gap between dispatch groups."""

    starts: dict[int, int] = {}
    for index, participant in enumerate(participants):
        focal_rank = (
            int(participant["focal_dp_rank"]) * 2
            + int(participant["focal_pp_rank"])
        )
        matches = [
            row for row in rows
            if int(row["operation_index"]) == index
            and int(row["global_rank"]) == focal_rank
        ]
        if len(matches) != 1:
            raise ValueError(
                f"operation {index}: expected one common-focal start on rank "
                f"{focal_rank}, got {len(matches)}"
            )
        starts[index] = int(matches[0]["gpu_start_global_ns"])
    victim = int(spec["victim_index"])
    aggressors = [int(index) for index in spec["aggressor_indices"]]
    if spec["dispatch_mode"] == "victim_first":
        gap_ns = min(starts[index] for index in aggressors) - starts[victim]
    elif spec["dispatch_mode"] == "aggressor_first":
        gap_ns = starts[victim] - max(starts[index] for index in aggressors)
    else:
        raise ValueError(f"unsupported dispatch mode: {spec['dispatch_mode']}")
    return gap_ns / 1e6


def configured_start_gap_ms(
    delays_ms: dict[int, float], spec: dict[str, Any]
) -> float:
    victim = int(spec["victim_index"])
    aggressors = [int(index) for index in spec["aggressor_indices"]]
    if spec["dispatch_mode"] == "victim_first":
        return min(delays_ms[index] for index in aggressors) - delays_ms[victim]
    if spec["dispatch_mode"] == "aggressor_first":
        return delays_ms[victim] - max(delays_ms[index] for index in aggressors)
    raise ValueError(f"unsupported dispatch mode: {spec['dispatch_mode']}")


def summarize_corrected_start_anchor_confirmation(
    measurements: list[dict[str, Any]],
    *,
    constraint_limited: bool,
    tolerance_pct: float = GEOMETRY_TOLERANCE_PCT,
) -> dict[str, Any]:
    """Apply the v2 K3 median-and-two-of-three confirmation rule."""

    if len(measurements) != START_ANCHOR_REPETITIONS:
        raise ValueError("start-anchor confirmation requires exactly K=3")
    errors = [
        float(row["scheduled_geometry_error_pct"]) for row in measurements
    ]
    overlaps = [float(row["scheduled_overlap_pct"]) for row in measurements]
    passing = [abs(error) <= tolerance_pct for error in errors]
    median_error = statistics.median(errors)
    robust = bool(
        abs(median_error) <= tolerance_pct
        and sum(passing) >= START_ANCHOR_MINIMUM_PASSING
    )
    if robust and constraint_limited:
        resolution = "constraint_limited_confirmed_within_10_pct_points"
    elif robust:
        resolution = "corrected_delay_confirmed"
    else:
        resolution = "corrected_delay_confirmation_unresolved"
    disposition = (
        RUNTIME_START_GEOMETRY_CONFIRMED
        if robust else RUNTIME_START_GEOMETRY_UNRESOLVED
    )
    return {
        "schema_version": 1,
        "policy": START_ANCHOR_CONFIRMATION_POLICY,
        "repetitions": START_ANCHOR_REPETITIONS,
        "minimum_passing_repetitions": START_ANCHOR_MINIMUM_PASSING,
        "unscored": True,
        "measured_counterfactual_overlap_pct_by_repetition": overlaps,
        "measured_geometry_error_pct_by_repetition": errors,
        "median_measured_counterfactual_overlap_pct": statistics.median(overlaps),
        "median_measured_geometry_error_pct": median_error,
        "passing_repetition_count": sum(passing),
        "robustly_within_tolerance": robust,
        "constraint_limited": bool(constraint_limited),
        "resolution_status": resolution,
        "runtime_start_geometry_disposition": disposition,
    }


def runtime_start_geometry_sample_fields(
    disposition: str,
    resolution_status: str | None,
) -> dict[str, Any]:
    """Return the explicit, fail-closed per-treatment scoring contract."""

    allowed = {
        RUNTIME_START_GEOMETRY_CONFIRMED,
        RUNTIME_START_GEOMETRY_UNRESOLVED,
        RUNTIME_START_GEOMETRY_NOT_APPLICABLE_STATIC,
    }
    if disposition not in allowed:
        raise ValueError(f"invalid runtime-start disposition {disposition!r}")
    scoring_eligible = disposition == RUNTIME_START_GEOMETRY_CONFIRMED
    return {
        "runtime_start_geometry_disposition_explicit": True,
        "runtime_start_geometry_disposition": disposition,
        "runtime_start_geometry_treatment_executed": True,
        "runtime_start_geometry_scoring_eligible": scoring_eligible,
        # Retained v13 aliases.
        "start_anchor_resolution_status": resolution_status,
        "start_anchor_scoring_allowed": scoring_eligible,
    }



def execute_runtime_start_anchor(
    spec: dict[str, Any],
    sample_index: int,
    participants: list[dict[str, Any]],
    operation_count: int,
    geometry_max_relative_shift_ms: float,
    local_pre_stages: list[dict[str, Any]],
    run_stage: Any,
    gather_rows: Any,
    consensus: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Run the unscored preliminary/corrected K3 start-anchor protocol."""

    pre_rows = gather_rows([
        row for stage in local_pre_stages for row in stage["rows"]
    ])
    pre_clock_provenance = common_clock_provenance(pre_rows)
    if not bool(spec["target_feasible_from_duration"]):
        if bool(spec.get("target_feasible_after_constraints")):
            raise RuntimeError(
                "duration-infeasible stage cannot remain feasible after "
                "cumulative constraints"
            )
        fields = runtime_start_geometry_sample_fields(
            RUNTIME_START_GEOMETRY_NOT_APPLICABLE_STATIC,
            "isolated_duration_bound_infeasible",
        )
        evidence = {
            "schema_version": 1,
            "policy": "static_calibrated_delays_v1",
            **pre_clock_provenance,
            "dynamic_rebase_applied": False,
            "static_policy_reason": "duration_bound_infeasible",
            "sample_index": sample_index,
            "treatment_mode": spec["mode"],
            "target_feasible_from_duration": False,
            "target_feasible_after_constraints": bool(
                spec.get("target_feasible_after_constraints", False)
            ),
            "target_feasible_after_runtime_calibration": False,
            "runtime_geometry_calibration_status": (
                "isolated_duration_bound_infeasible"
            ),
            "final_delays_ms": dict(spec["delays_ms"]),
            **fields,
        }
        return dict(spec["delays_ms"]), {
            **fields,
            "delay_rebase_evidence": evidence,
            "delay_rebase_evidence_sha256": canonical_sha256(evidence),
        }

    pre_durations = {
        index: max(
            float(row["duration_ms"])
            for row in matched_baseline_rows(
                pre_rows, "isolated_pre", spec["mode"], index, sample_index
            )
        )
        for index in range(operation_count)
    }
    initial_delays = {
        int(index): float(value) for index, value in spec["delays_ms"].items()
    }
    preliminary_measurements = []
    preliminary_rows = []
    for repetition_index in range(START_ANCHOR_REPETITIONS):
        local = run_stage(
            list(range(operation_count)),
            f"start_anchor::{spec['mode']}::repeat={repetition_index}",
            sample_index,
            order_override=spec["order"],
            delays_ms=initial_delays,
        )
        global_rows = [
            {**row, "repetition_index": repetition_index}
            for row in gather_rows(local["rows"])
        ]
        preliminary_rows.extend(global_rows)
        measurement = measure_runtime_geometry(
            global_rows, participants, spec, pre_durations
        )
        measurement.update({
            "repetition_index": repetition_index,
            "relative_shift_ms": 0.0,
            "actual_focal_gpu_start_gap_ms": focal_start_gap_ms(
                global_rows, participants, spec
            ),
        })
        preliminary_measurements.append(measurement)
    preliminary_summary = {
        "relative_shift_ms": 0.0,
        "scheduled_geometry_error_pct": statistics.median(
            row["scheduled_geometry_error_pct"]
            for row in preliminary_measurements
        ),
        "geometry_error_pct": statistics.median(
            row["geometry_error_pct"] for row in preliminary_measurements
        ),
        "scheduled_overlap_error_ms": statistics.median(
            (
                row["scheduled_overlap_pct"]
                - float(spec["requested_overlap_pct"])
            ) / 100.0 * pre_durations[int(spec["victim_index"])]
            for row in preliminary_measurements
        ),
    }
    lower, upper = relative_shift_bounds_ms(
        spec, initial_delays, geometry_max_relative_shift_ms
    )
    feedback = choose_geometry_feedback_shift(
        [preliminary_summary],
        float(spec["requested_overlap_pct"]),
        pre_durations[int(spec["victim_index"])],
        lower, upper,
        tolerance_pct=0.0,
    )
    correction_ms = float(feedback["next_shift_ms"])
    corrected_delays = apply_relative_geometry_shift(
        spec, initial_delays, correction_ms
    )
    constraint_limited = bool(feedback["stalled"])
    preliminary_decision = {
        "policy": START_ANCHOR_POLICY,
        "sample_index": sample_index,
        "treatment_mode": spec["mode"],
        "initial_delays_ms": initial_delays,
        "corrected_delays_ms": corrected_delays,
        "correction_ms": correction_ms,
        "constraint_limited": constraint_limited,
        "preliminary_geometry_errors_pct": [
            row["scheduled_geometry_error_pct"]
            for row in preliminary_measurements
        ],
    }
    preliminary_consensus = consensus(
        preliminary_decision, "start_anchor_preliminary_correction"
    )

    confirmation_measurements = []
    confirmation_rows = []
    for repetition_index in range(START_ANCHOR_REPETITIONS):
        local = run_stage(
            list(range(operation_count)),
            f"start_anchor_confirmation::{spec['mode']}::repeat={repetition_index}",
            sample_index,
            order_override=spec["order"],
            delays_ms=corrected_delays,
        )
        global_rows = [
            {**row, "repetition_index": repetition_index}
            for row in gather_rows(local["rows"])
        ]
        confirmation_rows.extend(global_rows)
        measurement = measure_runtime_geometry(
            global_rows, participants, spec, pre_durations
        )
        measurement.update({
            "repetition_index": repetition_index,
            "actual_focal_gpu_start_gap_ms": focal_start_gap_ms(
                global_rows, participants, spec
            ),
        })
        confirmation_measurements.append(measurement)
    confirmation = summarize_corrected_start_anchor_confirmation(
        confirmation_measurements, constraint_limited=constraint_limited
    )
    compound_clock_provenance = common_clock_provenance(
        [*pre_rows, *preliminary_rows, *confirmation_rows]
    )
    confirmation_clock_provenance = common_clock_provenance(confirmation_rows)
    focal_ranks = {
        index: int(participant["focal_dp_rank"]) * 2
        + int(participant["focal_pp_rank"])
        for index, participant in enumerate(participants)
    }
    preliminary_focal_rows = [
        row for row in preliminary_rows
        if int(row["global_rank"]) == focal_ranks[int(row["operation_index"])]
    ]
    confirmation_focal_rows = [
        row for row in confirmation_rows
        if int(row["global_rank"]) == focal_ranks[int(row["operation_index"])]
    ]
    preliminary_completion_rows = [
        {
            "repetition_index": int(row["repetition_index"]),
            "operation_index": int(row["operation_index"]),
            "participant_id": row["participant_id"],
            "global_rank": int(row["global_rank"]),
            "mode": row["mode"],
            "host_complete_global_ns": int(row["host_complete_global_ns"]),
            "clock_calibration_generation": int(
                row["clock_calibration_generation"]
            ),
            "clock_calibration_label": row["clock_calibration_label"],
            "clock_calibration_best_rtt_ns": int(
                row["clock_calibration_best_rtt_ns"]
            ),
            "clock_calibration_half_rtt_uncertainty_ns": int(
                row["clock_calibration_half_rtt_uncertainty_ns"]
            ),
        }
        for row in preliminary_rows
    ]
    confirmation_completion_rows = [
        {
            "repetition_index": int(row["repetition_index"]),
            "operation_index": int(row["operation_index"]),
            "participant_id": row["participant_id"],
            "global_rank": int(row["global_rank"]),
            "mode": row["mode"],
            "host_complete_global_ns": int(row["host_complete_global_ns"]),
            "clock_calibration_generation": int(
                row["clock_calibration_generation"]
            ),
            "clock_calibration_label": row["clock_calibration_label"],
            "clock_calibration_best_rtt_ns": int(
                row["clock_calibration_best_rtt_ns"]
            ),
            "clock_calibration_half_rtt_uncertainty_ns": int(
                row["clock_calibration_half_rtt_uncertainty_ns"]
            ),
        }
        for row in confirmation_rows
    ]
    confirmation.update({
        **confirmation_clock_provenance,
        "corrected_delays_ms": corrected_delays,
        "actual_focal_gpu_start_gaps_ms": [
            row["actual_focal_gpu_start_gap_ms"]
            for row in confirmation_measurements
        ],
        "raw_focal_start_rows_schema_version": 1,
        "raw_focal_start_rows": confirmation_focal_rows,
        "raw_focal_start_rows_sha256": canonical_sha256(
            confirmation_focal_rows
        ),
        "raw_focal_start_row_count": len(confirmation_focal_rows),
        "all_endpoint_completion_rows_schema_version": 1,
        "all_endpoint_completion_rows": confirmation_completion_rows,
        "all_endpoint_completion_rows_sha256": canonical_sha256(
            confirmation_completion_rows
        ),
        "latest_all_endpoint_complete_global_ns": max(
            row["host_complete_global_ns"]
            for row in confirmation_completion_rows
        ),
        "repetition_records": [
            {
                "repetition_index": index,
                "actual_focal_gpu_start_gap_ms": measurement[
                    "actual_focal_gpu_start_gap_ms"
                ],
                "measured_counterfactual_overlap_pct": measurement[
                    "scheduled_overlap_pct"
                ],
                "measured_geometry_error_pct": measurement[
                    "scheduled_geometry_error_pct"
                ],
                "within_tolerance": abs(
                    measurement["scheduled_geometry_error_pct"]
                ) <= GEOMETRY_TOLERANCE_PCT,
            }
            for index, measurement in enumerate(confirmation_measurements)
        ],
    })
    final_decision = {
        "policy": START_ANCHOR_CONFIRMATION_POLICY,
        "sample_index": sample_index,
        "treatment_mode": spec["mode"],
        "corrected_delays_ms": corrected_delays,
        "resolution_status": confirmation["resolution_status"],
        "runtime_start_geometry_disposition": confirmation[
            "runtime_start_geometry_disposition"
        ],
        "passing_repetition_count": confirmation["passing_repetition_count"],
        "median_measured_geometry_error_pct": confirmation[
            "median_measured_geometry_error_pct"
        ],
    }
    final_consensus = consensus(
        final_decision, "start_anchor_post_confirmation_decision"
    )
    fields = runtime_start_geometry_sample_fields(
        confirmation["runtime_start_geometry_disposition"],
        confirmation["resolution_status"],
    )
    start_anchor_evidence = {
        "schema_version": 1,
        "policy": START_ANCHOR_POLICY,
        **compound_clock_provenance,
        "repetitions": START_ANCHOR_REPETITIONS,
        "unscored": True,
        "sample_index": sample_index,
        "treatment_mode": spec["mode"],
        "configured_later_group_gap_ms": configured_start_gap_ms(
            initial_delays, spec
        ),
        "actual_focal_gpu_start_gaps_ms": [
            row["actual_focal_gpu_start_gap_ms"]
            for row in preliminary_measurements
        ],
        "launch_residuals_ms": [
            row["actual_focal_gpu_start_gap_ms"]
            - configured_start_gap_ms(initial_delays, spec)
            for row in preliminary_measurements
        ],
        "preliminary_decision": preliminary_decision,
        "preliminary_consensus": preliminary_consensus,
        "confirmation": confirmation,
        "post_confirmation_consensus": final_consensus,
        "raw_focal_start_rows_schema_version": 1,
        "raw_focal_start_rows": preliminary_focal_rows,
        "raw_focal_start_rows_sha256": canonical_sha256(
            preliminary_focal_rows
        ),
        "raw_focal_start_row_count": len(preliminary_focal_rows),
        "all_endpoint_completion_rows_schema_version": 1,
        "all_endpoint_completion_rows": preliminary_completion_rows,
        "all_endpoint_completion_rows_sha256": canonical_sha256(
            preliminary_completion_rows
        ),
        "latest_anchor_all_endpoint_complete_global_ns": max(
            row["host_complete_global_ns"] for row in preliminary_completion_rows
        ),
        "confirmation_rows_schema_version": 1,
        "confirmation_row_count": len(confirmation_rows),
        "confirmation_rows": confirmation_rows,
        "confirmation_rows_sha256": canonical_sha256(confirmation_rows),
        "final_delays_ms": corrected_delays,
        "resolution_status": confirmation["resolution_status"],
    }
    evidence = {
        "schema_version": 1,
        "policy": "adjacent_pre_runtime_start_rebase_v2",
        **compound_clock_provenance,
        "dynamic_rebase_applied": True,
        "static_policy_reason": None,
        "sample_index": sample_index,
        "treatment_mode": spec["mode"],
        "pre_durations_ms": pre_durations,
        "pre_rows_schema_version": 1,
        "pre_rows": pre_rows,
        "pre_rows_sha256": canonical_sha256(pre_rows),
        "pre_row_count": len(pre_rows),
        "selected_calibration_delays_ms": initial_delays,
        "final_delays_ms": corrected_delays,
        "start_anchor_policy": START_ANCHOR_POLICY,
        "start_anchor_evidence": start_anchor_evidence,
        "start_anchor_evidence_sha256": canonical_sha256(start_anchor_evidence),
        **fields,
    }
    return corrected_delays, {
        **fields,
        "delay_rebase_evidence": evidence,
        "delay_rebase_evidence_sha256": canonical_sha256(evidence),
    }

def summarize_geometry_confirmation_trials(
    measurements: list[dict[str, Any]],
    tolerance_pct: float,
    drift_tolerance_pct: float = BASELINE_DRIFT_TOLERANCE_PCT,
) -> dict[str, Any]:
    """Reduce repeated fixed-shift geometry trials without cherry-picking.

    Individual counterfactual geometry measurements retain the ordinary
    tolerance, while the candidate median must meet the tighter centering
    tolerance before feedback freezes its shift. Concurrent overlap remains
    in the legacy output fields as an outcome only.
    """

    if len(measurements) < GEOMETRY_CONFIRMATION_MINIMUM_PASSING:
        raise ValueError("too few runtime geometry confirmation measurements")
    shifts = {float(row["relative_shift_ms"]) for row in measurements}
    if len(shifts) != 1:
        raise ValueError("geometry confirmations must use one fixed relative shift")
    scheduled_errors = [
        float(row.get("scheduled_geometry_error_pct", row["geometry_error_pct"]))
        for row in measurements
    ]
    concurrent_errors = [
        float(row["geometry_error_pct"]) for row in measurements
    ]
    drifts = [float(row["paired_baseline_drift_pct"]) for row in measurements]
    geometry_baseline_drifts = [
        float(row.get(
            "geometry_baseline_max_abs_drift_pct",
            abs(float(row["paired_baseline_drift_pct"])),
        ))
        for row in measurements
    ]
    geometry_passing = [
        abs(value) <= tolerance_pct for value in scheduled_errors
    ]
    baseline_stable = [
        value <= drift_tolerance_pct for value in geometry_baseline_drifts
    ]
    jointly_passing = [
        geometry_ok and drift_ok
        for geometry_ok, drift_ok in zip(geometry_passing, baseline_stable)
    ]
    median_scheduled_error = statistics.median(scheduled_errors)
    median_concurrent_error = statistics.median(concurrent_errors)
    median_drift = statistics.median(drifts)
    median_geometry_baseline_drift = statistics.median(
        geometry_baseline_drifts
    )
    summary = {
        "relative_shift_ms": shifts.pop(),
        "union_overlap_ms": statistics.median(
            float(row["union_overlap_ms"]) for row in measurements
        ),
        "realized_overlap_pct": statistics.median(
            float(row["realized_overlap_pct"]) for row in measurements
        ),
        "geometry_error_pct": median_concurrent_error,
        "overlap_error_ms": statistics.median(
            float(row["overlap_error_ms"]) for row in measurements
        ),
        "scheduled_overlap_ms": statistics.median(
            float(row.get("scheduled_overlap_ms", row["union_overlap_ms"]))
            for row in measurements
        ),
        "scheduled_overlap_pct": statistics.median(
            float(row.get("scheduled_overlap_pct", row["realized_overlap_pct"]))
            for row in measurements
        ),
        "scheduled_geometry_error_pct": median_scheduled_error,
        "scheduled_overlap_error_ms": statistics.median(
            float(row.get("scheduled_overlap_error_ms", row["overlap_error_ms"]))
            for row in measurements
        ),
        "counterfactual_overlap_ms": statistics.median(
            float(row.get("counterfactual_overlap_ms", row["union_overlap_ms"]))
            for row in measurements
        ),
        "counterfactual_overlap_pct": statistics.median(
            float(row.get(
                "counterfactual_overlap_pct", row["realized_overlap_pct"]
            ))
            for row in measurements
        ),
        "counterfactual_geometry_error_pct": median_scheduled_error,
        "geometry_validity_basis": (
            GEOMETRY_VALIDITY_BASIS
            if all("scheduled_geometry_error_pct" in row for row in measurements)
            else "legacy_concurrent_intervals_fallback"
        ),
        "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
        "confirmation_trial_count": len(measurements),
        # Kept as the authoritative acceptance count for downstream metadata.
        "confirmation_passing_count": sum(jointly_passing),
        "geometry_passing_count": sum(geometry_passing),
        "baseline_stable_count": sum(baseline_stable),
        "jointly_passing_count": sum(jointly_passing),
        "confirmation_minimum_passing": GEOMETRY_CONFIRMATION_MINIMUM_PASSING,
        # Legacy error ranges describe concurrent/outcome geometry.
        "confirmation_error_min_pct": min(concurrent_errors),
        "confirmation_error_max_pct": max(concurrent_errors),
        "confirmation_scheduled_error_min_pct": min(scheduled_errors),
        "confirmation_scheduled_error_max_pct": max(scheduled_errors),
        "confirmation_drift_min_pct": min(drifts),
        "confirmation_drift_max_pct": max(drifts),
        "confirmation_geometry_baseline_max_abs_drift_min_pct": min(
            geometry_baseline_drifts
        ),
        "confirmation_geometry_baseline_max_abs_drift_max_pct": max(
            geometry_baseline_drifts
        ),
        "paired_baseline_ms": statistics.median(
            float(row["paired_baseline_ms"]) for row in measurements
        ),
        "paired_baseline_drift_pct": median_drift,
        "geometry_baseline_max_abs_drift_pct": (
            median_geometry_baseline_drift
        ),
        "geometry_baseline_stability_basis": (
            "maximum_absolute_pre_post_drift_across_all_operations"
        ),
        "confirmation_measurements": measurements,
    }
    summary["baseline_feedback_eligible"] = bool(
        median_geometry_baseline_drift <= drift_tolerance_pct
        and sum(baseline_stable) >= GEOMETRY_CONFIRMATION_MINIMUM_PASSING
    )
    summary["median_centered"] = bool(
        abs(median_scheduled_error) <= GEOMETRY_CENTERING_TOLERANCE_PCT
    )
    summary["robustly_converged"] = bool(
        summary["median_centered"]
        and median_geometry_baseline_drift <= drift_tolerance_pct
        and sum(jointly_passing) >= GEOMETRY_CONFIRMATION_MINIMUM_PASSING
    )
    return summary


def choose_geometry_candidate_feedback(
    candidate: dict[str, Any],
    feedback_history: list[dict[str, float]],
    requested_overlap_pct: float,
    minimum_shift_ms: float,
    maximum_shift_ms: float,
) -> dict[str, Any]:
    """Steer only from candidates with a stable matched denominator."""

    current_shift_ms = float(candidate["relative_shift_ms"])
    if not bool(candidate.get("baseline_feedback_eligible", False)):
        return {
            "converged": False,
            "next_shift_ms": current_shift_ms,
            "method": "repeat_baseline_unstable_candidate",
            "stalled": False,
            "feedback_eligible": False,
        }
    feedback = choose_geometry_feedback_shift(
        [*feedback_history, candidate],
        requested_overlap_pct,
        float(candidate["paired_baseline_ms"]),
        minimum_shift_ms,
        maximum_shift_ms,
        tolerance_pct=GEOMETRY_CENTERING_TOLERANCE_PCT,
    )
    feedback["feedback_eligible"] = True
    return feedback


def matched_baseline_rows(
    rows: list[dict[str, Any]],
    phase: str,
    treatment_mode: str,
    operation_index: int,
    sample_index: int,
) -> list[dict[str, Any]]:
    """Return the isolated rows paired with exactly one treatment cell.

    New runs tag every adjacent isolated bracket with ``bracket_for_mode`` so
    two treatments issued in the same adaptive batch can never share or
    overwrite a baseline.  Untagged rows are accepted only as a compatibility
    path for older fixed-result unit fixtures; a mixture of tagged and
    untagged rows is deliberately not ambiguous.
    """

    candidates = [
        row for row in rows
        if row["mode"] == phase
        and int(row["operation_index"]) == operation_index
        and int(row["sample_index"]) == sample_index
    ]
    tagged = [
        row for row in candidates
        if row.get("bracket_for_mode") == treatment_mode
    ]
    if tagged:
        return tagged
    if candidates and all(row.get("bracket_for_mode") is None for row in candidates):
        return candidates
    raise ValueError(
        f"missing matched {phase} rows for treatment {treatment_mode}, "
        f"operation {operation_index}, sample {sample_index}"
    )


def adaptive_cell_key(
    assignment: dict[str, Any], spec: dict[str, Any]
) -> AdaptiveBenchmarkCellKey:
    """Return the controller identity corresponding to one stage spec."""

    return AdaptiveBenchmarkCellKey(
        str(assignment["case_key"]),
        int(spec["victim_index"]),
        float(spec["requested_overlap_pct"]),
        str(spec["dispatch_mode"]),
    )


def adaptive_plan_batches(
    assignment: dict[str, Any],
    stage_specs: list[dict[str, Any]],
    plan: dict[AdaptiveBenchmarkCellKey, int],
) -> list[list[dict[str, Any]]]:
    """Expand a controller plan into deterministic treatment-stage batches.

    A stage appears exactly ``plan[cell]`` times. Stopped cells, which are
    absent from the plan, are never returned and therefore never launched.
    The matrix order is stable on every rank; per-attempt seeded shuffling may
    subsequently permute each returned batch identically.
    """

    by_key: dict[AdaptiveBenchmarkCellKey, dict[str, Any]] = {}
    for spec in stage_specs:
        key = adaptive_cell_key(assignment, spec)
        if key in by_key:
            raise ValueError(f"duplicate adaptive stage cell {key.canonical}")
        by_key[key] = spec
    unknown = sorted(set(plan) - set(by_key))
    if unknown:
        raise ValueError(
            "adaptive plan contains unknown cells: "
            + ", ".join(key.canonical for key in unknown)
        )
    if any(int(attempts) <= 0 for attempts in plan.values()):
        raise ValueError("adaptive plan attempts must be positive")
    maximum = max((int(attempts) for attempts in plan.values()), default=0)
    return [
        [
            spec for spec in stage_specs
            if int(plan.get(adaptive_cell_key(assignment, spec), 0)) > offset
        ]
        for offset in range(maximum)
    ]


def build_adaptive_controller_rows(
    assignment: dict[str, Any],
    participants: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    active_specs: list[dict[str, Any]],
    sample_index: int,
    *,
    correct: bool,
) -> list[dict[str, Any]]:
    """Reduce one globally gathered attempt into controller sample rows."""

    def duration(mode: str, operation_index: int) -> float:
        matches = [
            row for row in global_rows
            if row["mode"] == mode
            and int(row["operation_index"]) == operation_index
            and int(row["sample_index"]) == sample_index
        ]
        if not matches:
            raise ValueError(
                f"missing {mode} rows for operation {operation_index}, "
                f"sample {sample_index}"
            )
        return max(float(row["duration_ms"]) for row in matches)

    reduced = []
    for spec in active_specs:
        victim = int(spec["victim_index"])
        baseline_pre_by_operation = {
            index: max(
                float(row["duration_ms"])
                for row in matched_baseline_rows(
                    global_rows,
                    "isolated_pre",
                    spec["mode"],
                    index,
                    sample_index,
                )
            )
            for index in range(len(participants))
        }
        baseline_post_by_operation = {
            index: max(
                float(row["duration_ms"])
                for row in matched_baseline_rows(
                    global_rows,
                    "isolated_post",
                    spec["mode"],
                    index,
                    sample_index,
                )
            )
            for index in range(len(participants))
        }
        matched_baselines_ms = {
            index: (
                baseline_pre_by_operation[index]
                + baseline_post_by_operation[index]
            ) / 2.0
            for index in range(len(participants))
        }
        geometry_baseline_drifts_pct = {
            index: (
                baseline_post_by_operation[index]
                / baseline_pre_by_operation[index]
                - 1.0
            ) * 100.0
            for index in range(len(participants))
        }
        geometry_baseline_max_abs_drift_pct = max(
            abs(value) for value in geometry_baseline_drifts_pct.values()
        )
        baseline_pre = baseline_pre_by_operation[victim]
        baseline_post = baseline_post_by_operation[victim]
        baseline = matched_baselines_ms[victim]
        if baseline <= 0.0 or baseline_pre <= 0.0:
            raise ValueError("adaptive victim baseline must be positive")
        mode_rows = [
            row for row in global_rows
            if row["mode"] == spec["mode"]
            and int(row["sample_index"]) == sample_index
        ]
        start_contract_keys = (
            "runtime_start_geometry_disposition_explicit",
            "runtime_start_geometry_disposition",
            "runtime_start_geometry_treatment_executed",
            "runtime_start_geometry_scoring_eligible",
            "start_anchor_resolution_status",
            "start_anchor_scoring_allowed",
            "delay_rebase_evidence",
            "delay_rebase_evidence_sha256",
        )
        start_contracts = [
            {key: row.get(key) for key in start_contract_keys}
            for row in mode_rows
        ]
        if not start_contracts:
            raise ValueError("adaptive treatment emitted no runtime-start contract")
        start_metadata = start_contracts[0]
        if any(candidate != start_metadata for candidate in start_contracts[1:]):
            raise ValueError(
                "adaptive runtime-start disposition differs across endpoints"
            )
        geometry = measure_runtime_geometry(
            mode_rows, participants, spec, matched_baselines_ms
        )
        pre_geometry = measure_runtime_geometry(
            mode_rows, participants, spec, baseline_pre_by_operation
        )
        post_geometry = measure_runtime_geometry(
            mode_rows, participants, spec, baseline_post_by_operation
        )
        victim_baseline_stable = bool(
            abs((baseline_post / baseline_pre - 1.0) * 100.0)
            <= BASELINE_DRIFT_TOLERANCE_PCT
        )
        pre_post_counterfactual_geometry_robust = bool(
            abs(float(pre_geometry["scheduled_geometry_error_pct"]))
            <= GEOMETRY_TOLERANCE_PCT
            and abs(float(post_geometry["scheduled_geometry_error_pct"]))
            <= GEOMETRY_TOLERANCE_PCT
        )
        concurrent = duration(spec["mode"], victim)
        feasible = bool(
            spec.get(
                "target_feasible_after_runtime_calibration",
                spec["target_feasible_from_isolated"],
            )
        )
        scheduled_geometry_error = float(
            geometry["scheduled_geometry_error_pct"]
        )
        reduced.append({
            "case_id": assignment["case_id"],
            "case_key": assignment["case_key"],
            "sample_index": sample_index,
            **start_metadata,
            "victim_index": victim,
            "requested_overlap_pct": float(spec["requested_overlap_pct"]),
            "dispatch_mode": spec["dispatch_mode"],
            "target_feasible_after_runtime_calibration": feasible,
            "valid": (
                bool(start_metadata["runtime_start_geometry_scoring_eligible"])
                and feasible
                and abs(scheduled_geometry_error) <= GEOMETRY_TOLERANCE_PCT
                and victim_baseline_stable
                and pre_post_counterfactual_geometry_robust
            ),
            # Legacy/concurrent field remains an outcome.
            "geometry_error_pct": float(geometry["geometry_error_pct"]),
            "concurrent_overlap_ms": float(geometry["concurrent_overlap_ms"]),
            "realized_overlap_pct": float(geometry["realized_overlap_pct"]),
            "scheduled_overlap_ms": float(geometry["scheduled_overlap_ms"]),
            "scheduled_overlap_pct": float(geometry["scheduled_overlap_pct"]),
            "scheduled_geometry_error_pct": scheduled_geometry_error,
            "counterfactual_overlap_pct": float(
                geometry["counterfactual_overlap_pct"]
            ),
            "counterfactual_overlap_ms": float(
                geometry["counterfactual_overlap_ms"]
            ),
            "counterfactual_geometry_error_pct": scheduled_geometry_error,
            "geometry_validity_error_pct": scheduled_geometry_error,
            "geometry_validity_basis": GEOMETRY_VALIDITY_BASIS,
            "geometry_baseline_drifts_pct": geometry_baseline_drifts_pct,
            "geometry_baseline_max_abs_drift_pct": (
                geometry_baseline_max_abs_drift_pct
            ),
            "geometry_baseline_max_abs_drift_role": "diagnostic_only",
            "baseline_stability_policy": (
                "victim_relative_drift_and_pre_post_geometry_robust_v1"
            ),
            "victim_baseline_stable": victim_baseline_stable,
            "pre_baseline_scheduled_overlap_pct": float(
                pre_geometry["scheduled_overlap_pct"]
            ),
            "pre_baseline_scheduled_geometry_error_pct": float(
                pre_geometry["scheduled_geometry_error_pct"]
            ),
            "post_baseline_scheduled_overlap_pct": float(
                post_geometry["scheduled_overlap_pct"]
            ),
            "post_baseline_scheduled_geometry_error_pct": float(
                post_geometry["scheduled_geometry_error_pct"]
            ),
            "pre_post_counterfactual_geometry_robust": (
                pre_post_counterfactual_geometry_robust
            ),
            "geometry_baseline_stability_basis": (
                "maximum_absolute_pre_post_drift_across_all_operations"
            ),
            "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
            "geometry_tolerance_pct": GEOMETRY_TOLERANCE_PCT,
            "baseline_drift_pct": (
                baseline_post / baseline_pre - 1.0
            ) * 100.0,
            "baseline_measurement_method": BASELINE_MEASUREMENT_METHOD,
            "baseline_bracket_for_mode": spec["mode"],
            "correct": bool(correct),
            "victim_isolated_us": baseline * 1_000.0,
            "victim_slowdown_pct": (concurrent / baseline - 1.0) * 100.0,
        })
    return reduced


def build_adaptive_round_accounting(
    plan_payload: dict[str, int],
    controller_rows: list[dict[str, Any]],
    before_cells: dict[str, dict[str, Any]],
    after_cells: dict[str, dict[str, Any]],
    controller_config: AdaptiveBenchmarkControllerConfig,
) -> dict[str, Any]:
    """Reconcile one adaptive GPU batch and label every completed attempt.

    The controller may hit a hard valid-sample or total-attempt cap partway
    through an already issued batch. Those completed tail attempts remain in
    the raw corpus for diagnostics, but they must be distinguishable from
    attempts ingested by the controller and from valid samples used by its CI.
    """

    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in controller_rows:
        key = AdaptiveBenchmarkCellKey.from_row(row).canonical
        rows_by_cell.setdefault(key, []).append(row)
    unknown = sorted(set(rows_by_cell) - set(plan_payload))
    if unknown:
        raise RuntimeError(
            "adaptive batch completed unplanned cells: " + ", ".join(unknown)
        )

    per_cell: dict[str, dict[str, int | bool]] = {}
    dispositions: list[dict[str, Any]] = []
    for cell_key in sorted(plan_payload):
        if cell_key not in before_cells or cell_key not in after_cells:
            raise RuntimeError(
                f"adaptive accounting missing controller state for {cell_key}"
            )
        issued = int(plan_payload[cell_key])
        members = rows_by_cell.get(cell_key, [])
        completed = len(members)
        ingested = (
            int(after_cells[cell_key]["total_attempts"])
            - int(before_cells[cell_key]["total_attempts"])
        )
        discarded = (
            int(after_cells[cell_key]["discarded_after_stop"])
            - int(before_cells[cell_key]["discarded_after_stop"])
        )
        reconciled = issued == completed == ingested + discarded
        if not reconciled:
            raise RuntimeError(
                "adaptive attempt reconciliation failed for "
                f"{cell_key}: issued={issued}, completed={completed}, "
                f"ingested={ingested}, discarded_after_stop={discarded}"
            )
        per_cell[cell_key] = {
            "issued": issued,
            "completed": completed,
            "ingested": ingested,
            "discarded_after_stop": discarded,
            "reconciled": True,
        }
        for ordinal, row in enumerate(members):
            if ordinal < ingested:
                validity = classify_aggregate_trial(row, controller_config)
                disposition = (
                    "ingested_valid" if validity.valid else "ingested_invalid"
                )
                invalid_reasons = list(validity.reasons)
                valid_sample: bool | None = bool(validity.valid)
            else:
                disposition = "discarded_after_stop"
                invalid_reasons = []
                valid_sample = None
            dispositions.append({
                "cell_key": cell_key,
                "sample_index": int(row["sample_index"]),
                "disposition": disposition,
                "controller_ingested": ordinal < ingested,
                "controller_valid_sample": valid_sample,
                "invalid_reasons": invalid_reasons,
            })

    totals = {
        field: sum(int(row[field]) for row in per_cell.values())
        for field in (
            "issued", "completed", "ingested", "discarded_after_stop"
        )
    }
    totals["reconciled"] = bool(
        totals["issued"]
        == totals["completed"]
        == totals["ingested"] + totals["discarded_after_stop"]
    )
    return {
        "per_cell": per_cell,
        "totals": totals,
        "sample_dispositions": dispositions,
    }


def annotate_adaptive_aggregate_samples(
    samples: list[dict[str, Any]],
    adaptive_diagnostics: dict[str, Any] | None,
) -> None:
    """Attach controller retention/validity state to adaptive raw samples."""

    if adaptive_diagnostics is None:
        return
    disposition_rows = adaptive_diagnostics.get("sample_dispositions")
    # Backward compatibility for pre-accounting adaptive documents.
    if disposition_rows is None:
        return
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for row in disposition_rows:
        identity = (str(row["cell_key"]), int(row["sample_index"]))
        if identity in by_identity:
            raise RuntimeError(
                "duplicate adaptive sample disposition for "
                f"{identity[0]} sample {identity[1]}"
            )
        by_identity[identity] = row
    observed: set[tuple[str, int]] = set()
    for sample in samples:
        identity = (
            AdaptiveBenchmarkCellKey.from_row(sample).canonical,
            int(sample["sample_index"]),
        )
        disposition = by_identity.get(identity)
        if disposition is None:
            raise RuntimeError(
                "missing adaptive sample disposition for "
                f"{identity[0]} sample {identity[1]}"
            )
        observed.add(identity)
        sample.update({
            "adaptive_controller_disposition": disposition["disposition"],
            "adaptive_controller_ingested": bool(
                disposition["controller_ingested"]
            ),
            "adaptive_controller_valid_sample": disposition[
                "controller_valid_sample"
            ],
            "adaptive_controller_invalid_reasons": list(
                disposition["invalid_reasons"]
            ),
        })
    extra = sorted(set(by_identity) - observed)
    if extra:
        raise RuntimeError(
            "adaptive diagnostics contain dispositions without aggregate "
            "samples: "
            + ", ".join(f"{key}@{index}" for key, index in extra[:8])
        )


RUNTIME_HEALTH_NUMERIC_FIELDS = (
    "sm_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "gpu_utilization_pct",
    "throttle_reason_mask",
    "total_memory_bytes",
    "used_memory_bytes",
    "free_memory_bytes",
    "ecc_volatile_corrected",
    "ecc_volatile_uncorrected",
    "ecc_aggregate_corrected",
    "ecc_aggregate_uncorrected",
    "pcie_replay_counter",
    "memory_utilization_pct",
)


def parse_assigned_gpu_index(gpu_ids: list[Any]) -> int | None:
    """Return a physical NVML index for Ray's common integer GPU identifiers."""
    if len(gpu_ids) != 1:
        return None
    raw = str(gpu_ids[0])
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if str(value) == raw else None


def unavailable_runtime_health_record(
    *,
    phase: str,
    global_rank: int,
    node_ip: str,
    assigned_gpu_index: int | None,
    error: str,
) -> dict[str, Any]:
    timestamp_ns = time.time_ns()
    return {
        "phase": phase,
        "status": "unavailable",
        "global_rank": int(global_rank),
        "node_ip": node_ip,
        "assigned_gpu_index": assigned_gpu_index,
        "timestamp_ns": timestamp_ns,
        "timestamp_utc": datetime.fromtimestamp(
            timestamp_ns / 1e9, timezone.utc
        ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "error": error,
        "provider": None,
        "device": None,
    }


def capture_actor_runtime_health(
    collector: Any,
    *,
    phase: str,
    global_rank: int,
    node_ip: str,
    assigned_gpu_index: int | None,
) -> dict[str, Any]:
    """Capture one actor's assigned GPU outside benchmark timing intervals."""
    try:
        indices = None if assigned_gpu_index is None else (assigned_gpu_index,)
        snapshot = collector.collect(device_indices=indices)
        if len(snapshot.devices) != 1:
            raise RuntimeError(
                "runtime-health selection returned "
                f"{len(snapshot.devices)} devices; expected exactly one"
            )
        return {
            "phase": phase,
            "status": "ok",
            "global_rank": int(global_rank),
            "node_ip": node_ip,
            "assigned_gpu_index": assigned_gpu_index,
            "timestamp_ns": snapshot.timestamp_ns,
            "timestamp_utc": snapshot.timestamp_utc,
            "error": None,
            "provider": snapshot.provider,
            "device": snapshot.devices[0].to_dict(),
        }
    except Exception as error:
        return unavailable_runtime_health_record(
            phase=phase,
            global_rank=global_rank,
            node_ip=node_ip,
            assigned_gpu_index=assigned_gpu_index,
            error=f"{type(error).__name__}: {error}",
        )


def aggregate_runtime_health_audit(
    rank_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build compact cross-rank pre/post health summaries for one case."""
    records = [
        result.get("runtime_health", {})
        for result in sorted(rank_results, key=lambda row: int(row["global_rank"]))
    ]
    phase_records = {
        phase: [record.get(phase, {}) for record in records]
        for phase in ("pre_run", "post_run")
    }
    status_counts = {
        phase: dict(sorted(Counter(
            str(record.get("status", "missing"))
            for record in phase_records[phase]
        ).items()))
        for phase in phase_records
    }
    provider_counts = {
        phase: dict(sorted(Counter(
            str(record.get("provider") or "unavailable")
            for record in phase_records[phase]
        ).items()))
        for phase in phase_records
    }
    metric_summaries: dict[str, Any] = {}
    for metric in RUNTIME_HEALTH_NUMERIC_FIELDS:
        values_by_phase: dict[str, list[float]] = {}
        status_by_phase: dict[str, dict[str, int]] = {}
        for phase, observations in phase_records.items():
            values = []
            statuses: Counter[str] = Counter()
            for observation in observations:
                device = observation.get("device")
                if not isinstance(device, dict):
                    statuses[str(observation.get("status", "missing"))] += 1
                    continue
                if metric == "memory_utilization_pct":
                    status = (
                        "ok"
                        if device.get(metric) is not None
                        else "unsupported"
                    )
                else:
                    status = str(
                        device.get("metric_status", {}).get(metric, "missing")
                    )
                statuses[status] += 1
                if status == "ok" and device.get(metric) is not None:
                    values.append(float(device[metric]))
            values_by_phase[phase] = values
            status_by_phase[phase] = dict(sorted(statuses.items()))
        paired_changes = []
        for record in records:
            before = record.get("pre_run", {}).get("device")
            after = record.get("post_run", {}).get("device")
            if not isinstance(before, dict) or not isinstance(after, dict):
                continue
            before_value, after_value = before.get(metric), after.get(metric)
            if before_value is not None and after_value is not None:
                paired_changes.append(float(after_value) - float(before_value))
        metric_summaries[metric] = {
            "pre_run": summarize(values_by_phase["pre_run"]),
            "post_run": summarize(values_by_phase["post_run"]),
            "paired_post_minus_pre": summarize(paired_changes),
            "status_counts": status_by_phase,
        }

    throttle_counts = {}
    for phase, observations in phase_records.items():
        reasons: Counter[str] = Counter()
        for observation in observations:
            device = observation.get("device")
            if isinstance(device, dict):
                reasons.update(device.get("throttle_reasons") or ())
        throttle_counts[phase] = dict(sorted(reasons.items()))

    gpu_uuids = sorted({
        str(record[phase]["device"]["gpu_uuid"])
        for record in records
        for phase in ("pre_run", "post_run")
        if isinstance(record.get(phase, {}).get("device"), dict)
        and record[phase]["device"].get("gpu_uuid") is not None
    })
    complete_actor_count = sum(
        record.get("pre_run", {}).get("status") == "ok"
        and record.get("post_run", {}).get("status") == "ok"
        for record in records
    )
    return {
        "schema_version": 1,
        "actor_count": len(records),
        "complete_actor_count": complete_actor_count,
        "all_actor_pairs_complete": complete_actor_count == len(records),
        "status_counts": status_counts,
        "provider_counts": provider_counts,
        "unique_gpu_uuids": gpu_uuids,
        "unique_gpu_count": len(gpu_uuids),
        "metric_summaries": metric_summaries,
        "throttle_reason_counts": throttle_counts,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def fixed_sample_indices(numel: int, limit: int = 9) -> list[int]:
    """Return deterministic, evenly distributed flat indices without scanning data."""
    if numel < 0:
        raise ValueError("numel must be non-negative")
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    if numel <= limit:
        return list(range(numel))
    return sorted({
        (sample * (numel - 1)) // (limit - 1)
        for sample in range(limit)
    })


def flat_index_coordinate(flat_index: int, shape: list[int]) -> list[int]:
    """Convert a row-major flat index to a coordinate using pure arithmetic."""
    numel = math.prod(shape)
    if flat_index < 0 or flat_index >= numel:
        raise IndexError(f"flat index {flat_index} outside tensor with {numel} elements")
    coordinate = [0] * len(shape)
    remainder = flat_index
    for dimension in range(len(shape) - 1, -1, -1):
        width = shape[dimension]
        if width <= 0:
            raise ValueError(f"invalid tensor shape {shape}")
        coordinate[dimension] = remainder % width
        remainder //= width
    return coordinate


def _scalar_json_value(value: Any) -> bool | int | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "__float__"):
        return float(value)
    raise TypeError(f"unsupported output scalar type {type(value)!r}")


def compact_output_signature(value: Any, sample_limit: int = 9) -> dict[str, Any]:
    """Sample a compute result recursively without copying the full output."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": [
                {"key": str(key), "value": compact_output_signature(value[key], sample_limit)}
                for key in sorted(value, key=str)
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "sequence_type": type(value).__name__,
            "items": [compact_output_signature(item, sample_limit) for item in value],
        }
    if not all(hasattr(value, attribute) for attribute in ("shape", "dtype", "numel")):
        scalar = _scalar_json_value(value)
        return {
            "kind": "scalar",
            "dtype": type(value).__name__,
            "finite": not isinstance(scalar, float) or math.isfinite(scalar),
            "value": scalar,
        }

    shape = [int(dimension) for dimension in value.shape]
    numel = int(value.numel())
    samples = []
    finite = True
    for flat_index in fixed_sample_indices(numel, sample_limit):
        coordinate = flat_index_coordinate(flat_index, shape)
        scalar_tensor = value[tuple(coordinate)] if coordinate else value
        scalar = _scalar_json_value(scalar_tensor.item())
        scalar_finite = not isinstance(scalar, float) or math.isfinite(scalar)
        finite = finite and scalar_finite
        samples.append({
            "flat_index": flat_index,
            "coordinate": coordinate,
            "value": scalar,
            "finite": scalar_finite,
        })
    return {
        "kind": "tensor",
        "shape": shape,
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": numel,
        "finite": finite,
        "samples": samples,
    }


def compare_output_signatures(
    reference: dict[str, Any],
    observed: dict[str, Any],
    *,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Compare compact recursive signatures and report each validation facet."""
    checks = {
        "kind": reference.get("kind") == observed.get("kind"),
        "shape": True,
        "dtype": True,
        "finite": True,
        "sample_values": True,
        "structure": True,
    }
    mismatches: list[str] = []

    def fail(check: str, path: str) -> None:
        checks[check] = False
        mismatches.append(path)

    def visit(expected: dict[str, Any], actual: dict[str, Any], path: str) -> None:
        if expected.get("kind") != actual.get("kind"):
            fail("kind", f"{path}:kind")
            return
        kind = expected.get("kind")
        if kind == "none":
            return
        if kind == "mapping":
            expected_items = expected.get("items", [])
            actual_items = actual.get("items", [])
            if [item.get("key") for item in expected_items] != [
                item.get("key") for item in actual_items
            ]:
                fail("structure", f"{path}:mapping_keys")
                return
            for expected_item, actual_item in zip(expected_items, actual_items):
                visit(
                    expected_item["value"], actual_item["value"],
                    f"{path}.{expected_item['key']}",
                )
            return
        if kind == "sequence":
            expected_items = expected.get("items", [])
            actual_items = actual.get("items", [])
            if (
                expected.get("sequence_type") != actual.get("sequence_type")
                or len(expected_items) != len(actual_items)
            ):
                fail("structure", f"{path}:sequence")
                return
            for index, (expected_item, actual_item) in enumerate(
                zip(expected_items, actual_items)
            ):
                visit(expected_item, actual_item, f"{path}[{index}]")
            return
        if kind == "scalar":
            if expected.get("dtype") != actual.get("dtype"):
                fail("dtype", f"{path}:dtype")
            expected_finite = bool(expected.get("finite"))
            actual_finite = bool(actual.get("finite"))
            if not expected_finite or not actual_finite:
                fail("finite", f"{path}:finite")
                return
            if not math.isclose(
                float(expected.get("value")), float(actual.get("value")),
                rel_tol=relative_tolerance, abs_tol=absolute_tolerance,
            ):
                fail("sample_values", f"{path}:value")
            return
        if kind != "tensor":
            fail("structure", f"{path}:unknown_kind")
            return
        if expected.get("shape") != actual.get("shape") or (
            expected.get("numel") != actual.get("numel")
        ):
            fail("shape", f"{path}:shape")
        if expected.get("dtype") != actual.get("dtype"):
            fail("dtype", f"{path}:dtype")
        if not bool(expected.get("finite")) or not bool(actual.get("finite")):
            fail("finite", f"{path}:finite")
        expected_samples = expected.get("samples", [])
        actual_samples = actual.get("samples", [])
        expected_locations = [sample.get("flat_index") for sample in expected_samples]
        actual_locations = [sample.get("flat_index") for sample in actual_samples]
        if expected_locations != actual_locations:
            fail("structure", f"{path}:sample_locations")
            return
        for expected_sample, actual_sample in zip(expected_samples, actual_samples):
            sample_path = f"{path}[{expected_sample.get('flat_index')}]"
            if not bool(expected_sample.get("finite")) or not bool(actual_sample.get("finite")):
                fail("finite", f"{sample_path}:finite")
                continue
            if not math.isclose(
                float(expected_sample["value"]), float(actual_sample["value"]),
                rel_tol=relative_tolerance, abs_tol=absolute_tolerance,
            ):
                fail("sample_values", f"{sample_path}:value")

    visit(reference, observed, "output")
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mismatches": mismatches,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }



def grouped_gemm_prepare_semantics(
    participants: list[dict[str, Any]], operation_index: int,
) -> dict[str, Any]:
    """Tie a zero-work prepare marker to the adjacent grouped-MM launch(es)."""
    grouped_aliases = {
        "K-28e33c0c27e1",
        "K-49abe359818e",
        "K-7e2ed2672834",
    }
    adjacent = []
    for candidate in (operation_index - 1, operation_index + 1):
        if candidate < 0 or candidate >= len(participants):
            continue
        participant = participants[candidate]
        if (
            participant.get("kernel_alias") in grouped_aliases
            and participant.get("stream_role") == "compute/default"
        ):
            adjacent.append({
                "operation_index": candidate,
                "participant_id": participant.get("participant_id"),
                "kernel_alias": participant.get("kernel_alias"),
            })
    return {
        "passed": bool(adjacent),
        "validation_status": (
            "validated_by_adjacent_grouped_mm_semantics"
            if adjacent else "missing_adjacent_grouped_mm"
        ),
        "adjacent_grouped_mm_operations": adjacent,
    }


def _interval_union_coverage_pct(
    victim_index: int,
    aggressor_indices: list[int],
    starts_ms: dict[int, float],
    durations_ms: dict[int, float],
) -> float:
    """Predict victim coverage from operation start and isolated durations."""
    victim_left = starts_ms[victim_index]
    victim_right = victim_left + durations_ms[victim_index]
    covered = []
    for index in aggressor_indices:
        left = max(victim_left, starts_ms[index])
        right = min(victim_right, starts_ms[index] + durations_ms[index])
        if right > left:
            covered.append((left, right))
    merged: list[tuple[float, float]] = []
    for left, right in sorted(covered):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    overlap_ms = sum(right - left for left, right in merged)
    return min(100.0, overlap_ms / durations_ms[victim_index] * 100.0)



def _union_overlap_ns(
    operation_intervals_ns: dict[int, tuple[int, int]],
    spec: dict[str, Any],
) -> int:
    """Return victim coverage by the union of aggressor intervals."""

    victim = int(spec["victim_index"])
    victim_left, victim_right = operation_intervals_ns[victim]
    covered: list[tuple[int, int]] = []
    for aggressor in spec["aggressor_indices"]:
        other_left, other_right = operation_intervals_ns[int(aggressor)]
        left = max(victim_left, other_left)
        right = min(victim_right, other_right)
        if right > left:
            covered.append((left, right))
    merged: list[tuple[int, int]] = []
    for left, right in sorted(covered):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return sum(right - left for left, right in merged)


def _per_aggressor_overlap_metrics_ns(
    operation_intervals_ns: dict[int, tuple[int, int]],
    counterfactual_intervals_ns: dict[int, tuple[int, int]],
    spec: dict[str, Any],
    isolated_victim_ms: float,
) -> list[dict[str, Any]]:
    """Report outcome and pre-treatment coverage for each aggressor."""

    victim = int(spec["victim_index"])
    actual_victim = operation_intervals_ns[victim]
    scheduled_victim = counterfactual_intervals_ns[victim]
    metrics = []
    for aggressor_value in spec["aggressor_indices"]:
        aggressor = int(aggressor_value)
        actual = operation_intervals_ns[aggressor]
        scheduled = counterfactual_intervals_ns[aggressor]
        actual_intersection_ms = max(
            0,
            min(actual_victim[1], actual[1])
            - max(actual_victim[0], actual[0]),
        ) / 1e6
        scheduled_intersection_ms = max(
            0,
            min(scheduled_victim[1], scheduled[1])
            - max(scheduled_victim[0], scheduled[0]),
        ) / 1e6
        actual_coverage_pct = min(
            100.0, actual_intersection_ms / isolated_victim_ms * 100.0
        )
        scheduled_coverage_pct = min(
            100.0, scheduled_intersection_ms / isolated_victim_ms * 100.0
        )
        metrics.append({
            "operation_index": aggressor,
            "intersection_ms": actual_intersection_ms,
            "victim_coverage_pct": actual_coverage_pct,
            "scheduled_intersection_ms": scheduled_intersection_ms,
            "scheduled_victim_coverage_pct": scheduled_coverage_pct,
            "counterfactual_intersection_ms": scheduled_intersection_ms,
            "counterfactual_victim_coverage_pct": scheduled_coverage_pct,
        })
    return metrics


def measure_runtime_geometry(
    rows: list[dict[str, Any]],
    participants: list[dict[str, Any]],
    spec: dict[str, Any],
    isolated_victim_ms: dict[int, float] | float,
) -> dict[str, Any]:
    """Measure outcome and pre-treatment geometry on the global clock.

    New callers pass a matched isolated duration for every operation. Those
    durations are joined to the treatment's actual globally aligned GPU starts
    to form counterfactual intervals. Concurrent GPU ends remain observable
    outcomes and deliberately do not define calibration or sample validity.

    A scalar victim duration is accepted only for compatibility with callers
    that predate counterfactual geometry; in that legacy mode the scheduled
    aliases equal concurrent geometry and the provenance says so explicitly.
    """

    isolated_durations_ms = isolated_victim_ms
    legacy_scalar_duration = not isinstance(isolated_durations_ms, dict)
    if legacy_scalar_duration:
        isolated_victim_ms = float(isolated_durations_ms)
        matched_durations_ms: dict[int, float] | None = None
    else:
        matched_durations_ms = {
            int(index): float(duration)
            for index, duration in isolated_durations_ms.items()
        }
        expected = set(range(len(participants)))
        if set(matched_durations_ms) != expected:
            raise ValueError(
                "counterfactual geometry requires one matched isolated "
                "duration per operation"
            )
        if any(duration <= 0.0 for duration in matched_durations_ms.values()):
            raise ValueError("matched isolated durations must be positive")
        isolated_victim_ms = matched_durations_ms[int(spec["victim_index"])]
    if isolated_victim_ms <= 0.0:
        raise ValueError("isolated victim duration must be positive")
    operation_intervals_ns: dict[int, tuple[int, int]] = {}
    for index, participant in enumerate(participants):
        focal_rank = (
            int(participant["focal_dp_rank"]) * 2
            + int(participant["focal_pp_rank"])
        )
        matches = [
            row for row in rows
            if int(row["operation_index"]) == index
            and int(row["global_rank"]) == focal_rank
        ]
        if len(matches) != 1:
            raise ValueError(
                f"operation {index}: expected one focal interval on rank "
                f"{focal_rank}, got {len(matches)}"
            )
        left = int(matches[0]["gpu_start_global_ns"])
        right = int(matches[0]["gpu_end_global_ns"])
        if right < left:
            raise ValueError(f"operation {index}: GPU interval ends before it starts")
        operation_intervals_ns[index] = (left, right)

    victim = int(spec["victim_index"])
    victim_left, victim_right = operation_intervals_ns[victim]
    union_overlap_ms = _union_overlap_ns(operation_intervals_ns, spec) / 1e6
    realized_overlap_pct = min(
        100.0, union_overlap_ms / isolated_victim_ms * 100.0
    )
    requested_overlap_pct = float(spec["requested_overlap_pct"])
    if matched_durations_ms is None:
        counterfactual_intervals_ns = dict(operation_intervals_ns)
        geometry_validity_basis = "legacy_concurrent_intervals_fallback"
    else:
        counterfactual_intervals_ns = {
            index: (
                left,
                left + int(round(matched_durations_ms[index] * 1e6)),
            )
            for index, (left, _right) in operation_intervals_ns.items()
        }
        geometry_validity_basis = GEOMETRY_VALIDITY_BASIS
    counterfactual_overlap_ms = (
        _union_overlap_ns(counterfactual_intervals_ns, spec) / 1e6
    )
    counterfactual_overlap_pct = min(
        100.0, counterfactual_overlap_ms / isolated_victim_ms * 100.0
    )
    counterfactual_error_pct = (
        counterfactual_overlap_pct - requested_overlap_pct
    )
    per_aggressor_overlap = _per_aggressor_overlap_metrics_ns(
        operation_intervals_ns,
        counterfactual_intervals_ns,
        spec,
        float(isolated_victim_ms),
    )
    return {
        # Backward-compatible concurrent/outcome fields.
        "union_overlap_ms": union_overlap_ms,
        "realized_overlap_pct": realized_overlap_pct,
        "geometry_error_pct": realized_overlap_pct - requested_overlap_pct,
        "concurrent_overlap_ms": union_overlap_ms,
        "concurrent_geometry_error_pct": (
            realized_overlap_pct - requested_overlap_pct
        ),
        # Explicit pre-treatment validity fields. Scheduled and
        # counterfactual names are intentional aliases for downstream clarity.
        "scheduled_overlap_ms": counterfactual_overlap_ms,
        "scheduled_overlap_pct": counterfactual_overlap_pct,
        "scheduled_geometry_error_pct": counterfactual_error_pct,
        "counterfactual_overlap_ms": counterfactual_overlap_ms,
        "counterfactual_overlap_pct": counterfactual_overlap_pct,
        "counterfactual_geometry_error_pct": counterfactual_error_pct,
        "geometry_validity_error_pct": counterfactual_error_pct,
        "geometry_validity_basis": geometry_validity_basis,
        "counterfactual_geometry_available": matched_durations_ms is not None,
        "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
        "victim_concurrent_duration_ms": (victim_right - victim_left) / 1e6,
        "operation_intervals_global_ns": {
            str(index): {"start": left, "end": right}
            for index, (left, right) in operation_intervals_ns.items()
        },
        "counterfactual_operation_intervals_global_ns": {
            str(index): {"start": left, "end": right}
            for index, (left, right) in counterfactual_intervals_ns.items()
        },
        "per_aggressor_overlap": per_aggressor_overlap,
    }


def relative_shift_bounds_ms(
    spec: dict[str, Any],
    base_delays_ms: dict[int, float],
    max_relative_shift_ms: float,
) -> tuple[float, float]:
    """Return legal bounds for moving only the later dispatch group."""
    if max_relative_shift_ms < 0.0:
        raise ValueError("maximum relative shift must be nonnegative")
    victim = int(spec["victim_index"])
    aggressors = [int(index) for index in spec["aggressor_indices"]]
    if not aggressors:
        raise ValueError("runtime geometry calibration requires an aggressor")
    if spec["dispatch_mode"] == "victim_first":
        initial_gap_ms = (
            min(base_delays_ms[index] for index in aggressors)
            - base_delays_ms[victim]
        )
    elif spec["dispatch_mode"] == "aggressor_first":
        initial_gap_ms = base_delays_ms[victim] - max(
            base_delays_ms[index] for index in aggressors
        )
    else:
        raise ValueError(f"unsupported dispatch mode: {spec['dispatch_mode']}")
    if initial_gap_ms < -1e-7:
        raise ValueError("base delays do not preserve declared dispatch order")
    return -max(0.0, initial_gap_ms), max_relative_shift_ms


def apply_relative_geometry_shift(
    spec: dict[str, Any],
    base_delays_ms: dict[int, float],
    relative_shift_ms: float,
) -> dict[int, float]:
    """Apply an order-preserving victim/aggressor-group relative shift."""
    delays = {int(index): float(value) for index, value in base_delays_ms.items()}
    victim = int(spec["victim_index"])
    aggressors = [int(index) for index in spec["aggressor_indices"]]
    if spec["dispatch_mode"] == "victim_first":
        for index in aggressors:
            delays[index] += relative_shift_ms
    elif spec["dispatch_mode"] == "aggressor_first":
        delays[victim] += relative_shift_ms
    else:
        raise ValueError(f"unsupported dispatch mode: {spec['dispatch_mode']}")
    ordered_delays = [delays[int(index)] for index in spec["order"]]
    if any(
        right + 1e-7 < left
        for left, right in zip(ordered_delays, ordered_delays[1:])
    ):
        raise ValueError("relative shift violates declared dispatch order")
    if min(delays.values()) < -1e-7:
        raise ValueError("relative shift produces a negative host delay")
    return delays


def choose_geometry_feedback_shift(
    history: list[dict[str, float]],
    requested_overlap_pct: float,
    isolated_victim_ms: float,
    minimum_shift_ms: float,
    maximum_shift_ms: float,
    tolerance_pct: float = GEOMETRY_TOLERANCE_PCT,
) -> dict[str, Any]:
    """Choose the next bounded delay using proportional feedback or bisection.

    Counterfactual overlap time, rather than concurrent duration, prevents
    interference from steering its own treatment. Legacy history rows without
    scheduled fields retain their prior behavior deliberately.
    """
    if not history:
        raise ValueError("geometry feedback requires at least one measurement")
    if isolated_victim_ms <= 0.0:
        raise ValueError("isolated victim duration must be positive")
    if minimum_shift_ms > maximum_shift_ms:
        raise ValueError("invalid relative-shift bounds")
    latest = history[-1]
    latest_error_pct = float(
        latest.get("scheduled_geometry_error_pct", latest["geometry_error_pct"])
    )
    current_shift_ms = float(latest["relative_shift_ms"])
    if abs(latest_error_pct) <= tolerance_pct:
        return {
            "converged": True,
            "next_shift_ms": current_shift_ms,
            "method": "within_tolerance",
            "stalled": False,
        }

    high = [
        row for row in history
        if float(row.get(
            "scheduled_geometry_error_pct", row["geometry_error_pct"]
        )) > tolerance_pct
    ]
    low = [
        row for row in history
        if float(row.get(
            "scheduled_geometry_error_pct", row["geometry_error_pct"]
        )) < -tolerance_pct
    ]
    ordered_brackets = [
        (above, below)
        for above in high
        for below in low
        if float(above["relative_shift_ms"]) < float(below["relative_shift_ms"])
    ]
    if ordered_brackets:
        above, below = min(
            ordered_brackets,
            key=lambda pair: (
                float(pair[1]["relative_shift_ms"])
                - float(pair[0]["relative_shift_ms"])
            ),
        )
        candidate_ms = (
            float(above["relative_shift_ms"])
            + float(below["relative_shift_ms"])
        ) / 2.0
        method = "bracket_bisection"
    else:
        if latest.get("scheduled_overlap_error_ms") is not None:
            overlap_error_ms = float(latest["scheduled_overlap_error_ms"])
        elif latest.get("overlap_error_ms") is not None:
            overlap_error_ms = float(latest["overlap_error_ms"])
        else:
            target_overlap_ms = (
                requested_overlap_pct / 100.0 * isolated_victim_ms
            )
            measured_overlap_ms = float(latest["union_overlap_ms"])
            overlap_error_ms = measured_overlap_ms - target_overlap_ms
        candidate_ms = current_shift_ms + overlap_error_ms
        method = "overlap_time_feedback"
    candidate_ms = min(maximum_shift_ms, max(minimum_shift_ms, candidate_ms))
    stalled = math.isclose(candidate_ms, current_shift_ms, abs_tol=1e-6)
    return {
        "converged": False,
        "next_shift_ms": candidate_ms,
        "method": method,
        "stalled": stalled,
    }

def compensate_launch_latencies(
    desired_gpu_starts_ms: dict[int, float],
    launch_latencies_ms: dict[int, float],
    dispatch_order: list[int],
) -> dict[str, Any]:
    """Map desired GPU starts to legal host launch delays.

    A launch issued at host delay h is expected to reach the GPU at
    h + gate_to_start. Subtracting the isolated gate-to-start calibration
    removes this systematic bias. A common shift makes all delays nonnegative
    without changing overlap. Finally, an order projection prevents a later
    declared participant from being launched before an earlier one.
    """
    if set(dispatch_order) != set(desired_gpu_starts_ms):
        raise ValueError("dispatch order and desired starts must name the same operations")
    if set(dispatch_order) != set(launch_latencies_ms):
        raise ValueError("dispatch order and launch latencies must name the same operations")
    raw_delays = {
        index: desired_gpu_starts_ms[index] - launch_latencies_ms[index]
        for index in dispatch_order
    }
    origin_shift_ms = max(0.0, -min(raw_delays.values()))
    shifted_delays = {
        index: raw_delays[index] + origin_shift_ms
        for index in dispatch_order
    }
    host_delays: dict[int, float] = {}
    clamped_indices = []
    previous_delay = 0.0
    for position, index in enumerate(dispatch_order):
        legal_delay = max(0.0, shifted_delays[index])
        if position:
            legal_delay = max(previous_delay, legal_delay)
        if legal_delay > shifted_delays[index] + 1e-9:
            clamped_indices.append(index)
        host_delays[index] = legal_delay
        previous_delay = legal_delay
    predicted_gpu_starts = {
        index: host_delays[index] + launch_latencies_ms[index]
        for index in dispatch_order
    }
    return {
        "host_delays_ms": host_delays,
        "predicted_gpu_starts_ms": predicted_gpu_starts,
        "raw_host_delays_ms": raw_delays,
        "origin_shift_ms": origin_shift_ms,
        "order_clamped_indices": clamped_indices,
    }


def build_overlap_stage_specs(
    participants: list[dict[str, Any]],
    isolated_estimates_ms: dict[int, float],
    isolated_gate_to_start_ms: dict[int, float],
    coverage_targets: list[float],
) -> list[dict[str, Any]]:
    """Construct latency-compensated, order-constrained overlap stages."""
    validate_coverage_targets(coverage_targets)
    if any(duration <= 0.0 for duration in isolated_estimates_ms.values()):
        raise ValueError("isolated operation durations must be positive")
    if set(isolated_estimates_ms) != set(isolated_gate_to_start_ms):
        raise ValueError("duration and gate-to-start calibrations must match")
    if any(latency < 0.0 for latency in isolated_gate_to_start_ms.values()):
        raise ValueError("gate-to-start calibrations must be nonnegative")
    if participants and set(isolated_estimates_ms) != set(range(len(participants))):
        raise ValueError("calibrations must cover every participant")

    if any(item.get("role") == "victim" for item in participants):
        victim_indices = [
            index for index, item in enumerate(participants)
            if item.get("role") == "victim"
        ]
    else:
        victim_indices = list(range(len(participants)))

    stage_specs = []
    for target in coverage_targets:
        fraction = target / 100.0
        for victim in victim_indices:
            aggressors = [
                index for index in range(len(participants)) if index != victim
            ]
            victim_ms = isolated_estimates_ms[victim]
            all_natural_offsets = {
                index: float(
                    participants[index].get("natural_group_offset_ms", 0.0)
                )
                for index in range(len(participants))
            }
            natural_offsets = {
                index: all_natural_offsets[index]
                for index in aggressors
            }
            group_origin_ms = min(natural_offsets.values())
            ordered_aggressors = sorted(
                aggressors, key=lambda index: (natural_offsets[index], index)
            )
            aggressor_span_ms = max(
                natural_offsets[index] - group_origin_ms
                + isolated_estimates_ms[index]
                for index in aggressors
            )
            predicted_max_pct = min(100.0, aggressor_span_ms / victim_ms * 100.0)
            duration_feasible = target <= predicted_max_pct + 1e-6
            for dispatch_mode in ("victim_first", "aggressor_first"):
                if dispatch_mode == "victim_first":
                    base_delay = max(0.0, victim_ms * (1.0 - fraction))
                    desired_starts = {
                        index: base_delay + natural_offsets[index] - group_origin_ms
                        for index in aggressors
                    }
                    desired_starts[victim] = 0.0
                    order = [victim, *ordered_aggressors]
                else:
                    victim_delay = max(0.0, aggressor_span_ms - fraction * victim_ms)
                    desired_starts = {
                        index: natural_offsets[index] - group_origin_ms
                        for index in aggressors
                    }
                    desired_starts[victim] = victim_delay
                    order = [*ordered_aggressors, victim]
                compensation = compensate_launch_latencies(
                    desired_starts,
                    isolated_gate_to_start_ms,
                    order,
                )
                predicted_scheduled_pct = _interval_union_coverage_pct(
                    victim,
                    aggressors,
                    compensation["predicted_gpu_starts_ms"],
                    isolated_estimates_ms,
                )
                constraint_error_pct = predicted_scheduled_pct - target
                projection_feasible = (
                    abs(constraint_error_pct) <= GEOMETRY_TOLERANCE_PCT
                )
                constraints_feasible = (
                    duration_feasible and projection_feasible
                )
                mode = f"coverage_{target:g}_v{victim}_{dispatch_mode}"
                stage_specs.append({
                    "mode": mode,
                    "victim_index": victim,
                    "aggressor_indices": aggressors,
                    "dispatch_mode": dispatch_mode,
                    "requested_overlap_pct": target,
                    "predicted_max_overlap_pct": predicted_max_pct,
                    "isolated_estimates_ms": dict(isolated_estimates_ms),
                    "natural_group_offsets_ms": all_natural_offsets,
                    "aggressor_span_ms": aggressor_span_ms,
                    "duration_feasibility_tolerance_pct_points": 1e-6,
                    "duration_feasibility_formula": (
                        "min(100, max(offset_i-min_aggressor_offset+"
                        "isolated_duration_i)/victim_isolated_duration*100)"
                    ),
                    "predicted_scheduled_overlap_pct": predicted_scheduled_pct,
                    "predicted_constraint_error_pct": constraint_error_pct,
                    "target_feasible_from_duration": duration_feasible,
                    "dispatch_order_projection_feasible": projection_feasible,
                    "target_feasible_after_constraints": constraints_feasible,
                    "target_feasible_from_isolated": constraints_feasible,
                    "order": order,
                    "delays_ms": compensation["host_delays_ms"],
                    "desired_gpu_starts_ms": desired_starts,
                    "isolated_gate_to_start_ms": dict(isolated_gate_to_start_ms),
                    "predicted_gpu_starts_ms": compensation["predicted_gpu_starts_ms"],
                    "unconstrained_host_delays_ms": compensation["raw_host_delays_ms"],
                    "host_delay_origin_shift_ms": compensation["origin_shift_ms"],
                    "order_clamped_indices": compensation["order_clamped_indices"],
                })
    return stage_specs


def parse_event_id(event_id: str) -> tuple[int, int, int]:
    fields = event_id.split(".")
    if len(fields) < 4:
        raise ValueError(event_id)
    return int(fields[0][1:]), int(fields[1][1:]), int(fields[2][1:])


def _participant_id(participant: dict[str, Any]) -> str:
    return f"{participant['event_id']}:{participant['kernel_alias']}"


def load_cases(
    matrix_path: Path,
    requested: list[str],
) -> list[dict[str, Any]]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    result = []
    for token in requested:
        case_id, _, variant = token.partition(":")
        source_key = "multi_assignments" if case_id.startswith("M") else "pair_assignments"
        candidates = [row for row in matrix[source_key] if row["case_id"] == case_id]
        if variant:
            candidates = [row for row in candidates if row.get("variant") == variant]
        elif len(candidates) > 1:
            raise ValueError(f"{case_id} requires an explicit variant")
        if len(candidates) != 1:
            raise ValueError(f"{token}: expected one matrix assignment, got {len(candidates)}")
        assignment = dict(candidates[0])
        assignment["case_key"] = token
        participants = []
        p2p_slot = 0
        for participant in assignment["participants"]:
            item = dict(participant)
            dp_rank, pp_rank, iteration = parse_event_id(item["event_id"])
            item.update(
                {
                    "participant_id": _participant_id(item),
                    "focal_dp_rank": dp_rank,
                    "focal_pp_rank": pp_rank,
                    "profiler_iteration": iteration,
                }
            )
            if item["category"] in {"p2p_send", "p2p_recv"}:
                item["p2p_slot"] = p2p_slot
                p2p_slot += 1
            participants.append(item)
        assignment["participants"] = participants
        result.append(assignment)
    return result


def attach_trace_start_offsets(
    assignments: list[dict[str, Any]], events_path: Path,
) -> None:
    wanted = {
        item["event_id"]
        for assignment in assignments
        if assignment["case_id"].startswith("M")
        for item in assignment["participants"]
    }
    starts = {}
    if wanted:
        csv.field_size_limit(sys.maxsize)
        with events_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                event_id = row["event_id"]
                if event_id in wanted:
                    starts[event_id] = float(row["start_us"])
                    if len(starts) == len(wanted):
                        break
        missing = sorted(wanted - starts.keys())
        if missing:
            raise ValueError(f"missing multi-aggressor trace events: {missing}")
    for assignment in assignments:
        if not assignment["case_id"].startswith("M"):
            for item in assignment["participants"]:
                item["natural_group_offset_ms"] = 0.0
            continue
        aggressors = [
            item for item in assignment["participants"]
            if item.get("role") == "aggressor"
        ]
        origin = min(starts[item["event_id"]] for item in aggressors)
        for item in assignment["participants"]:
            item["natural_group_offset_ms"] = (
                (starts[item["event_id"]] - origin) / 1000.0
                if item.get("role") == "aggressor" else 0.0
            )


def communicator_plan(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    dp_pp = sorted({
        int(item["focal_pp_rank"])
        for assignment in assignments for item in assignment["participants"]
        if item["category"] in {"ag", "rs"}
    })
    ep_pp = sorted({
        int(item["focal_pp_rank"])
        for assignment in assignments for item in assignment["participants"]
        if item["category"] == "a2a"
    })
    p2p = sorted({
        (int(item["focal_dp_rank"]), int(item.get("p2p_slot", 0)))
        for assignment in assignments for item in assignment["participants"]
        if item["category"] in {"p2p_send", "p2p_recv"}
    })
    return {"dp_pp_ranks": dp_pp, "ep_pp_ranks": ep_pp, "p2p_groups": p2p}


def load_a2a_profiles(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {row["event_id"]: row for row in document["payloads"]}


def scale_split_matrix(profile: dict[str, Any], seq_len: int) -> dict[int, dict[str, Any]]:
    scale = seq_len / BASE_SEQ
    base = [row["input_split_rows"] for row in profile["ranks"]]
    sends = [[int(round(value * scale)) for value in row] for row in base]
    result = {}
    for source in range(8):
        input_rows = sends[source]
        output_rows = [sends[peer][source] for peer in range(8)]
        result[source] = {
            "input_split_rows": input_rows,
            "output_split_rows": output_rows,
            "input_numel": sum(input_rows) * MODEL_CONFIG["dim"],
            "output_numel": sum(output_rows) * MODEL_CONFIG["dim"],
        }
    return result


class PairwiseOverlapRank:
    """Fresh one-GPU process with all topology communicators."""

    def __init__(self, global_rank: int, expected_node_ip: str) -> None:
        self.environment_audit = sanitize_nccl_environment(
            phase=f"pairwise_worker_rank_{global_rank}_before_torch_import"
        )
        assert_default_nccl_environment()
        import ray
        import torch

        self.ray = ray
        self.torch = torch
        self.global_rank = global_rank
        self.expected_node_ip = expected_node_ip
        torch.cuda.set_device(0)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.node_ip = ray.util.get_node_ip_address()
        self.clock_offset_ns = 0
        self.clock_rtt_ns = None
        self.clock_calibration_generation = 0
        self.clock_calibration_current = None
        self.clock_calibration_history: list[dict[str, Any]] = []
        self.initialized = False
        self.assigned_gpu_index = parse_assigned_gpu_index(
            list(ray.get_gpu_ids())
        )
        self.runtime_health_collector = None
        self.runtime_health_initialization_error = None

    def metadata(self) -> dict[str, Any]:
        props = self.torch.cuda.get_device_properties(0)
        clock_metadata = (
            {}
            if self.clock_calibration_current is None
            else dict(self.clock_calibration_current)
        )
        return {
            "global_rank": self.global_rank,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "expected_node_ip": self.expected_node_ip,
            "hostname": socket.gethostname(),
            "ray_gpu_ids": [str(value) for value in self.ray.get_gpu_ids()],
            "runtime_health_assigned_nvml_index": self.assigned_gpu_index,
            "cuda_name": props.name,
            "cuda_total_memory": props.total_memory,
            "cuda_sm_count": props.multi_processor_count,
            "torch_version": self.torch.__version__,
            "cuda_version": self.torch.version.cuda,
            "nccl_version": self.torch.cuda.nccl.version(),
            "environment_audit": self.environment_audit,
            "effective_nccl_overrides": nccl_behavior_environment(),
            "retained_nccl_debug_variables": nccl_debug_environment(),
            "imported_worker_implementation_sha256": (
                implementation_sha256()
            ),
            "clock_offset_ns": self.clock_offset_ns,
            "clock_best_rtt_ns": self.clock_rtt_ns,
            **clock_metadata,
        }

    def _capture_runtime_health(self, phase: str) -> dict[str, Any]:
        if (
            self.runtime_health_collector is None
            and self.runtime_health_initialization_error is None
        ):
            try:
                from scripts.partial_overlap_interference_benchmarks.runtime_health import (
                    RuntimeHealthCollector,
                )

                self.runtime_health_collector = RuntimeHealthCollector().open()
            except Exception as error:
                self.runtime_health_initialization_error = (
                    f"{type(error).__name__}: {error}"
                )
        if self.runtime_health_collector is None:
            return unavailable_runtime_health_record(
                phase=phase,
                global_rank=self.global_rank,
                node_ip=self.node_ip,
                assigned_gpu_index=self.assigned_gpu_index,
                error=(
                    self.runtime_health_initialization_error
                    or "runtime-health collector unavailable"
                ),
            )
        return capture_actor_runtime_health(
            self.runtime_health_collector,
            phase=phase,
            global_rank=self.global_rank,
            node_ip=self.node_ip,
            assigned_gpu_index=self.assigned_gpu_index,
        )

    def rendezvous(self) -> tuple[str, int]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.node_ip, 0))
            return self.node_ip, sock.getsockname()[1]

    def initialize(
        self, master_addr: str, master_port: int, topology: dict[str, Any],
        comm_plan: dict[str, Any],
    ) -> None:
        import torch.distributed as dist

        assert_default_nccl_environment()
        topo = PPOuterTopology(
            node_ips=tuple(topology["node_ips"]),
            gpus_per_node=int(topology["gpus_per_node"]),
            physical_pp=int(topology["physical_pp"]),
            dp=int(topology["dp"]),
            ep=int(topology["ep"]),
        )
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=self.global_rank,
            world_size=topo.world_size,
            timeout=timedelta(minutes=30),
        )
        self.dp_groups = {}
        self.ep_groups = {}
        self.pp_groups = {}
        for pp_rank in comm_plan["dp_pp_ranks"]:
            members = topo.dp_groups[pp_rank]
            group = dist.new_group(list(members), backend="nccl")
            if self.global_rank in members:
                self.dp_groups[pp_rank] = group
        for pp_rank in comm_plan["ep_pp_ranks"]:
            members = topo.ep_groups[pp_rank]
            group = dist.new_group(list(members), backend="nccl")
            if self.global_rank in members:
                self.ep_groups[pp_rank] = group
        for dp_rank, slot in comm_plan["p2p_groups"]:
            members = topo.pp_pairs[dp_rank]
            group = dist.new_group(list(members), backend="nccl")
            if self.global_rank in members:
                self.pp_groups[(dp_rank, slot)] = group
        dist.barrier()
        self.initialized = True

    def set_clock_generation_floor(self, generation: int) -> None:
        if self.clock_calibration_current is not None:
            raise RuntimeError("clock generation floor must precede calibration")
        self.clock_calibration_generation = max(
            self.clock_calibration_generation, int(generation)
        )

    def calibrate_clock(
        self,
        clock_actor: Any,
        probes: int = 12,
        label: str = "benchmark_start",
    ) -> dict[str, Any]:
        """Refresh the global-clock mapping and retain its uncertainty."""

        best = None
        for _ in range(probes):
            before = time.monotonic_ns()
            remote = int(self.ray.get(clock_actor.now_ns.remote()))
            after = time.monotonic_ns()
            rtt = after - before
            offset = remote - ((before + after) // 2)
            if best is None or rtt < best[0]:
                best = (rtt, offset)
        assert best is not None
        self.clock_rtt_ns, self.clock_offset_ns = best
        self.clock_calibration_generation += 1
        local_ns = time.monotonic_ns()
        record = {
            "global_rank": self.global_rank,
            "clock_calibration_best_rtt_ns": int(best[0]),
            "clock_offset_ns": int(best[1]),
            "clock_calibration_generation": self.clock_calibration_generation,
            "clock_calibration_timestamp_global_ns": int(
                local_ns + self.clock_offset_ns
            ),
            "clock_calibration_local_monotonic_ns": int(local_ns),
            "clock_calibration_label": str(label),
            "clock_calibration_half_rtt_uncertainty_ns": int(
                math.ceil(best[0] / 2)
            ),
        }
        self.clock_calibration_current = dict(record)
        self.clock_calibration_history.append(dict(record))
        return record

    def _global_now_ns(self) -> int:
        return time.monotonic_ns() + self.clock_offset_ns

    def _clock_row_metadata(self) -> dict[str, Any]:
        if self.clock_calibration_current is None:
            raise RuntimeError("global clock has not been calibrated")
        return dict(self.clock_calibration_current)

    def current_clock_calibration(self) -> dict[str, Any]:
        if self.clock_calibration_current is None:
            raise RuntimeError("global clock has not been calibrated")
        return dict(self.clock_calibration_current)

    def _compute(self, participant: dict[str, Any], seq_len: int) -> dict[str, Any] | None:
        if self.global_rank != participant["focal_dp_rank"] * 2 + participant["focal_pp_rank"]:
            return None
        torch = self.torch
        alias = participant["kernel_alias"]
        compute_class = KERNEL_TO_COMPUTE.get(alias)
        if compute_class == "vocab_projection_gemm":
            dim = MODEL_CONFIG["dim"]
            qdim = MODEL_CONFIG["n_heads"] * MODEL_CONFIG["head_dim"]
            hidden = torch.ones((seq_len, dim), dtype=torch.bfloat16, device="cuda")
            q_gradient = torch.ones((seq_len, qdim), dtype=torch.bfloat16, device="cuda")

            def launch() -> Any:
                return torch.mm(q_gradient.transpose(0, 1), hidden)
            owned = [hidden, q_gradient]
            metadata = {
                "operation": "Q projection weight-gradient GEMM",
                "lhs": [qdim, seq_len], "rhs": [seq_len, dim],
            }
        elif compute_class == "expert_grouped_gemm":
            groups = MODEL_CONFIG["num_experts"] // 8
            rows_per_group = 2 * MODEL_CONFIG["moe_inter_dim"]
            reduction = (
                8 * seq_len * MODEL_CONFIG["top_k"]
                // MODEL_CONFIG["num_experts"]
            )
            lhs = torch.full(
                (groups * rows_per_group + 1, reduction), 0.01,
                dtype=torch.bfloat16, device="cuda",
            )
            rhs = torch.full(
                (groups, reduction, MODEL_CONFIG["dim"]), 0.01,
                dtype=torch.bfloat16, device="cuda",
            )
            offsets = torch.arange(
                rows_per_group, groups * rows_per_group + 1, rows_per_group,
                dtype=torch.int32, device="cuda",
            )

            def launch() -> Any:
                return torch._grouped_mm(lhs, rhs, offs=offsets)

            owned = [lhs, rhs, offsets]
            metadata = {
                "operation": "deterministic expert grouped dW GEMM",
                "groups": groups,
                "per_group": [
                    [rows_per_group, reduction],
                    [reduction, MODEL_CONFIG["dim"]],
                ],
                "input_fill": 0.01,
            }
        elif compute_class == "sdpa_backward":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            shape = (
                1, MODEL_CONFIG["n_heads"], seq_len, MODEL_CONFIG["head_dim"]
            )
            q = torch.full(shape, 0.01, dtype=torch.bfloat16, device="cuda")
            k = torch.full_like(q, 0.01)
            v = torch.full_like(q, 0.01)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                saved = torch.ops.aten._cudnn_attention_forward.default(
                    q, k, v, None, None, None, seq_len, seq_len,
                    True, 0.0, True, False, scale=None,
                )
            output, logsumexp = saved[0], saved[1]
            gradient = torch.ones_like(output)

            def launch() -> Any:
                return torch.ops.aten._cudnn_attention_backward.default(
                    gradient, q, k, v, output, logsumexp,
                    saved[6], saved[7], None, saved[2], saved[3],
                    saved[4], saved[5], 0.0, True, scale=None,
                )

            owned = [q, k, v, gradient]
            owned.extend(value for value in saved if isinstance(value, torch.Tensor))
            metadata = {
                "operation": "deterministic sdpa_backward",
                "qkv_shape": list(shape),
                "causal": True,
                "input_fill": 0.01,
            }
        elif compute_class == "routing_permutation":
            grid_x = int(participant["signature"]["grid"][0])
            anchor_numel = grid_x * 128 * 4
            rows = int(round(
                anchor_numel / MODEL_CONFIG["dim"] * seq_len / BASE_SEQ
            ))
            output = torch.empty(
                (rows, MODEL_CONFIG["dim"]),
                dtype=torch.bfloat16, device="cuda",
            )
            source = torch.ones_like(output)
            index = torch.arange(
                rows - 1, -1, -1, dtype=torch.long, device="cuda"
            )

            def launch() -> Any:
                output[index] = source
                return output
            owned = [output, source, index]
            metadata = {
                "operation": "event-specific routed BF16 index_put",
                "rows": rows,
                "anchor_grid_x": grid_x,
            }
        elif compute_class == "expert_activation_pointwise":
            numel = int(round(167_538 * 1024 * seq_len / BASE_SEQ))
            launch, owned, metadata = make_swiglu_backward(torch, numel=numel)
            launch()
            torch.cuda.synchronize()
        elif compute_class in {
            "expert_grouped_gemm_forward", "expert_grouped_gemm_backward_input",
        }:
            groups = MODEL_CONFIG["num_experts"]
            rows_per_group = seq_len * MODEL_CONFIG["top_k"] // groups
            rows = groups * rows_per_group
            dim = MODEL_CONFIG["dim"]
            width2 = 2 * MODEL_CONFIG["moe_inter_dim"]
            offsets = torch.arange(
                rows_per_group, rows + 1, rows_per_group,
                dtype=torch.int32, device="cuda",
            )
            if compute_class == "expert_grouped_gemm_forward":
                lhs = torch.ones((rows + 1, dim), dtype=torch.bfloat16, device="cuda")
                weight = torch.ones((groups, width2, dim), dtype=torch.bfloat16, device="cuda")

                def launch() -> Any:
                    return torch._grouped_mm(lhs, weight.transpose(-2, -1), offs=offsets)
            else:
                lhs_storage = torch.ones(
                    (rows, width2), dtype=torch.bfloat16, device="cuda"
                )
                lhs = lhs_storage.transpose(0, 1)
                weight = torch.ones(
                    (rows, dim), dtype=torch.bfloat16, device="cuda"
                )

                def launch() -> Any:
                    return torch._grouped_mm(lhs, weight, offs=offsets)
            owned = [lhs, weight, offsets]
            if compute_class == "expert_grouped_gemm_backward_input":
                owned.append(lhs_storage)
            metadata = {
                "operation": compute_class,
                "groups": groups,
                "rows_per_group": rows_per_group,
                "dim": dim,
                "width2": width2,
            }
        elif compute_class == "grouped_gemm_prepare":
            # The selected prepare event is emitted by the following high-level
            # grouped-MM call.  Keep its trace position as a zero-work serial
            # participant instead of launching an additional GEMM.
            def launch() -> None:
                return None
            owned = []
            metadata = {
                "operation": "prepare emitted by adjacent grouped MM",
                "launches_kernel": False,
            }
        elif compute_class == "elementwise_add":
            numel = int(round(188_272 * 1024 * seq_len / BASE_SEQ))
            launch, owned, metadata = make_bf16_add(torch, numel=numel)
            launch()
            torch.cuda.synchronize()
        else:
            raise ValueError(f"unsupported compute alias {alias}")
        torch.cuda.synchronize()
        return {
            "kind": "compute",
            "participant": participant,
            "launch": launch,
            "owned": owned,
            "metadata": metadata,
        }

    def _communication(
        self,
        participant: dict[str, Any],
        seq_len: int,
        a2a_profile: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        import torch.distributed as dist

        torch = self.torch
        category = participant["category"]
        focal_dp = participant["focal_dp_rank"]
        focal_pp = participant["focal_pp_rank"]
        local_dp = self.global_rank // 2
        local_pp = self.global_rank % 2
        signature = participant["signature"]
        dtype = {"BFloat16": torch.bfloat16, "Float": torch.float32}[signature["dtype"]]
        owned = []
        if category in {"ag", "rs", "a2a"} and local_pp != focal_pp:
            return None
        if category == "ag":
            input_numel = int(signature["in_msg_nelems"])
            source = torch.full((input_numel,), local_dp + 1, dtype=dtype, device="cuda")
            output = torch.empty((input_numel * 8,), dtype=dtype, device="cuda")
            owned = [source, output]

            def launch() -> Any:
                return dist.all_gather_into_tensor(
                    output, source, group=self.dp_groups[focal_pp], async_op=True
                )

            def validate() -> bool:
                return all(float(output[index * input_numel].item()) == index + 1 for index in range(8))
            runtime = {"input_numel": input_numel, "output_numel": input_numel * 8}
        elif category == "rs":
            input_numel = int(signature["in_msg_nelems"])
            if input_numel % 8:
                raise ValueError(f"RS size is not divisible by 8: {input_numel}")
            source = torch.full((input_numel,), local_dp + 1, dtype=dtype, device="cuda")
            output = torch.empty((input_numel // 8,), dtype=dtype, device="cuda")
            owned = [source, output]

            def launch() -> Any:
                return dist.reduce_scatter_tensor(
                    output, source, group=self.dp_groups[focal_pp], async_op=True
                )

            def validate() -> bool:
                return float(output[0].item()) == 36.0
            runtime = {"input_numel": input_numel, "output_numel": input_numel // 8}
        elif category == "a2a":
            if a2a_profile is None:
                raise ValueError(f"missing A2A profile for {participant['event_id']}")
            scaled = scale_split_matrix(a2a_profile, seq_len)[local_dp]
            input_rows = scaled["input_split_rows"]
            output_rows = scaled["output_split_rows"]
            dim = MODEL_CONFIG["dim"]
            source_chunks = [
                torch.full((rows * dim,), local_dp, dtype=dtype, device="cuda")
                for rows in input_rows
            ]
            source = torch.cat(source_chunks)
            output = torch.empty((scaled["output_numel"],), dtype=dtype, device="cuda")
            owned = [source, output, *source_chunks]
            input_splits = [rows * dim for rows in input_rows]
            output_splits = [rows * dim for rows in output_rows]

            def launch() -> Any:
                return dist.all_to_all_single(
                    output, source,
                    output_split_sizes=output_splits,
                    input_split_sizes=input_splits,
                    group=self.ep_groups[focal_pp], async_op=True,
                )

            def validate() -> bool:
                offset = 0
                for peer, count in enumerate(output_splits):
                    if count and float(output[offset].item()) != peer:
                        return False
                    offset += count
                return True
            runtime = scaled
        elif category in {"p2p_send", "p2p_recv"}:
            if local_dp != focal_dp:
                return None
            if local_pp not in {focal_pp, 1 - focal_pp}:
                return None
            numel = int(round(int(signature["in_msg_nelems"]) * seq_len / BASE_SEQ))
            slot = int(participant.get("p2p_slot", 0))
            group = self.pp_groups[(focal_dp, slot)]
            peer = focal_dp * 2 + (1 - local_pp)
            focal_global = focal_dp * 2 + focal_pp
            focal_sends = category == "p2p_send"
            sends = (self.global_rank == focal_global) == focal_sends
            value = float(self.global_rank + 11 if sends else peer + 11)
            tensor = torch.full((numel,), value if sends else -1.0, dtype=dtype, device="cuda")
            owned = [tensor]

            def launch() -> Any:
                op = dist.P2POp(dist.isend if sends else dist.irecv, tensor, peer, group=group)
                return dist.batch_isend_irecv([op])

            def validate() -> bool:
                return sends or float(tensor[0].item()) == float(peer + 11)
            runtime = {"numel": numel, "sends": sends, "peer": peer, "slot": slot}
        else:
            raise ValueError(category)
        torch.cuda.synchronize()
        return {
            "kind": "communication",
            "participant": participant,
            "launch": launch,
            "validate": validate,
            "owned": owned,
            "metadata": runtime,
        }

    def _operation(
        self,
        participant: dict[str, Any],
        seq_len: int,
        a2a_profiles: dict[str, Any],
    ) -> dict[str, Any] | None:
        if participant["category"] in COMM_CATEGORIES:
            return self._communication(
                participant, seq_len, a2a_profiles.get(participant["event_id"])
            )
        return self._compute(participant, seq_len)

    def benchmark(
        self,
        assignment: dict[str, Any],
        seq_len: int,
        a2a_profiles: dict[str, Any],
        warmups: int,
        iterations: int,
        stage_lead_ns: int,
        coverage_targets: list[float],
        profile_dir: str | None,
        geometry_calibration_trials: int,
        geometry_max_relative_shift_ms: float,
        clock_actor: Any,
        adaptive_sampling_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        if not self.initialized:
            raise RuntimeError("worker not initialized")
        torch = self.torch
        dist.barrier()
        runtime_health_pre = self._capture_runtime_health("pre_run")
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        participants = assignment["participants"]
        operations = [self._operation(item, seq_len, a2a_profiles) for item in participants]
        compute_stream = torch.cuda.Stream(priority=0)
        streams = [
            (
                compute_stream
                if op is not None
                and participants[index]["stream_role"] == "compute/default"
                else torch.cuda.Stream(priority=0) if op is not None else None
            )
            for index, op in enumerate(operations)
        ]
        control_stream = torch.cuda.Stream(priority=-1)
        if self.clock_calibration_current is None:
            raise RuntimeError("benchmark requires an initialized global clock")
        case_clock_calibration_history: list[dict[str, Any]] = []

        def maybe_refresh_clock(
            label: str, *, force: bool = False
        ) -> dict[str, Any]:
            """Refresh every rank together at a compound-stage boundary."""

            if label not in CLOCK_REFRESH_BOUNDARIES:
                raise ValueError(f"unsupported clock refresh boundary {label!r}")
            assert self.clock_calibration_current is not None
            local_age_ns = (
                time.monotonic_ns()
                - int(
                    self.clock_calibration_current[
                        "clock_calibration_local_monotonic_ns"
                    ]
                )
            )
            refresh = torch.tensor(
                [int(clock_refresh_required(local_age_ns, force=force))],
                dtype=torch.int32,
            )
            dist.all_reduce(refresh, op=dist.ReduceOp.MAX)
            if int(refresh.item()):
                record = self.calibrate_clock(
                    clock_actor, CLOCK_CALIBRATION_PROBES, label
                )
                case_clock_calibration_history.append(dict(record))
            dist.barrier()
            generations = [None] * dist.get_world_size()
            dist.all_gather_object(
                generations, self.clock_calibration_generation
            )
            if len(set(generations)) != 1:
                raise RuntimeError(
                    f"clock calibration generations diverged: {generations}"
                )
            return self._clock_row_metadata()

        # Every case starts with a fresh generation; age-triggered refreshes
        # then occur only at safe compound-stage boundaries.
        maybe_refresh_clock("benchmark_start", force=True)

        def run_stage(
            active: list[int], mode: str, sample_index: int,
            *, order_override: list[int] | None = None,
            delays_ms: dict[int, float] | None = None,
            capture_compute_outputs: bool = False,
            capture_communication_correctness: bool = False,
            bracket_for_mode: str | None = None,
            row_metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            anchor = torch.cuda.Event(enable_timing=True)
            anchor_before_local_ns = time.monotonic_ns()
            anchor.record()
            anchor.synchronize()
            anchor_after_local_ns = time.monotonic_ns()
            anchor_global_ns = (
                (anchor_before_local_ns + anchor_after_local_ns) // 2
                + self.clock_offset_ns
            )
            clock_metadata = self._clock_row_metadata()
            dist.barrier()
            latest_start = torch.tensor(
                [self._global_now_ns()], dtype=torch.int64
            )
            dist.all_reduce(latest_start, op=dist.ReduceOp.MAX)
            stage_origin_ns = int(latest_start.item()) + stage_lead_ns

            def wait_until_global(target_ns: int) -> None:
                while True:
                    remaining_ns = target_ns - self._global_now_ns()
                    if remaining_ns <= 0:
                        return
                    if remaining_ns > 500_000:
                        time.sleep((remaining_ns - 200_000) / 1e9)

            wait_until_global(stage_origin_ns)
            gate = torch.cuda.Event(enable_timing=True)
            records: dict[int, dict[str, Any]] = {}
            with torch.cuda.stream(control_stream):
                gate.record()
            order = (
                list(order_override) if order_override is not None
                else (active if mode != "reverse" else list(reversed(active)))
            )
            configured_delays = delays_ms or {}
            for index in order:
                op = operations[index]
                stream = streams[index]
                if op is None or stream is None:
                    continue
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                delay_ms = float(configured_delays.get(index, 0.0))
                wait_until_global(
                    stage_origin_ns + int(round(delay_ms * 1e6))
                )
                stream.wait_event(gate)
                with torch.cuda.stream(stream):
                    start.record()
                    host_launch_ns = self._global_now_ns()
                    result = op["launch"]()
                    if op["kind"] == "compute":
                        end.record()
                records[index] = {
                    "start": start,
                    "end": end,
                    "result": result,
                    "host_launch_ns": host_launch_ns,
                }
            # Queue completion only after every participant communicator was launched.
            for index in order:
                if index not in records:
                    continue
                op = operations[index]
                stream = streams[index]
                record = records[index]
                if op["kind"] == "communication":
                    with torch.cuda.stream(stream):
                        works = record["result"]
                        if not isinstance(works, (list, tuple)):
                            works = [works]
                        for work in works:
                            work.wait()
                        record["end"].record()
            pending = set(records)
            while pending:
                for index in list(pending):
                    if records[index]["end"].query():
                        records[index]["host_complete_ns"] = self._global_now_ns()
                        pending.remove(index)
            torch.cuda.synchronize()
            timebase_metadata = gpu_timebase_metadata(
                anchor_before_local_ns=anchor_before_local_ns,
                anchor_after_local_ns=anchor_after_local_ns,
                clock_offset_ns=self.clock_offset_ns,
                calibration_local_monotonic_ns=int(
                    clock_metadata[
                        "clock_calibration_local_monotonic_ns"
                    ]
                ),
                gate_elapsed_ms=float(anchor.elapsed_time(gate)),
            )
            output_signatures = []
            communication_correctness = []
            if capture_compute_outputs:
                for index in active:
                    op = operations[index]
                    if op is None:
                        continue
                    if op["kind"] == "compute":
                        output_signatures.append({
                            "operation_index": index,
                            "participant_id": participants[index]["participant_id"],
                            "global_rank": self.global_rank,
                            "mode": mode,
                            "sample_index": sample_index,
                            # Sampling happens after end-event completion and is
                            # intentionally excluded from every timed interval.
                            "signature": compact_output_signature(
                                records[index]["result"]
                            ),
                        })
                    elif capture_communication_correctness:
                        communication_correctness.append({
                            "operation_index": index,
                            "participant_id": participants[index]["participant_id"],
                            "global_rank": self.global_rank,
                            "mode": mode,
                            "sample_index": sample_index,
                            "validation_method": "per_treatment_communication_payload_check",
                            "validation_status": "validated",
                            "passed": bool(op["validate"]()),
                        })
            dist.barrier()
            rows = []
            for index in active:
                op = operations[index]
                if op is None:
                    continue
                record = records[index]
                rows.append(
                    {
                        "operation_index": index,
                        "participant_id": participants[index]["participant_id"],
                        "category": participants[index]["category"],
                        "global_rank": self.global_rank,
                        "sample_index": sample_index,
                        "mode": mode,
                        "bracket_for_mode": bracket_for_mode,
                        "configured_delay_ms": float(configured_delays.get(index, 0.0)),
                        "duration_ms": float(record["start"].elapsed_time(record["end"])),
                        "gate_to_start_ms": float(gate.elapsed_time(record["start"])),
                        "gate_to_end_ms": float(gate.elapsed_time(record["end"])),
                        "host_launch_global_ns": record["host_launch_ns"],
                        "stage_origin_global_ns": stage_origin_ns,
                        "launch_origin_global_ns": stage_origin_ns,
                        "configured_launch_target_global_ns": (
                            stage_origin_ns + int(round(
                                float(configured_delays.get(index, 0.0)) * 1e6
                            ))
                        ),
                        "gpu_start_global_ns": int(anchor_global_ns + anchor.elapsed_time(record["start"]) * 1e6),
                        "gpu_end_global_ns": int(anchor_global_ns + anchor.elapsed_time(record["end"]) * 1e6),
                        "host_complete_global_ns": record["host_complete_ns"],
                        **timebase_metadata,
                        **clock_metadata,
                        **(row_metadata or {}),
                    }
                )
            return {
                "rows": rows,
                "compute_output_signatures": output_signatures,
                "communication_correctness": communication_correctness,
            }

        def run_odd_k_baseline(
            index: int, phase: str, sample_index: int, treatment_mode: str
        ) -> dict[str, Any]:
            return collapse_odd_k_stage_measurements([
                run_stage(
                    [index], phase, sample_index,
                    bracket_for_mode=treatment_mode,
                )
                for _ in range(BASELINE_MEASUREMENT_REPETITIONS)
            ])

        def gather_runtime_rows(
            local_rows: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_rows)
            return [row for rank_rows in gathered for row in rank_rows]

        def runtime_decision_consensus(
            payload: dict[str, Any], label: str
        ) -> dict[str, Any]:
            digest = canonical_sha256(payload)
            local_record = {
                "global_rank": self.global_rank,
                **self._clock_row_metadata(),
                "decision_global_ns": self._global_now_ns(),
                "decision_digest_sha256": digest,
            }
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_record)
            if any(
                candidate["decision_digest_sha256"] != digest
                for candidate in gathered
            ):
                raise RuntimeError(
                    f"{label} differs across distributed ranks: {gathered}"
                )
            consensus_clock = common_clock_provenance(gathered)
            return {
                "label": label,
                **consensus_clock,
                "decision_clock_uncertainty_ns": (
                    consensus_clock[
                        "clock_calibration_max_half_rtt_uncertainty_ns"
                    ]
                ),
                "world_size": dist.get_world_size(),
                "decision_digest_sha256": digest,
                "actor_decision_digests": {
                    str(row["global_rank"]): row["decision_digest_sha256"]
                    for row in gathered
                },
                "actor_decision_records": gathered,
                "decision_global_ns_min": min(
                    row["decision_global_ns"] for row in gathered
                ),
                "decision_global_ns_max": max(
                    row["decision_global_ns"] for row in gathered
                ),
                "all_ranks_agree": True,
            }

        # Prewarm and calibrate isolated critical-path durations.  The final
        # isolated calibration result is sampled after its timed end event and
        # becomes the deterministic per-operation correctness reference.
        calibration = []
        compute_correctness_references: dict[int, dict[str, Any]] = {}
        calibration_iterations = max(2, min(5, warmups + 1))
        for index in range(len(operations)):
            for warmup in range(warmups):
                run_stage([index], "isolated_warmup", -1000 - warmup)
            for calibration_index in range(calibration_iterations):
                stage = run_stage(
                    [index], "calibration", -500 - calibration_index,
                    capture_compute_outputs=(
                        calibration_index == calibration_iterations - 1
                    ),
                )
                calibration.append(stage)
                for snapshot in stage["compute_output_signatures"]:
                    compute_correctness_references[index] = snapshot
        for index, op in enumerate(operations):
            if (
                op is not None and op["kind"] == "compute"
                and index not in compute_correctness_references
            ):
                raise RuntimeError(
                    f"missing isolated compute correctness reference for operation {index}"
                )
        local_calibration_rows = [
            row for stage in calibration for row in stage["rows"]
        ]
        gathered_calibration = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_calibration, local_calibration_rows)
        all_calibration_rows = [
            row for rank_rows in gathered_calibration for row in rank_rows
        ]
        isolated_estimates = {}
        isolated_gate_to_start = {}
        for index in range(len(operations)):
            per_iteration = []
            per_iteration_gate_to_start = []
            focal_rank = (
                int(participants[index]["focal_dp_rank"]) * 2
                + int(participants[index]["focal_pp_rank"])
            )
            for calibration_index in range(calibration_iterations):
                matches = [
                    row for row in all_calibration_rows
                    if row["operation_index"] == index
                    and row["sample_index"] == -500 - calibration_index
                ]
                if not matches:
                    raise RuntimeError(f"missing calibration for operation {index}")
                per_iteration.append(max(row["duration_ms"] for row in matches))
                focal_matches = [
                    row for row in matches if row["global_rank"] == focal_rank
                ]
                if len(focal_matches) != 1:
                    raise RuntimeError(
                        f"operation {index}: expected one focal-rank calibration "
                        f"row on rank {focal_rank}, got {len(focal_matches)}"
                    )
                per_iteration_gate_to_start.append(
                    focal_matches[0]["gate_to_start_ms"]
                )
            isolated_estimates[index] = statistics.median(per_iteration)
            isolated_gate_to_start[index] = statistics.median(
                per_iteration_gate_to_start
            )

        stage_specs = build_overlap_stage_specs(
            participants,
            isolated_estimates,
            isolated_gate_to_start,
            coverage_targets,
        )

        for warmup in range(warmups):
            for spec in stage_specs:
                run_stage(
                    list(range(len(operations))), spec["mode"], -2000 - warmup,
                    order_override=spec["order"], delays_ms=spec["delays_ms"],
                )

        # Correct residual host-launch-to-GPU-start error with a bounded number
        # of measured trials. All actors gather globally aligned focal
        # intervals and therefore make the same deterministic update.
        for spec_index, spec in enumerate(stage_specs):
            base_delays_ms = {
                int(index): float(value)
                for index, value in spec["delays_ms"].items()
            }
            spec["pre_runtime_calibration_delays_ms"] = dict(base_delays_ms)
            spec["geometry_calibration_history"] = []
            spec["geometry_calibration_candidate_shifts_requested"] = (
                geometry_calibration_trials
            )
            spec["geometry_confirmation_trials_per_candidate"] = (
                GEOMETRY_CONFIRMATION_TRIALS
            )
            spec["geometry_tolerance_pct"] = GEOMETRY_TOLERANCE_PCT
            spec["geometry_centering_tolerance_pct"] = (
                GEOMETRY_CENTERING_TOLERANCE_PCT
            )
            spec["geometry_validity_basis"] = GEOMETRY_VALIDITY_BASIS
            spec["concurrent_overlap_role"] = CONCURRENT_OVERLAP_ROLE
            # Only the structural duration bound proves infeasibility. A
            # dispatch-order projection miss and runtime nonconvergence retain
            # explicit unresolved status and receive no acceptance exemption.
            if not spec["target_feasible_from_duration"]:
                spec["target_feasible_after_runtime_calibration"] = False
                spec["geometry_cell_status"] = "proven_infeasible"
                spec["geometry_infeasibility_basis"] = "duration_limit"
                spec["runtime_geometry_calibration_status"] = (
                    "isolated_duration_bound_infeasible"
                )
                spec["runtime_relative_shift_ms"] = 0.0
                continue
            if not spec["target_feasible_after_constraints"]:
                spec["target_feasible_after_runtime_calibration"] = False
                spec["geometry_cell_status"] = "calibration_unresolved"
                spec["geometry_infeasibility_basis"] = None
                spec["runtime_geometry_calibration_status"] = (
                    "dispatch_order_projection_unresolved"
                )
                spec["runtime_relative_shift_ms"] = 0.0
                continue
            if geometry_calibration_trials == 0:
                spec["target_feasible_after_runtime_calibration"] = True
                spec["geometry_cell_status"] = "calibration_disabled"
                spec["geometry_infeasibility_basis"] = None
                spec["runtime_geometry_calibration_status"] = "disabled"
                spec["runtime_relative_shift_ms"] = 0.0
                continue

            minimum_shift_ms, maximum_shift_ms = relative_shift_bounds_ms(
                spec, base_delays_ms, geometry_max_relative_shift_ms
            )
            spec["runtime_relative_shift_bounds_ms"] = {
                "minimum": minimum_shift_ms,
                "maximum": maximum_shift_ms,
            }
            relative_shift_ms = 0.0
            stop_reason = "bounded_trials_exhausted"
            feedback_history = []
            for trial_index in range(geometry_calibration_trials):
                trial_delays_ms = apply_relative_geometry_shift(
                    spec, base_delays_ms, relative_shift_ms
                )
                confirmations = []
                for confirmation_index in range(GEOMETRY_CONFIRMATION_TRIALS):
                    maybe_refresh_clock("geometry_confirmation")
                    calibration_sample_index = geometry_confirmation_sample_index(
                        spec_index,
                        trial_index,
                        confirmation_index,
                        geometry_calibration_trials,
                    )
                    confirmation_rng = random.Random(
                        f"{assignment['case_key']}:geometry:"
                        f"{spec['mode']}:{calibration_sample_index}"
                    )
                    prime_order = list(range(len(operations)))
                    confirmation_rng.shuffle(prime_order)
                    for index in prime_order:
                        run_stage(
                            [index], "sample_prime", calibration_sample_index
                        )
                    pre_order, post_order = focal_adjacent_bracket_orders(
                        spec, len(operations)
                    )
                    local_confirmation_stages = []
                    for index in pre_order:
                        local_confirmation_stages.append(run_odd_k_baseline(
                            index, "isolated_pre", calibration_sample_index,
                            spec["mode"],
                        ))
                    local_stage = run_stage(
                        list(range(len(operations))),
                        geometry_calibration_mode(spec["mode"]),
                        calibration_sample_index,
                        order_override=spec["order"],
                        delays_ms=trial_delays_ms,
                    )
                    local_confirmation_stages.append(local_stage)
                    for index in post_order:
                        local_confirmation_stages.append(run_odd_k_baseline(
                            index, "isolated_post", calibration_sample_index,
                            spec["mode"],
                        ))
                    gathered_rows = [None] * dist.get_world_size()
                    dist.all_gather_object(
                        gathered_rows,
                        [
                            row
                            for stage in local_confirmation_stages
                            for row in stage["rows"]
                        ],
                    )
                    global_rows = [
                        row for rank_rows in gathered_rows for row in rank_rows
                    ]
                    baseline_pre_by_operation = {
                        index: max(
                            float(row["duration_ms"])
                            for row in matched_baseline_rows(
                                global_rows,
                                "isolated_pre",
                                spec["mode"],
                                index,
                                calibration_sample_index,
                            )
                        )
                        for index in range(len(operations))
                    }
                    baseline_post_by_operation = {
                        index: max(
                            float(row["duration_ms"])
                            for row in matched_baseline_rows(
                                global_rows,
                                "isolated_post",
                                spec["mode"],
                                index,
                                calibration_sample_index,
                            )
                        )
                        for index in range(len(operations))
                    }
                    paired_baselines_ms = {
                        index: (
                            baseline_pre_by_operation[index]
                            + baseline_post_by_operation[index]
                        ) / 2.0
                        for index in range(len(operations))
                    }
                    paired_baseline_drifts_pct = {
                        index: (
                            baseline_post_by_operation[index]
                            / baseline_pre_by_operation[index]
                            - 1.0
                        ) * 100.0
                        for index in range(len(operations))
                    }
                    victim = int(spec["victim_index"])
                    baseline_pre = baseline_pre_by_operation[victim]
                    baseline_post = baseline_post_by_operation[victim]
                    paired_baseline_ms = paired_baselines_ms[victim]
                    treatment_rows = geometry_treatment_rows(
                        global_rows, spec["mode"], calibration_sample_index
                    )
                    measurement = measure_runtime_geometry(
                        treatment_rows,
                        participants,
                        spec,
                        paired_baselines_ms,
                    )
                    measurement.update({
                        "trial_index": trial_index,
                        "confirmation_index": confirmation_index,
                        "calibration_sample_index": calibration_sample_index,
                        "relative_shift_ms": relative_shift_ms,
                        "delays_ms": trial_delays_ms,
                        "paired_baseline_pre_ms": baseline_pre,
                        "paired_baseline_post_ms": baseline_post,
                        "paired_baseline_ms": paired_baseline_ms,
                        "paired_baselines_ms": paired_baselines_ms,
                        "paired_baseline_drifts_pct": (
                            paired_baseline_drifts_pct
                        ),
                        "geometry_baseline_max_abs_drift_pct": max(
                            abs(value)
                            for value in paired_baseline_drifts_pct.values()
                        ),
                        "paired_baseline_drift_pct": (
                            paired_baseline_drifts_pct[victim]
                        ),
                        "overlap_error_ms": (
                            float(measurement["union_overlap_ms"])
                            - float(spec["requested_overlap_pct"])
                            / 100.0
                            * paired_baseline_ms
                        ),
                        "scheduled_overlap_error_ms": (
                            float(measurement["scheduled_overlap_ms"])
                            - float(spec["requested_overlap_pct"])
                            / 100.0
                            * paired_baseline_ms
                        ),
                        "geometry_validity_basis": GEOMETRY_VALIDITY_BASIS,
                        "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
                        "baseline_measurement_method": (
                            BASELINE_MEASUREMENT_METHOD
                        ),
                    })
                    confirmations.append(measurement)
                candidate = summarize_geometry_confirmation_trials(
                    confirmations, GEOMETRY_TOLERANCE_PCT
                )
                candidate.update({
                    "trial_index": trial_index,
                    "delays_ms": trial_delays_ms,
                })
                spec["geometry_calibration_history"].append(candidate)
                feedback = choose_geometry_candidate_feedback(
                    candidate,
                    feedback_history,
                    float(spec["requested_overlap_pct"]),
                    minimum_shift_ms,
                    maximum_shift_ms,
                )
                candidate["feedback_method"] = feedback["method"]
                if not feedback["feedback_eligible"]:
                    stop_reason = "baseline_unstable_confirmations"
                    continue
                feedback_history.append(candidate)
                if candidate["robustly_converged"]:
                    candidate["feedback_method"] = "robust_confirmation"
                    stop_reason = "converged"
                    break
                if feedback["converged"]:
                    # A median can be centered while individual confirmations
                    # or their matched baselines fail. Repeat the same shift.
                    candidate["feedback_method"] = (
                        "repeat_unstable_centered_candidate"
                    )
                    continue
                if feedback["stalled"]:
                    stop_reason = "feedback_stalled_at_bound"
                    break
                relative_shift_ms = float(feedback["next_shift_ms"])

            history = spec["geometry_calibration_history"]
            best = min(
                history,
                key=lambda row: (
                    not bool(row["robustly_converged"]),
                    not bool(row["baseline_feedback_eligible"]),
                    abs(float(row["scheduled_geometry_error_pct"])),
                    -int(row["confirmation_passing_count"]),
                    int(row["trial_index"]),
                ),
            )
            best_shift_ms = float(best["relative_shift_ms"])
            spec["delays_ms"] = apply_relative_geometry_shift(
                spec, base_delays_ms, best_shift_ms
            )
            spec["runtime_relative_shift_ms"] = best_shift_ms
            spec["runtime_geometry_selected_trial_index"] = int(
                best["trial_index"]
            )
            spec["runtime_calibrated_overlap_pct"] = float(
                best["realized_overlap_pct"]
            )
            spec["runtime_calibrated_geometry_error_pct"] = float(
                best["geometry_error_pct"]
            )
            spec["runtime_calibrated_scheduled_overlap_pct"] = float(
                best["scheduled_overlap_pct"]
            )
            spec["runtime_calibrated_scheduled_geometry_error_pct"] = float(
                best["scheduled_geometry_error_pct"]
            )
            spec["runtime_calibrated_counterfactual_overlap_pct"] = float(
                best["counterfactual_overlap_pct"]
            )
            spec[
                "runtime_calibrated_counterfactual_geometry_error_pct"
            ] = float(best["counterfactual_geometry_error_pct"])
            spec["runtime_geometry_paired_baseline_ms"] = float(
                best["paired_baseline_ms"]
            )
            spec["runtime_geometry_paired_baseline_drift_pct"] = float(
                best["paired_baseline_drift_pct"]
            )
            spec["runtime_geometry_baseline_max_abs_drift_pct"] = float(
                best["geometry_baseline_max_abs_drift_pct"]
            )
            spec["runtime_geometry_baseline_stability_basis"] = (
                best["geometry_baseline_stability_basis"]
            )
            spec["runtime_geometry_confirmation_trial_count"] = int(
                best["confirmation_trial_count"]
            )
            spec["runtime_geometry_confirmation_passing_count"] = int(
                best["confirmation_passing_count"]
            )
            spec["runtime_geometry_confirmation_geometry_passing_count"] = int(
                best["geometry_passing_count"]
            )
            spec["runtime_geometry_confirmation_baseline_stable_count"] = int(
                best["baseline_stable_count"]
            )
            spec["runtime_geometry_confirmation_jointly_passing_count"] = int(
                best["jointly_passing_count"]
            )
            spec["runtime_geometry_baseline_feedback_eligible"] = bool(
                best["baseline_feedback_eligible"]
            )
            spec["runtime_geometry_confirmation_error_min_pct"] = float(
                best["confirmation_error_min_pct"]
            )
            spec["runtime_geometry_confirmation_error_max_pct"] = float(
                best["confirmation_error_max_pct"]
            )
            spec[
                "runtime_geometry_confirmation_scheduled_error_min_pct"
            ] = float(best["confirmation_scheduled_error_min_pct"])
            spec[
                "runtime_geometry_confirmation_scheduled_error_max_pct"
            ] = float(best["confirmation_scheduled_error_max_pct"])
            spec["runtime_geometry_median_centered"] = bool(
                best["median_centered"]
            )
            spec["runtime_geometry_confirmation_drift_min_pct"] = float(
                best["confirmation_drift_min_pct"]
            )
            spec["runtime_geometry_confirmation_drift_max_pct"] = float(
                best["confirmation_drift_max_pct"]
            )
            spec[
                "runtime_geometry_confirmation_geometry_baseline_"
                "max_abs_drift_min_pct"
            ] = float(
                best[
                    "confirmation_geometry_baseline_max_abs_drift_min_pct"
                ]
            )
            spec[
                "runtime_geometry_confirmation_geometry_baseline_"
                "max_abs_drift_max_pct"
            ] = float(
                best[
                    "confirmation_geometry_baseline_max_abs_drift_max_pct"
                ]
            )
            converged = bool(best["robustly_converged"])
            spec["runtime_geometry_robustly_converged"] = converged
            spec["target_feasible_after_runtime_calibration"] = converged
            spec["geometry_cell_status"] = (
                "confirmed_feasible" if converged else "calibration_unresolved"
            )
            spec["geometry_infeasibility_basis"] = None
            spec["runtime_geometry_calibration_status"] = (
                "converged" if converged else stop_reason
            )

        def treatment_correctness(stage: dict[str, Any]) -> list[dict[str, Any]]:
            checks = list(stage["communication_correctness"])
            for snapshot in stage["compute_output_signatures"]:
                index = int(snapshot["operation_index"])
                reference = compute_correctness_references[index]
                if participants[index]["kernel_alias"] == "K-c3ef3bbe1a3e":
                    comparison = grouped_gemm_prepare_semantics(participants, index)
                    validation_method = "adjacent_grouped_mm_semantics"
                else:
                    comparison = compare_output_signatures(
                        reference["signature"], snapshot["signature"]
                    )
                    comparison["validation_status"] = (
                        "validated" if comparison["passed"] else "mismatch"
                    )
                    validation_method = "compact_isolated_reference_signature"
                checks.append({
                    "operation_index": index,
                    "participant_id": snapshot["participant_id"],
                    "global_rank": self.global_rank,
                    "mode": snapshot["mode"],
                    "sample_index": snapshot["sample_index"],
                    "validation_method": validation_method,
                    "reference_mode": reference["mode"],
                    "reference_sample_index": reference["sample_index"],
                    "observed_signature": snapshot["signature"],
                    **comparison,
                })
            return checks

        samples: list[dict[str, Any]] = []
        correctness: list[dict[str, Any]] = []
        adaptive_diagnostics = None
        last_sample_index = -1
        if adaptive_sampling_config is None:
            for sample_index in range(iterations):
                last_sample_index = sample_index
                rng = random.Random(
                    f"{assignment['case_key']}:{sample_index}"
                )
                prime_order = list(range(len(operations)))
                rng.shuffle(prime_order)
                for index in prime_order:
                    # This deliberately unmeasured launch moves every operation
                    # into the same active GPU/NCCL state before any baseline
                    # or treatment in the sample is recorded.
                    run_stage([index], "sample_prime", sample_index)
                treatment_order = list(stage_specs)
                rng.shuffle(treatment_order)
                for spec in treatment_order:
                    maybe_refresh_clock("fixed_treatment")
                    pre_order, post_order = focal_adjacent_bracket_orders(
                        spec, len(operations)
                    )
                    local_pre_stages = [
                        run_odd_k_baseline(
                            index, "isolated_pre", sample_index, spec["mode"]
                        )
                        for index in pre_order
                    ]
                    samples.extend(local_pre_stages)
                    treatment_delays, start_metadata = execute_runtime_start_anchor(
                        spec,
                        sample_index,
                        participants,
                        len(operations),
                        geometry_max_relative_shift_ms,
                        local_pre_stages,
                        run_stage,
                        gather_runtime_rows,
                        runtime_decision_consensus,
                    )
                    stage = run_stage(
                        list(range(len(operations))), spec["mode"], sample_index,
                        order_override=spec["order"], delays_ms=treatment_delays,
                        capture_compute_outputs=True,
                        capture_communication_correctness=True,
                        row_metadata=start_metadata,
                    )
                    samples.append(stage)
                    correctness.extend(treatment_correctness(stage))
                    for index in post_order:
                        samples.append(run_odd_k_baseline(
                            index, "isolated_post", sample_index, spec["mode"]
                        ))
        else:
            sampling_payload = dict(adaptive_sampling_config["sampling"])
            controller_config = AdaptiveBenchmarkControllerConfig(
                sampling=AdaptiveSamplingConfig(**sampling_payload),
                drift_threshold_pct=float(
                    adaptive_sampling_config["drift_threshold_pct"]
                ),
                default_geometry_tolerance_pct=float(
                    adaptive_sampling_config["default_geometry_tolerance_pct"]
                ),
                max_total_attempt_multiplier=float(
                    adaptive_sampling_config["max_total_attempt_multiplier"]
                ),
                attempt_batch_size=int(
                    adaptive_sampling_config["attempt_batch_size"]
                ),
            )
            controller = AdaptiveBenchmarkController(controller_config)
            for spec in stage_specs:
                key = adaptive_cell_key(assignment, spec)
                controller.register_cell(
                    key, isolated_estimates[int(spec["victim_index"])]
                )
                geometry_cell_status = spec.get("geometry_cell_status")
                if geometry_cell_status == "proven_infeasible":
                    controller.mark_cell_infeasible(
                        key, proof_basis="duration_bound"
                    )
                elif geometry_cell_status == "calibration_unresolved":
                    controller.mark_cell_calibration_unresolved(key)
                elif geometry_cell_status not in {
                    "confirmed_feasible", "calibration_disabled"
                }:
                    raise RuntimeError(
                        f"unknown geometry cell status {geometry_cell_status!r} "
                        f"for {key.canonical}"
                    )

            round_history = []
            adaptive_sample_dispositions = []
            next_sample_index = 0
            round_index = 0
            while not controller.all_stopped:
                plan = controller.next_batch_plan()
                plan_payload = {
                    key.canonical: int(attempts)
                    for key, attempts in sorted(plan.items())
                }
                gathered_plans = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_plans, plan_payload)
                if any(candidate != plan_payload for candidate in gathered_plans):
                    raise RuntimeError(
                        "adaptive treatment plans differ across distributed ranks"
                    )
                batches = adaptive_plan_batches(assignment, stage_specs, plan)
                if not batches:
                    raise RuntimeError(
                        "adaptive controller is active but issued an empty plan"
                    )
                round_controller_rows = []
                round_samples = []
                for active_specs in batches:
                    sample_index = next_sample_index
                    next_sample_index += 1
                    last_sample_index = sample_index
                    rng = random.Random(
                        f"{assignment['case_key']}:adaptive:{sample_index}"
                    )
                    local_stages = []
                    prime_order = list(range(len(operations)))
                    rng.shuffle(prime_order)
                    for index in prime_order:
                        run_stage([index], "sample_prime", sample_index)
                    treatment_order = list(active_specs)
                    rng.shuffle(treatment_order)
                    local_treatment_correctness = []
                    for spec in treatment_order:
                        maybe_refresh_clock("adaptive_treatment")
                        pre_order, post_order = focal_adjacent_bracket_orders(
                            spec, len(operations)
                        )
                        local_pre_stages = [
                            run_odd_k_baseline(
                                index, "isolated_pre", sample_index, spec["mode"]
                            )
                            for index in pre_order
                        ]
                        local_stages.extend(local_pre_stages)
                        treatment_delays, start_metadata = execute_runtime_start_anchor(
                            spec,
                            sample_index,
                            participants,
                            len(operations),
                            geometry_max_relative_shift_ms,
                            local_pre_stages,
                            run_stage,
                            gather_runtime_rows,
                            runtime_decision_consensus,
                        )
                        stage = run_stage(
                            list(range(len(operations))),
                            spec["mode"],
                            sample_index,
                            order_override=spec["order"],
                            delays_ms=treatment_delays,
                            capture_compute_outputs=True,
                            capture_communication_correctness=True,
                            row_metadata=start_metadata,
                        )
                        local_stages.append(stage)
                        checks = treatment_correctness(stage)
                        correctness.extend(checks)
                        local_treatment_correctness.extend(checks)
                        for index in post_order:
                            local_stages.append(run_odd_k_baseline(
                                index, "isolated_post", sample_index, spec["mode"]
                            ))
                    samples.extend(local_stages)
                    round_samples.append(sample_index)

                    local_payload = {
                        "rows": [
                            row for stage in local_stages for row in stage["rows"]
                        ],
                        "correctness": local_treatment_correctness,
                    }
                    gathered_payloads = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered_payloads, local_payload)
                    global_correctness = [
                        row
                        for payload in gathered_payloads
                        for row in payload["correctness"]
                    ]
                    failures = [
                        row for row in global_correctness
                        if row.get("passed") is not True
                    ]
                    if failures:
                        raise RuntimeError(
                            "adaptive treatment correctness failed: "
                            + json.dumps(failures[:8], sort_keys=True)
                        )
                    global_rows = [
                        row
                        for payload in gathered_payloads
                        for row in payload["rows"]
                    ]
                    round_controller_rows.extend(
                        build_adaptive_controller_rows(
                            assignment,
                            participants,
                            global_rows,
                            active_specs,
                            sample_index,
                            correct=True,
                        )
                    )
                before_cells = {
                    row.cell_key: row.to_dict()
                    for row in controller.decisions()
                }
                ingestion = controller.ingest_rows(round_controller_rows)
                after_cells = {
                    row.cell_key: row.to_dict()
                    for row in controller.decisions()
                }
                accounting = build_adaptive_round_accounting(
                    plan_payload,
                    round_controller_rows,
                    before_cells,
                    after_cells,
                    controller_config,
                )
                if accounting["totals"] != {
                    "issued": int(ingestion["rows_seen"]),
                    "completed": int(ingestion["rows_seen"]),
                    "ingested": int(ingestion["rows_ingested"]),
                    "discarded_after_stop": int(
                        ingestion["rows_discarded_after_stop"]
                    ),
                    "reconciled": True,
                }:
                    raise RuntimeError(
                        "adaptive round accounting disagrees with controller "
                        f"ingestion counters: {accounting['totals']} vs {ingestion}"
                    )
                adaptive_sample_dispositions.extend(
                    accounting["sample_dispositions"]
                )
                round_history.append({
                    "round_index": round_index,
                    "planned_attempts": plan_payload,
                    "sample_indices": round_samples,
                    "ingestion": ingestion,
                    "attempt_accounting": {
                        "per_cell": accounting["per_cell"],
                        "totals": accounting["totals"],
                    },
                })
                round_index += 1
            adaptive_diagnostics = controller.diagnostics()
            adaptive_diagnostics["rounds"] = round_history
            cell_keys = [
                row["cell_key"] for row in adaptive_diagnostics["cells"]
            ]
            accounting_fields = {
                "issued_attempt_counts": "issued",
                "completed_attempt_counts": "completed",
                "ingested_attempt_counts": "ingested",
                "discarded_after_stop_counts": "discarded_after_stop",
            }
            for output_field, accounting_field in accounting_fields.items():
                adaptive_diagnostics[output_field] = {
                    cell_key: sum(
                        int(round_row["attempt_accounting"]["per_cell"]
                            .get(cell_key, {}).get(accounting_field, 0))
                        for round_row in round_history
                    )
                    for cell_key in cell_keys
                }
            adaptive_diagnostics["actual_attempt_counts"] = dict(
                adaptive_diagnostics["completed_attempt_counts"]
            )
            adaptive_diagnostics["actual_attempt_counts_semantics"] = (
                "completed treatment attempts; use ingested_attempt_counts "
                "for controller-retained attempts"
            )
            issued_total = sum(
                adaptive_diagnostics["issued_attempt_counts"].values()
            )
            completed_total = sum(
                adaptive_diagnostics["completed_attempt_counts"].values()
            )
            ingested_total = sum(
                adaptive_diagnostics["ingested_attempt_counts"].values()
            )
            discarded_total = sum(
                adaptive_diagnostics["discarded_after_stop_counts"].values()
            )
            adaptive_diagnostics["attempt_reconciliation"] = {
                "issued": issued_total,
                "completed": completed_total,
                "ingested": ingested_total,
                "discarded_after_stop": discarded_total,
                "passed": bool(
                    issued_total
                    == completed_total
                    == ingested_total + discarded_total
                ),
            }
            if not adaptive_diagnostics["attempt_reconciliation"]["passed"]:
                raise RuntimeError(
                    "adaptive run-level attempt reconciliation failed: "
                    + json.dumps(
                        adaptive_diagnostics["attempt_reconciliation"],
                        sort_keys=True,
                    )
                )
            if (
                ingested_total
                != int(adaptive_diagnostics["summary"]["total_attempts"])
                or discarded_total
                != int(adaptive_diagnostics["summary"]["discarded_after_stop"])
            ):
                raise RuntimeError(
                    "adaptive explicit accounting disagrees with controller "
                    "diagnostic summary"
                )
            adaptive_diagnostics["sample_dispositions"] = sorted(
                adaptive_sample_dispositions,
                key=lambda row: (row["cell_key"], row["sample_index"]),
            )

        for index, op in enumerate(operations):
            if op is not None and op["kind"] == "communication":
                correctness.append(
                    {
                        "operation_index": index,
                        "participant_id": participants[index]["participant_id"],
                        "global_rank": self.global_rank,
                        "mode": "final_state_after_retained_samples",
                        "sample_index": last_sample_index,
                        "validation_method": "existing_communication_payload_check",
                        "validation_status": "validated",
                        "passed": bool(op["validate"]()),
                    }
                )

        validated_operation_indices = {
            int(row["operation_index"]) for row in correctness
            if row.get("passed") is True
        }
        for index, reference in compute_correctness_references.items():
            if index in validated_operation_indices:
                continue
            correctness.append({
                "operation_index": index,
                "participant_id": participants[index]["participant_id"],
                "global_rank": self.global_rank,
                "mode": "isolated_reference_self_consistency",
                "sample_index": int(reference["sample_index"]),
                "validation_method": "isolated_reference_self_consistency",
                "validation_status": "validated",
                "reference_mode": reference["mode"],
                "reference_sample_index": reference["sample_index"],
                "observed_signature": reference["signature"],
                "passed": True,
            })
        failed_correctness = [
            row for row in correctness if not bool(row.get("passed"))
        ]
        if failed_correctness:
            raise RuntimeError(
                "participant correctness failed: "
                + json.dumps(failed_correctness[:8], sort_keys=True)
            )

        # One CUPTI trace per GPU. It is a kernel-identity gate, not timing input.
        trace_path = None
        if profile_dir:
            from torch.profiler import ProfilerActivity, profile

            dist.barrier()
            profile_spec = max(
                stage_specs, key=lambda item: item["requested_overlap_pct"]
            )
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                run_stage(
                    list(range(len(operations))), "profile", -4000,
                    order_override=profile_spec["order"],
                    delays_ms=profile_spec["delays_ms"],
                )
            if any(op is not None for op in operations):
                trace_path = str(
                    Path(profile_dir)
                    / f"{assignment['case_key'].replace(':', '_')}_rank{self.global_rank}.json"
                )
                Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(trace_path)
            dist.barrier()

        flat_rows = [row for sample in samples for row in sample["rows"]]
        metadata = [
            None if op is None else {
                "operation_index": index,
                "participant_id": participants[index]["participant_id"],
                "global_rank": self.global_rank,
                "metadata": op["metadata"],
            }
            for index, op in enumerate(operations)
            if op is not None
        ]
        for op in operations:
            if op is not None:
                op["owned"].clear()
        del operations
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()
        runtime_health_post = self._capture_runtime_health("post_run")
        dist.barrier()
        return {
            "global_rank": self.global_rank,
            "rows": flat_rows,
            "stage_specs": stage_specs,
            "isolated_estimates_ms": isolated_estimates,
            "isolated_gate_to_start_ms": isolated_gate_to_start,
            "correctness": correctness,
            "compute_correctness_references": [
                compute_correctness_references[index]
                for index in sorted(compute_correctness_references)
            ],
            "operation_metadata": metadata,
            "trace_path": trace_path,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "adaptive_sampling": adaptive_diagnostics,
            "clock_calibration_history": case_clock_calibration_history,
            "clock_calibration_final": self.current_clock_calibration(),
            "runtime_health": {
                "pre_run": runtime_health_pre,
                "post_run": runtime_health_post,
            },
        }

    def shutdown(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        self.initialized = False
        if self.runtime_health_collector is not None:
            self.runtime_health_collector.close()
            self.runtime_health_collector = None


def aggregate_case(
    assignment: dict[str, Any],
    rank_results: list[dict[str, Any]],
    iterations: int | None,
) -> dict[str, Any]:
    rows = [row for result in rank_results for row in result["rows"]]
    participants = assignment["participants"]
    samples = []
    stage_specs = rank_results[0]["stage_specs"]
    if any(result["stage_specs"] != stage_specs for result in rank_results[1:]):
        raise ValueError("rank stage specifications differ")
    adaptive_sampling = rank_results[0].get("adaptive_sampling")
    if any(
        result.get("adaptive_sampling") != adaptive_sampling
        for result in rank_results[1:]
    ):
        raise ValueError("rank adaptive-sampling diagnostics differ")
    sample_indices_by_mode = {
        spec["mode"]: sorted({
            int(row["sample_index"])
            for row in rows if row["mode"] == spec["mode"]
        })
        for spec in stage_specs
    }
    if adaptive_sampling is None:
        if iterations is None:
            raise ValueError("fixed sampling requires an iteration count")
        expected = list(range(iterations))
        for mode, observed in sample_indices_by_mode.items():
            if observed != expected:
                raise ValueError(
                    f"fixed stage {mode} has sample indices {observed}; "
                    f"expected {expected}"
                )
    isolated_pre: dict[tuple[int, int, str], float] = {}
    isolated_post: dict[tuple[int, int, str], float] = {}
    for spec in stage_specs:
        treatment_mode = str(spec["mode"])
        for index in range(len(participants)):
            for sample_index in sample_indices_by_mode[treatment_mode]:
                for phase, destination in (
                    ("isolated_pre", isolated_pre),
                    ("isolated_post", isolated_post),
                ):
                    matches = matched_baseline_rows(
                        rows,
                        phase,
                        treatment_mode,
                        index,
                        sample_index,
                    )
                    destination[(index, sample_index, treatment_mode)] = max(
                        row["duration_ms"] for row in matches
                    )
    for spec in stage_specs:
        mode = spec["mode"]
        victim = int(spec["victim_index"])
        for sample_index in sample_indices_by_mode[mode]:
            mode_rows = [
                row for row in rows
                if row["mode"] == mode and row["sample_index"] == sample_index
            ]
            start_contract_keys = (
                "runtime_start_geometry_disposition_explicit",
                "runtime_start_geometry_disposition",
                "runtime_start_geometry_treatment_executed",
                "runtime_start_geometry_scoring_eligible",
                "start_anchor_resolution_status",
                "start_anchor_scoring_allowed",
                "delay_rebase_evidence",
                "delay_rebase_evidence_sha256",
            )
            start_contracts = [
                {key: row.get(key) for key in start_contract_keys}
                for row in mode_rows
            ]
            if not start_contracts:
                raise ValueError("treatment emitted no runtime-start contract")
            start_metadata = start_contracts[0]
            if any(candidate != start_metadata for candidate in start_contracts[1:]):
                raise ValueError(
                    "runtime-start disposition/evidence differs across endpoints"
                )
            op_intervals = {}
            for index in range(len(participants)):
                matches = [row for row in mode_rows if row["operation_index"] == index]
                if not matches:
                    raise ValueError(f"missing concurrent rows for operation {index}")
                focal = (
                    participants[index]["focal_dp_rank"] * 2
                    + participants[index]["focal_pp_rank"]
                )
                focal_rows = [row for row in matches if row["global_rank"] == focal]
                representative = (
                    focal_rows[0] if focal_rows
                    else max(matches, key=lambda row: row["duration_ms"])
                )
                op_intervals[index] = {
                    "start_ms": representative["gpu_start_global_ns"] / 1e6,
                    "end_ms": representative["gpu_end_global_ns"] / 1e6,
                    "local_duration_ms": representative["duration_ms"],
                    "max_rank_duration_ms": max(row["duration_ms"] for row in matches),
                    "earliest_host_launch_global_ns": min(
                        row["host_launch_global_ns"] for row in matches
                    ),
                    "latest_host_complete_global_ns": max(
                        row["host_complete_global_ns"] for row in matches
                    ),
                }
            # Normalize globally aligned timestamps for readable per-stage
            # output while preserving cross-actor interval relationships.
            interval_origin_ms = min(
                interval["start_ms"] for interval in op_intervals.values()
            )
            for interval in op_intervals.values():
                interval["start_ms"] -= interval_origin_ms
                interval["end_ms"] -= interval_origin_ms
            matched_baselines_ms = {
                index: (
                    isolated_pre[(index, sample_index, mode)]
                    + isolated_post[(index, sample_index, mode)]
                ) / 2.0
                for index in range(len(participants))
            }
            geometry_baseline_drifts_pct = {
                index: (
                    isolated_post[(index, sample_index, mode)]
                    / isolated_pre[(index, sample_index, mode)]
                    - 1.0
                ) * 100.0
                for index in range(len(participants))
            }
            geometry_baseline_max_abs_drift_pct = max(
                abs(value) for value in geometry_baseline_drifts_pct.values()
            )
            baseline_pre = isolated_pre[(victim, sample_index, mode)]
            baseline_post = isolated_post[(victim, sample_index, mode)]
            baseline = matched_baselines_ms[victim]
            baseline_drift_pct = (baseline_post / baseline_pre - 1.0) * 100.0
            baseline_pre_by_operation = {
                index: isolated_pre[(index, sample_index, mode)]
                for index in range(len(participants))
            }
            baseline_post_by_operation = {
                index: isolated_post[(index, sample_index, mode)]
                for index in range(len(participants))
            }
            compound_clock_rows = list(mode_rows)
            for phase in ("isolated_pre", "isolated_post"):
                for index in range(len(participants)):
                    compound_clock_rows.extend(matched_baseline_rows(
                        rows, phase, mode, index, sample_index
                    ))
            compound_clock_provenance = common_clock_provenance(
                compound_clock_rows
            )
            evidence_generation = start_metadata[
                "delay_rebase_evidence"
            ].get("clock_calibration_generation")
            if evidence_generation != compound_clock_provenance[
                "clock_calibration_generation"
            ]:
                raise RuntimeError(
                    "adjacent-pre/anchor/treatment/post crossed clock "
                    "calibration generations"
                )
            pre_geometry = measure_runtime_geometry(
                mode_rows, participants, spec, baseline_pre_by_operation
            )
            post_geometry = measure_runtime_geometry(
                mode_rows, participants, spec, baseline_post_by_operation
            )
            victim_baseline_stable = bool(
                abs(baseline_drift_pct) <= BASELINE_DRIFT_TOLERANCE_PCT
            )
            pre_post_counterfactual_geometry_robust = bool(
                abs(float(pre_geometry["scheduled_geometry_error_pct"]))
                <= GEOMETRY_TOLERANCE_PCT
                and abs(float(post_geometry["scheduled_geometry_error_pct"]))
                <= GEOMETRY_TOLERANCE_PCT
            )
            victim_interval = op_intervals[victim]
            covered = []
            per_aggressor_overlap = []
            for aggressor in spec["aggressor_indices"]:
                other = op_intervals[int(aggressor)]
                left = max(victim_interval["start_ms"], other["start_ms"])
                right = min(victim_interval["end_ms"], other["end_ms"])
                intersection_ms = max(0.0, right - left)
                per_aggressor_overlap.append({
                    "operation_index": int(aggressor),
                    "participant_id": participants[int(aggressor)]["participant_id"],
                    "family": participants[int(aggressor)]["category"],
                    "intersection_ms": intersection_ms,
                })
                if right > left:
                    covered.append((left, right))
            merged = []
            for left, right in sorted(covered):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            overlap_ms = sum(right - left for left, right in merged)
            counterfactual_intervals_ns = {
                index: (
                    int(round(interval["start_ms"] * 1e6)),
                    int(round(
                        (interval["start_ms"] + matched_baselines_ms[index])
                        * 1e6
                    )),
                )
                for index, interval in op_intervals.items()
            }
            scheduled_overlap_ms = (
                _union_overlap_ns(counterfactual_intervals_ns, spec) / 1e6
            )
            scheduled_overlap_pct = min(
                100.0, scheduled_overlap_ms / baseline * 100.0
            )
            scheduled_geometry_error_pct = (
                scheduled_overlap_pct - spec["requested_overlap_pct"]
            )
            concurrent = victim_interval["max_rank_duration_ms"]
            per_aggressor_counterfactual = {
                int(metric["operation_index"]): metric
                for metric in _per_aggressor_overlap_metrics_ns(
                    {
                        index: (
                            int(round(interval["start_ms"] * 1e6)),
                            int(round(interval["end_ms"] * 1e6)),
                        )
                        for index, interval in op_intervals.items()
                    },
                    counterfactual_intervals_ns,
                    spec,
                    baseline,
                )
            }
            for item in per_aggressor_overlap:
                metric = per_aggressor_counterfactual[
                    int(item["operation_index"])
                ]
                for field in (
                    "victim_coverage_pct",
                    "scheduled_intersection_ms",
                    "scheduled_victim_coverage_pct",
                    "counterfactual_intersection_ms",
                    "counterfactual_victim_coverage_pct",
                ):
                    item[field] = metric[field]
            cumulative_overlap_ms = sum(
                item["intersection_ms"] for item in per_aggressor_overlap
            )
            victim_left = victim_interval["start_ms"]
            victim_right = victim_interval["end_ms"]
            width_boundaries = {victim_left, victim_right}
            clipped_intervals = []
            for aggressor in spec["aggressor_indices"]:
                other = op_intervals[int(aggressor)]
                left = max(victim_left, other["start_ms"])
                right = min(victim_right, other["end_ms"])
                if right > left:
                    clipped_intervals.append((left, right))
                    width_boundaries.update((left, right))
            concurrency_width_ms: dict[str, float] = {}
            ordered_boundaries = sorted(width_boundaries)
            for left, right in zip(ordered_boundaries, ordered_boundaries[1:]):
                if right <= left:
                    continue
                midpoint = (left + right) / 2.0
                width = sum(
                    interval_left <= midpoint < interval_right
                    for interval_left, interval_right in clipped_intervals
                )
                key = str(width)
                concurrency_width_ms[key] = (
                    concurrency_width_ms.get(key, 0.0) + right - left
                )
            participant_outcomes = []
            for index, interval in op_intervals.items():
                isolated_ms = (
                    isolated_pre[(index, sample_index, mode)]
                    + isolated_post[(index, sample_index, mode)]
                ) / 2.0
                concurrent_ms = interval["max_rank_duration_ms"]
                participant_outcomes.append({
                    "operation_index": index,
                    "participant_id": participants[index]["participant_id"],
                    "family": participants[index]["category"],
                    "isolated_ms": isolated_ms,
                    "concurrent_ms": concurrent_ms,
                    "slowdown_pct": (concurrent_ms / isolated_ms - 1.0) * 100.0,
                    "start_ms": interval["start_ms"],
                    "end_ms": interval["end_ms"],
                    "counterfactual_end_ms": (
                        interval["start_ms"] + isolated_ms
                    ),
                })
            aggressor_intervals = [
                op_intervals[int(index)] for index in spec["aggressor_indices"]
            ]
            aggressor_group_span_ms = (
                max(item["end_ms"] for item in aggressor_intervals)
                - min(item["start_ms"] for item in aggressor_intervals)
            )
            group_makespan_ms = (
                max(item["end_ms"] for item in op_intervals.values())
                - min(item["start_ms"] for item in op_intervals.values())
            )
            isolated_aggressor_work_ms = sum(
                (
                    isolated_pre[(int(index), sample_index, mode)]
                    + isolated_post[(int(index), sample_index, mode)]
                ) / 2.0
                for index in spec["aggressor_indices"]
            )
            isolated_serial_work_ms = baseline + isolated_aggressor_work_ms
            ideal_overlap_ms = max(baseline, aggressor_group_span_ms)
            realized_overlap_pct = min(
                100.0, overlap_ms / baseline * 100.0
            )
            geometry_error_pct = (
                realized_overlap_pct - spec["requested_overlap_pct"]
            )
            geometry_tolerance_pct = GEOMETRY_TOLERANCE_PCT
            target_feasible_after_runtime_calibration = bool(
                spec.get(
                    "target_feasible_after_runtime_calibration",
                    spec["target_feasible_from_isolated"],
                )
            )
            valid = (
                bool(start_metadata["runtime_start_geometry_scoring_eligible"])
                and target_feasible_after_runtime_calibration
                and abs(scheduled_geometry_error_pct) <= geometry_tolerance_pct
                and victim_baseline_stable
                and pre_post_counterfactual_geometry_robust
            )
            samples.append({
                "case_id": assignment["case_id"],
                "case_key": assignment["case_key"],
                "seq_len": BASE_SEQ,
                "sample_index": sample_index,
                "stage_mode": mode,
                "dispatch_mode": spec["dispatch_mode"],
                "dispatch_order": mode,
                "victim_index": victim,
                "victim_family": participants[victim]["category"],
                "aggressor_family": "+".join(
                    participants[int(index)]["category"]
                    for index in spec["aggressor_indices"]
                ),
                "requested_overlap_pct": spec["requested_overlap_pct"],
                "predicted_max_overlap_pct": spec["predicted_max_overlap_pct"],
                "predicted_scheduled_overlap_pct": (
                    spec["predicted_scheduled_overlap_pct"]
                ),
                "predicted_constraint_error_pct": (
                    spec["predicted_constraint_error_pct"]
                ),
                "target_feasible_from_duration": (
                    spec["target_feasible_from_duration"]
                ),
                "dispatch_order_projection_feasible": spec.get(
                    "dispatch_order_projection_feasible"
                ),
                "target_feasible_after_constraints": (
                    spec["target_feasible_after_constraints"]
                ),
                "target_feasible_from_isolated": spec["target_feasible_from_isolated"],
                "target_feasible_after_runtime_calibration": (
                    target_feasible_after_runtime_calibration
                ),
                "runtime_geometry_calibration_status": spec.get(
                    "runtime_geometry_calibration_status", "not_available"
                ),
                **start_metadata,
                **compound_clock_provenance,
                "runtime_geometry_robustly_converged": spec.get(
                    "runtime_geometry_robustly_converged"
                ),
                "runtime_geometry_confirmation_trial_count": spec.get(
                    "runtime_geometry_confirmation_trial_count"
                ),
                "runtime_geometry_confirmation_passing_count": spec.get(
                    "runtime_geometry_confirmation_passing_count"
                ),
                "runtime_geometry_confirmation_geometry_passing_count": spec.get(
                    "runtime_geometry_confirmation_geometry_passing_count"
                ),
                "runtime_geometry_confirmation_baseline_stable_count": spec.get(
                    "runtime_geometry_confirmation_baseline_stable_count"
                ),
                "runtime_geometry_confirmation_jointly_passing_count": spec.get(
                    "runtime_geometry_confirmation_jointly_passing_count"
                ),
                "runtime_geometry_baseline_feedback_eligible": spec.get(
                    "runtime_geometry_baseline_feedback_eligible"
                ),
                "runtime_geometry_confirmation_error_min_pct": spec.get(
                    "runtime_geometry_confirmation_error_min_pct"
                ),
                "runtime_geometry_confirmation_error_max_pct": spec.get(
                    "runtime_geometry_confirmation_error_max_pct"
                ),
                "runtime_geometry_confirmation_scheduled_error_min_pct": spec.get(
                    "runtime_geometry_confirmation_scheduled_error_min_pct"
                ),
                "runtime_geometry_confirmation_scheduled_error_max_pct": spec.get(
                    "runtime_geometry_confirmation_scheduled_error_max_pct"
                ),
                "runtime_geometry_median_centered": spec.get(
                    "runtime_geometry_median_centered"
                ),
                "runtime_geometry_confirmation_drift_min_pct": spec.get(
                    "runtime_geometry_confirmation_drift_min_pct"
                ),
                "runtime_geometry_confirmation_drift_max_pct": spec.get(
                    "runtime_geometry_confirmation_drift_max_pct"
                ),
                "runtime_geometry_confirmation_geometry_baseline_max_abs_drift_min_pct": spec.get(
                    "runtime_geometry_confirmation_geometry_baseline_max_abs_drift_min_pct"
                ),
                "runtime_geometry_confirmation_geometry_baseline_max_abs_drift_max_pct": spec.get(
                    "runtime_geometry_confirmation_geometry_baseline_max_abs_drift_max_pct"
                ),
                "runtime_geometry_paired_baseline_ms": spec.get(
                    "runtime_geometry_paired_baseline_ms"
                ),
                "runtime_geometry_paired_baseline_drift_pct": spec.get(
                    "runtime_geometry_paired_baseline_drift_pct"
                ),
                "runtime_geometry_baseline_max_abs_drift_pct": spec.get(
                    "runtime_geometry_baseline_max_abs_drift_pct"
                ),
                "runtime_geometry_baseline_stability_basis": spec.get(
                    "runtime_geometry_baseline_stability_basis"
                ),
                "runtime_calibrated_overlap_pct": spec.get(
                    "runtime_calibrated_overlap_pct"
                ),
                "runtime_calibrated_geometry_error_pct": spec.get(
                    "runtime_calibrated_geometry_error_pct"
                ),
                "runtime_calibrated_scheduled_overlap_pct": spec.get(
                    "runtime_calibrated_scheduled_overlap_pct"
                ),
                "runtime_calibrated_scheduled_geometry_error_pct": spec.get(
                    "runtime_calibrated_scheduled_geometry_error_pct"
                ),
                "runtime_calibrated_counterfactual_overlap_pct": spec.get(
                    "runtime_calibrated_counterfactual_overlap_pct"
                ),
                "runtime_calibrated_counterfactual_geometry_error_pct": spec.get(
                    "runtime_calibrated_counterfactual_geometry_error_pct"
                ),
                "runtime_relative_shift_ms": spec.get(
                    "runtime_relative_shift_ms", 0.0
                ),
                "realized_overlap_pct": realized_overlap_pct,
                "geometry_error_pct": geometry_error_pct,
                "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
                "scheduled_overlap_pct": scheduled_overlap_pct,
                "scheduled_geometry_error_pct": scheduled_geometry_error_pct,
                "counterfactual_overlap_pct": scheduled_overlap_pct,
                "counterfactual_geometry_error_pct": (
                    scheduled_geometry_error_pct
                ),
                "geometry_validity_error_pct": scheduled_geometry_error_pct,
                "geometry_validity_basis": GEOMETRY_VALIDITY_BASIS,
                "baseline_durations_pre_ms": baseline_pre_by_operation,
                "baseline_durations_post_ms": baseline_post_by_operation,
                "geometry_baseline_drifts_pct": geometry_baseline_drifts_pct,
                "geometry_baseline_max_abs_drift_pct": (
                    geometry_baseline_max_abs_drift_pct
                ),
                "geometry_baseline_max_abs_drift_role": "diagnostic_only",
                "baseline_stability_policy": (
                    "victim_relative_drift_and_pre_post_geometry_robust_v1"
                ),
                "victim_baseline_stable": victim_baseline_stable,
                "pre_baseline_scheduled_overlap_pct": float(
                    pre_geometry["scheduled_overlap_pct"]
                ),
                "pre_baseline_scheduled_geometry_error_pct": float(
                    pre_geometry["scheduled_geometry_error_pct"]
                ),
                "post_baseline_scheduled_overlap_pct": float(
                    post_geometry["scheduled_overlap_pct"]
                ),
                "post_baseline_scheduled_geometry_error_pct": float(
                    post_geometry["scheduled_geometry_error_pct"]
                ),
                "pre_post_geometry_robust": (
                    pre_post_counterfactual_geometry_robust
                ),
                "pre_post_counterfactual_geometry_robust": (
                    pre_post_counterfactual_geometry_robust
                ),
                "pre_baseline_counterfactual_operation_intervals_ms": {
                    str(index): {
                        "start": op_intervals[index]["start_ms"],
                        "end": (
                            op_intervals[index]["start_ms"]
                            + baseline_pre_by_operation[index]
                        ),
                    }
                    for index in range(len(participants))
                },
                "post_baseline_counterfactual_operation_intervals_ms": {
                    str(index): {
                        "start": op_intervals[index]["start_ms"],
                        "end": (
                            op_intervals[index]["start_ms"]
                            + baseline_post_by_operation[index]
                        ),
                    }
                    for index in range(len(participants))
                },
                "geometry_baseline_stability_basis": (
                    "maximum_absolute_pre_post_drift_across_all_operations"
                ),
                "geometry_tolerance_pct": geometry_tolerance_pct,
                "geometry_centering_tolerance_pct": (
                    GEOMETRY_CENTERING_TOLERANCE_PCT
                ),
                "victim_isolated_pre_us": baseline_pre * 1000.0,
                "victim_isolated_post_us": baseline_post * 1000.0,
                "baseline_drift_pct": baseline_drift_pct,
                "baseline_measurement_method": BASELINE_MEASUREMENT_METHOD,
                "baseline_bracket_for_mode": mode,
                "victim_isolated_us": baseline * 1000.0,
                "victim_concurrent_us": concurrent * 1000.0,
                "victim_slowdown_pct": (concurrent / baseline - 1.0) * 100.0,
                "union_overlap_us": overlap_ms * 1000.0,
                "concurrent_overlap_us": overlap_ms * 1000.0,
                "scheduled_overlap_us": scheduled_overlap_ms * 1000.0,
                "counterfactual_overlap_us": scheduled_overlap_ms * 1000.0,
                "counterfactual_operation_intervals_ms": {
                    str(index): {
                        "start": op_intervals[index]["start_ms"],
                        "end": (
                            op_intervals[index]["start_ms"]
                            + matched_baselines_ms[index]
                        ),
                    }
                    for index in range(len(participants))
                },
                "cumulative_overlap_us": cumulative_overlap_ms * 1000.0,
                "per_aggressor_overlap": per_aggressor_overlap,
                "concurrency_width_us": {
                    key: value * 1000.0
                    for key, value in sorted(concurrency_width_ms.items())
                },
                "participant_outcomes": participant_outcomes,
                "aggressor_group_span_us": aggressor_group_span_ms * 1000.0,
                "group_makespan_us": group_makespan_ms * 1000.0,
                "isolated_serial_work_us": isolated_serial_work_ms * 1000.0,
                "ideal_overlap_us": ideal_overlap_ms * 1000.0,
                "makespan_inflation_over_ideal_pct": (
                    group_makespan_ms / ideal_overlap_ms - 1.0
                ) * 100.0,
                "serial_time_saved_pct": (
                    1.0 - group_makespan_ms / isolated_serial_work_ms
                ) * 100.0,
                "valid": valid,
            })
    annotate_adaptive_aggregate_samples(samples, adaptive_sampling)
    correctness = [
        row for result in rank_results for row in result["correctness"]
    ]
    required_operations = set(range(len(participants)))
    validated_operations = {
        int(row["operation_index"])
        for row in correctness if row.get("passed") is True
    }
    missing_correctness = sorted(required_operations - validated_operations)
    if missing_correctness:
        raise RuntimeError(
            f"correctness coverage missing operations {missing_correctness}"
        )
    if any(row.get("passed") is not True for row in correctness):
        raise RuntimeError("at least one correctness row did not pass")
    clock_calibration_history = sorted(
        [
            row
            for result in rank_results
            for row in result.get("clock_calibration_history", [])
        ],
        key=lambda row: (
            int(row["clock_calibration_generation"]),
            int(row["global_rank"]),
        ),
    )
    clock_calibration_final = sorted(
        [
            result["clock_calibration_final"]
            for result in rank_results
            if result.get("clock_calibration_final") is not None
        ],
        key=lambda row: int(row["global_rank"]),
    )
    if clock_calibration_history:
        structured_clock_calibration_history(
            clock_calibration_history,
            world_size=len(rank_results),
            session_index=0,
        )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "assignment": assignment,
        # Preserve every cell's calibration evidence even when the controller
        # correctly emits zero treatment samples for it.
        "stage_specs": stage_specs,
        "samples": samples,
        "participants": rows,
        "correctness": correctness,
        "compute_correctness_references": [
            row
            for result in rank_results
            for row in result.get("compute_correctness_references", [])
        ],
        "operation_metadata": [row for result in rank_results for row in result["operation_metadata"]],
        "trace_paths": [result["trace_path"] for result in rank_results if result["trace_path"]],
        "max_peak_memory_bytes": max(result["peak_memory_bytes"] for result in rank_results),
        "adaptive_sampling": adaptive_sampling,
        "clock_calibration_history": clock_calibration_history,
        "clock_calibration_final": clock_calibration_final,
        "runtime_health": {
            "audit": aggregate_runtime_health_audit(rank_results),
            "rank_snapshots": [
                {
                    "global_rank": int(result["global_rank"]),
                    **result.get("runtime_health", {}),
                }
                for result in sorted(
                    rank_results, key=lambda row: int(row["global_rank"])
                )
            ],
        },
    }


def default_checkpoint_path(output_path: Path) -> Path:
    """Place durable resume state next to the requested final document."""

    return output_path.with_name(f"{output_path.name}.checkpoint.json")


def build_restart_configuration_identity(
    args: argparse.Namespace,
    *,
    node_ips: tuple[str, ...],
    world_size: int,
    coverage_targets: list[float],
    adaptive_config: AdaptiveBenchmarkControllerConfig | None,
    assignments: list[dict[str, Any]],
    communicator_plan_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the fail-closed identity for checkpoint compatibility.

    Output and checkpoint paths are intentionally excluded: moving the durable
    artifacts does not change the experiment.  Input contents, case mappings,
    topology, sampling, scheduling, and profiler behavior are included.
    """

    def artifact(path: Path) -> dict[str, str]:
        return {
            "path": str(path.resolve()),
            "sha256": file_digest(path),
        }

    sampling = {
        "mode": "adaptive" if adaptive_config is not None else "fixed",
        "warmups": int(args.warmups),
        "fixed_iterations": (
            None if adaptive_config is not None else int(args.iterations)
        ),
        "adaptive": (
            None if adaptive_config is None else adaptive_config.to_dict()
        ),
    }
    return {
        "restart_identity_schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation_sha256": implementation_sha256(),
        "model": MODEL_NAME,
        "model_config": MODEL_CONFIG,
        "sequence_length": int(args.seq_len),
        "cases": assignments,
        "case_keys": [row["case_key"] for row in assignments],
        "topology": {
            "node_ips": list(node_ips),
            "world_size": int(world_size),
            "gpus_per_node": 8,
            "physical_pp": 2,
            "dp": 8,
            "ep": 8,
            "placement": "pp_outer",
        },
        "communicator_plan": communicator_plan_payload,
        "sampling": sampling,
        "coverage_targets_pct": coverage_targets,
        "scheduling": {
            "mechanism": "global_clock_aligned_host_launch",
            **strict_scheduling_provenance(),
            "baseline_measurement": BASELINE_MEASUREMENT_METHOD,
            "baseline_measurement_repetitions": BASELINE_MEASUREMENT_REPETITIONS,
            "geometry_confirmation_baseline_repetitions": (
                BASELINE_MEASUREMENT_REPETITIONS
            ),
            "same_dispatch_start_anchor_policy": START_ANCHOR_POLICY,
            "same_dispatch_start_anchor_repetitions": START_ANCHOR_REPETITIONS,
            "same_dispatch_start_anchor_preliminary_repetitions": (
                START_ANCHOR_REPETITIONS
            ),
            "same_dispatch_start_anchor_confirmation_policy": (
                START_ANCHOR_CONFIRMATION_POLICY
            ),
            "same_dispatch_start_anchor_confirmation_repetitions": (
                START_ANCHOR_REPETITIONS
            ),
            "same_dispatch_start_anchor_confirmation_minimum_passing": (
                START_ANCHOR_MINIMUM_PASSING
            ),
            "same_dispatch_start_anchor_unscored": True,
            "sample_priming": "one_unmeasured_launch_per_operation",
            "bracket_scope": "adjacent_per_treatment_cell",
            "bracket_operation_order": (
                "pre_nonvictims_then_victim__post_victim_then_nonvictims"
            ),
            "stage_lead_ms": float(args.stage_lead_ms),
            "geometry_feedback_candidate_shifts_max": int(
                args.geometry_calibration_trials
            ),
            "geometry_confirmation_trials_per_candidate": (
                GEOMETRY_CONFIRMATION_TRIALS
            ),
            "geometry_confirmation_minimum_passing": (
                GEOMETRY_CONFIRMATION_MINIMUM_PASSING
            ),
            "geometry_confirmation_baseline_measurement": (
                BASELINE_MEASUREMENT_METHOD
            ),
            "geometry_confirmation_joint_gate": (
                "abs_scheduled_geometry_error_le_10_and_victim_abs_matched_"
                "baseline_drift_le_5_and_pre_post_counterfactual_geometry_"
                "errors_le_10_and_abs_candidate_median_error_le_5"
            ),
            "geometry_confirmation_drift_tolerance_pct": 5.0,
            "geometry_candidate_median_centering_tolerance_pct": (
                GEOMETRY_CENTERING_TOLERANCE_PCT
            ),
            "geometry_validity_basis": GEOMETRY_VALIDITY_BASIS,
            "geometry_baseline_stability_basis": (
                "maximum_absolute_pre_post_drift_across_all_operations"
            ),
            "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
            "legacy_realized_overlap_fields_are_concurrent_outcomes": True,
            "geometry_confirmation_launches_max_per_feasible_stage": (
                int(args.geometry_calibration_trials)
                * GEOMETRY_CONFIRMATION_TRIALS
            ),
            "geometry_max_relative_shift_ms": float(
                args.geometry_max_relative_shift_ms
            ),
            "geometry_tolerance_pct": GEOMETRY_TOLERANCE_PCT,
        },
        "profiler": {
            "enabled": args.profile_dir is not None,
            "directory": (
                None
                if args.profile_dir is None
                else str(args.profile_dir.resolve())
            ),
        },
        "inputs": {
            "matrix": artifact(args.matrix),
            "a2a_profiles": artifact(args.a2a_profiles),
            "trace_events": artifact(args.trace_events),
        },
        "nccl_behavior": "defaults_only",
        "case_recovery_granularity": "case_boundary",
    }


def build_final_output_document(
    *,
    preview: dict[str, Any],
    checkpoint_state: dict[str, Any],
    checkpoint_path: Path,
    resumed: bool,
    skipped_case_keys: list[str],
    executed_case_keys: list[str],
    placement: dict[str, Any],
    clock_calibration: list[dict[str, Any]],
    driver_audit: dict[str, Any],
    publisher_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a fully reconciled output from durable checkpoint state."""

    restart_audit = checkpoint_audit(checkpoint_state, resumed=resumed)
    restart_audit.update({
        "checkpoint_path": str(checkpoint_path),
        "skipped_completed_case_keys": skipped_case_keys,
        "executed_case_keys": executed_case_keys,
        "execution_sessions": checkpoint_state.get("execution_sessions", []),
        "artifact_publication": "atomic_json_via_ray_cpu_actor",
        "artifact_publisher": publisher_receipt,
    })
    if not restart_audit["all_requested_cases_reconciled"]:
        raise RuntimeError("final output requested cases are not reconciled")
    placements = placement.get("ranks", [])
    sessions = checkpoint_state.get("execution_sessions", [])
    structured_history = [
        entry
        for session in sessions
        for entry in session.get("clock_calibration_history", [])
    ]
    structured_history.sort(key=lambda entry: int(entry["generation"]))
    latest_generation = max(
        (int(entry["generation"]) for entry in structured_history),
        default=0,
    )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "configuration_identity": checkpoint_state["configuration_identity"],
        "run_id": checkpoint_state["run_id"],
        "source_run_id": checkpoint_state["run_id"],
        "config_id": checkpoint_state["config_id"],
        "configuration": preview,
        "placement": placement,
        "clock_calibration_schema_version": CLOCK_CALIBRATION_SCHEMA_VERSION,
        "clock_calibration_generation": latest_generation,
        "clock_calibration": clock_calibration,
        "clock_calibration_history": structured_history,
        "results": ordered_results(checkpoint_state),
        "restart_provenance": restart_audit,
        "environment": {
            "driver_audit": driver_audit,
            "all_workers_default_nccl_behavior": bool(placements) and all(
                not row["effective_nccl_overrides"] for row in placements
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="10.65.10.75")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--node-ips", default="10.65.10.75,10.65.26.217")
    parser.add_argument(
        "--matrix", type=Path,
        default=Path("out/20260809_pairwise_overlap_foundation/EXACT_KERNEL_EXECUTION_MATRIX.json"),
    )
    parser.add_argument(
        "--a2a-profiles", type=Path,
        default=Path("out/20260809_pairwise_overlap_foundation/ragged_a2a_payloads.json"),
    )
    parser.add_argument("--cases", default="A04:C-GROUPED,A20,M02")
    parser.add_argument(
        "--trace-events", type=Path,
        default=Path("out/20260809_132048/kernel_overlap_events.csv"),
    )
    parser.add_argument("--seq-len", type=int, default=BASE_SEQ)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--baseline-repetitions",
        type=int,
        default=BASELINE_MEASUREMENT_REPETITIONS,
        help="Odd-K isolated baseline repetitions; authoritative policy requires 3",
    )
    parser.add_argument(
        "--adaptive-sampling",
        action="store_true",
        help=(
            "Replace fixed --iterations treatment sampling with per-cell "
            "bootstrap-CI convergence"
        ),
    )
    parser.add_argument(
        "--adaptive-target-ci-half-width-pp",
        type=float,
        default=1.5,
        help="Target 95%% median-CI half-width in percentage points",
    )
    parser.add_argument(
        "--adaptive-max-valid-samples",
        type=int,
        default=500,
        help="Hard valid-sample cap per adaptive cell",
    )
    parser.add_argument(
        "--adaptive-attempt-multiplier",
        type=float,
        default=3.0,
        help="Total-attempt cap as a multiple of the valid-sample cap",
    )
    parser.add_argument(
        "--adaptive-batch-size",
        type=int,
        default=10,
        help="Maximum attempts issued per active cell before reevaluating its CI",
    )
    parser.add_argument(
        "--stage-lead-ms", type=float, default=5.0,
        help="Host-clock lead time used to align ranks before each stage",
    )
    parser.add_argument(
        "--coverage-targets", default="100",
        help="Comma-separated requested victim coverage percentages",
    )
    parser.add_argument(
        "--geometry-calibration-trials",
        type=int,
        default=DEFAULT_GEOMETRY_CALIBRATION_TRIALS,
        help=(
            "Maximum geometry-feedback candidate shifts per feasible stage; "
            "each candidate runs three fixed-shift confirmation launches"
        ),
    )
    parser.add_argument(
        "--geometry-max-relative-shift-ms",
        type=float,
        default=DEFAULT_GEOMETRY_MAX_RELATIVE_SHIFT_MS,
        help="Maximum extra delay applied to the later dispatch group",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Atomic case-boundary checkpoint path (default: "
            "<output>.checkpoint.json)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching checkpoint and skip its completed cases",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Explicit run ID for a new checkpoint; on resume it must match "
            "the persisted run ID"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    driver_audit = sanitize_nccl_environment(phase="pairwise_driver_before_ray_or_torch_import")
    assert_default_nccl_environment()
    args = parse_args()
    node_ips = tuple(item.strip() for item in args.node_ips.split(",") if item.strip())
    if len(node_ips) != 2:
        raise ValueError("--node-ips requires two IPs")
    coverage_targets = [
        float(item.strip()) for item in args.coverage_targets.split(",")
        if item.strip()
    ]
    validate_coverage_targets(coverage_targets)
    if args.baseline_repetitions != BASELINE_MEASUREMENT_REPETITIONS:
        raise ValueError(
            "--baseline-repetitions must be 3 for the authoritative odd-K policy"
        )
    if args.geometry_calibration_trials < 0:
        raise ValueError("--geometry-calibration-trials must be nonnegative")
    if args.geometry_max_relative_shift_ms < 0.0:
        raise ValueError("--geometry-max-relative-shift-ms must be nonnegative")
    adaptive_config = None
    if args.adaptive_sampling:
        # A 100-sample floor applies to kernels at or below 10 ms, so values
        # below 100 cannot support every calibrated cell in a mixed case.
        if args.adaptive_max_valid_samples < 100:
            raise ValueError("--adaptive-max-valid-samples must be at least 100")
        adaptive_config = AdaptiveBenchmarkControllerConfig(
            sampling=AdaptiveSamplingConfig(
                target_ci_half_width_pp=args.adaptive_target_ci_half_width_pp,
                max_valid_samples=args.adaptive_max_valid_samples,
            ),
            max_total_attempt_multiplier=args.adaptive_attempt_multiplier,
            attempt_batch_size=args.adaptive_batch_size,
        )
    topology = PPOuterTopology(node_ips=node_ips)
    assignments = load_cases(
        args.matrix,
        [item.strip() for item in args.cases.split(",") if item.strip()],
    )
    attach_trace_start_offsets(assignments, args.trace_events)
    a2a_profiles = load_a2a_profiles(args.a2a_profiles)
    comm_plan = communicator_plan(assignments)
    restart_identity = build_restart_configuration_identity(
        args,
        node_ips=node_ips,
        world_size=topology.world_size,
        coverage_targets=coverage_targets,
        adaptive_config=adaptive_config,
        assignments=assignments,
        communicator_plan_payload=comm_plan,
    )
    config_id = run_config_id(restart_identity)
    output_path = args.output.resolve()
    checkpoint_path = (
        args.checkpoint or default_checkpoint_path(output_path)
    ).resolve()
    requested_case_keys = [row["case_key"] for row in assignments]
    preview = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation_sha256": restart_identity["implementation_sha256"],
        "model": MODEL_NAME,
        "model_config": MODEL_CONFIG,
        "sequence_length": args.seq_len,
        "coverage_targets_pct": coverage_targets,
        "sampling": {
            "mode": "adaptive" if adaptive_config is not None else "fixed",
            "fixed_iterations": (
                None if adaptive_config is not None else args.iterations
            ),
            "adaptive": (
                None if adaptive_config is None else adaptive_config.to_dict()
            ),
        },
        "cases": assignments,
        "topology": {
            "node_ips": node_ips,
            "world_size": topology.world_size,
            "gpus_per_node": 8,
            "physical_pp": 2,
            "dp": 8,
            "ep": 8,
            "placement": "pp_outer",
        },
        "communicator_plan": comm_plan,
        "scheduling": {
            "mechanism": "global_clock_aligned_host_launch",
            **strict_scheduling_provenance(),
            "baseline_measurement": BASELINE_MEASUREMENT_METHOD,
            "baseline_measurement_repetitions": BASELINE_MEASUREMENT_REPETITIONS,
            "geometry_confirmation_baseline_repetitions": (
                BASELINE_MEASUREMENT_REPETITIONS
            ),
            "same_dispatch_start_anchor_policy": START_ANCHOR_POLICY,
            "same_dispatch_start_anchor_repetitions": START_ANCHOR_REPETITIONS,
            "same_dispatch_start_anchor_preliminary_repetitions": (
                START_ANCHOR_REPETITIONS
            ),
            "same_dispatch_start_anchor_confirmation_policy": (
                START_ANCHOR_CONFIRMATION_POLICY
            ),
            "same_dispatch_start_anchor_confirmation_repetitions": (
                START_ANCHOR_REPETITIONS
            ),
            "same_dispatch_start_anchor_confirmation_minimum_passing": (
                START_ANCHOR_MINIMUM_PASSING
            ),
            "same_dispatch_start_anchor_unscored": True,
            "sample_priming": "one_unmeasured_launch_per_operation",
            "bracket_scope": "adjacent_per_treatment_cell",
            "bracket_operation_order": (
                "pre_nonvictims_then_victim__post_victim_then_nonvictims"
            ),
            "stage_lead_ms": args.stage_lead_ms,
            "device_delay_kernels": False,
            "geometry_feedback": {
                "candidate_shifts_max": args.geometry_calibration_trials,
                "confirmation_trials_per_candidate": (
                    GEOMETRY_CONFIRMATION_TRIALS
                ),
                "confirmation_minimum_passing": (
                    GEOMETRY_CONFIRMATION_MINIMUM_PASSING
                ),
                "confirmation_baseline_measurement": (
                    BASELINE_MEASUREMENT_METHOD
                ),
                "confirmation_baseline_repetitions": (
                    BASELINE_MEASUREMENT_REPETITIONS
                ),
                "confirmation_joint_gate": (
                    "abs_scheduled_geometry_error_le_10_and_victim_abs_matched_"
                "baseline_drift_le_5_and_pre_post_counterfactual_geometry_"
                "errors_le_10_and_abs_candidate_median_error_le_5"
                ),
                "confirmation_drift_tolerance_pct": 5.0,
                "candidate_median_centering_tolerance_pct": (
                    GEOMETRY_CENTERING_TOLERANCE_PCT
                ),
                "confirmation_launches_max_per_feasible_stage": (
                    args.geometry_calibration_trials
                    * GEOMETRY_CONFIRMATION_TRIALS
                ),
                "tolerance_pct": GEOMETRY_TOLERANCE_PCT,
                "validity_basis": GEOMETRY_VALIDITY_BASIS,
                "geometry_baseline_stability_basis": (
                    "maximum_absolute_pre_post_drift_across_all_operations"
                ),
                "counterfactual_duration_measurement": (
                    BASELINE_MEASUREMENT_METHOD
                ),
                "concurrent_overlap_role": CONCURRENT_OVERLAP_ROLE,
                "legacy_realized_overlap_fields_are_concurrent_outcomes": True,
                "validity_uses_counterfactual_geometry": True,
                "max_relative_shift_ms": args.geometry_max_relative_shift_ms,
                "uses_globally_aligned_focal_intervals": True,
                "preserves_dispatch_and_natural_group_order": True,
            },
        },
        "nccl_environment_audit": driver_audit,
        "restart": {
            "config_id": config_id,
            "checkpoint_path": str(checkpoint_path),
            "resume_requested": bool(args.resume),
            "requested_run_id": args.run_id,
            "recovery_granularity": "case_boundary",
            "mid_case_actor_state_recoverable": False,
        },
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    if args.resume:
        if not checkpoint_path.exists():
            raise ValueError(
                f"--resume requires existing checkpoint {checkpoint_path}"
            )
        checkpoint_state = validate_checkpoint(
            load_checkpoint(checkpoint_path),
            expected_config_id=config_id,
            expected_configuration_identity=restart_identity,
            expected_case_keys=requested_case_keys,
        )
        if args.run_id is not None and args.run_id != checkpoint_state["run_id"]:
            raise ValueError(
                "--run-id does not match resumed checkpoint: "
                f"{args.run_id!r} != {checkpoint_state['run_id']!r}"
            )
    else:
        if checkpoint_path.exists():
            raise ValueError(
                f"checkpoint {checkpoint_path} already exists; use --resume "
                "with an identical configuration or choose a new output"
            )
        if output_path.exists():
            raise ValueError(
                f"output {output_path} already exists without a resumable "
                "checkpoint; refusing to overwrite"
            )
        checkpoint_state = new_checkpoint(
            config_id=config_id,
            configuration_identity=restart_identity,
            requested_case_keys=requested_case_keys,
            run_id=args.run_id,
        )
    checkpoint_state.setdefault("execution_sessions", [])
    skipped_case_keys = [
        key for key in requested_case_keys
        if key in checkpoint_state["completed_cases"]
    ]
    pending_assignments = [
        assignment for assignment in assignments
        if assignment["case_key"] not in checkpoint_state["completed_cases"]
    ]
    preview["restart"]["run_id"] = checkpoint_state["run_id"]
    preview["restart"]["completed_cases_at_start"] = skipped_case_keys

    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    class GlobalClock:
        def now_ns(self) -> int:
            return time.monotonic_ns()

    class ArtifactPublisher:
        """Publish durable artifacts from a Ray node with writable Lustre."""

        def publish(self, path: str, document: dict[str, Any]) -> dict[str, Any]:
            destination = Path(path)
            atomic_write_json(destination, document)
            return {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "hostname": socket.gethostname(),
                "node_ip": socket.gethostbyname(socket.gethostname()),
            }

    ray.init(
        address=f"{args.address}:{args.port}",
        namespace=f"pairwise_overlap_benchmark_{os.getpid()}_{time.time_ns()}",
        runtime_env={"env_vars": nccl_debug_environment()},
        log_to_driver=False,
    )
    # Reserve no schedulable CPU so publication cannot consume one of the
    # exact 8-CPU placement bundles used by the 16 GPU rank actors. Pin it to
    # the first requested cluster node so writable-mount placement is explicit.
    publisher_type = ray.remote(num_cpus=0, max_restarts=0)(ArtifactPublisher)
    publisher = publisher_type.options(
        resources={f"node:{node_ips[0]}": 0.001}
    ).remote()
    publisher_receipt = ray.get(
        publisher.publish.remote(str(checkpoint_path), checkpoint_state),
        timeout=120,
    )

    # A fully committed resume is an idempotent output reconstruction. It must
    # not reserve GPUs or recreate NCCL communicators merely to rewrite JSON.
    if not pending_assignments:
        try:
            sessions = checkpoint_state.get("execution_sessions", [])
            if not sessions:
                raise RuntimeError(
                    "completed checkpoint lacks a prior execution session"
                )
            last_session = sessions[-1]
            output = build_final_output_document(
                preview=preview,
                checkpoint_state=checkpoint_state,
                checkpoint_path=checkpoint_path,
                resumed=args.resume,
                skipped_case_keys=skipped_case_keys,
                executed_case_keys=[],
                placement=last_session["placement"],
                clock_calibration=last_session["clock_calibration"],
                driver_audit=driver_audit,
                publisher_receipt=publisher_receipt,
            )
            ray.get(
                publisher.publish.remote(str(output_path), output), timeout=300
            )
            print(
                "Resume checkpoint already contains all requested cases; "
                "no GPUs reserved",
                flush=True,
            )
            print(output_path)
        finally:
            ray.kill(publisher, no_restart=True)
            ray.shutdown()
        return

    print("Ray connected; reserving two 8-GPU node bundles", flush=True)
    pg = placement_group(
        [
            {"CPU": 8, "GPU": 8, f"node:{node_ip}": 0.001}
            for node_ip in node_ips
        ],
        strategy="STRICT_SPREAD",
    )
    actor_type = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)(PairwiseOverlapRank)
    clock_type = ray.remote(num_cpus=0, max_concurrency=32)(GlobalClock)
    actors: list[Any] = []
    clock = None
    try:
        ray.get(pg.ready(), timeout=600)
        print("Placement group ready; starting 16 GPU actors", flush=True)
        for global_rank in range(topology.world_size):
            pp_rank = topology.physical_pp_rank(global_rank)
            actors.append(
                actor_type.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=pp_rank
                    )
                ).remote(global_rank, topology.node_ip_for_rank(global_rank))
            )
        placements = ray.get([actor.metadata.remote() for actor in actors], timeout=300)
        print("Actor placement validated; creating required communicators", flush=True)
        placement_validation = validate_observed_placement(topology, placements)
        placement_validation.update(
            validate_worker_implementation_provenance(
                placements, restart_identity["implementation_sha256"]
            )
        )
        master_addr, master_port = ray.get(actors[0].rendezvous.remote())
        ray.get(
            [actor.initialize.remote(master_addr, master_port, topology.__dict__, comm_plan) for actor in actors],
            timeout=1_800,
        )
        print("Communicators ready; calibrating shared clock", flush=True)
        clock = clock_type.remote()
        prior_clock_generation = max(
            (
                int(session.get("clock_calibration_generation", 0))
                for session in checkpoint_state["execution_sessions"]
            ),
            default=0,
        )
        ray.get([
            actor.set_clock_generation_floor.remote(prior_clock_generation)
            for actor in actors
        ], timeout=120)
        clock_calibration = ray.get([
            actor.calibrate_clock.remote(
                clock, CLOCK_CALIBRATION_PROBES, "suite_initial"
            )
            for actor in actors
        ], timeout=300)
        session_index = len(checkpoint_state["execution_sessions"])
        initial_clock_history = structured_clock_calibration_history(
            clock_calibration,
            world_size=topology.world_size,
            session_index=session_index,
        )
        placements = ray.get(
            [actor.metadata.remote() for actor in actors], timeout=300
        )
        placement_validation = validate_observed_placement(topology, placements)
        placement_validation.update(
            validate_worker_implementation_provenance(
                placements, restart_identity["implementation_sha256"]
            )
        )
        checkpoint_state["execution_sessions"].append({
            "session_index": session_index,
            "ray_address": f"{args.address}:{args.port}",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "placement": {
                "validation": placement_validation,
                "ranks": placements,
            },
            "clock_calibration_schema_version": (
                CLOCK_CALIBRATION_SCHEMA_VERSION
            ),
            "clock_calibration_generation": initial_clock_history[-1][
                "generation"
            ],
            "clock_calibration": clock_calibration,
            "clock_calibration_history": initial_clock_history,
            "artifact_publisher": publisher_receipt,
        })
        ray.get(
            publisher.publish.remote(str(checkpoint_path), checkpoint_state),
            timeout=120,
        )
        print("Clock calibrated; starting timed cases", flush=True)
        if skipped_case_keys:
            print(
                "Resume validated; skipping completed cases: "
                + ", ".join(skipped_case_keys),
                flush=True,
            )
        for assignment in pending_assignments:
            print(f"Running {assignment['case_key']}", flush=True)
            rank_results = ray.get(
                [
                    actor.benchmark.remote(
                        assignment,
                        args.seq_len,
                        a2a_profiles,
                        args.warmups,
                        args.iterations,
                        int(round(args.stage_lead_ms * 1_000_000)),
                        coverage_targets,
                        str(args.profile_dir) if args.profile_dir else None,
                        args.geometry_calibration_trials,
                        args.geometry_max_relative_shift_ms,
                        clock,
                        (
                            None
                            if adaptive_config is None
                            else adaptive_config.to_dict()
                        ),
                    )
                    for actor in actors
                ],
                timeout=3_600,
            )
            aggregated = aggregate_case(
                assignment,
                rank_results,
                args.iterations if adaptive_config is None else None,
            )
            aggregated["clock_calibration_session_index"] = session_index
            case_clock_history = structured_clock_calibration_history(
                aggregated["clock_calibration_history"],
                world_size=topology.world_size,
                session_index=session_index,
            )
            current_session = checkpoint_state["execution_sessions"][-1]
            history_by_generation = {
                int(entry["generation"]): entry
                for entry in current_session["clock_calibration_history"]
            }
            for entry in case_clock_history:
                generation = int(entry["generation"] )
                previous = history_by_generation.get(generation)
                if previous is not None and previous != entry:
                    raise RuntimeError(
                        f"clock generation {generation} changed across cases"
                    )
                history_by_generation[generation] = entry
            current_session["clock_calibration_history"] = [
                history_by_generation[generation]
                for generation in sorted(history_by_generation)
            ]
            clock_calibration = aggregated["clock_calibration_final"]
            current_session["clock_calibration"] = clock_calibration
            current_session["clock_calibration_generation"] = max(
                int(entry["generation"])
                for entry in current_session["clock_calibration_history"]
            )
            placements = ray.get(
                [actor.metadata.remote() for actor in actors], timeout=300
            )
            placement_validation = validate_observed_placement(
                topology, placements
            )
            placement_validation.update(
                validate_worker_implementation_provenance(
                    placements, restart_identity["implementation_sha256"]
                )
            )
            current_session["placement"] = {
                "validation": placement_validation,
                "ranks": placements,
            }
            for sample in aggregated["samples"]:
                sample["seq_len"] = args.seq_len
            aggregated = annotate_and_reconcile_case(
                aggregated,
                run_id=checkpoint_state["run_id"],
                config_id=config_id,
            )
            checkpoint_state = commit_case(checkpoint_state, aggregated)
            ray.get(
                publisher.publish.remote(
                    str(checkpoint_path), checkpoint_state
                ),
                timeout=300,
            )
            medians = summarize([
                row["victim_slowdown_pct"] for row in aggregated["samples"]
            ])
            print(json.dumps({
                "case": assignment["case_key"],
                "slowdown_pct": medians,
                "checkpoint_generation": checkpoint_state[
                    "checkpoint_generation"
                ],
            }))
        output = build_final_output_document(
            preview=preview,
            checkpoint_state=checkpoint_state,
            checkpoint_path=checkpoint_path,
            resumed=args.resume,
            skipped_case_keys=skipped_case_keys,
            executed_case_keys=[
                row["case_key"] for row in pending_assignments
            ],
            placement={"validation": placement_validation, "ranks": placements},
            clock_calibration=clock_calibration,
            driver_audit=driver_audit,
            publisher_receipt=publisher_receipt,
        )
        ray.get(
            publisher.publish.remote(str(output_path), output), timeout=300
        )
        print(output_path)
    finally:
        if actors:
            try:
                ray.get([actor.shutdown.remote() for actor in actors], timeout=120)
            except Exception:
                pass
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if clock is not None:
            ray.kill(clock, no_restart=True)
        if publisher is not None:
            ray.kill(publisher, no_restart=True)
        remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
