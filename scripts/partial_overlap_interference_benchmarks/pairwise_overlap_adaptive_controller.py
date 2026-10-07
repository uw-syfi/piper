# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU-side adaptive controller for pairwise-overlap benchmark samples.

The GPU runner produces one aggregate row per attempted treatment.  This
module groups those rows into independently sampled cells, applies the corpus
validity gates, and feeds only accepted slowdown observations to
``AdaptiveSamplingCell``.  It deliberately has no Ray, CUDA, or PyTorch
dependency, which keeps stopping decisions deterministic and easy to test.

A cell is ``(case, victim_index, requested_overlap_pct, dispatch_mode)``.
Sequence length is normally fixed for one runner invocation; callers that mix
sequence lengths should include it in ``case_key`` or use one controller per
sequence length.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
import statistics
from typing import Any, Iterable, Mapping

from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_sampling import (
    AdaptiveSamplingCell,
    AdaptiveSamplingConfig,
)


DEFAULT_DRIFT_THRESHOLD_PCT = 5.0
DEFAULT_GEOMETRY_TOLERANCE_PCT = 10.0
DEFAULT_MAX_TOTAL_ATTEMPT_MULTIPLIER = 3.0
DEFAULT_ATTEMPT_BATCH_SIZE = 10
PRE_SAMPLING_READY = "ready"
PRE_SAMPLING_PROVEN_INFEASIBLE = "proven_infeasible"
PRE_SAMPLING_CALIBRATION_UNRESOLVED = "calibration_unresolved"
INFEASIBILITY_PROOF_DURATION_BOUND = "duration_bound"
COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS = (
    "counterfactual_intervals_from_actual_global_gpu_starts_and_"
    "matched_isolated_pre_post_durations"
)
GEOMETRY_BASELINE_STABILITY_BASIS = (
    "maximum_absolute_pre_post_drift_across_all_operations"
)


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _finite_float_mapping(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    parsed: dict[str, float] = {}
    for key, item in value.items():
        number = _finite_float(item)
        if number is None:
            return None
        parsed[str(key)] = number
    return parsed


@dataclass(frozen=True, order=True)
class AdaptiveBenchmarkCellKey:
    """Stable identity for one independently sampled overlap cell."""

    case_key: str
    victim_index: int
    requested_overlap_pct: float
    dispatch_mode: str

    def __post_init__(self) -> None:
        if not self.case_key:
            raise ValueError("case_key must be non-empty")
        if self.victim_index < 0:
            raise ValueError("victim_index must be non-negative")
        if not math.isfinite(self.requested_overlap_pct):
            raise ValueError("requested_overlap_pct must be finite")
        if not self.dispatch_mode:
            raise ValueError("dispatch_mode must be non-empty")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "AdaptiveBenchmarkCellKey":
        case_key = str(row.get("case_key") or row.get("case_id") or "")
        victim = _finite_float(row.get("victim_index"))
        target = _finite_float(row.get("requested_overlap_pct"))
        dispatch_mode = str(row.get("dispatch_mode") or "")
        if victim is None or int(victim) != victim:
            raise ValueError("sample victim_index must be an integer")
        if target is None:
            raise ValueError("sample requested_overlap_pct must be finite")
        return cls(case_key, int(victim), target, dispatch_mode)

    @property
    def canonical(self) -> str:
        return (
            f"{self.case_key}:v{self.victim_index}:"
            f"o{self.requested_overlap_pct:g}:{self.dispatch_mode}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateTrialValidity:
    """Validity classification for one aggregate treatment row."""

    valid: bool
    slowdown_pp: float | None
    reasons: tuple[str, ...]
    explicitly_infeasible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveBenchmarkControllerConfig:
    """Corpus gates and stopping limits for the adaptive controller."""

    sampling: AdaptiveSamplingConfig = field(default_factory=AdaptiveSamplingConfig)
    drift_threshold_pct: float = DEFAULT_DRIFT_THRESHOLD_PCT
    default_geometry_tolerance_pct: float = DEFAULT_GEOMETRY_TOLERANCE_PCT
    max_total_attempt_multiplier: float = DEFAULT_MAX_TOTAL_ATTEMPT_MULTIPLIER
    attempt_batch_size: int = DEFAULT_ATTEMPT_BATCH_SIZE

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.drift_threshold_pct)) or float(
            self.drift_threshold_pct
        ) < 0.0:
            raise ValueError("drift_threshold_pct must be finite and non-negative")
        if not math.isfinite(float(self.default_geometry_tolerance_pct)) or float(
            self.default_geometry_tolerance_pct
        ) < 0.0:
            raise ValueError(
                "default_geometry_tolerance_pct must be finite and non-negative"
            )
        multiplier = float(self.max_total_attempt_multiplier)
        if not math.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError("max_total_attempt_multiplier must be at least 1")
        if int(self.attempt_batch_size) <= 0:
            raise ValueError("attempt_batch_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sampling"] = asdict(self.sampling)
        return result


def classify_aggregate_trial(
    row: Mapping[str, Any],
    config: AdaptiveBenchmarkControllerConfig,
    *,
    default_correctness: bool | None = None,
) -> AggregateTrialValidity:
    """Apply geometry, drift, correctness, and finite-measurement gates.

    ``valid`` from the runner is treated as a geometry gate, then checked
    independently against scheduled counterfactual error and tolerance.
    Legacy rows without the new field deliberately fall back to their
    historical concurrent ``geometry_error_pct`` semantics.
    """

    reasons: list[str] = []
    feasible = _optional_bool(
        row.get(
            "target_feasible_after_runtime_calibration",
            row.get("target_feasible_from_isolated"),
        )
    )
    explicitly_infeasible = feasible is False
    if explicitly_infeasible:
        reasons.append("geometry_target_infeasible")

    runner_valid = _optional_bool(row.get("valid"))
    if runner_valid is not True:
        reasons.append(
            "runner_geometry_invalid"
            if runner_valid is False
            else "runner_geometry_unknown"
        )

    counterfactual_schema_fields = (
        "geometry_validity_error_pct",
        "scheduled_geometry_error_pct",
        "counterfactual_geometry_error_pct",
        "geometry_validity_basis",
        "geometry_baseline_drifts_pct",
        "geometry_baseline_max_abs_drift_pct",
        "geometry_baseline_stability_basis",
    )
    has_counterfactual_error = any(
        row.get(field) not in (None, "", {})
        for field in counterfactual_schema_fields
    )
    geometry_error = _finite_float(
        row.get(
            "geometry_validity_error_pct",
            row.get("scheduled_geometry_error_pct"),
        )
        if has_counterfactual_error
        else row.get("geometry_error_pct")
    )
    if has_counterfactual_error and row.get("geometry_validity_basis") != (
        COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
    ):
        reasons.append("geometry_validity_basis_mismatch")
    if has_counterfactual_error:
        scheduled_error = _finite_float(
            row.get("scheduled_geometry_error_pct")
        )
        validity_error = _finite_float(
            row.get("geometry_validity_error_pct")
        )
        counterfactual_error = _finite_float(
            row.get("counterfactual_geometry_error_pct")
        )
        if any(
            value is None
            for value in (
                scheduled_error,
                validity_error,
                counterfactual_error,
            )
        ) or any(
            not math.isclose(
                value,
                validity_error,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for value in (scheduled_error, counterfactual_error)
        ):
            reasons.append("geometry_error_alias_mismatch")
    geometry_tolerance = _finite_float(row.get("geometry_tolerance_pct"))
    if geometry_tolerance is None:
        geometry_tolerance = float(config.default_geometry_tolerance_pct)
    if geometry_error is None:
        reasons.append("geometry_error_unknown")
    elif abs(geometry_error) > geometry_tolerance:
        reasons.append("geometry_out_of_tolerance")

    if has_counterfactual_error:
        geometry_baseline_drift = _finite_float(
            row.get("geometry_baseline_max_abs_drift_pct")
        )
        if row.get("geometry_baseline_stability_basis") != (
            GEOMETRY_BASELINE_STABILITY_BASIS
        ):
            reasons.append("geometry_baseline_stability_basis_mismatch")
        per_operation_drifts = _finite_float_mapping(
            row.get("geometry_baseline_drifts_pct")
        )
        recomputed_max_drift = (
            None
            if per_operation_drifts is None
            else max(abs(value) for value in per_operation_drifts.values())
        )
        if (
            recomputed_max_drift is None
            or geometry_baseline_drift is None
            or not math.isclose(
                recomputed_max_drift,
                geometry_baseline_drift,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            reasons.append("geometry_baseline_drift_summary_mismatch")
        if geometry_baseline_drift is None:
            reasons.append("geometry_baseline_drift_unknown")
        elif geometry_baseline_drift > float(config.drift_threshold_pct):
            reasons.append("geometry_baseline_drift_out_of_tolerance")

    drift = _finite_float(row.get("baseline_drift_pct"))
    if drift is None:
        reasons.append("baseline_drift_unknown")
    elif abs(drift) > float(config.drift_threshold_pct):
        reasons.append("baseline_drift_out_of_tolerance")

    correct = _optional_bool(row.get("correct"))
    if correct is None:
        correct = default_correctness
    if correct is not True:
        reasons.append("correctness_failed" if correct is False else "correctness_unknown")

    slowdown = _finite_float(row.get("victim_slowdown_pct"))
    if slowdown is None:
        reasons.append("slowdown_unknown_or_nonfinite")

    return AggregateTrialValidity(
        valid=not reasons,
        slowdown_pp=slowdown if not reasons else None,
        reasons=tuple(reasons),
        explicitly_infeasible=explicitly_infeasible,
    )


def _isolated_duration_ms(row: Mapping[str, Any]) -> float | None:
    milliseconds = _finite_float(row.get("isolated_duration_ms"))
    if milliseconds is None:
        milliseconds = _finite_float(row.get("victim_isolated_ms"))
    if milliseconds is None:
        microseconds = _finite_float(row.get("victim_isolated_us"))
        milliseconds = None if microseconds is None else microseconds / 1_000.0
    if milliseconds is None or milliseconds <= 0.0:
        return None
    return milliseconds


@dataclass(frozen=True)
class AdaptiveBenchmarkCellDecision:
    """Serializable convergence and attempt-budget state for one cell."""

    cell: dict[str, Any]
    cell_key: str
    isolated_duration_ms: float
    total_attempts: int
    valid_samples: int
    invalid_samples: int
    invalid_reason_counts: dict[str, int]
    discarded_after_stop: int
    minimum_valid_samples: int
    max_valid_samples: int
    max_total_attempts: int
    remaining_total_attempts: int
    target_ci_half_width_pp: float
    median_pp: float | None
    ci95_low_pp: float | None
    ci95_high_pp: float | None
    ci95_half_width_pp: float | None
    max_endpoint_distance_pp: float | None
    precision_met: bool
    converged: bool
    explicitly_infeasible: bool
    pre_sampling_status: str
    infeasibility_proof_basis: str | None
    should_continue: bool
    stop_reason: str | None
    recommended_next_attempts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ControllerCell:
    def __init__(
        self,
        key: AdaptiveBenchmarkCellKey,
        isolated_duration_ms: float,
        config: AdaptiveBenchmarkControllerConfig,
    ) -> None:
        self.key = key
        self.config = config
        self.sampling = AdaptiveSamplingCell(
            key.canonical, isolated_duration_ms, config.sampling
        )
        self.total_attempts = 0
        self.invalid_reason_counts: Counter[str] = Counter()
        self.discarded_after_stop = 0
        self.explicitly_infeasible = False
        self.pre_sampling_status = PRE_SAMPLING_READY
        self.infeasibility_proof_basis: str | None = None

    @property
    def max_total_attempts(self) -> int:
        return int(
            math.ceil(
                self.config.sampling.max_valid_samples
                * self.config.max_total_attempt_multiplier
            )
        )

    def add(self, validity: AggregateTrialValidity) -> bool:
        """Add a trial, returning false if it arrived after this cell stopped."""

        # Do not bootstrap after every row. The controller evaluates CI
        # convergence once per ingested batch; only hard limits must be
        # enforced within a batch.
        hard_stopped = (
            self.explicitly_infeasible
            or self.pre_sampling_status == PRE_SAMPLING_CALIBRATION_UNRESOLVED
            or self.total_attempts >= self.max_total_attempts
            or len(self.sampling.valid_samples_pp)
            >= self.config.sampling.max_valid_samples
        )
        if hard_stopped:
            self.discarded_after_stop += 1
            return False
        self.total_attempts += 1
        if validity.valid:
            self.sampling.add_sample(validity.slowdown_pp, valid=True)
        else:
            self.sampling.add_sample(None, valid=False)
            self.invalid_reason_counts.update(validity.reasons)
        return True

    def decision(self) -> AdaptiveBenchmarkCellDecision:
        base = self.sampling.decision()
        exhausted_attempts = self.total_attempts >= self.max_total_attempts
        if self.explicitly_infeasible:
            should_continue = False
            stop_reason = "geometry_target_proven_infeasible"
        elif self.pre_sampling_status == PRE_SAMPLING_CALIBRATION_UNRESOLVED:
            should_continue = False
            stop_reason = "geometry_calibration_unresolved"
        elif base.converged:
            should_continue = False
            stop_reason = base.stop_reason
        elif not base.should_continue:
            should_continue = False
            stop_reason = base.stop_reason
        elif exhausted_attempts:
            should_continue = False
            stop_reason = "max_total_attempts"
        else:
            should_continue = True
            stop_reason = None
        remaining = max(0, self.max_total_attempts - self.total_attempts)
        recommended = (
            min(int(self.config.attempt_batch_size), remaining)
            if should_continue
            else 0
        )
        return AdaptiveBenchmarkCellDecision(
            cell=self.key.to_dict(),
            cell_key=self.key.canonical,
            isolated_duration_ms=self.sampling.isolated_duration_ms,
            total_attempts=self.total_attempts,
            valid_samples=base.valid_samples,
            invalid_samples=base.invalid_samples,
            invalid_reason_counts=dict(sorted(self.invalid_reason_counts.items())),
            discarded_after_stop=self.discarded_after_stop,
            minimum_valid_samples=base.minimum_valid_samples,
            max_valid_samples=base.max_valid_samples,
            max_total_attempts=self.max_total_attempts,
            remaining_total_attempts=remaining,
            target_ci_half_width_pp=base.target_ci_half_width_pp,
            median_pp=base.median_pp,
            ci95_low_pp=base.ci95_low_pp,
            ci95_high_pp=base.ci95_high_pp,
            ci95_half_width_pp=base.ci95_half_width_pp,
            max_endpoint_distance_pp=base.max_endpoint_distance_pp,
            precision_met=base.precision_met,
            converged=base.converged,
            explicitly_infeasible=self.explicitly_infeasible,
            pre_sampling_status=self.pre_sampling_status,
            infeasibility_proof_basis=self.infeasibility_proof_basis,
            should_continue=should_continue,
            stop_reason=stop_reason,
            recommended_next_attempts=recommended,
        )


class AdaptiveBenchmarkController:
    """Incremental CPU-side controller over aggregate benchmark rows."""

    def __init__(
        self, config: AdaptiveBenchmarkControllerConfig | None = None
    ) -> None:
        self.config = config or AdaptiveBenchmarkControllerConfig()
        self._cells: dict[AdaptiveBenchmarkCellKey, _ControllerCell] = {}

    def register_cell(
        self, key: AdaptiveBenchmarkCellKey, isolated_duration_ms: float
    ) -> None:
        duration = float(isolated_duration_ms)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("isolated_duration_ms must be finite and positive")
        if key in self._cells:
            existing = self._cells[key].sampling.isolated_duration_ms
            if not math.isclose(existing, duration, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"cell {key.canonical} already registered with duration {existing}"
                )
            return
        self._cells[key] = _ControllerCell(key, duration, self.config)

    def mark_cell_infeasible(
        self,
        key: AdaptiveBenchmarkCellKey,
        *,
        proof_basis: str,
    ) -> None:
        """Stop a registered cell whose calibrated geometry is unattainable.

        Geometry calibration is not a retained treatment attempt, so this
        transition deliberately leaves both attempt and invalid-sample counts
        at zero while preserving an explicit terminal reason.
        """

        if key not in self._cells:
            raise KeyError(f"cell {key.canonical} is not registered")
        if proof_basis != INFEASIBILITY_PROOF_DURATION_BOUND:
            raise ValueError(
                "only a structural duration bound currently proves geometry "
                "infeasibility"
            )
        cell = self._cells[key]
        cell.explicitly_infeasible = True
        cell.pre_sampling_status = PRE_SAMPLING_PROVEN_INFEASIBLE
        cell.infeasibility_proof_basis = proof_basis
        cell.invalid_reason_counts["geometry_target_proven_infeasible"] += 1

    def mark_cell_calibration_unresolved(
        self, key: AdaptiveBenchmarkCellKey
    ) -> None:
        """Terminate an unsampled cell without granting infeasibility exemption."""

        if key not in self._cells:
            raise KeyError(f"cell {key.canonical} is not registered")
        cell = self._cells[key]
        if cell.explicitly_infeasible:
            raise ValueError(
                f"cell {key.canonical} is already proven infeasible"
            )
        cell.pre_sampling_status = PRE_SAMPLING_CALIBRATION_UNRESOLVED
        cell.invalid_reason_counts["geometry_calibration_unresolved"] += 1

    def ingest_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        default_correctness: bool | None = None,
    ) -> dict[str, int]:
        """Group and ingest aggregate rows, returning ingestion counters.

        New cells use the median isolated victim duration in their first batch.
        A runner can avoid any first-batch duration ambiguity by calling
        :meth:`register_cell` with its isolated calibration estimate.
        """

        grouped: dict[AdaptiveBenchmarkCellKey, list[Mapping[str, Any]]] = {}
        for row in rows:
            key = AdaptiveBenchmarkCellKey.from_row(row)
            grouped.setdefault(key, []).append(row)

        for key, members in grouped.items():
            if key in self._cells:
                continue
            durations = [
                duration
                for duration in (_isolated_duration_ms(row) for row in members)
                if duration is not None
            ]
            if not durations:
                raise ValueError(
                    f"cell {key.canonical} has no positive isolated victim duration"
                )
            self.register_cell(key, statistics.median(durations))

        counters = Counter(
            {"rows_seen": 0, "rows_ingested": 0, "rows_discarded_after_stop": 0}
        )
        for key, members in grouped.items():
            cell = self._cells[key]
            # One stopping decision per batch keeps bootstrap cost bounded.
            # A GPU runner should call this method after each issued batch.
            if not cell.decision().should_continue:
                discarded = len(members)
                cell.discarded_after_stop += discarded
                counters["rows_seen"] += discarded
                counters["rows_discarded_after_stop"] += discarded
                continue
            for row in members:
                counters["rows_seen"] += 1
                validity = classify_aggregate_trial(
                    row, self.config, default_correctness=default_correctness
                )
                if cell.add(validity):
                    counters["rows_ingested"] += 1
                else:
                    counters["rows_discarded_after_stop"] += 1
        return dict(counters)

    def decisions(self) -> list[AdaptiveBenchmarkCellDecision]:
        return [
            self._cells[key].decision()
            for key in sorted(self._cells)
        ]

    def cells_needing_samples(self) -> list[AdaptiveBenchmarkCellKey]:
        return [
            key for key in sorted(self._cells)
            if self._cells[key].decision().should_continue
        ]

    def next_batch_plan(self) -> dict[AdaptiveBenchmarkCellKey, int]:
        return {
            key: self._cells[key].decision().recommended_next_attempts
            for key in self.cells_needing_samples()
        }

    @property
    def all_stopped(self) -> bool:
        return bool(self._cells) and not self.cells_needing_samples()

    def diagnostics(self) -> dict[str, Any]:
        decisions = self.decisions()
        return {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "summary": {
                "cell_count": len(decisions),
                "active_cell_count": sum(row.should_continue for row in decisions),
                "converged_cell_count": sum(row.converged for row in decisions),
                "stopped_without_convergence_count": sum(
                    (not row.should_continue and not row.converged)
                    for row in decisions
                ),
                "total_attempts": sum(row.total_attempts for row in decisions),
                "valid_samples": sum(row.valid_samples for row in decisions),
                "invalid_samples": sum(row.invalid_samples for row in decisions),
                "discarded_after_stop": sum(
                    row.discarded_after_stop for row in decisions
                ),
                "all_stopped": self.all_stopped,
            },
            "cells": [row.to_dict() for row in decisions],
        }

    def serialize_diagnostics(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.diagnostics(), indent=indent, sort_keys=True, allow_nan=False
        )


__all__ = [
    "AdaptiveBenchmarkCellDecision",
    "AdaptiveBenchmarkCellKey",
    "AdaptiveBenchmarkController",
    "AdaptiveBenchmarkControllerConfig",
    "AggregateTrialValidity",
    "DEFAULT_ATTEMPT_BATCH_SIZE",
    "DEFAULT_DRIFT_THRESHOLD_PCT",
    "DEFAULT_GEOMETRY_TOLERANCE_PCT",
    "DEFAULT_MAX_TOTAL_ATTEMPT_MULTIPLIER",
    "INFEASIBILITY_PROOF_DURATION_BOUND",
    "PRE_SAMPLING_CALIBRATION_UNRESOLVED",
    "PRE_SAMPLING_PROVEN_INFEASIBLE",
    "PRE_SAMPLING_READY",
    "classify_aggregate_trial",
]
